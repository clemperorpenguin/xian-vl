# Xian-VL — Core Vision-Language orchestration engine.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

"""Translate recognized lines in a single text call.

Once OCR has the text, translation is a text problem — which means it can go to
the NPU (see :mod:`xian.omni_router`) and leave the GPU alone.  All lines go in
one request so the model sees them as one screen and keeps pronouns and
terminology consistent between them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from xian.timeout import CHAT_AUX_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

__all__ = ["apply_glossary", "batch_translate", "parse_translations"]

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Concurrent requests on the dedicated-MT path. One line per request keeps the
#: output aligned with the input; a handful in flight keeps the NPU busy without
#: queueing the whole screen behind one slow line.
_MT_CONCURRENCY = 4


def _system_prompt(source_lang: str, target_lang: str, glossary: dict[str, str] | None) -> str:
    prompt = (
        f"You translate {source_lang} text from a video game screen into {target_lang}.\n"
        "The user sends a JSON object mapping line numbers to source text.\n"
        'Reply with ONLY a JSON object of the same keys mapping to translations: {"1": "...", "2": "..."}.\n'
        "Translate every key. Keep translations about as short as the original so they fit "
        "the on-screen space they came from. The lines are from one screen, so keep names "
        "and terminology consistent between them.\n"
        "No markdown, no commentary, no explanation."
    )
    if glossary:
        lines = "\n".join(f"- {src}: {dst}" for src, dst in glossary.items())
        prompt += f"\n\nUse this terminology:\n{lines}"
    return prompt


def parse_translations(raw: str, count: int) -> list[str]:
    """Parse the model's numbered-JSON reply into an ordered list.

    Missing or unparseable entries come back as empty strings so callers can
    fall back to the original text rather than dropping the line.
    """
    out = [""] * count
    if not raw:
        return out
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]

    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return out
    try:
        mapping = json.loads(match.group(0))
    except (ValueError, TypeError):
        logger.debug("Batch translation reply was not valid JSON: %.200s", raw)
        return out
    if not isinstance(mapping, dict):
        return out

    for key, value in mapping.items():
        try:
            index = int(str(key).strip()) - 1
        except ValueError:
            continue
        if 0 <= index < count and isinstance(value, str):
            out[index] = value.strip()
    return out


def apply_glossary(text: str, glossary: dict[str, str] | None) -> str:
    """Substitute known terms into the *source* text before translating.

    The instruction-following path passes the glossary as prompt text.  A
    dedicated MT model has no instructions to follow, so the only lever is the
    input itself: replacing 名劍山莊 with "Mingjian Manor" before the call gets
    the established rendering through, because MT models pass unknown Latin
    tokens along rather than re-translating them.  Longest term first, so a term
    that contains a shorter one is not clobbered by it.
    """
    if not glossary:
        return text
    for src in sorted(glossary, key=len, reverse=True):
        if src and src in text:
            text = text.replace(src, glossary[src])
    return text


async def _translate_one(processor, model: str, text: str, source_lang: str, target_lang: str) -> str:
    """Translate a single line with a bare prompt.  Returns "" on failure."""
    try:
        response = await asyncio.wait_for(
            processor.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": f"Translate the following {source_lang} text to {target_lang}.\n\n{text}",
                }],
                max_tokens=256,
                temperature=0.0,
            ),
            timeout=CHAT_AUX_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Translation timed out for a line of %d chars", len(text))
        return ""
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return ""

    choice = response.choices[0] if response.choices else None
    return ((choice.message.content or "") if choice else "").strip()


async def _batch_translate_mt(
    processor,
    model: str,
    texts: list[str],
    source_lang: str,
    target_lang: str,
    glossary: dict[str, str] | None,
) -> list[str]:
    """Translate via a dedicated MT model: one line per request, in parallel.

    A translation-only model has no way to return several lines keyed by number
    — asked to, it translates the request.  One line per call costs more
    requests but keeps the output aligned with the input by construction, and
    these models are small enough (and NPU-resident often enough) that the
    round-trips overlap into roughly the same wall time.
    """
    semaphore = asyncio.Semaphore(_MT_CONCURRENCY)

    async def run(text: str) -> str:
        async with semaphore:
            return await _translate_one(
                processor, model, apply_glossary(text, glossary), source_lang, target_lang
            )

    results = await asyncio.gather(*(run(text) for text in texts), return_exceptions=True)
    return [
        original if isinstance(result, BaseException) or not result else result
        for result, original in zip(results, texts)
    ]


async def batch_translate(
    processor,
    texts: list[str],
    source_lang: str,
    target_lang: str,
    *,
    glossary: dict[str, str] | None = None,
) -> list[str]:
    """Translate every line in one call, preserving order.

    Any line the model fails to return falls back to its source text: showing
    the original is far better than showing a blank box over it.
    """
    if not texts:
        return []
    if not processor.client:
        return list(texts)

    model = processor.get_translation_model_name()
    if processor.router.is_translation_only(model):
        return await _batch_translate_mt(processor, model, texts, source_lang, target_lang, glossary)

    payload = json.dumps({str(i + 1): text for i, text in enumerate(texts)}, ensure_ascii=False)

    try:
        response = await asyncio.wait_for(
            processor.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _system_prompt(source_lang, target_lang, glossary)},
                    {"role": "user", "content": payload},
                ],
                max_tokens=max(256, 64 * len(texts)),
                temperature=0.1,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=CHAT_AUX_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Batch translation timed out for %d lines", len(texts))
        return list(texts)
    except Exception as exc:
        logger.warning("Batch translation failed: %s", exc)
        return list(texts)

    choice = response.choices[0] if response.choices else None
    content = (choice.message.content or "") if choice else ""
    translations = parse_translations(content, len(texts))

    return [translated or original for translated, original in zip(translations, texts)]
