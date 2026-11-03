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

"""Line translation through Hy-MT2.

The OCR pipeline's other half.  Once a local reader supplies text and boxes,
the network is only asked to translate — no geometry, no JSON schema, no
instruction-following — which is a far lower bar and is why a 1.8B model is
enough for it.

Hy-MT2 is a machine-translation model with its own published instruction
formats, and it gets those verbatim rather than something improvised.  Three of
them are used here:

* **Default** — target language, and an explicit "only output the translated
  result", because anything else it says lands on the screen as a translation.
* **Terminology** — reference pairs before the request, which is how a glossary
  reaches an MT model.  MAGE's wiki glossary is keyed by the source-language
  name and valued by the canonical target-language one, which is exactly the
  shape this template wants.
* **Style** — used when the user has configured a translation style.

The source language is deliberately *not* named: the published default format
does not carry one, and the model detects it.

Everything goes in a single user turn.  There is no system prompt, because an
MT model translates instructions rather than following them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from xian.translation_cache import TranslationCache

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONCURRENCY",
    "LineTranslator",
    "TRANSLATION_MODEL",
    "TRANSLATION_MODELS",
    "NPU_TRANSLATION_MODEL",
    "TranslationModelUnavailable",
    "build_translation_prompt",
    "install_hint",
    "is_supported_translation_model",
    "relevant_terms",
    "translation_pull_body",
]

#: The line translator, pinned to Hy-MT2.  A specific model rather than a
#: routed one: the prompt shapes above are Hy-MT2's own published formats, the
#: absence of a system turn is a property of machine-translation models, and
#: silently running any of it against whatever the router happened to pick
#: produces quality problems that read as OCR problems.
TRANSLATION_MODEL = "Hy-MT2-1.8B-GGUF-Q4_K_M"

#: The 1.8B again, built for the AMD NPU by FastFlowLM.  Slower per line than
#: the GPU build, but it leaves the GPU to the game, which is the point: on an
#: APU the game and the translator otherwise share one GPU.  Not in Lemonade's
#: catalogue, so it is registered as a user model; see
#: :func:`translation_pull_body`.
NPU_TRANSLATION_MODEL = "user.Hy-MT2-Flash-1.8B"
NPU_TRANSLATION_CHECKPOINT = "hy-mt2-flash:1.8b"

#: The models on offer, GPU sizes fastest first, then the NPU build.  Every
#: text translation in the app uses one of these; nothing else is offered,
#: because the pipeline is built around them.
TRANSLATION_MODELS = (
    "Hy-MT2-1.8B-GGUF-Q4_K_M",
    "Hy-MT2-7B-GGUF-Q4_K_M",
    NPU_TRANSLATION_MODEL,
)

#: Models served one request at a time.  The NPU runs requests in turn, so the
#: per-line fan-out only queues them, and the queueing costs: eight lines took
#: 4.6 s sent together against 2.5 s sent one after another.  The batched path
#: is no help either — this build copies the delimiter instruction into its
#: first line.
SERIAL_TRANSLATION_MODELS = frozenset({NPU_TRANSLATION_MODEL})


def is_supported_translation_model(model_id: str | None) -> bool:
    """True for a model this pipeline is willing to translate with."""
    return model_id in TRANSLATION_MODELS


def translation_pull_body(model_id: str) -> dict | None:
    """The ``POST /v1/pull`` body that registers a model the server lacks.

    ``None`` for the catalogue models, which the server pulls by name.
    """
    if model_id != NPU_TRANSLATION_MODEL:
        return None
    return {
        "model_name": NPU_TRANSLATION_MODEL,
        "recipe": "flm",
        "checkpoints": {"main": NPU_TRANSLATION_CHECKPOINT},
        "labels": [],
        "recipe_options": {},
    }


def install_hint(model_id: str) -> str:
    """The command that installs a translation model by hand."""
    if model_id == NPU_TRANSLATION_MODEL:
        return (
            f"lemonade pull {NPU_TRANSLATION_MODEL} --recipe flm "
            f"--checkpoint main {NPU_TRANSLATION_CHECKPOINT}"
        )
    return f"lemonade pull {model_id}"

#: In-flight requests.  Local serving may or may not parallelise; see
#: ``batched`` below for what happens when it does not.
DEFAULT_CONCURRENCY = 8

#: Separator for the batched fallback path.  Long and punctuation-heavy so no
#: translator rewrites it and no source text contains it.
BATCH_SEPARATOR = "\n##|||##\n"

#: Hy-MT2's published default instruction format.
DEFAULT_TEMPLATE = (
    "Translate the following text into {target_lang}. Note that you should only output "
    "the translated result without any additional explanation:\n\n{source_text}"
)

#: Its style variant, used when a translation style is configured.
STYLE_TEMPLATE = (
    "Please translate the following text into {target_lang}. Note that the translation "
    "style must strictly conform to [{target_style}]:\n\n{source_text}"
)

#: Its terminology variant: reference pairs, then the request.
TERMINOLOGY_PREFIX = "Reference the following translations:\n{pairs}\n\n"

#: Its delimiter variant, for the batched path, where losing one separator
#: misaligns every line after it.
DELIMITER_TEMPLATE = (
    "Please accurately translate the following text into {target_lang}.\n"
    "You must retain the exact same number of delimiters in the translation. Strictly do "
    "not omit, escape, or translate these symbols, and pay close attention to their "
    "placement.\n\n{source_text}"
)

#: Most glossary terms are irrelevant to any given line, and a wiki with five
#: hundred entries would otherwise put all of them in front of every request.
MAX_TERMS_PER_REQUEST = 12


class TranslationModelUnavailable(RuntimeError):
    """The pinned translation model is not installed on the server."""


def relevant_terms(text: str, glossary: dict[str, str] | None) -> dict[str, str]:
    """The glossary entries this particular line actually needs.

    Filtered by presence in the source text: an MT model given a hundred
    irrelevant reference pairs spends the whole request on them, and the terms
    that mattered are no more prominent than the ones that did not.  Longest
    first, so a two-word term outranks a one-word term it contains.
    """
    if not glossary or not text:
        return {}
    found = {
        source: target
        for source, target in glossary.items()
        if source and target and source in text
    }
    ordered = sorted(found.items(), key=lambda item: -len(item[0]))
    return dict(ordered[:MAX_TERMS_PER_REQUEST])


def build_translation_prompt(
    text: str,
    target_lang: str,
    *,
    glossary: dict[str, str] | None = None,
    style: str | None = None,
    delimited: bool = False,
) -> str:
    """One user turn, in Hy-MT2's own published instruction format.

    The source language is not named: the published default format does not
    carry one.
    """
    if delimited:
        body = DELIMITER_TEMPLATE.format(target_lang=target_lang, source_text=text)
    elif style:
        body = STYLE_TEMPLATE.format(target_lang=target_lang, target_style=style, source_text=text)
    else:
        body = DEFAULT_TEMPLATE.format(target_lang=target_lang, source_text=text)

    terms = relevant_terms(text, glossary)
    if terms:
        pairs = "\n".join(f"{source} translates to {target}" for source, target in terms.items())
        return TERMINOLOGY_PREFIX.format(pairs=pairs) + body
    return body


@dataclass
class LineTranslator:
    """Translates recognized lines, cache-first, one request per line.

    Holds the cache, so a translator that lives as long as a live session
    accumulates hits across it.
    """

    processor: object
    model: str = TRANSLATION_MODEL
    concurrency: int = DEFAULT_CONCURRENCY
    #: A translation style, from the user's settings, rendered through
    #: Hy-MT2's style template.
    style: str | None = None
    #: Fall back to the router's choice when the pin is missing.  Off by
    #: default: a quiet substitution is how a user ends up debugging
    #: translation quality that has nothing to do with the model they think
    #: they are running.
    allow_fallback: bool = False
    #: One request carrying every line, split on a separator.  Kept because
    #: the fan-out's advantage depends on the server actually serving
    #: concurrently, which a single-slot backend does not; see the plan's
    #: risk 4.  Off by default.
    batched: bool = False
    cache: TranslationCache = field(default_factory=TranslationCache)
    glossary: dict[str, str] = field(default_factory=dict)
    max_tokens: int = 512

    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _resolved: str | None = field(default=None, init=False, repr=False)

    # ── model resolution ─────────────────────────────────────────────

    def resolve_model(self) -> str:
        """The model id to send to, or an error that says what to install."""
        if self._resolved is not None:
            return self._resolved

        if not is_supported_translation_model(self.model):
            # A stale setting, or a caller passing the chat model through.
            # Falling back quietly is how the whole pipeline ends up running
            # on a model none of its prompts were written for.
            logger.warning("%s is not a supported translation model; using %s",
                           self.model, TRANSLATION_MODEL)
            self.model = TRANSLATION_MODEL

        router = getattr(self.processor, "router", None)
        installed = list(getattr(router, "downloaded_model_ids", []) or []) if router else []

        if not installed or self.model in installed:
            # An empty list means discovery has not run, not that the model is
            # absent; failing there would break a first tick that would have
            # worked.
            self._resolved = self.model
            return self._resolved

        if self.allow_fallback and hasattr(self.processor, "get_translation_model_name"):
            fallback = self.processor.get_translation_model_name()
            logger.warning("%s is not installed; falling back to %s", self.model, fallback)
            self._resolved = fallback
            return self._resolved

        raise TranslationModelUnavailable(
            f"{self.model} is not installed on the server. "
            f"Install it from Settings, or run: {install_hint(self.model)}"
        )

    # ── requests ─────────────────────────────────────────────────────

    async def _complete(self, prompt: str) -> str:
        client = self.processor.client
        response = await client.chat.completions.create(
            model=self.resolve_model(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    async def _translate_one(self, text: str, target_lang: str) -> str:
        prompt = build_translation_prompt(
            text, target_lang, glossary=self.glossary, style=self.style
        )
        async with self._semaphore:
            try:
                raw = await self._complete(prompt)
            except TranslationModelUnavailable:
                # A configuration problem, not a transient one: it is the same
                # for every line and the user has to act on it.
                raise
            except Exception as exc:
                logger.warning("translation failed for %r: %s", text[:40], exc)
                return text  # the source is a better overlay than a blank one
        return raw or text

    async def _translate_batched(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """One request for every line, split back by separator.

        Index-mapped, so a dropped separator misaligns everything after it —
        which is why the length is checked and the whole batch falls back to
        the source rather than painting shifted translations.
        """
        joined = BATCH_SEPARATOR.join(texts)
        prompt = build_translation_prompt(
            joined, target_lang, glossary=self.glossary, delimited=True
        )
        try:
            raw = await self._complete(prompt)
        except Exception as exc:
            logger.warning("batched translation failed: %s", exc)
            return list(texts)

        parts = [part.strip() for part in raw.split(BATCH_SEPARATOR.strip())]
        if len(parts) != len(texts):
            logger.warning("batched translation returned %d parts for %d lines", len(parts), len(texts))
            return list(texts)
        return [part or original for part, original in zip(parts, texts)]

    async def translate_lines(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        """Translate every line, returning one result per input in order."""
        if not texts:
            return []

        results: list[str | None] = [None] * len(texts)
        pending: list[int] = []
        for index, text in enumerate(texts):
            if not text.strip():
                results[index] = text
                continue
            cached = self.cache.get(text, source_lang, target_lang)
            if cached is not None:
                results[index] = cached
            else:
                pending.append(index)

        if pending:
            # Up front, and outside the per-line error handling: a missing
            # model is the same answer for every line, and letting it fall
            # through the fan-out's fallback turns a fixable configuration
            # error into a screen of untranslated text with a warning in a log
            # nobody is reading.
            self.resolve_model()

            if self._semaphore is None:
                serial = self.resolve_model() in SERIAL_TRANSLATION_MODELS
                self._semaphore = asyncio.Semaphore(1 if serial else self.concurrency)

            missing = [texts[index] for index in pending]
            if self.batched:
                translated = await self._translate_batched(missing, source_lang, target_lang)
            else:
                translated = await asyncio.gather(
                    *(self._translate_one(text, target_lang) for text in missing)
                )

            for index, translation in zip(pending, translated):
                results[index] = translation
                if translation != texts[index]:
                    # A result identical to its source is either an untranslated
                    # failure or a proper noun; caching it would make a failure
                    # permanent for the rest of the session.
                    self.cache.put(texts[index], source_lang, target_lang, translation)

        return [result if result is not None else text for result, text in zip(results, texts)]
