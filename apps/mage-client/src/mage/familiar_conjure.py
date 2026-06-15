# MAGE — Gaming HUD for real-time screen translation.
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

"""Conjure a custom familiar from a natural-language prompt via the local
Lemonade LLM.

The model's only job is to emit a JSON "art recipe" (see
:mod:`mage.familiar_recipe`). Because every recipe field is enum-bounded and
``coerce_recipe`` snaps anything invalid to a safe default, even a sloppy model
reply yields a renderable familiar. We additionally extract the first JSON
object from the reply and retry once with the error fed back, so transient
formatting slips don't surface to the user.

Runs on a ``QThread`` and dispatches the actual call onto the processor's async
engine — the same path the translator uses — so the GUI never blocks.
"""

import asyncio
import json
import logging
import re

from PyQt6.QtCore import QThread, pyqtSignal

from mage.familiar_recipe import coerce_recipe, schema_hint

logger = logging.getLogger(__name__)

CONJURE_TIMEOUT_S = 60
CONJURE_MAX_TOKENS = 400

SYSTEM_PROMPT = (
    "You are a sprite designer for a small desktop mascot (a 'familiar'). "
    "Given a short description, respond with ONLY a single JSON object describing "
    "the creature — no markdown, no backticks, no commentary. "
    "Choose every value ONLY from the allowed lists below. Pick the `transit` "
    "that best fits: winged or floating creatures use \"fly\", small or magical "
    "ones use \"teleport\", cats/lizards/climbers use \"climb\". Use a small, "
    "harmonious palette of #rrggbb hex colors that read well at a tiny size.\n\n"
    + schema_hint()
    + "\nReturn exactly this shape:\n"
    '{"name": str, "palette": {"primary": hex, "secondary": hex, "accent": hex, '
    '"skin": hex}, "body": str, "head": str, "ears": str, "eyes": str, '
    '"features": [str], "headwear": str, "accessory": str, "transit": str, '
    '"float_down": bool, "glow_color": hex}'
)

_RETRY_NUDGE = (
    "That was not a single valid JSON object matching the schema. "
    "Reply with ONLY the JSON object — no prose, no backticks."
)


def _extract_json_object(text: str):
    """Return the first balanced ``{...}`` parsed from *text*, or ``None``."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    # Tolerate trailing commas, a common small-model slip.
                    try:
                        return json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
                    except Exception:
                        return None
    return None


def parse_recipe(text: str):
    """Extract + coerce a recipe from raw model output. ``None`` if no JSON."""
    obj = _extract_json_object(text or "")
    return coerce_recipe(obj) if obj is not None else None


class ConjureWorker(QThread):
    """Turn a text prompt into a validated art recipe, off the GUI thread."""

    conjured = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, processor, prompt: str, retries: int = 1):
        super().__init__()
        self.processor = processor
        self.prompt = prompt
        self.retries = retries

    async def _complete(self, messages) -> str:
        response = await asyncio.wait_for(
            self.processor.client.chat.completions.create(
                model=self.processor.get_model_name(),
                messages=messages,
                max_tokens=CONJURE_MAX_TOKENS,
                temperature=0.7,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=CONJURE_TIMEOUT_S,
        )
        choice = response.choices[0] if response.choices else None
        return (choice.message.content or "").strip() if choice else ""

    async def _run_async(self) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Description: {self.prompt}"},
        ]
        last = ""
        for _ in range(self.retries + 1):
            out = await self._complete(messages)
            recipe = parse_recipe(out)
            if recipe is not None:
                logger.info("Conjured familiar '%s' from prompt: %s",
                            recipe.get("name"), self.prompt)
                return recipe
            last = out
            messages.append({"role": "assistant", "content": out})
            messages.append({"role": "user", "content": _RETRY_NUDGE})
        raise ValueError("Model did not return a valid recipe. Last output: "
                         + (last[:200] or "<empty>"))

    def run(self):
        try:
            future = self.processor.engine.submit(self._run_async())
            recipe = future.result(timeout=CONJURE_TIMEOUT_S * (self.retries + 1) + 5)
            self.conjured.emit(recipe)
        except Exception as e:
            logger.error("ConjureWorker error: %s", e)
            self.error.emit(str(e))
