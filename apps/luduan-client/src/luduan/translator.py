# Luduan — EPUB to RoboBook narration CLI.
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

"""Lemonade-backed translation engine for Luduan.

Replaces the original Luduan ``translator.py`` which loaded Qwen
directly onto the GPU.  All inference now goes through Lemonade
Server's ``/v1/chat/completions`` endpoint via the xian-vl pipeline.
"""

import asyncio
import logging

from openai import AsyncOpenAI

from shared_types.constants import DEFAULT_API_URL, DEFAULT_MODEL
from xian.omni_router import OmniModelRouter

logger = logging.getLogger(__name__)


class DocumentTranslator:
    """Translates document passages via Lemonade's chat completions API.

    Parameters
    ----------
    base_url:
        Lemonade API base URL.
    model:
        Model name to use for translation.
    source_lang:
        Source language for the system prompt.
    target_lang:
        Target language for the system prompt.
    router:
        Active model router.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        source_lang: str = "Chinese",
        target_lang: str = "English",
        router: OmniModelRouter | None = None,
    ) -> None:
        import os
        api_url = os.environ.get("LEMONADE_API_URL", base_url)
        if router:
            api_url = router.api_url
            model = router.llm()
        self._client = AsyncOpenAI(base_url=api_url, api_key="not-needed")
        self._model = model
        self._source = source_lang
        self._target = target_lang

    async def translate(self, text: str) -> str:
        """Translate a single passage.

        Returns the translated text, or the original on failure.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are an expert {self._source}-to-{self._target} translator "
                            f"specialising in web novels (wuxia, xianxia, xuanhuan). "
                            f"Translate the following passage naturally. Preserve proper nouns. "
                            f"Output ONLY the translation."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or text
        except Exception:
            logger.exception("Translation failed for passage (len=%d)", len(text))
            return text

    async def translate_batch(
        self,
        passages: list[str],
        *,
        concurrency: int = 8,
    ) -> list[str]:
        """Translate multiple passages with a bounded concurrency limit."""
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(p: str) -> str:
            async with sem:
                return await self.translate(p)

        return await asyncio.gather(*(_one(p) for p in passages))
