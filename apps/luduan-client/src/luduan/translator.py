"""Lemonade-backed translation engine for Luduan.

Replaces the original Luduan ``translator.py`` which loaded Qwen
directly onto the GPU.  All inference now goes through Lemonade
Server's ``/v1/chat/completions`` endpoint via the xian-vl pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from openai import AsyncOpenAI

from shared_types.constants import DEFAULT_API_URL

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
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        model: str = "",
        source_lang: str = "Chinese",
        target_lang: str = "English",
    ) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
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
            )
            return response.choices[0].message.content or text
        except Exception:
            logger.exception("Translation failed for passage (len=%d)", len(text))
            return text

    async def translate_batch(self, passages: list[str]) -> list[str]:
        """Translate multiple passages concurrently."""
        tasks = [self.translate(p) for p in passages]
        return await asyncio.gather(*tasks)
