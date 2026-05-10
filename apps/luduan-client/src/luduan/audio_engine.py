"""Lemonade-backed TTS engine for Luduan.

Replaces the original ``audio_engine.py`` which loaded Qwen3-TTS
directly.  All speech synthesis now goes through Lemonade's
``POST /v1/audio/speech`` endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from xian.lemonade_client import LemonadeClient

logger = logging.getLogger(__name__)


class AudioEngine:
    """Generates speech audio for translated passages via Lemonade TTS.

    Parameters
    ----------
    voice:
        TTS voice identifier (e.g. ``"af_heart"``).
    """

    def __init__(self, voice: str = "af_heart") -> None:
        self._voice = voice
        self._client = LemonadeClient()

    async def close(self) -> None:
        await self._client.close()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to WAV audio bytes via Lemonade TTS."""
        logger.info("Synthesizing %d chars → TTS …", len(text))
        return await self._client.tts(text, voice=self._voice)

    async def synthesize_to_file(self, text: str, output_path: Path) -> int:
        """Synthesize and write to disk.  Returns file size in bytes."""
        audio_bytes = await self.synthesize(text)
        output_path.write_bytes(audio_bytes)
        logger.info("Wrote %s (%d bytes)", output_path.name, len(audio_bytes))
        return len(audio_bytes)
