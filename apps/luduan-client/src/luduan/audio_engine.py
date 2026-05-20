"""Lemonade-backed TTS engine for Luduan.

Replaces the original ``audio_engine.py`` which loaded Qwen3-TTS
directly.  All speech synthesis now goes through Lemonade's
``POST /v1/audio/speech`` endpoint.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from pathlib import Path

from xian.lemonade_client import LemonadeClient

logger = logging.getLogger(__name__)


def concatenate_wavs(wav_datas: list[bytes]) -> bytes:
    """Concatenate multiple WAV byte chunks into a single valid WAV byte chunk.

    Uses Python's built-in wave module to correctly merge headers and audio data.
    """
    if not wav_datas:
        return b""
    if len(wav_datas) == 1:
        return wav_datas[0]

    try:
        # Read parameters from the first WAV chunk
        first_io = io.BytesIO(wav_datas[0])
        with wave.open(first_io, "rb") as w_in:
            params = w_in.getparams()

        out_io = io.BytesIO()
        with wave.open(out_io, "wb") as w_out:
            w_out.setparams(params)
            for data in wav_datas:
                data_io = io.BytesIO(data)
                with wave.open(data_io, "rb") as w_in:
                    # Check if audio parameters match
                    if w_in.getparams()[:3] != params[:3]:
                        logger.warning("WAV concatenation parameter mismatch: %s vs %s", w_in.getparams(), params)
                    w_out.writeframes(w_in.readframes(w_in.getnframes()))

        return out_io.getvalue()
    except Exception as e:
        logger.error("Failed to concatenate WAV files using wave module: %s. Falling back to header slicing.", e)
        # Fallback to manual raw concatenation by stripping headers
        # Standard PCM WAV header is 44 bytes.
        parts = [wav_datas[0]]
        for data in wav_datas[1:]:
            if len(data) > 44:
                parts.append(data[44:])
            else:
                parts.append(data)
        # Adjust total file size and data chunk size in the first header
        combined = b"".join(parts)
        if len(combined) >= 44:
            # RIFF chunk size at offset 4 (total size - 8)
            total_size = len(combined) - 8
            # data subchunk size at offset 40 (total size - 44)
            data_size = len(combined) - 44

            combined_bytes = bytearray(combined)
            combined_bytes[4:8] = total_size.to_bytes(4, byteorder="little")
            combined_bytes[40:44] = data_size.to_bytes(4, byteorder="little")
            return bytes(combined_bytes)
        return combined


class AudioEngine:
    """Generates speech audio for translated passages via Lemonade TTS.

    Parameters
    ----------
    voice:
        TTS voice identifier (e.g. ``"af_heart"``).
    """

    def __init__(self, voice: str = "af_heart") -> None:
        self._voice = voice
        import os
        from shared_types.constants import DEFAULT_API_URL
        base_url = os.environ.get("LEMONADE_API_URL", DEFAULT_API_URL)
        base_url_no_v1 = base_url.removesuffix("/v1")
        self._client = LemonadeClient(base_url=base_url_no_v1)

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
