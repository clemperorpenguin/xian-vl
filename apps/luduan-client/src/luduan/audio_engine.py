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

"""Lemonade-backed TTS engine for Luduan.

Replaces the original ``audio_engine.py`` which loaded Qwen3-TTS
directly.  All speech synthesis now goes through Lemonade's
``POST /v1/audio/speech`` endpoint.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from pathlib import Path

from xian.lemonade_client import LemonadeClient
from xian.omni_router import OmniModelRouter

logger = logging.getLogger(__name__)

# soundfile (bundling libsndfile) decodes the 32-bit *float* WAV that Lemonade's
# kokoro TTS returns — which Python's stdlib ``wave`` module rejects ("unknown
# format: 3") — and concatenates without trusting the streamed 0xFFFFFFFF data
# size in the header. Falls back to the stdlib path if it is unavailable.
try:
    import numpy as np
    import soundfile as sf

    _HAVE_SF = True
except Exception:  # pragma: no cover - optional accel path
    _HAVE_SF = False


def get_wav_duration_ms(wav_bytes: bytes) -> float:
    """Return the duration of a WAV chunk in milliseconds."""
    if _HAVE_SF:
        try:
            info = sf.info(io.BytesIO(wav_bytes))
            if info.samplerate > 0:
                return (info.frames / info.samplerate) * 1000.0
        except Exception as e:
            logger.debug("soundfile duration probe failed, using header math: %s", e)

    if len(wav_bytes) > 44:
        try:
            # Stdlib fallback. WAV header fields:
            # Channels at bytes 22-23, sample rate at 24-27, bits at 34-35.
            channels = struct.unpack("<H", wav_bytes[22:24])[0]
            sample_rate = struct.unpack("<I", wav_bytes[24:28])[0]
            bits_per_sample = struct.unpack("<H", wav_bytes[34:36])[0]

            bytes_per_second = sample_rate * channels * (bits_per_sample // 8)
            data_size = len(wav_bytes) - 44
            if bytes_per_second > 0:
                return (data_size / bytes_per_second) * 1000.0
        except Exception as e:
            logger.error("Failed to parse WAV header: %s", e)
    return 0.0


def concatenate_wavs(wav_datas: list[bytes]) -> bytes:
    """Concatenate multiple WAV byte chunks into a single valid WAV byte chunk."""
    if not wav_datas:
        return b""
    if len(wav_datas) == 1:
        return wav_datas[0]

    if _HAVE_SF:
        try:
            blocks = []
            sample_rate = None
            for data in wav_datas:
                samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    logger.warning("WAV sample-rate mismatch: %s vs %s", sr, sample_rate)
                blocks.append(samples)
            merged = np.concatenate(blocks)
            out_io = io.BytesIO()
            # Emit standard 16-bit PCM so every downstream consumer can read it.
            sf.write(out_io, merged, int(sample_rate or 24000), format="WAV", subtype="PCM_16")
            return out_io.getvalue()
        except Exception as e:
            logger.warning("soundfile concatenation failed (%s); falling back to stdlib.", e)

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
    router:
        Active model router for TTS endpoint mapping.
    """

    def __init__(self, voice: str = "af_heart", router: OmniModelRouter | None = None) -> None:
        self._voice = voice
        self._router = router
        import os
        from shared_types.constants import DEFAULT_API_URL
        base_url = os.environ.get("LEMONADE_API_URL", DEFAULT_API_URL)
        if router:
            base_url = router.api_url
        base_url_no_v1 = base_url.removesuffix("/v1")
        self._client = LemonadeClient(base_url=base_url_no_v1)

    async def close(self) -> None:
        await self._client.close()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to WAV audio bytes via Lemonade TTS."""
        logger.info("Synthesizing %d chars → TTS …", len(text))
        model_id = self._router.tts() if self._router else ""
        return await self._client.tts(text, voice=self._voice, model=model_id)

    async def synthesize_to_file(self, text: str, output_path: Path) -> int:
        """Synthesize and write to disk.  Returns file size in bytes."""
        audio_bytes = await self.synthesize(text)
        output_path.write_bytes(audio_bytes)
        logger.info("Wrote %s (%d bytes)", output_path.name, len(audio_bytes))
        return len(audio_bytes)
