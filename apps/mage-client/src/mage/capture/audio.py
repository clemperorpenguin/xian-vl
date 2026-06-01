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

"""System audio capture for Cinematic Mode.

Captures a short clip of system audio (what the speakers are playing)
using PipeWire's ``pw-record`` or PulseAudio's ``parecord``.  The
captured audio is then sent to Lemonade's ASR endpoint for
transcription and fused with the screen's OCR output.

This module is Linux-only.  On other platforms it is a no-op stub.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from mage.utils.env import clean_subprocess_env

logger = logging.getLogger(__name__)


def _find_recorder() -> str | None:
    """Return the path to a working system-audio recorder, or *None*."""
    for cmd in ("pw-record", "parecord"):
        path = shutil.which(cmd)
        if path:
            return cmd
    return None


async def capture_system_audio(
    duration_seconds: float = 5.0,
    sample_rate: int = 16000,
) -> bytes | None:
    """Record *duration_seconds* of system monitor audio.

    Returns raw WAV bytes, or ``None`` if no recorder is available.
    The WAV is mono 16-bit at *sample_rate* Hz — suitable for
    ``POST /v1/audio/transcriptions``.
    """
    recorder = _find_recorder()
    if recorder is None:
        logger.warning(
            "No audio recorder found (tried pw-record, parecord). "
            "Cinematic Mode audio capture is disabled."
        )
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        outpath = tmp.name

    try:
        if recorder == "pw-record":
            cmd = [
                "pw-record",
                "--rate", str(sample_rate),
                "--channels", "1",
                "--format", "s16",
                outpath,
            ]
        else:  # parecord
            cmd = [
                "parecord",
                "--rate", str(sample_rate),
                "--channels", "1",
                "--format", "s16le",
                "--device", "@DEFAULT_MONITOR@",
                outpath,
            ]

        logger.info("Recording %s s of system audio via %s …", duration_seconds, recorder)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=clean_subprocess_env(),
        )

        # Let it record for the requested duration, then terminate
        await asyncio.sleep(duration_seconds)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Audio capture process did not exit on SIGTERM. Killing it.")
            proc.kill()
            await proc.wait()

        audio_path = Path(outpath)
        if audio_path.exists() and audio_path.stat().st_size > 44:  # > WAV header
            return audio_path.read_bytes()

        logger.warning("Audio capture produced an empty file.")
        return None

    except Exception:
        logger.exception("Audio capture failed")
        return None
    finally:
        Path(outpath).unlink(missing_ok=True)


class ContinuousAudioStreamer:
    """Streams continuous system audio from parecord, chunking by VAD."""
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        silence_threshold_rms: int = 300, 
        min_chunk_sec: float = 1.0, 
        max_chunk_sec: float = 5.0, 
        silence_duration_sec: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.silence_threshold_rms = silence_threshold_rms
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec
        self.silence_duration_sec = silence_duration_sec
        self._running = False
        self._proc = None

    async def start(self):
        recorder = _find_recorder()
        if not recorder:
            raise RuntimeError("No audio recorder found.")
        
        if recorder == "pw-record":
            cmd = ["pw-record", "--rate", str(self.sample_rate), "--channels", "1", "--format", "s16", "-"]
        else:
            cmd = ["parecord", "--rate", str(self.sample_rate), "--channels", "1", "--format", "s16le", "--device", "@DEFAULT_MONITOR@", "-"]
            
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=clean_subprocess_env(),
        )
        self._running = True

    async def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            self._proc = None

    async def read_chunks(self):
        """Yields WAV-encoded audio bytes whenever a chunk is ready."""
        if not self._proc or not self._proc.stdout:
            return

        chunk_bytes = bytearray()
        bytes_per_sample = 2
        bytes_per_sec = self.sample_rate * bytes_per_sample
        read_size = int(self.sample_rate * 0.1) * bytes_per_sample # 100ms
        
        silence_bytes = 0
        silence_limit_bytes = int(self.silence_duration_sec * bytes_per_sec)
        max_limit_bytes = int(self.max_chunk_sec * bytes_per_sec)
        min_limit_bytes = int(self.min_chunk_sec * bytes_per_sec)

        while self._running:
            try:
                # Use a timeout. If the audio system suspends the sink (like PulseAudio does
                # when no audio is playing), readexactly will block forever.
                # A timeout lets us inject synthetic silence or just trigger the VAD flush.
                data = await asyncio.wait_for(self._proc.stdout.readexactly(read_size), timeout=0.5)
                
                chunk_bytes.extend(data)
                
                samples_count = len(data) // 2
                if samples_count > 0:
                    samples = struct.unpack(f'<{samples_count}h', data[:samples_count*2])
                    rms = math.sqrt(sum(s*s for s in samples) / samples_count)
                else:
                    rms = 0
            except asyncio.TimeoutError:
                # No audio produced. Treat as pure silence.
                data = b'\x00' * read_size
                chunk_bytes.extend(data)
                rms = 0
            except asyncio.IncompleteReadError as e:
                data = e.partial
                if not data:
                    break
                chunk_bytes.extend(data)
                rms = 0
            except Exception:
                break
            
            if rms < self.silence_threshold_rms:
                silence_bytes += len(data)
            else:
                silence_bytes = 0

            if len(chunk_bytes) >= max_limit_bytes or (silence_bytes >= silence_limit_bytes and len(chunk_bytes) >= min_limit_bytes):
                wav_data = self._pcm_to_wav(bytes(chunk_bytes))
                if wav_data:
                    yield wav_data
                chunk_bytes.clear()
                silence_bytes = 0

    def _pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """Wrap raw s16le PCM in a valid WAV header."""
        num_channels = 1
        bytes_per_sample = 2
        sample_rate = self.sample_rate
        byte_rate = sample_rate * num_channels * bytes_per_sample
        block_align = num_channels * bytes_per_sample
        data_size = len(pcm_data)
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bytes_per_sample * 8,
            b'data',
            data_size
        )
        return header + pcm_data


def play_audio(audio_bytes: bytes):
    """Play WAV audio bytes using pw-play, paplay, or aplay."""
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path
    
    player = None
    for cmd in ("pw-play", "paplay", "aplay"):
        if shutil.which(cmd):
            player = cmd
            break
            
    if not player:
        logger.warning("No audio player found (pw-play, paplay, aplay). Cannot play audio.")
        return
        
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        subprocess.run([player, tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=clean_subprocess_env())
    except Exception as e:
        logger.error("Failed to play audio with %s: %s", player, e)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def play_audio_async(audio_bytes: bytes):
    """Play WAV audio bytes in a background thread."""
    import threading
    t = threading.Thread(target=play_audio, args=(audio_bytes,), daemon=True)
    t.start()
