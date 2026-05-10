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
import shutil
import subprocess
import tempfile
from pathlib import Path

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
                "--target", "0",  # default monitor source
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
        )

        # Let it record for the requested duration, then terminate
        await asyncio.sleep(duration_seconds)
        proc.terminate()
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
