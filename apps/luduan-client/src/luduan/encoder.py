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

"""Opus encoding and KOReader manifest generation.

Takes translated audio segments and produces:
1. A single ``.opus`` file (concatenated and encoded).
2. A ``.audio.json`` sidecar file for KOReader synchronised playback.

This module wraps ``opusenc`` (CLI) or falls back to ``ffmpeg``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """A single narrated passage with timing metadata."""

    chapter_index: int
    paragraph_index: int
    text: str
    wav_path: Path
    offset_ms: int = 0
    duration_ms: int = 0


def encode_opus(
    wav_path: Path,
    output_path: Path,
    *,
    bitrate: int = 48,
) -> bool:
    """Encode a WAV file to Opus.

    Tries ``opusenc`` first, then ``ffmpeg`` as fallback.

    Returns ``True`` on success.
    """
    if shutil.which("opusenc"):
        cmd = [
            "opusenc",
            "--bitrate", str(bitrate),
            str(wav_path),
            str(output_path),
        ]
    elif shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-c:a", "libopus",
            "-b:a", f"{bitrate}k",
            str(output_path),
        ]
    else:
        logger.error("Neither opusenc nor ffmpeg found. Cannot encode Opus.")
        return False

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        logger.exception("Opus encoding failed")
        return False


def generate_koreader_manifest(
    segments: list[AudioSegment],
    opus_filename: str,
) -> dict:
    """Generate a KOReader ``.audio.json`` sidecar manifest.

    The manifest maps text passages to time offsets in the Opus file,
    enabling synchronised audio playback on e-ink devices.
    """
    entries = []
    for seg in segments:
        entries.append({
            "chapter": seg.chapter_index,
            "paragraph": seg.paragraph_index,
            "text": seg.text,
            "offset_ms": seg.offset_ms,
            "duration_ms": seg.duration_ms,
        })

    return {
        "audio_file": opus_filename,
        "format": "opus",
        "segments": entries,
    }


def write_manifest(manifest: dict, output_path: Path) -> None:
    """Write the KOReader manifest to disk."""
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Wrote KOReader manifest: %s", output_path.name)
