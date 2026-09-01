# Xian-VL — Real-Time Vision-Language Assistant for Gaming Environments.
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

"""Locating the benchmark corpus of captured game frames.

The live path is a chain of judgement calls — is this frame different from the
last one, did OCR find the text, does a tick fit in the budget — and every one
of them is calibrated against what a real game screen looks like.  Synthetic
images cannot answer them: a drawn rectangle on flat grey is trivially
detectable and trivially different from the next one.  So the benchmarks run
against real captures.

The corpus is deliberately not in the repository (it is hundreds of megabytes
of game screenshots).  Point ``XIAN_SCREENSHOT_CORPUS`` at a directory of
frames, or leave them in ``../game_screenshots`` beside the checkout.  With no
corpus every benchmark skips, which is what happens in CI.

Plain functions rather than fixtures, and a module rather than a ``conftest``,
because the two benchmark files live in different workspace packages: pytest
picks its rootdir from the arguments it is given, so a shared ``conftest`` is
loaded when the whole suite runs and quietly missing when someone runs one
file.  An import cannot be missed.

Environment:
  ``XIAN_SCREENSHOT_CORPUS``  directory of frames (default ``../game_screenshots``)
  ``XIAN_BENCH_MAX_FRAMES``   cap the frame count for a quicker pass
  ``XIAN_BENCH_STRICT``       hold latency to the real tick budget rather than
                              the loose "did something break" bound.  Off by
                              default: wall-clock is a property of the machine,
                              and a busy shared runner should not fail a build.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

__all__ = ["CORPUS_ENV_VAR", "corpus_dir", "corpus_paths", "strict_budget"]

CORPUS_ENV_VAR = "XIAN_SCREENSHOT_CORPUS"
MAX_FRAMES_ENV_VAR = "XIAN_BENCH_MAX_FRAMES"
STRICT_ENV_VAR = "XIAN_BENCH_STRICT"

#: Beside the checkout, which is where the frames are captured to by default.
DEFAULT_CORPUS = Path(__file__).resolve().parent.parent / "game_screenshots"

FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def corpus_dir() -> Path:
    """The directory of captured game frames, or skip the calling test."""
    configured = os.environ.get(CORPUS_ENV_VAR)
    corpus = Path(configured).expanduser() if configured else DEFAULT_CORPUS

    if not corpus.is_dir():
        pytest.skip(
            f"No screenshot corpus at {corpus}. "
            f"Set {CORPUS_ENV_VAR} to a directory of captured game frames."
        )
    return corpus


def corpus_paths() -> list[Path]:
    """Every frame in the corpus, in capture order.

    Sorting by name puts timestamped captures in the order they were taken,
    which is what makes consecutive pairs meaningful to the change gate.
    """
    corpus = corpus_dir()
    paths = sorted(p for p in corpus.iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
    if not paths:
        pytest.skip(f"Screenshot corpus {corpus} contains no images")

    limit = os.environ.get(MAX_FRAMES_ENV_VAR, "")
    if limit.isdigit():
        paths = paths[: int(limit)]
    return paths


def strict_budget() -> bool:
    """Whether latency assertions hold to the real tick budget."""
    return os.environ.get(STRICT_ENV_VAR, "").lower() in ("1", "true", "yes")
