# Xian-VL — Core Vision-Language orchestration engine.
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

"""Grounding geometry, checked against real captured frames.

The vision model reports boxes normalized 0-1000 against whatever image it was
sent, so the overlay's accuracy rests on two mappings: the downscale that
prepares a frame, and the rescale back to capture pixels. Both are pure
arithmetic, and both are wrong in ways that only show up at real aspect ratios
and real sizes — a 2880x1800 desktop, a 291x442 dialogue crop, the odd
2869x1786 window.

Skips without the corpus (see :mod:`benchmark_corpus`), so CI is unaffected.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_corpus import corpus_paths  # noqa: E402

from shared_types.constants import QWEN_MAX_DIMENSION
from xian.grounding import TextRegion, preprocess_for_grounding, rescale_regions

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="session")
def screenshot_paths() -> list[Path]:
    return corpus_paths()


# ── the vision-model path's geometry ─────────────────────────────────

def test_grounding_preprocess_caps_size_without_distorting_it(screenshot_paths):
    """Boxes come back normalized, so any aspect change silently offsets them."""
    for path in screenshot_paths:
        with Image.open(path) as image:
            original = image.convert("RGB")
            prepared = preprocess_for_grounding(original)

        assert max(prepared.size) <= QWEN_MAX_DIMENSION, f"{path.name} exceeds the model's limit"

        before = original.size[0] / original.size[1]
        after = prepared.size[0] / prepared.size[1]
        assert abs(before - after) / before < 0.01, (
            f"{path.name}: aspect ratio moved {before:.4f} -> {after:.4f}"
        )


def test_regions_survive_the_round_trip_back_to_capture_pixels(screenshot_paths):
    """A box parsed against the scaled frame has to land on the same glyphs.

    ``ground_and_translate`` maps boxes from the prepared image back to the
    caller's coordinates; at 2880x1800 that is a 1.5x jump, so an error here is
    tens of pixels of misplaced overlay.
    """
    for path in screenshot_paths:
        with Image.open(path) as image:
            full_size = image.convert("RGB").size
            prepared_size = preprocess_for_grounding(image.convert("RGB")).size

        if prepared_size == full_size:
            continue

        # A box over the middle of the prepared frame, in its pixel space.
        source = TextRegion(
            (
                int(prepared_size[0] * 0.25), int(prepared_size[1] * 0.40),
                int(prepared_size[0] * 0.75), int(prepared_size[1] * 0.46),
            ),
            "原文", "translated",
        )
        scaled = rescale_regions([source], prepared_size, full_size)
        assert scaled, f"{path.name}: rescaling dropped the region"

        back = rescale_regions(scaled, full_size, prepared_size)[0]
        drift = max(abs(a - b) for a, b in zip(source.box, back.box))
        assert drift <= 2, f"{path.name}: round trip moved the box by {drift}px"
