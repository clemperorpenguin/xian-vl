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

"""Change gating and colour sampling for continuous in-place translation.

These are the pure decision functions — no Qt event loop, no screen capture.
"""

import os
import sys

import imagehash
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.live_lens import (  # noqa: E402
    LIVE_HASH_SIZE,
    LiveLensWorker,
    LiveRegion,
    phash_distance,
    regions_signature,
    sample_background,
)


def _frame(text_rows=0, seed=0) -> Image.Image:
    """A synthetic HUD panel with a configurable number of text-like bars."""
    image = Image.new("RGB", (400, 200), (28, 30, 46))
    draw = ImageDraw.Draw(image)
    for i in range(text_rows):
        y = 20 + i * 30 + seed
        draw.rectangle([20, y, 360, y + 16], fill=(230, 230, 230))
    return image


def _hash(image: Image.Image):
    return imagehash.phash(image, hash_size=LIVE_HASH_SIZE)


class _Gate(LiveLensWorker):
    """Exercise the change gate without constructing a QThread's async loop."""

    def __init__(self, threshold=6):
        # Deliberately skip LiveLensWorker.__init__ (it needs a processor and
        # a QThread); only the gate's state matters here.
        self.change_threshold = threshold
        self._clean_hash = None
        self._painted_hash = None
        self._last_signature = ()
        self._needs_clean_frame = False


def test_first_frame_always_translates():
    gate = _Gate()
    assert gate._should_translate(_hash(_frame(2))) is True
    assert gate._needs_clean_frame is False


def test_an_unchanged_frame_is_skipped():
    gate = _Gate()
    frame = _frame(2)
    gate._clean_hash = _hash(frame)

    assert gate._should_translate(_hash(frame)) is False


def test_new_dialogue_on_screen_triggers_a_translation():
    gate = _Gate()
    gate._clean_hash = _hash(_frame(2))

    assert gate._should_translate(_hash(_frame(5))) is True


def test_our_own_overlay_does_not_look_like_a_content_change():
    """The overlay sits inside the captured region; without this the loop
    would re-translate its own output forever."""
    gate = _Gate()
    clean = _frame(2)
    painted = _frame(2, seed=1)  # what the screen looks like once we render
    gate._clean_hash = _hash(clean)
    gate._painted_hash = _hash(painted)

    assert gate._should_translate(_hash(painted)) is False


def test_a_real_change_while_the_overlay_is_up_requests_a_clean_frame():
    gate = _Gate()
    gate._clean_hash = _hash(_frame(2))
    gate._painted_hash = _hash(_frame(2, seed=1))

    assert gate._should_translate(_hash(_frame(6))) is True
    assert gate._needs_clean_frame is True, "must recapture without the overlay"


def test_a_change_with_no_overlay_up_needs_no_clean_frame():
    gate = _Gate()
    gate._clean_hash = _hash(_frame(2))

    assert gate._should_translate(_hash(_frame(6))) is True
    assert gate._needs_clean_frame is False


def test_threshold_of_zero_reacts_to_any_difference():
    gate = _Gate(threshold=0)
    frame = _frame(2)
    gate._clean_hash = _hash(frame)

    assert gate._should_translate(_hash(frame)) is False
    assert gate._should_translate(_hash(_frame(3))) is True


def test_phash_distance_treats_a_missing_hash_as_maximally_different():
    assert phash_distance(None, _hash(_frame(1))) > 1000
    assert phash_distance(_hash(_frame(1)), None) > 1000


def test_identical_renders_share_a_signature():
    a = [LiveRegion((0, 0, 10, 10), "文", "text")]
    b = [LiveRegion((0, 0, 10, 10), "文", "text")]
    c = [LiveRegion((0, 0, 10, 10), "文", "other")]
    d = [LiveRegion((0, 5, 10, 15), "文", "text")]

    assert regions_signature(a) == regions_signature(b)
    assert regions_signature(a) != regions_signature(c), "text changed"
    assert regions_signature(a) != regions_signature(d), "position changed"


def test_background_is_sampled_outside_the_box_not_from_the_glyphs():
    """Sampling inside would average in the text being covered."""
    image = Image.new("RGB", (200, 100), (20, 40, 60))
    ImageDraw.Draw(image).rectangle([50, 40, 150, 60], fill=(255, 255, 255))

    fill = sample_background(image, (50, 40, 150, 60))

    assert fill == (20, 40, 60)


def test_background_sampling_survives_a_box_flush_against_the_edge():
    image = Image.new("RGB", (100, 50), (10, 10, 10))
    assert sample_background(image, (0, 0, 100, 50)) == (0, 0, 0)


def test_background_sampling_handles_greyscale_frames():
    image = Image.new("L", (120, 60), 128)
    assert sample_background(image, (30, 20, 90, 40)) == (128, 128, 128)
