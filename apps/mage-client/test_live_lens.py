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

import io
import os
import sys

import imagehash

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.live_lens import (  # noqa: E402
    CHANGE_THRESHOLD_RATIO,
    DEFAULT_CHANGE_THRESHOLD,
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

    def __init__(self, threshold=DEFAULT_CHANGE_THRESHOLD, painted=()):
        # Deliberately skip LiveLensWorker.__init__ (it needs a processor and
        # a QThread); only the gate's state matters here.
        self.change_threshold = threshold
        self._clean_hash = None
        self._painted_boxes = list(painted)
        self._last_signature = ()
        self._warned_untranslated = False


def test_first_frame_always_translates():
    gate = _Gate()
    assert gate._should_translate(_hash(_frame(2))) is True


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
    """The overlay sits inside the captured region.

    Without masking, the loop re-translates its own output forever. The painted
    areas are blanked in both the reference and the candidate, so whatever we
    drew there cannot register as the game changing.
    """
    clean = _frame(2)
    painted_box = (20, 20, 360, 120)
    gate = _Gate(painted=[painted_box])
    gate._clean_hash = gate._masked_hash(clean)

    # The screen now carries our translations across that whole band.
    painted = clean.copy()
    ImageDraw.Draw(painted).rectangle(painted_box, fill=(200, 30, 30))

    assert gate._should_translate(gate._masked_hash(painted)) is False


def test_a_change_outside_the_painted_area_still_triggers_a_translation():
    """Masking must not blind the gate to the rest of the region."""
    clean = _frame(2)
    painted_box = (20, 20, 360, 60)
    gate = _Gate(painted=[painted_box])
    gate._clean_hash = gate._masked_hash(clean)

    changed = clean.copy()
    ImageDraw.Draw(changed).rectangle([20, 150, 380, 190], fill=(255, 255, 255))

    assert gate._should_translate(gate._masked_hash(changed)) is True


def test_masking_is_a_no_op_before_anything_is_painted():
    gate = _Gate()
    frame = _frame(2)
    assert gate._mask_painted(frame) is frame


def test_the_threshold_is_a_fraction_of_the_hash_bits():
    """Expressed as a ratio so it survives a change of hash size.

    Measured on the screenshot corpus: re-captures of one screen reach 20 and
    distinct screens start at 22, so the threshold has to sit between them.
    """
    assert DEFAULT_CHANGE_THRESHOLD == round(LIVE_HASH_SIZE**2 * CHANGE_THRESHOLD_RATIO)
    assert 20 <= DEFAULT_CHANGE_THRESHOLD < 22


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


# ── silent-failure detection ─────────────────────────────────────────

class _Publisher(LiveLensWorker):
    """_publish without a QThread, capturing what would reach the UI."""

    def __init__(self):
        self.change_threshold = DEFAULT_CHANGE_THRESHOLD
        self._clean_hash = None
        self._painted_boxes = []
        self._last_signature = ()
        self._warned_untranslated = False
        self._session_recorder = None
        self._fill_frame = None
        self._fill_cache = {}
        self._region_hash_cache = {}
        self._retained = []
        self._carry = []
        self.rect = None
        self._served_rect = None
        self.errors = []
        self.emitted = []

    # Stand in for the Qt signals.
    class _Sig:
        def __init__(self, sink):
            self._sink = sink

        def emit(self, *args):
            self._sink.append(args)

    @property
    def error(self):
        return self._Sig(self.errors)

    @property
    def regions_ready(self):
        return self._Sig(self.emitted)

    def _capture_scale(self, frame):
        return 1.0


def _region(original, translated):
    from xian.grounding import TextRegion

    return TextRegion((10, 10, 100, 40), original, translated)


def test_an_all_untranslated_screen_is_reported():
    """Every failure path falls back to the source text on purpose.

    That makes a model which cannot follow the prompt look identical to a
    working overlay — the user sees their own game text, neatly boxed, and no
    error anywhere. One small model really did answer with an empty json fence
    and drop all twelve lines through.
    """
    worker = _Publisher()
    frame = Image.new("RGB", (200, 100), (30, 30, 30))

    worker._publish([_region("前往長安城", "前往長安城"), _region("已完成", "已完成")], frame)

    assert worker.errors, "the user must be told nothing was translated"
    assert "translated" in worker.errors[0][0]


def test_a_partly_translated_screen_is_not_reported():
    worker = _Publisher()
    frame = Image.new("RGB", (200, 100), (30, 30, 30))

    worker._publish([_region("前往長安城", "Head to the city"), _region("已完成", "已完成")], frame)

    assert worker.errors == []


def test_the_untranslated_warning_fires_only_once():
    """A whole session of untranslatable frames should not spam."""
    worker = _Publisher()
    frame = Image.new("RGB", (200, 100), (30, 30, 30))

    worker._publish([_region("已完成", "已完成")], frame)
    worker._publish([_region("確定", "確定")], frame)

    assert len(worker.errors) == 1


def test_publishing_records_the_boxes_the_change_gate_must_mask():
    worker = _Publisher()
    frame = Image.new("RGB", (200, 100), (30, 30, 30))

    worker._publish([_region("前往", "Go")], frame)

    assert worker._painted_boxes == [(10, 10, 100, 40)]


def test_background_is_sampled_once_per_box_per_frame():
    """Streaming republishes the whole list on every batch.

    Each batch re-sampled every box already published, so the sampling cost
    grew with the square of the region count — worst exactly where there are
    most regions, which is a whole-screen capture.
    """
    import mage.live_lens as live_lens

    worker = _Publisher()
    frame = Image.new("RGB", (400, 300), (30, 30, 30))

    calls = []
    original = live_lens.sample_background

    def counting(image, box):
        calls.append(box)
        return original(image, box)

    live_lens.sample_background = counting
    try:
        from xian.grounding import TextRegion

        first = _region("一", "one")
        second = TextRegion((10, 60, 100, 90), "二", "two")
        worker._publish([first], frame)
        worker._publish([first, second], frame)
    finally:
        live_lens.sample_background = original

    assert calls == [first.box, second.box]


def test_a_new_frame_resamples_its_own_backgrounds():
    """The cache is per frame; the next capture has different pixels."""
    worker = _Publisher()
    region = _region("一", "one")

    worker._publish([region], Image.new("RGB", (400, 300), (10, 10, 10)))
    dark = worker.emitted[-1][0][0].fill

    worker._last_signature = ()
    worker._publish([region], Image.new("RGB", (400, 300), (240, 240, 240)))
    light = worker.emitted[-1][0][0].fill

    assert dark != light


# ── region retention ─────────────────────────────────────────────────

def _text_frame(colour=(30, 30, 30), mark=None):
    """A frame with enough structure that a crop of it is not flat."""
    frame = Image.new("RGB", (400, 300), colour)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((10, 10, 100, 40), fill=(200, 200, 200))
    draw.rectangle((20, 18, 40, 32), fill=(20, 20, 20))
    if mark:
        draw.rectangle(mark, fill=(255, 0, 0))
    return frame


def test_an_unchanged_region_keeps_its_translation_without_inference():
    """The tooltip case: read once, then kept while its pixels stay put."""
    worker = _Publisher()
    frame = _text_frame()

    worker._publish([_region("前往", "Go")], frame)

    # A second capture of the same screen, re-encoded as the capture path does.
    buffer = io.BytesIO()
    frame.save(buffer, "JPEG", quality=85)
    buffer.seek(0)
    again = Image.open(buffer).convert("RGB")

    survivors = worker._surviving_regions(again)

    assert [r.translated for r in survivors] == ["Go"]


def test_a_region_whose_text_changed_is_not_retained():
    worker = _Publisher()
    worker._publish([_region("前往", "Go")], _text_frame())

    # The same box now holds something else entirely.
    replaced = _text_frame(mark=(10, 10, 100, 40))

    assert worker._surviving_regions(replaced) == []


def test_a_failed_call_keeps_the_regions_that_are_still_valid():
    """Blanking the overlay because one call timed out throws away work that
    is demonstrably still correct — the pixels have not moved."""
    worker = _Publisher()
    frame = _text_frame()
    worker._publish([_region("前往", "Go")], frame)

    worker._carry = worker._surviving_regions(frame)
    worker._remember(worker._carry, frame)

    assert [r.translated for r in worker._carry] == ["Go"]
    assert len(worker._retained) == 1


def test_carried_regions_are_painted_alongside_fresh_ones():
    from xian.grounding import TextRegion

    worker = _Publisher()
    frame = _text_frame()
    worker._publish([_region("前往", "Go")], frame)
    worker._carry = worker._surviving_regions(frame)

    worker._last_signature = ()
    worker._publish([TextRegion((200, 200, 300, 240), "確定", "Confirm")], frame)

    painted = {r.translated for r in worker.emitted[-1][0]}
    assert painted == {"Confirm", "Go"}


def test_a_carried_region_never_paints_over_a_fresh_one():
    """The overlay fills before it draws, so two boxes over the same pixels
    means one translation is painted out. A fresh reading wins.

    The fresh box here overlaps the carried one without being the same line —
    a box that *is* the same line takes the retention path below instead.
    """
    from xian.grounding import TextRegion

    worker = _Publisher()
    frame = _text_frame()
    worker._publish([_region("前往", "Go")], frame)
    worker._carry = worker._surviving_regions(frame)
    assert worker._carry, "the carried region should have survived"

    worker._last_signature = ()
    worker._publish([TextRegion((60, 10, 160, 40), "確定", "Confirm")], frame)

    assert [r.translated for r in worker.emitted[-1][0]] == ["Confirm"]


def test_unchanged_text_keeps_the_wording_it_already_had():
    """The same pixels re-read come back worded differently each time, and the
    overlay shimmers. Identical text should not change its translation."""
    from xian.grounding import TextRegion

    worker = _Publisher()
    frame = _text_frame()
    worker._publish([_region("前往", "Go")], frame)
    worker._carry = worker._surviving_regions(frame)

    worker._last_signature = ()
    worker._publish([TextRegion((11, 11, 99, 39), "前往", "Head over")], frame)

    assert [r.translated for r in worker.emitted[-1][0]] == ["Go"]
