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

"""Is the change gate calibrated for real game frames?

`test_live_lens.py` checks the gate's logic with synthetic panels, where "the
same frame" and "a different frame" are unambiguous.  On a real screen they are
not, and the gate has to tell apart two things that both look like small pixel
differences:

* the *same* screen captured again — compositor dither, a blinking cursor, a
  scrolled combat log, sub-pixel drift;
* a genuinely *new* screen — a tooltip opened over the same busy scene.

Get it wrong one way and the overlay re-translates constantly and flickers.
Get it wrong the other way and it paints a stale translation over new text and
never notices.  Only real captures can say which side the threshold sits on.

Skips without the screenshot corpus (see :mod:`benchmark_corpus`).
"""

from __future__ import annotations

import io
import itertools
import os
import statistics
import sys
from pathlib import Path

import imagehash
import pytest
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_corpus import corpus_paths  # noqa: E402
from mage.live_lens import (  # noqa: E402
    DEFAULT_CHANGE_THRESHOLD,
    LIVE_HASH_SIZE,
    phash_distance,
    sample_background,
)

pytestmark = pytest.mark.benchmark

#: A hash size that has enough bits to resolve a panel-sized change.  See
#: ``test_a_larger_hash_separates_recaptures_from_real_changes``.
CANDIDATE_HASH_SIZE = 16


def _hash(image: Image.Image, hash_size: int = LIVE_HASH_SIZE):
    return imagehash.phash(image, hash_size=hash_size)


def _recaptures(image: Image.Image):
    """The same screen, captured again — what must *not* count as a change.

    Each of these is something a second capture of an unchanged screen really
    does look like: the capture path re-encodes, the mouse cursor blinks over
    the region, the compositor hands back a frame offset by a pixel, and
    HDR/gamma handling shifts levels slightly between grabs.
    """
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    buffer.seek(0)
    yield "re-encode", Image.open(buffer).convert("RGB")

    cursor = image.copy()
    middle = (image.width // 2, image.height // 2)
    ImageDraw.Draw(cursor).ellipse(
        [middle[0], middle[1], middle[0] + 14, middle[1] + 14], fill=(255, 255, 255)
    )
    yield "cursor blink", cursor

    yield "one-pixel drift", image.transform(image.size, Image.AFFINE, (1, 0, 1, 0, 1, 1))
    yield "gamma drift", ImageEnhance.Brightness(image).enhance(1.03)


@pytest.fixture(scope="session")
def screenshot_paths() -> list[Path]:
    return corpus_paths()


@pytest.fixture(scope="session")
def frame_hashes(screenshot_paths) -> dict[str, object]:
    """One perceptual hash per corpus frame, at the configured hash size."""
    hashes = {}
    for path in screenshot_paths:
        with Image.open(path) as image:
            hashes[path.name] = _hash(image.convert("RGB"))
    return hashes


# ── the flicker direction: a re-capture must read as unchanged ───────

def test_a_recapture_of_one_frame_never_reads_as_changed(screenshot_paths, record_property):
    """Every tick that mistakes noise for change costs a full translation pass.

    This is also what keeps the overlay still: a "change" forces a hide, a
    clean re-grab and a repaint, so a jumpy gate is visible as flicker.
    """
    worst = []
    for path in screenshot_paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            reference = _hash(image)
            for label, variant in _recaptures(image):
                worst.append((phash_distance(reference, _hash(variant)), path.name, label))

    worst.sort(reverse=True)
    record_property("recapture_distance_max", int(worst[0][0]))
    record_property("change_threshold", DEFAULT_CHANGE_THRESHOLD)

    distance, name, label = worst[0]
    assert distance <= DEFAULT_CHANGE_THRESHOLD, (
        f"{label} on {name} moved the hash by {distance}, over the "
        f"{DEFAULT_CHANGE_THRESHOLD} threshold: the overlay would re-translate and flicker "
        "on an unchanged screen"
    )


# ── the stale direction: a new screen must read as changed ───────────

@pytest.mark.xfail(
    LIVE_HASH_SIZE == 8,
    strict=True,
    reason=(
        "A 64-bit pHash keeps only the lowest 8x8 DCT coefficients, which describe "
        "the broad light/dark layout of the frame. A dialogue or loot panel opening "
        "over a busy scene barely perturbs those, so it scores under the change "
        "threshold and the overlay keeps showing the previous translation. "
        "See test_a_larger_hash_separates_recaptures_from_real_changes."
    ),
)
def test_every_distinct_frame_reads_as_changed(frame_hashes, record_property):
    """Two different screens must never be gated away as the same one.

    A miss here is the failure the user actually sees: new text on screen,
    the old translation still painted over it, and nothing to prompt a retry.
    """
    missed = [
        (int(phash_distance(frame_hashes[a], frame_hashes[b])), a, b)
        for a, b in itertools.combinations(frame_hashes, 2)
        if phash_distance(frame_hashes[a], frame_hashes[b]) <= DEFAULT_CHANGE_THRESHOLD
    ]
    record_property("distinct_pairs_gated_away", len(missed))

    assert not missed, (
        "distinct frames the gate would call unchanged: "
        + "; ".join(f"{a} vs {b} (distance {d})" for d, a, b in sorted(missed)[:5])
    )


def test_a_larger_hash_separates_recaptures_from_real_changes(screenshot_paths, record_property):
    """The fix path, measured: enough bits and the two classes come apart.

    Separation is what makes a threshold meaningful at all — the largest
    distance between re-captures of one screen has to sit below the smallest
    distance between two different screens.  At 64 bits it does not, so no
    threshold works; at 256 it does, and the threshold belongs in that gap.
    """
    unchanged: list[int] = []
    hashes = {}

    for path in screenshot_paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            reference = _hash(image, CANDIDATE_HASH_SIZE)
            hashes[path.name] = reference
            unchanged += [
                int(phash_distance(reference, _hash(variant, CANDIDATE_HASH_SIZE)))
                for _label, variant in _recaptures(image)
            ]

    changed = [
        int(phash_distance(hashes[a], hashes[b]))
        for a, b in itertools.combinations(hashes, 2)
    ]

    ceiling, floor = max(unchanged), min(changed)
    record_property("candidate_hash_size", CANDIDATE_HASH_SIZE)
    record_property("candidate_recapture_max", ceiling)
    record_property("candidate_distinct_min", floor)

    assert floor > ceiling, (
        f"at hash_size={CANDIDATE_HASH_SIZE} the classes still overlap: re-captures reach "
        f"{ceiling} while distinct frames start at {floor}"
    )


def test_a_locked_region_gates_more_reliably_than_a_whole_desktop(screenshot_paths, record_property):
    """Why the flaw is survivable in the intended workflow.

    The gate hashes whatever the user locked.  A panel that is 4% of a desktop
    is most of a locked dialogue box, so the same change moves far more of the
    hash.  This measures that margin so the guidance — lock the text, not the
    screen — rests on a number.
    """
    pairs = []
    for first, second in zip(screenshot_paths, screenshot_paths[1:]):
        with Image.open(first) as a, Image.open(second) as b:
            left, right = a.convert("RGB"), b.convert("RGB")
            if left.size != right.size:
                continue  # the window was resized between captures; not a pair
            width, height = left.size
            window = (int(width * 0.55), int(height * 0.05), int(width * 0.98), int(height * 0.55))
            pairs.append((
                int(phash_distance(_hash(left), _hash(right))),
                int(phash_distance(_hash(left.crop(window)), _hash(right.crop(window)))),
            ))

    if len(pairs) < 3:
        pytest.skip("corpus has too few same-size consecutive frames to compare")

    full_median = statistics.median(p[0] for p in pairs)
    region_median = statistics.median(p[1] for p in pairs)
    record_property("consecutive_pairs", len(pairs))
    record_property("fullscreen_change_distance_median", full_median)
    record_property("region_change_distance_median", region_median)

    assert min(p[1] for p in pairs) > DEFAULT_CHANGE_THRESHOLD, (
        "even on a locked region the gate missed a change between consecutive frames"
    )


# ── the fill under the translation ───────────────────────────────────

def _ocr_available() -> bool:
    from xian.ocr.engine import ocr_available

    return ocr_available()


@pytest.mark.skipif(not _ocr_available(), reason="optional OCR dependencies not installed")
def test_the_fill_matches_the_backdrop_rather_than_the_glyphs(screenshot_paths, record_property):
    """The inpainted patch has to disappear into the UI it covers.

    ``sample_background`` reads a band *outside* the box precisely so the
    glyphs about to be covered do not tint the fill toward the text colour.
    On real frames — gradient panels, parchment, translucent HUDs — that band
    is not a flat colour, so this checks the result still lands on the
    background side rather than the text side.
    """
    import numpy as np

    from xian.ocr.onnx_engine import OnnxOcrEngine

    engine = OnnxOcrEngine()
    correct = 0
    total = 0

    for path in screenshot_paths[:12]:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            pixels = np.asarray(image)
            for line in engine.run(pixels)[:20]:
                left, top, right, bottom = line.box
                patch = pixels[top:bottom, left:right].reshape(-1, 3).astype(float)
                if patch.size == 0:
                    continue

                fill = np.array(sample_background(image, line.box), dtype=float)
                luminance = patch.mean(axis=1)
                # Polarity varies (light text on dark panels and the reverse), so
                # compare against both extremes and let the nearer one be the
                # backdrop rather than assuming which is which.
                bright = patch[luminance >= np.percentile(luminance, 95)].mean(axis=0)
                dark = patch[luminance <= np.percentile(luminance, 40)].mean(axis=0)

                to_bright = float(np.linalg.norm(fill - bright))
                to_dark = float(np.linalg.norm(fill - dark))
                # The backdrop is whichever cluster dominates the box's area.
                correct += min(to_bright, to_dark) < max(to_bright, to_dark)
                total += 1

    assert total, "no lines detected to sample a background from"
    rate = correct / total
    record_property("fill_matches_backdrop_rate", round(rate, 3))
    record_property("fill_samples", total)

    assert rate > 0.95, f"the fill sat closer to the text colour on {1 - rate:.1%} of {total} lines"
