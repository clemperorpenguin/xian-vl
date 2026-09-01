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

"""Does the OCR sidecar actually work on real game frames?

`test_ocr_sidecar.py` checks the contract — polygons reduce to boxes, replies
map back to input order — with hand-built inputs.  It cannot answer the
question the experimental branch actually rests on: on a real 5-megapixel HUD
full of small Chinese glyphs, does a local detector find the text, put boxes on
it, and finish fast enough to call the result "live"?

So these run the real engine over a corpus of captured frames and measure it.
They skip without the corpus (see :mod:`benchmark_corpus`) or without the
optional OCR dependencies, so CI is unaffected.

Latency assertions are loose by default — wall-clock is a property of the
machine, and a busy shared runner should not fail a build.  ``XIAN_BENCH_STRICT=1``
holds them to the budget the live loop actually has.
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_corpus import corpus_paths, strict_budget as corpus_strict_budget  # noqa: E402

from shared_types.constants import QWEN_MAX_DIMENSION
from xian.grounding import (
    TextRegion,
    preprocess_for_grounding,
    rescale_regions,
    suppress_overlapping_regions,
)
from xian.ocr.engine import merge_adjacent_lines, ocr_available
from xian.timeout import vision_timeout_for_mode

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(not ocr_available(), reason="optional OCR dependencies not installed"),
]

#: The live loop's polling interval (``mage.live_lens.DEFAULT_INTERVAL_MS``).
#: Duplicated rather than imported: importing it would drag PyQt6 and the whole
#: client application into the engine's test environment.
LIVE_TICK_MS = 700

#: Frames at or under this many pixels stand in for a locked region — a
#: dialogue box or quest panel — rather than a whole captured desktop.
REGION_MEGAPIXELS = 1.0

#: How far past the tick a loose run may go before something is really wrong.
LOOSE_BUDGET_MULTIPLE = 4


@dataclass
class FrameRun:
    """One frame through the detector, with what it cost."""

    path: Path
    size: tuple[int, int]
    lines: list
    elapsed_ms: float

    @property
    def megapixels(self) -> float:
        return self.size[0] * self.size[1] / 1e6

    @property
    def is_region_sized(self) -> bool:
        return self.megapixels <= REGION_MEGAPIXELS

    @property
    def text(self) -> str:
        return "".join(line.text for line in self.lines)


def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"


def _edge_density(gray: np.ndarray) -> float:
    """Mean absolute neighbour difference — high on glyphs, low on flat art."""
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return 0.0
    horizontal = np.abs(np.diff(gray.astype(np.int16), axis=1)).mean()
    vertical = np.abs(np.diff(gray.astype(np.int16), axis=0)).mean()
    return float(horizontal + vertical)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def screenshot_paths() -> list[Path]:
    return corpus_paths()


@pytest.fixture(scope="session")
def strict_budget() -> bool:
    return corpus_strict_budget()


@pytest.fixture(scope="session")
def ocr_engine():
    """One engine for the whole session, warmed so timings exclude model load."""
    from xian.ocr.onnx_engine import OnnxOcrEngine

    engine = OnnxOcrEngine()
    engine.run(np.full((64, 256, 3), 255, dtype=np.uint8))
    return engine


@pytest.fixture(scope="session")
def frame_runs(screenshot_paths, ocr_engine) -> list[FrameRun]:
    """Run every corpus frame once and keep the results, not the pixels.

    Session-scoped because a pass over the corpus costs tens of seconds, and
    frame-scoped because holding 30 decoded 5-megapixel frames would cost half
    a gigabyte of resident memory.
    """
    runs = []
    for path in screenshot_paths:
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        started = time.perf_counter()
        lines = ocr_engine.run(frame)
        runs.append(
            FrameRun(path, (frame.shape[1], frame.shape[0]), lines,
                     (time.perf_counter() - started) * 1000)
        )
    return runs


# ── detection and recognition ────────────────────────────────────────

def test_every_frame_yields_text(frame_runs):
    """A game screenshot always has *something* readable on it."""
    empty = [run.path.name for run in frame_runs if not run.lines]
    assert not empty, f"the detector found no text at all in: {empty}"


def test_recognized_text_is_predominantly_the_source_script(frame_runs):
    """Guards against a detector that fires on textures and returns noise.

    These are Chinese-client captures, so most of what comes back should be
    Han characters.  Latin runs are real (file paths, ability keys, numbers),
    so this is a majority test, not a purity test.
    """
    han = sum(1 for run in frame_runs for char in run.text if _is_cjk(char))
    letters = sum(1 for run in frame_runs for char in run.text if char.isalpha() or _is_cjk(char))

    assert letters > 0
    assert han / letters > 0.5, f"only {han}/{letters} recognized letters were Han"


def test_recognition_confidence_stays_high_on_real_frames(frame_runs):
    """Small anti-aliased HUD glyphs are the hard case; scores should hold up.

    A drop here is the early warning that recognition has degraded — long
    before the translation output visibly turns to nonsense.
    """
    scores = [line.confidence for run in frame_runs for line in run.lines]
    assert scores

    mean = statistics.mean(scores)
    weak = sum(1 for score in scores if score < 0.8) / len(scores)

    assert mean > 0.85, f"mean recognition confidence fell to {mean:.3f}"
    assert weak < 0.15, f"{weak:.1%} of lines scored under 0.8"


# ── geometry: the boxes have to be paintable ─────────────────────────

def test_boxes_stay_inside_the_frame_and_have_area(frame_runs):
    """The overlay paints these coordinates directly; a bad box is a visible bug."""
    for run in frame_runs:
        width, height = run.size
        for line in run.lines:
            left, top, right, bottom = line.box
            assert 0 <= left < right <= width, f"{run.path.name}: box {line.box} escapes width {width}"
            assert 0 <= top < bottom <= height, f"{run.path.name}: box {line.box} escapes height {height}"
            assert line.is_valid()


def test_boxes_land_on_text_rather_than_background(frame_runs, screenshot_paths):
    """Boxes should cover glyphs, not scenery.

    Compared against same-sized patches taken elsewhere in the same frame, so
    the control has the same content and the same box dimensions — a fixed
    threshold would just measure how busy the game's art is.  Aggregate
    medians, because individual tails overlap: a detailed tree canopy really is
    edgier than one large glyph.
    """
    rng = np.random.default_rng(7)
    inside: list[float] = []
    control: list[float] = []

    for run in frame_runs:
        if not run.lines:
            continue
        with Image.open(run.path) as image:
            gray = np.asarray(image.convert("L"))
        height, width = gray.shape

        for line in run.lines:
            left, top, right, bottom = line.box
            inside.append(_edge_density(gray[top:bottom, left:right]))

            box_width, box_height = right - left, bottom - top
            patch_left = int(rng.integers(0, max(1, width - box_width)))
            patch_top = int(rng.integers(0, max(1, height - box_height)))
            control.append(
                _edge_density(gray[patch_top:patch_top + box_height, patch_left:patch_left + box_width])
            )

    ratio = statistics.median(inside) / max(1e-6, statistics.median(control))
    assert ratio > 2.5, (
        f"boxed regions are only {ratio:.2f}x edgier than the rest of the frame; "
        "the detector may be firing on textures"
    )


def test_line_merging_is_stable_on_real_detections(frame_runs):
    """Merging must converge, or one fill can swallow a whole panel.

    ``merge_adjacent_lines`` joins fragments of one spaced-out label.  Run over
    its own output it should change nothing: if a second pass keeps merging,
    the rule is transitive in a way that will eventually paint one rectangle
    over unrelated UI.
    """
    for run in frame_runs:
        once = merge_adjacent_lines(run.lines)
        twice = merge_adjacent_lines(once)
        assert [line.box for line in once] == [line.box for line in twice], (
            f"{run.path.name}: merging did not converge ({len(once)} -> {len(twice)} lines)"
        )


def _covering_pairs(boxes, threshold: float = 0.5):
    """Box pairs where one covers more than *threshold* of the other's area."""
    offenders = []
    for i, first in enumerate(boxes):
        for second in boxes[i + 1:]:
            overlap_x = min(first[2], second[2]) - max(first[0], second[0])
            overlap_y = min(first[3], second[3]) - max(first[1], second[1])
            if overlap_x <= 0 or overlap_y <= 0:
                continue
            smaller = min(
                (first[2] - first[0]) * (first[3] - first[1]),
                (second[2] - second[0]) * (second[3] - second[1]),
            )
            covered = overlap_x * overlap_y / max(1, smaller)
            if covered > threshold:
                offenders.append((covered, first, second))
    return offenders


def test_overlapping_detections_are_suppressed_before_painting(frame_runs, record_property):
    """Two fills over the same pixels means one translation is painted away.

    The detector itself still returns overlapping boxes on tightly stacked UI —
    a chat log, a character list — and always will; that is a property of the
    detector, not a defect the pipeline can prevent upstream. What matters is
    that they never reach ``InpaintOverlay.paintEvent``, which fills and then
    draws each region in list order, so a later fill lands on top of an earlier
    translation and blanks it.

    Both the vision and OCR paths run their regions through
    ``suppress_overlapping_regions`` for exactly this reason. This checks the
    output of that, not the raw detections — and records the raw rate so the
    size of the problem it is solving stays visible.
    """
    raw_offenders = 0
    boxes_seen = 0
    survivors: list[tuple[float, str, tuple, tuple]] = []

    for run in frame_runs:
        lines = merge_adjacent_lines(run.lines)
        boxes_seen += len(lines)
        raw_offenders += len(_covering_pairs([line.box for line in lines]))

        painted = suppress_overlapping_regions(
            [TextRegion(line.box, line.text, line.text) for line in lines]
        )
        survivors += [
            (covered, run.path.name, first, second)
            for covered, first, second in _covering_pairs([r.box for r in painted])
        ]

    record_property("boxes_inspected", boxes_seen)
    record_property("overlapping_pairs_from_detector", raw_offenders)
    record_property("overlapping_pairs_after_suppression", len(survivors))

    survivors.sort(reverse=True)
    assert not survivors, (
        f"{len(survivors)} box pairs still cover each other by more than half after "
        f"suppression (detector produced {raw_offenders}). Worst: "
        + "; ".join(f"{name} {first} vs {second} ({covered:.0%})"
                    for covered, name, first, second in survivors[:3])
    )


# ── latency: the whole reason the sidecar exists ─────────────────────

def test_a_locked_region_is_read_within_the_live_tick(frame_runs, strict_budget, record_property):
    """The claim under test: local OCR keeps up with the polling interval.

    A tick that overruns is not a crash — single flight just drops the ones
    that arrive while it works — but the overlay stops tracking the screen,
    which is the entire feature.
    """
    regions = [run for run in frame_runs if run.is_region_sized]
    if not regions:
        pytest.skip("corpus has no region-sized frames to stand in for a locked region")

    timings = sorted(run.elapsed_ms for run in regions)
    median = statistics.median(timings)
    slowest = timings[-1]
    record_property("region_ocr_ms_median", round(median, 1))
    record_property("region_ocr_ms_max", round(slowest, 1))
    record_property("region_frames", len(regions))

    budget = LIVE_TICK_MS if strict_budget else LIVE_TICK_MS * LOOSE_BUDGET_MULTIPLE
    assert median < budget, (
        f"median {median:.0f}ms over {len(regions)} region-sized frames "
        f"(slowest {slowest:.0f}ms) against a {LIVE_TICK_MS}ms tick"
    )


def test_a_full_screen_frame_stays_far_under_the_vision_timeout(
    frame_runs, strict_budget, record_property
):
    """The sidecar's other claim: it is an order of magnitude under the VLM path.

    A whole 5-megapixel desktop is the worst case, and it does *not* fit the
    tick — the honest number is recorded here rather than asserted away.  What
    is asserted is the comparison that justified splitting OCR out of the
    vision model at all.
    """
    full = [run for run in frame_runs if not run.is_region_sized]
    if not full:
        pytest.skip("corpus has no full-screen frames")

    timings = sorted(run.elapsed_ms for run in full)
    median = statistics.median(timings)
    vision_budget_ms = vision_timeout_for_mode("Game") * 1000

    record_property("fullscreen_ocr_ms_median", round(median, 1))
    record_property("fullscreen_ocr_ms_max", round(timings[-1], 1))
    record_property("fullscreen_fits_live_tick", bool(median < LIVE_TICK_MS))

    # Loose by default for the same reason as the region budget: OCR saturates
    # every core, so a benchmark sharing the machine with anything else reads
    # several times slower than one that has it to itself.
    budget = vision_budget_ms / (10 if strict_budget else 2)
    assert median < budget, (
        f"full-frame OCR median {median:.0f}ms is no longer an order of magnitude "
        f"under the {vision_budget_ms:.0f}ms vision-call budget it replaces"
    )


def test_cost_tracks_content_not_just_pixels(frame_runs):
    """Detection scales with area, recognition with line count.

    This is why a busy inventory screen costs more than an empty landscape at
    the same resolution, and why "how big a region can I lock" is the wrong
    question on its own.  If this correlation disappears, the recognition stage
    has stopped doing per-line work — i.e. it is dropping lines.
    """
    busy = [run for run in frame_runs if len(run.lines) >= 50]
    sparse = [run for run in frame_runs if 0 < len(run.lines) <= 10]
    if not busy or not sparse:
        pytest.skip("corpus lacks both busy and sparse frames")

    assert statistics.median(r.elapsed_ms for r in busy) > statistics.median(
        r.elapsed_ms for r in sparse
    )


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
