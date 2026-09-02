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

"""The live path, end to end, against a real vision model.

Every other benchmark here measures a piece: the change gate on a corpus,
capture against a display, grounding geometry as pure arithmetic. This measures
the thing the overlay actually does — one changed frame in, located
translations out — using the real encoder, the real preprocessing, and the real
streaming parse.

It is the measurement that retired the local OCR sidecar. Grounding a locked
region came in around 1.0s against roughly 2.25s for OCR plus a separate batch
translation, with the first line painted at ~460ms rather than after the whole
chain, so the sidecar was carrying no weight.

Needs a Lemonade server with a **vision-capable** model installed; skips
otherwise. ``XIAN_BENCH_MODEL`` names one explicitly.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_corpus import corpus_paths, strict_budget as corpus_strict_budget  # noqa: E402

from xian.grounding import ground_and_translate  # noqa: E402
from xian.lemonade_url import normalize_lemonade_api_base_url  # noqa: E402
from xian.omni_router import is_translation_only_model  # noqa: E402
from xian.pipeline import VLProcessor  # noqa: E402

pytestmark = [pytest.mark.benchmark, pytest.mark.anyio]

#: The live loop's polling interval (``mage.live_lens.DEFAULT_INTERVAL_MS``).
LIVE_TICK_MS = 700

#: Frames at or under this many megapixels stand in for a locked region.
REGION_MEGAPIXELS = 1.0

#: What the removed OCR sidecar cost on a region-sized frame: ~342ms to detect
#: and read, plus a separate batch translation (~1.9s measured on a small NPU
#: model). Kept as the historical bar this path had to clear.
OCR_SIDECAR_TOTAL_MS = 2250


def _base_url() -> str:
    return normalize_lemonade_api_base_url(
        os.environ.get("XIAN_LEMONADE_URL", "http://localhost:13305")
    )


def _vision_model() -> str | None:
    """An installed model the server says can read an image.

    Translation-only models are excluded even when they carry a ``vision``
    label — TranslateGemma does, and is an image-text-to-text model, but it
    follows no instructions and cannot answer a grounding prompt. This is the
    same guard ``OmniModelRouter.vision()`` applies, for the same reason.
    """
    configured = os.environ.get("XIAN_BENCH_MODEL")
    if configured:
        return configured
    try:
        response = httpx.get(f"{_base_url()}/models", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception:
        return None

    for model in models:
        model_id = model.get("id") or ""
        if is_translation_only_model(model_id):
            continue
        if "vision" in (model.get("labels") or []):
            return model_id
    return None


class _Processor:
    """The slice of ``VLProcessor`` that grounding uses.

    ``encode_image`` is bound from the real class rather than reimplemented, so
    the JPEG path being measured is the one that ships.
    """

    encode_image = VLProcessor.encode_image

    def __init__(self, client, model: str):
        self.client = client
        self._model = model

    def get_vision_model_name(self) -> str:
        return self._model


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def strict_budget() -> bool:
    return corpus_strict_budget()


@pytest.fixture(scope="module")
def vision_model() -> str:
    model = _vision_model()
    if not model:
        pytest.skip(
            f"No vision-capable model installed on {_base_url()}. "
            "Install one, or set XIAN_BENCH_MODEL."
        )
    return model


@asynccontextmanager
async def open_processor(model: str):
    """A client created and closed inside the running loop.

    Deliberately not a fixture: the client owns a connection pool bound to the
    loop it was made on, and this suite runs under both pytest-asyncio (the
    workspace sets asyncio_mode = "auto") and an anyio marker. A fixture ends up
    straddling their loops and dies in teardown with "Event loop is closed",
    failing tests whose bodies already passed. Opening it inside the test body
    keeps creation, use and close on one loop.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=_base_url(), api_key="lemonade", max_retries=0)
    try:
        yield _Processor(client, model)
    finally:
        await client.close()


@pytest.fixture(scope="session")
def region_frames() -> list[Path]:
    """Corpus frames small enough to stand in for a locked region."""
    frames = []
    for path in corpus_paths():
        with Image.open(path) as image:
            if image.width * image.height / 1e6 <= REGION_MEGAPIXELS:
                frames.append(path)
    if not frames:
        pytest.skip("corpus has no region-sized frames")
    return frames


async def _ground(processor, path: Path, **kwargs):
    with Image.open(path) as opened:
        frame = opened.convert("RGB")
    started = time.perf_counter()
    regions = await ground_and_translate(processor, frame, "Chinese", "English", **kwargs)
    return (time.perf_counter() - started) * 1000, regions


# ── does it work at all? ─────────────────────────────────────────────

async def test_a_real_frame_yields_located_translations(vision_model, region_frames, record_property):
    """The whole feature in one assertion: text found, placed, and in English."""
    async with open_processor(vision_model) as processor:
        elapsed, regions = await _ground(processor, region_frames[0])

    record_property("ground_ms", round(elapsed, 1))
    record_property("regions", len(regions))

    assert regions, "no text regions came back from a frame that has text on it"
    for region in regions:
        assert region.is_valid()
        left, top, right, bottom = region.box
        assert 0 <= left < right and 0 <= top < bottom


async def test_translations_are_not_silent_fallbacks(vision_model, region_frames, record_property):
    """``parse_regions`` fills a missing translation with the original.

    That is the right behaviour — the source beats a blank box — but it means a
    model that cannot follow the grounding prompt produces a screen of Chinese
    painted over Chinese and no error. Same failure mode the batch translator
    has, checked the same way.
    """
    async with open_processor(vision_model) as processor:
        _elapsed, regions = await _ground(processor, region_frames[0])
    if not regions:
        pytest.skip("no regions to judge")

    untranslated = [r for r in regions if r.translated.strip() == r.original.strip()]
    record_property("fallback_regions", len(untranslated))
    record_property("total_regions", len(regions))

    assert len(untranslated) < len(regions), (
        f"all {len(regions)} regions came back as their own source text"
    )


# ── latency ──────────────────────────────────────────────────────────

async def test_grounding_a_locked_region_fits_the_live_budget(
    vision_model, region_frames, strict_budget, record_property
):
    """One vision call replacing OCR plus a translation call.

    Loose by default — wall-clock belongs to the machine — with
    ``XIAN_BENCH_STRICT=1`` holding it to the tick the loop actually polls on.
    """
    timings = []
    async with open_processor(vision_model) as processor:
        for path in region_frames[:5]:
            elapsed, _regions = await _ground(processor, path)
            timings.append(elapsed)

    median = statistics.median(timings)
    record_property("ground_ms_median", round(median, 1))
    record_property("ground_ms_max", round(max(timings), 1))
    record_property("frames", len(timings))
    record_property("fits_live_tick", bool(median < LIVE_TICK_MS))
    record_property("retired_ocr_sidecar_total_ms", OCR_SIDECAR_TOTAL_MS)

    budget = LIVE_TICK_MS if strict_budget else LIVE_TICK_MS * 10
    assert median < budget, (
        f"grounding a region-sized frame took {median:.0f}ms against a {LIVE_TICK_MS}ms tick"
    )


async def test_streaming_paints_the_first_line_well_before_the_last(
    vision_model, region_frames, record_property
):
    """The Phase 2 claim, measured.

    Streaming does not make the call faster; it makes the *first* translation
    appear sooner, which is what the user perceives as responsiveness. If the
    first region does not arrive meaningfully before the whole response, the
    added parsing complexity is not buying anything.
    """
    first_region_at: list[float] = []
    started = time.perf_counter()

    def on_region(_regions):
        if not first_region_at:
            first_region_at.append((time.perf_counter() - started) * 1000)

    with Image.open(region_frames[0]) as opened:
        frame = opened.convert("RGB")
    async with open_processor(vision_model) as processor:
        regions = await ground_and_translate(
            processor, frame, "Chinese", "English", on_region=on_region
        )
    total_ms = (time.perf_counter() - started) * 1000

    if not first_region_at:
        pytest.skip("the model returned no parseable region while streaming")

    record_property("time_to_first_region_ms", round(first_region_at[0], 1))
    record_property("total_ms", round(total_ms, 1))
    record_property("regions", len(regions))

    assert first_region_at[0] < total_ms, "the first region should land before the last"


# ── whole-screen capture ─────────────────────────────────────────────

#: Frames at or above this many megapixels stand in for a whole-screen capture.
SCREEN_MEGAPIXELS = 4.0


@pytest.fixture(scope="session")
def screen_frames() -> list[Path]:
    """Corpus frames big enough to stand in for a whole-screen capture.

    A different problem from a locked region, and the one that broke in
    practice: the text is a far smaller fraction of the frame, so the downscale
    that a region tolerates renders whole-screen glyphs unreadable.
    """
    frames = []
    for path in corpus_paths():
        with Image.open(path) as image:
            if image.width * image.height / 1e6 >= SCREEN_MEGAPIXELS:
                frames.append(path)
    if not frames:
        pytest.skip("corpus has no screen-sized frames")
    return frames


async def test_no_region_ever_covers_the_whole_screen(
    vision_model, screen_frames, record_property
):
    """The reported failure: a screen filled with one flat colour.

    The overlay fills every box it is handed, so a box the size of the frame
    hides the entire game behind a single rectangle carrying one line of text.
    Whatever the model returns, nothing that large may reach the caller.
    """
    from xian.grounding import MAX_FRAME_COVER

    worst = 0.0
    async with open_processor(vision_model) as processor:
        for path in screen_frames[:3]:
            with Image.open(path) as opened:
                frame = opened.convert("RGB")
            regions = await ground_and_translate(processor, frame, "Chinese", "English")
            frame_area = frame.width * frame.height
            for region in regions:
                worst = max(worst, region.width * region.height / frame_area)

    record_property("largest_region_frame_cover", round(worst, 3))
    assert worst < MAX_FRAME_COVER, (
        f"a region covered {worst:.0%} of the frame; the overlay would paint over the game"
    )


async def test_a_whole_screen_reads_better_than_at_the_locked_region_budget(
    vision_model, screen_frames, record_property
):
    """Why :data:`LIVE_MIN_SCALE` exists, measured rather than asserted.

    1024 across a 2880-wide capture leaves a measured 23px UI glyph at 8.2px.
    The request still succeeds and boxes still come back, so the only way to
    see the damage is to count what survives at each budget.

    Totalled over several frames rather than compared per frame: region counts
    from a generative model move around, and the sum is the stable statistic.
    """
    from shared_types.constants import LIVE_MAX_DIMENSION
    from xian.grounding import live_max_dimension

    flat_total = adaptive_total = 0
    flat_ms: list[float] = []
    adaptive_ms: list[float] = []

    async with open_processor(vision_model) as processor:
        for path in screen_frames[:3]:
            with Image.open(path) as opened:
                frame = opened.convert("RGB")

            started = time.perf_counter()
            flat = await ground_and_translate(
                processor, frame, "Chinese", "English", max_dimension=LIVE_MAX_DIMENSION
            )
            flat_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            adaptive = await ground_and_translate(processor, frame, "Chinese", "English")
            adaptive_ms.append((time.perf_counter() - started) * 1000)

            flat_total += sum(1 for r in flat if r.translated.strip() != r.original.strip())
            adaptive_total += sum(
                1 for r in adaptive if r.translated.strip() != r.original.strip()
            )

    record_property("budget_flat_px", LIVE_MAX_DIMENSION)
    record_property("budget_adaptive_px", live_max_dimension((2880, 1800)))
    record_property("translated_regions_flat", flat_total)
    record_property("translated_regions_adaptive", adaptive_total)
    record_property("ground_ms_flat_median", round(statistics.median(flat_ms), 1))
    record_property("ground_ms_adaptive_median", round(statistics.median(adaptive_ms), 1))

    assert adaptive_total >= flat_total, (
        f"the larger budget found fewer translated regions ({adaptive_total}) than "
        f"the flat {LIVE_MAX_DIMENSION}px one ({flat_total}) — LIVE_MIN_SCALE is not earning its latency"
    )
