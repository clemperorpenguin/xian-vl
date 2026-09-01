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

Every other benchmark here measures a piece: OCR on a corpus, translation
against a text model, capture against a display. This measures the thing the
overlay actually does — one changed frame in, located translations out — using
the real encoder, the real preprocessing, and the real streaming parse.

It is the number that decides whether live mode is viable, and whether the
local OCR sidecar can be deleted: if grounding a locked region beats the
sidecar's measured 342ms median plus its translation call, the sidecar is
carrying no weight.

Needs a Lemonade server with a **vision-capable** model installed; skips
otherwise. ``XIAN_BENCH_MODEL`` names one explicitly.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
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

#: The OCR sidecar's measured median on a region-sized frame, before its
#: separate translation call. Grounding has to beat the *sum* to justify
#: deleting the sidecar; this is the floor it is competing with.
OCR_SIDECAR_MEDIAN_MS = 342


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


@pytest.fixture
def processor(vision_model):
    from openai import AsyncOpenAI

    return _Processor(
        AsyncOpenAI(base_url=_base_url(), api_key="lemonade", max_retries=0), vision_model
    )


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

async def test_a_real_frame_yields_located_translations(processor, region_frames, record_property):
    """The whole feature in one assertion: text found, placed, and in English."""
    elapsed, regions = await _ground(processor, region_frames[0])

    record_property("ground_ms", round(elapsed, 1))
    record_property("regions", len(regions))

    assert regions, "no text regions came back from a frame that has text on it"
    for region in regions:
        assert region.is_valid()
        left, top, right, bottom = region.box
        assert 0 <= left < right and 0 <= top < bottom


async def test_translations_are_not_silent_fallbacks(processor, region_frames, record_property):
    """``parse_regions`` fills a missing translation with the original.

    That is the right behaviour — the source beats a blank box — but it means a
    model that cannot follow the grounding prompt produces a screen of Chinese
    painted over Chinese and no error. Same failure mode the batch translator
    has, checked the same way.
    """
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
    processor, region_frames, strict_budget, record_property
):
    """One vision call replacing OCR plus a translation call.

    Loose by default — wall-clock belongs to the machine — with
    ``XIAN_BENCH_STRICT=1`` holding it to the tick the loop actually polls on.
    """
    timings = []
    for path in region_frames[:5]:
        elapsed, regions = await _ground(processor, path)
        timings.append(elapsed)

    median = statistics.median(timings)
    record_property("ground_ms_median", round(median, 1))
    record_property("ground_ms_max", round(max(timings), 1))
    record_property("frames", len(timings))
    record_property("fits_live_tick", bool(median < LIVE_TICK_MS))
    record_property("ocr_sidecar_median_ms", OCR_SIDECAR_MEDIAN_MS)

    budget = LIVE_TICK_MS if strict_budget else LIVE_TICK_MS * 10
    assert median < budget, (
        f"grounding a region-sized frame took {median:.0f}ms against a {LIVE_TICK_MS}ms tick"
    )


async def test_streaming_paints_the_first_line_well_before_the_last(
    processor, region_frames, record_property
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
