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

"""The other half of a live tick: translating the lines OCR found.

`test_ocr_sidecar.py` drives :func:`batch_translate` against a mocked client,
so it proves the request is shaped right and the reply is parsed right.  It
cannot see the two things that decide whether live mode is usable:

* **Silent fallback.** Every failure path in ``batch_translate`` returns the
  *source* text — deliberately, because the original beats a blank box.  That
  makes a broken prompt, a timed-out model and a model that ignores the JSON
  contract all look identical to a working pipeline holding a Chinese-only
  screen.  Only a real model can tell them apart.
* **Fit.** The overlay paints each translation into the box its source came
  from.  A translation twice the width shrinks or wraps until it is unreadable,
  so "correct" is not enough — it has to be short.

Needs a Lemonade server with a chat model installed; skips otherwise.  Point
``XIAN_LEMONADE_URL`` at it (default ``http://localhost:13305``) and optionally
name a model with ``XIAN_BENCH_MODEL``.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark_corpus import strict_budget as corpus_strict_budget  # noqa: E402

from xian.lemonade_url import normalize_lemonade_api_base_url  # noqa: E402
from xian.ocr.translate import batch_translate  # noqa: E402
from xian.omni_router import is_translation_only_model  # noqa: E402

pytestmark = [pytest.mark.benchmark, pytest.mark.anyio]

#: Same tick the live loop polls on (``mage.live_lens.DEFAULT_INTERVAL_MS``).
LIVE_TICK_MS = 700

#: A screen's worth of lines from a real game HUD: quest text, menu labels,
#: item names, a chat line.  Fixed rather than OCR'd so this benchmark runs
#: without the optional OCR dependencies — it is measuring the translator.
SAMPLE_LINES = [
    "前往長安城尋找李掌櫃",
    "已完成",
    "確定",
    "取消",
    "背包已滿",
    "等級75 法師",
    "選擇伺服器",
    "創建新角色",
    "系統區域語言設置不符合要求，請手動修復",
    "你獲得了【雷鳴脊背雷象】",
    "生命值",
    "刪除角色",
]

#: How much longer than its source a translation may run before it stops
#: fitting the box it has to be painted into.  English against Chinese is
#: legitimately longer per character — a Han character is a whole word — so
#: this is measured in rendered width, roughly two Latin characters per Han
#: character before the overlay has to start shrinking the font.
MAX_WIDTH_EXPANSION = 2.5


def _lemonade_base_url() -> str:
    return normalize_lemonade_api_base_url(
        os.environ.get("XIAN_LEMONADE_URL", "http://localhost:13305")
    )


def _installed_chat_model() -> str | None:
    """A chat model this Lemonade actually has, or None.

    ``/v1/models`` lists what is installed and loadable; the catalogue of
    everything downloadable is a different endpoint, and asking for a model
    from it would trigger a multi-gigabyte pull mid-test.
    """
    configured = os.environ.get("XIAN_BENCH_MODEL")
    if configured:
        return configured
    try:
        response = httpx.get(f"{_lemonade_base_url()}/models", timeout=5.0)
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception:
        return None

    for model in models:
        labels = model.get("labels") or []
        if not labels or "chat" in labels:
            return model.get("id")
    return models[0].get("id") if models else None


class _Processor:
    """The slice of ``VLProcessor`` that :func:`batch_translate` actually uses.

    Constructing the real processor would start the async engine and its
    background thread; this benchmark only needs a client and a model name, and
    routes through the real ``is_translation_only_model`` so the MT path is
    chosen exactly as it would be in the app.
    """

    def __init__(self, client, model: str):
        self.client = client
        self._model = model
        self.router = self

    def get_translation_model_name(self) -> str:
        return self._model

    def is_translation_only(self, model_id: str | None) -> bool:
        return is_translation_only_model(model_id)


def _rendered_width(text: str) -> float:
    """Approximate painted width in Latin-character units.

    A Han character occupies about two Latin characters at the same point size,
    which is what makes a naive character-count comparison misleading here.
    """
    return sum(2.0 if "一" <= char <= "鿿" else 1.0 for char in text)


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def strict_budget() -> bool:
    return corpus_strict_budget()


@pytest.fixture(scope="module")
def translation_model() -> str:
    model = _installed_chat_model()
    if not model:
        pytest.skip(
            f"No model installed on the Lemonade server at {_lemonade_base_url()}. "
            "Install one (`lemonade pull <model>`) or set XIAN_BENCH_MODEL."
        )
    return model


@pytest.fixture
def processor(translation_model):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=_lemonade_base_url(), api_key="lemonade", max_retries=0)
    return _Processor(client, translation_model)


@pytest.fixture(scope="module")
def translated(translation_model) -> list[str]:
    """One real translation pass, shared by the tests that inspect its output."""
    import asyncio

    from openai import AsyncOpenAI

    async def run():
        client = AsyncOpenAI(base_url=_lemonade_base_url(), api_key="lemonade", max_retries=0)
        return await batch_translate(
            _Processor(client, translation_model), SAMPLE_LINES, "Chinese", "English"
        )

    return asyncio.run(run())


# ── the silent-fallback failure mode ─────────────────────────────────

def test_no_line_falls_back_to_its_source_text(translated, record_property):
    """Every fallback path returns the original, so untranslated == failed.

    This is the check that a mocked test cannot make: with a real model, a line
    that comes back byte-identical to its Chinese source means the request
    failed, the reply missed that key, or the model ignored the JSON contract —
    and the overlay would paint Chinese over Chinese, looking like it worked.
    """
    untouched = [
        source for source, result in zip(SAMPLE_LINES, translated) if source == result
    ]
    record_property("fallback_lines", len(untouched))
    record_property("total_lines", len(SAMPLE_LINES))

    assert not untouched, (
        f"{len(untouched)} of {len(SAMPLE_LINES)} lines came back as their own source text: "
        f"{untouched[:4]}"
    )


def test_translations_are_in_the_target_language(translated):
    """A reply that is still Han characters is not a translation.

    Catches a model that echoes, transliterates, or answers in the source
    language — all of which pass a "did we get a string back" check.
    """
    for source, result in zip(SAMPLE_LINES, translated):
        assert result.strip(), f"empty translation for {source!r}"
        han = sum(1 for char in result if "一" <= char <= "鿿")
        assert han / len(result) < 0.3, f"{source!r} -> {result!r} is still mostly Han"


def test_translations_fit_the_boxes_they_are_painted_into(translated, record_property):
    """The prompt asks for on-screen brevity; this checks the model obeyed.

    An over-long translation is not a wrong translation, but the overlay can
    only shrink the font so far before wrapping a HUD label into unreadable
    two-point text.
    """
    ratios = [
        _rendered_width(result) / max(1.0, _rendered_width(source))
        for source, result in zip(SAMPLE_LINES, translated)
    ]
    record_property("width_expansion_median", round(statistics.median(ratios), 2))
    record_property("width_expansion_max", round(max(ratios), 2))

    worst = max(zip(ratios, SAMPLE_LINES, translated))
    assert statistics.median(ratios) < MAX_WIDTH_EXPANSION, (
        f"translations run {statistics.median(ratios):.1f}x the width of their source; "
        f"worst: {worst[1]!r} -> {worst[2]!r}"
    )


# ── latency: does the other half of the tick fit? ────────────────────

async def test_a_screens_worth_of_lines_translates_within_the_live_budget(
    processor, strict_budget, record_property
):
    """Translation shares the tick with OCR, and it is the larger half.

    The sidecar's whole argument is that OCR plus a text-only translation beats
    one vision call.  OCR is measured in ``test_screenshot_benchmark.py``; this
    is the rest of the bill.
    """
    started = time.perf_counter()
    results = await batch_translate(processor, SAMPLE_LINES, "Chinese", "English")
    elapsed_ms = (time.perf_counter() - started) * 1000

    record_property("batch_translate_ms", round(elapsed_ms, 1))
    record_property("lines", len(SAMPLE_LINES))
    record_property("fits_live_tick", bool(elapsed_ms < LIVE_TICK_MS))

    assert len(results) == len(SAMPLE_LINES)
    budget = LIVE_TICK_MS if strict_budget else LIVE_TICK_MS * 10
    assert elapsed_ms < budget, (
        f"translating {len(SAMPLE_LINES)} lines took {elapsed_ms:.0f}ms "
        f"against a {LIVE_TICK_MS}ms tick"
    )


async def test_batching_the_screen_beats_one_request_per_line(processor, record_property):
    """Why every line goes in one request.

    ``batch_translate`` sends the whole screen as numbered JSON so the model
    keeps names consistent between lines.  That choice only holds up if it is
    also not slower than the obvious alternative — one request per line, which
    is what the dedicated-MT path is forced to do.
    """
    started = time.perf_counter()
    await batch_translate(processor, SAMPLE_LINES, "Chinese", "English")
    batched_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    for line in SAMPLE_LINES:
        await batch_translate(processor, [line], "Chinese", "English")
    serial_ms = (time.perf_counter() - started) * 1000

    record_property("batched_ms", round(batched_ms, 1))
    record_property("one_per_line_ms", round(serial_ms, 1))

    assert batched_ms < serial_ms, (
        f"one batched call ({batched_ms:.0f}ms) was no faster than "
        f"{len(SAMPLE_LINES)} separate ones ({serial_ms:.0f}ms)"
    )
