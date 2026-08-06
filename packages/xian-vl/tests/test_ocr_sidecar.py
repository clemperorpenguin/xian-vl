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

"""Local OCR line handling and batch translation.

The ONNX models themselves are an optional dependency, so the tests that need
them skip when they are absent; everything else is exercised unconditionally.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from xian.ocr.engine import OcrLine, merge_adjacent_lines, ocr_available
from xian.ocr.onnx_engine import PROVIDER_PREFERENCE, _polygon_to_box, select_providers
from xian.ocr.translate import batch_translate, parse_translations


# ── line geometry ────────────────────────────────────────────────────

def test_detector_quadrilaterals_reduce_to_axis_aligned_boxes():
    # A slightly rotated detection, as detectors actually return them.
    polygon = [[10, 20], [110, 24], [110, 44], [10, 40]]
    assert _polygon_to_box(polygon, 200, 100) == (10, 20, 110, 44)


def test_boxes_are_clamped_to_the_frame():
    polygon = [[-30, -10], [400, -10], [400, 300], [-30, 300]]
    assert _polygon_to_box(polygon, 200, 100) == (0, 0, 200, 100)


@pytest.mark.parametrize("polygon", [[], [[5, 5]], "nonsense", [[0, 0], [0, 0], [0, 0], [0, 0]]])
def test_degenerate_polygons_are_rejected(polygon):
    assert _polygon_to_box(polygon, 200, 100) is None


def test_fragments_of_one_spaced_out_label_are_merged():
    """Detectors split wide letter-spacing into pieces; one fill looks better."""
    lines = [
        OcrLine((10, 20, 60, 40), "純陽"),
        OcrLine((68, 20, 120, 40), "宮"),
    ]
    merged = merge_adjacent_lines(lines)

    assert len(merged) == 1
    assert merged[0].box == (10, 20, 120, 40)
    assert merged[0].text == "純陽 宮"


def test_separate_rows_are_not_merged():
    lines = [OcrLine((10, 20, 60, 40), "first"), OcrLine((10, 60, 60, 80), "second")]
    assert len(merge_adjacent_lines(lines)) == 2


def test_distant_fragments_on_the_same_row_are_not_merged():
    lines = [OcrLine((10, 20, 60, 40), "left"), OcrLine((300, 20, 360, 40), "right")]
    assert len(merge_adjacent_lines(lines)) == 2


def test_merging_is_a_no_op_for_a_single_line():
    lines = [OcrLine((10, 20, 60, 40), "solo")]
    assert merge_adjacent_lines(lines) == lines


# ── provider selection ───────────────────────────────────────────────

def test_cpu_is_always_a_candidate_even_with_nothing_installed(monkeypatch):
    monkeypatch.setattr("xian.ocr.onnx_engine.available_providers", lambda: [])
    assert select_providers() == ["CPUExecutionProvider"]


def test_npu_is_preferred_over_gpu_and_cpu(monkeypatch):
    monkeypatch.setattr(
        "xian.ocr.onnx_engine.available_providers",
        lambda: ["CPUExecutionProvider", "ROCMExecutionProvider", "VitisAIExecutionProvider"],
    )
    assert select_providers()[0] == "VitisAIExecutionProvider"


def test_npu_can_be_declined(monkeypatch):
    monkeypatch.setattr(
        "xian.ocr.onnx_engine.available_providers",
        lambda: ["VitisAIExecutionProvider", "CPUExecutionProvider"],
    )
    assert "VitisAIExecutionProvider" not in select_providers(prefer_npu=False)


def test_provider_preference_puts_cpu_last():
    assert PROVIDER_PREFERENCE[-1] == "CPUExecutionProvider"


# ── batch translation ────────────────────────────────────────────────

def test_numbered_replies_map_back_to_input_order():
    raw = json.dumps({"1": "Head to the gate", "2": "Press F", "3": "Level up"})
    assert parse_translations(raw, 3) == ["Head to the gate", "Press F", "Level up"]


def test_out_of_order_and_partial_replies_still_land_correctly():
    raw = json.dumps({"3": "third", "1": "first"})
    assert parse_translations(raw, 3) == ["first", "", "third"]


@pytest.mark.parametrize("raw", ["", "no json", "[1,2]", "{bad json", "null"])
def test_unusable_replies_yield_blanks(raw):
    assert parse_translations(raw, 2) == ["", ""]


def test_parser_ignores_keys_outside_the_requested_range():
    raw = json.dumps({"0": "zero", "1": "one", "9": "nine", "x": "letter"})
    assert parse_translations(raw, 2) == ["one", ""]


def test_fenced_and_thinking_wrapped_replies_still_parse():
    payload = json.dumps({"1": "translated"})
    assert parse_translations(f"<think>hmm</think>```json\n{payload}\n```", 1) == ["translated"]


def _processor(reply: str):
    processor = MagicMock()
    choice = MagicMock()
    choice.message.content = reply
    response = MagicMock()
    response.choices = [choice]
    processor.client.chat.completions.create = AsyncMock(return_value=response)
    processor.get_model_name.return_value = "test-llm"
    return processor


@pytest.mark.anyio
async def test_all_lines_are_translated_in_one_request():
    processor = _processor(json.dumps({"1": "Gate", "2": "Market"}))

    result = await batch_translate(processor, ["城門", "市場"], "Chinese", "English")

    assert result == ["Gate", "Market"]
    assert processor.client.chat.completions.create.await_count == 1, "one call keeps terminology consistent"


@pytest.mark.anyio
async def test_a_missing_translation_falls_back_to_the_source_text():
    """A blank box over the original is worse than leaving the original."""
    processor = _processor(json.dumps({"1": "Gate"}))

    assert await batch_translate(processor, ["城門", "市場"], "Chinese", "English") == ["Gate", "市場"]


@pytest.mark.anyio
async def test_a_failed_request_leaves_every_line_readable():
    processor = _processor("")
    processor.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("server down"))

    assert await batch_translate(processor, ["城門"], "Chinese", "English") == ["城門"]


@pytest.mark.anyio
async def test_no_lines_means_no_request():
    processor = _processor("{}")
    assert await batch_translate(processor, [], "Chinese", "English") == []
    processor.client.chat.completions.create.assert_not_awaited()


@pytest.mark.anyio
async def test_glossary_terms_reach_the_translator():
    processor = _processor(json.dumps({"1": "Pure Yang"}))

    await batch_translate(
        processor, ["純陽"], "Chinese", "English", glossary={"純陽": "Pure Yang"}
    )

    system = processor.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "純陽: Pure Yang" in system


# ── optional dependency ──────────────────────────────────────────────

def test_availability_probe_does_not_raise_when_deps_are_missing():
    assert isinstance(ocr_available(), bool)


@pytest.mark.skipif(not ocr_available(), reason="optional OCR dependencies not installed")
def test_engine_reads_rendered_text_and_reports_its_provider():
    from PIL import Image, ImageDraw

    from xian.ocr.onnx_engine import OnnxOcrEngine

    image = Image.new("RGB", (480, 120), "white")
    ImageDraw.Draw(image).text((20, 40), "HELLO WORLD", fill="black")

    engine = OnnxOcrEngine()
    lines = engine.run(image)

    assert engine.provider.endswith("ExecutionProvider")
    assert any("HELLO" in ln.text.upper() for ln in lines)
