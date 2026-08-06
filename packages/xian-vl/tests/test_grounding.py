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

"""Located-text parsing for the in-place translation overlay."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from xian.grounding import (
    TextRegion,
    ground_and_translate,
    parse_regions,
    preprocess_for_grounding,
    rescale_regions,
)

SAMPLE = [
    {"box": [100, 200, 500, 260], "original": "前往城門", "translated": "Head to the city gate"},
    {"box": [0, 900, 1000, 1000], "original": "按 F 繼續", "translated": "Press F to continue"},
]


def test_boxes_map_from_the_0_1000_grid_into_pixels():
    regions = parse_regions(json.dumps(SAMPLE), 1920, 1080)

    assert len(regions) == 2
    assert regions[0].box == (192, 216, 960, 280)
    assert regions[0].translated == "Head to the city gate"
    assert regions[1].box == (0, 972, 1920, 1080)


@pytest.mark.parametrize("wrapper", [
    "```json\n{payload}\n```",
    "Here are the regions:\n{payload}\nThat is all.",
    "<think>The image shows a quest log.</think>{payload}",
    "```\n{payload}\n```",
])
def test_parser_tolerates_the_wrappers_models_actually_emit(wrapper):
    raw = wrapper.format(payload=json.dumps(SAMPLE))
    assert len(parse_regions(raw, 1000, 1000)) == 2


@pytest.mark.parametrize("raw", ["", "no json here", "[", "{}", "null", "[1, 2, 3]"])
def test_unusable_responses_yield_no_regions(raw):
    assert parse_regions(raw, 800, 600) == []


def test_inverted_and_out_of_range_boxes_are_normalized():
    raw = json.dumps([
        {"box": [500, 260, 100, 200], "translated": "reversed corners"},
        {"box": [-50, -50, 1200, 1200], "translated": "overflowing"},
    ])
    regions = parse_regions(raw, 400, 300)

    # x: 100..500 of 1000 across 400px; y: 200..260 of 1000 down 300px.
    assert regions[0].box == (40, 60, 200, 78)
    assert regions[1].box == (0, 0, 400, 300)


def test_regions_without_usable_text_or_area_are_dropped():
    raw = json.dumps([
        {"box": [10, 10, 20, 20], "original": "文", "translated": ""},
        {"box": [10, 10, 10, 20], "translated": "zero width"},
        {"box": [10, 10], "translated": "truncated box"},
        {"box": ["x", "y", "z", "w"], "translated": "non-numeric"},
        "not an object",
    ])
    regions = parse_regions(raw, 1000, 1000)

    # Only the first survives: a blank translation falls back to the original.
    assert [r.translated for r in regions] == ["文"]


def test_grounding_preprocessing_does_not_pad_to_a_square():
    """Padding would offset every box the model reports."""
    prepared = preprocess_for_grounding(Image.new("RGB", (1600, 400)), max_dimension=800)

    assert prepared.size == (800, 200)
    assert prepared.width != prepared.height


def test_preprocessing_leaves_small_frames_alone():
    assert preprocess_for_grounding(Image.new("RGB", (300, 120))).size == (300, 120)


def test_rescale_maps_regions_back_to_the_original_frame():
    regions = [TextRegion((100, 50, 200, 100), "文", "text")]

    scaled = rescale_regions(regions, (400, 200), (800, 400))

    assert scaled[0].box == (200, 100, 400, 200)


@pytest.mark.anyio
async def test_ground_and_translate_returns_boxes_in_source_pixels():
    """A downscaled frame must not leave boxes in the downscaled space."""
    from xian.pipeline import VLConfig, VLProcessor

    processor = VLProcessor(VLConfig())
    choice = MagicMock()
    choice.message.content = json.dumps([
        {"box": [0, 0, 500, 100], "original": "文", "translated": "text"}
    ])
    response = MagicMock()
    response.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    processor.engine = MagicMock()
    processor.engine.client = mock_client

    source = Image.new("RGB", (3840, 800))
    regions = await ground_and_translate(processor, source, "Chinese", "English")

    # 0-500 of 1000 across a 3840px-wide source is the left half.
    assert regions[0].box[2] == pytest.approx(1920, abs=4)
    assert regions[0].box[3] == pytest.approx(80, abs=4)


@pytest.mark.anyio
async def test_grounding_prompt_carries_the_glossary():
    from xian.pipeline import VLConfig, VLProcessor

    processor = VLProcessor(VLConfig())
    choice = MagicMock()
    choice.message.content = "[]"
    response = MagicMock()
    response.choices = [choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    processor.engine = MagicMock()
    processor.engine.client = mock_client

    await ground_and_translate(
        processor, Image.new("RGB", (200, 100)), "Chinese", "English",
        glossary={"純陽": "Pure Yang"},
    )

    system = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "純陽: Pure Yang" in system
