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


# ── live-path encoding and sizing ────────────────────────────────────

def test_the_live_path_sends_a_jpeg_not_a_png():
    """PNG costs ~200ms and ~3MB per frame on a full-size game frame.

    The live overlay re-sends on every screen change, so that is paid over and
    over. The data URL has to declare what was actually encoded, or the server
    is handed a JPEG labelled as a PNG.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from PIL import Image

    from xian.grounding import ground_and_translate

    processor = MagicMock()
    choice = MagicMock()
    choice.message.content = "[]"
    response = MagicMock()
    response.choices = [choice]
    processor.client.chat.completions.create = AsyncMock(return_value=response)
    processor.get_vision_model_name.return_value = "test-vlm"
    processor.encode_image.side_effect = lambda img, **kw: "BASE64"

    asyncio.run(ground_and_translate(
        processor, Image.new("RGB", (2880, 1800)), "Chinese", "English"
    ))

    assert processor.encode_image.call_args.kwargs["fmt"] == "JPEG"

    content = processor.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
    assert url.startswith("data:image/jpeg;base64,")


def test_the_live_path_downscales_further_than_the_one_shot_path():
    """Prefill scales with area, so the live frame is capped well below 1920."""
    from PIL import Image

    from shared_types.constants import LIVE_MAX_DIMENSION, QWEN_MAX_DIMENSION

    assert LIVE_MAX_DIMENSION < QWEN_MAX_DIMENSION

    prepared = preprocess_for_grounding(
        Image.new("RGB", (2880, 1800)), LIVE_MAX_DIMENSION, sharpen=False
    )
    assert max(prepared.size) == LIVE_MAX_DIMENSION
    # Aspect ratio has to survive: the boxes come back normalized against it.
    assert abs(prepared.size[0] / prepared.size[1] - 2880 / 1800) < 0.01


def test_sharpening_is_optional_and_leaves_the_pixels_alone_when_off():
    """A whole-image convolution is 17-19ms; the live path does not pay it."""
    from PIL import Image

    source = Image.new("RGB", (64, 64), (120, 130, 140))
    source.putpixel((32, 32), (255, 0, 0))

    assert preprocess_for_grounding(source, 128, sharpen=False).tobytes() == source.tobytes()
    assert preprocess_for_grounding(source, 128, sharpen=True).tobytes() != source.tobytes()


# ── overlap suppression ──────────────────────────────────────────────

def test_a_box_nested_inside_another_is_dropped():
    """The overlay fills then draws per region in order, so an overlapping
    fill paints out the translation drawn under it."""
    from xian.grounding import suppress_overlapping_regions

    outer = TextRegion((10, 10, 200, 60), "外", "outer")
    inner = TextRegion((20, 20, 80, 50), "内", "inner")

    kept = suppress_overlapping_regions([outer, inner])

    assert [r.translated for r in kept] == ["outer"]


def test_boxes_that_merely_touch_are_both_kept():
    from xian.grounding import suppress_overlapping_regions

    left = TextRegion((0, 0, 50, 20), "左", "left")
    right = TextRegion((50, 0, 100, 20), "右", "right")

    assert len(suppress_overlapping_regions([left, right])) == 2


def test_stacked_lines_with_slight_overlap_survive():
    """Adjacent rows of a chat log graze each other; that is not a conflict."""
    from xian.grounding import suppress_overlapping_regions

    first = TextRegion((0, 0, 200, 22), "一", "first")
    second = TextRegion((0, 20, 200, 42), "二", "second")

    assert len(suppress_overlapping_regions([first, second])) == 2


def test_reading_order_is_preserved_after_suppression():
    """Largest-first is only the selection order; output stays as given."""
    from xian.grounding import suppress_overlapping_regions

    regions = [
        TextRegion((0, 0, 40, 20), "一", "first"),
        TextRegion((0, 100, 300, 160), "二", "second"),
        TextRegion((0, 200, 60, 220), "三", "third"),
    ]
    assert [r.translated for r in suppress_overlapping_regions(regions)] == [
        "first", "second", "third",
    ]


# ── incremental streaming parse ──────────────────────────────────────

def test_complete_objects_are_pulled_from_a_partial_array():
    """A streamed response is not valid JSON until it ends; the overlay should
    not have to wait for the last box to paint the first."""
    from xian.grounding import scan_json_objects

    partial = '[{"box":[1,2,3,4],"original":"a","translated":"A"},{"box":[5,6'
    objects, consumed = scan_json_objects(partial)

    assert objects == ['{"box":[1,2,3,4],"original":"a","translated":"A"}']
    assert consumed == partial.index("},") + 1


def test_scanning_resumes_without_reemitting_earlier_objects():
    from xian.grounding import scan_json_objects

    buffer = '[{"a":1},'
    first, consumed = scan_json_objects(buffer)
    buffer += '{"b":2}]'
    second, _ = scan_json_objects(buffer, consumed)

    assert first == ['{"a":1}']
    assert second == ['{"b":2}']


def test_braces_inside_strings_do_not_close_an_object():
    from xian.grounding import scan_json_objects

    buffer = '[{"original":"a } brace","translated":"ok"}]'
    objects, _ = scan_json_objects(buffer)

    assert objects == ['{"original":"a } brace","translated":"ok"}']


def test_an_escaped_quote_does_not_end_the_string():
    from xian.grounding import scan_json_objects

    buffer = r'[{"original":"say \"hi\" }","translated":"ok"}]'
    objects, _ = scan_json_objects(buffer)

    assert objects == [r'{"original":"say \"hi\" }","translated":"ok"}']


def test_nested_objects_close_at_the_right_brace():
    from xian.grounding import scan_json_objects

    buffer = '[{"box":[1,2,3,4],"meta":{"n":1}},{"x":2}]'
    objects, _ = scan_json_objects(buffer)

    assert objects == ['{"box":[1,2,3,4],"meta":{"n":1}}', '{"x":2}']


def test_a_buffer_with_no_complete_object_yields_nothing():
    from xian.grounding import scan_json_objects

    assert scan_json_objects('[{"box":[1,2') == ([], 0)


# ── streaming end to end ─────────────────────────────────────────────

class _FakeStream:
    """An async iterator of OpenAI-shaped streaming chunks."""

    def __init__(self, pieces):
        self._pieces = list(pieces)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._pieces:
            raise StopAsyncIteration
        from unittest.mock import MagicMock

        chunk = MagicMock()
        chunk.choices[0].delta.content = self._pieces.pop(0)
        return chunk


def _streaming_processor(pieces):
    from unittest.mock import AsyncMock, MagicMock

    processor = MagicMock()
    processor.client.chat.completions.create = AsyncMock(return_value=_FakeStream(pieces))
    processor.get_vision_model_name.return_value = "test-vlm"
    processor.encode_image.side_effect = lambda img, **kw: "BASE64"
    return processor


def test_regions_are_published_as_they_arrive_not_only_at_the_end():
    """The point of streaming: the first line paints without waiting for the last."""
    import asyncio

    from PIL import Image

    from xian.grounding import ground_and_translate

    processor = _streaming_processor([
        '[{"box":[0,0,500,100],"original":"一","translated":"first"}',
        ',{"box":[0,200,500,300],"original":"二","translated":"second"}]',
    ])

    batches = []
    result = asyncio.run(ground_and_translate(
        processor, Image.new("RGB", (400, 400)), "Chinese", "English",
        on_region=lambda regions: batches.append([r.translated for r in regions]),
    ))

    assert batches == [["first"], ["first", "second"]], "should paint progressively"
    assert [r.translated for r in result] == ["first", "second"]


def test_a_stream_that_dies_midway_keeps_what_it_decoded():
    """A partial screen of translations beats an exception and a blank overlay."""
    import asyncio

    from PIL import Image

    from xian.grounding import ground_and_translate

    class _DyingStream(_FakeStream):
        async def __anext__(self):
            if not self._pieces:
                raise RuntimeError("connection reset")
            return await super().__anext__()

    from unittest.mock import AsyncMock, MagicMock

    processor = MagicMock()
    processor.client.chat.completions.create = AsyncMock(
        return_value=_DyingStream(['[{"box":[0,0,500,100],"original":"一","translated":"kept"}'])
    )
    processor.get_vision_model_name.return_value = "test-vlm"
    processor.encode_image.side_effect = lambda img, **kw: "BASE64"

    result = asyncio.run(ground_and_translate(
        processor, Image.new("RGB", (400, 400)), "Chinese", "English",
        on_region=lambda regions: None,
    ))

    assert [r.translated for r in result] == ["kept"]


def test_the_non_streaming_path_is_unchanged_without_a_callback():
    import asyncio

    from unittest.mock import AsyncMock, MagicMock

    from PIL import Image

    from xian.grounding import ground_and_translate

    processor = MagicMock()
    choice = MagicMock()
    choice.message.content = '[{"box":[0,0,500,100],"original":"一","translated":"only"}]'
    response = MagicMock()
    response.choices = [choice]
    processor.client.chat.completions.create = AsyncMock(return_value=response)
    processor.get_vision_model_name.return_value = "test-vlm"
    processor.encode_image.side_effect = lambda img, **kw: "BASE64"

    result = asyncio.run(ground_and_translate(
        processor, Image.new("RGB", (400, 400)), "Chinese", "English"
    ))

    assert [r.translated for r in result] == ["only"]
    assert "stream" not in processor.client.chat.completions.create.call_args.kwargs


# ── container and frame-sized boxes ──────────────────────────────────

def test_a_panel_box_does_not_delete_the_lines_inside_it():
    """The bug that emptied a whole-screen capture.

    Asked for one box per line, the model returns the panel as well. Selecting
    largest-first then kept the panel and dropped every line it covered, so the
    overlay painted one opaque slab carrying a single translation where five
    lines should have been.
    """
    from xian.grounding import suppress_overlapping_regions

    panel = TextRegion((0, 0, 400, 300), "面板", "panel")
    lines = [
        TextRegion((10, 10 + row * 50, 380, 50 + row * 50), f"第{row}", f"line {row}")
        for row in range(5)
    ]

    kept = suppress_overlapping_regions([panel, *lines])

    assert [r.translated for r in kept] == [f"line {row}" for row in range(5)]


def test_a_box_containing_only_one_other_is_still_the_survivor():
    """One nested box is two attempts at the same line, not a container.

    The larger of the pair is the better answer — it is the one likely to hold
    the whole line rather than a fragment — so this case must stay as it was.
    """
    from xian.grounding import suppress_overlapping_regions

    outer = TextRegion((10, 10, 200, 60), "外", "outer")
    inner = TextRegion((20, 20, 80, 50), "内", "inner")

    assert [r.translated for r in suppress_overlapping_regions([outer, inner])] == ["outer"]


def test_a_box_covering_the_whole_frame_is_dropped():
    """"World of Warcraft" over a screen filled with one flat colour.

    A box this size is the model boxing the picture. The overlay fills every
    box it is given, so painting this one hides the entire game behind a
    rectangle — the single worst thing the overlay can do.
    """
    from xian.grounding import suppress_overlapping_regions

    whole_frame = TextRegion((0, 0, 1000, 620), "魔獸世界", "World of Warcraft")
    line = TextRegion((40, 500, 300, 530), "背包已滿", "Bag is full")

    kept = suppress_overlapping_regions([whole_frame, line], frame_size=(1024, 640))

    assert [r.translated for r in kept] == ["Bag is full"]


def test_a_frame_sized_box_survives_when_it_is_the_only_one():
    """A region drawn tightly around one line is legitimately almost all frame.

    There is nothing else to show in that case, so the rule only fires where a
    frame-sized box is demonstrably spurious: alongside smaller ones.
    """
    from xian.grounding import suppress_overlapping_regions

    only = TextRegion((2, 2, 298, 58), "背包已滿", "Bag is full")

    kept = suppress_overlapping_regions([only], frame_size=(300, 60))

    assert [r.translated for r in kept] == ["Bag is full"]


def test_frame_sized_suppression_needs_a_frame_to_measure_against():
    """Without a frame size there is no such thing as "frame-sized".

    The same pair is then an ordinary overlap, and the larger box wins as it
    always did — which is why the one caller that knows the frame passes it.
    """
    from xian.grounding import suppress_overlapping_regions

    big = TextRegion((0, 0, 1000, 620), "魔獸世界", "World of Warcraft")
    line = TextRegion((40, 500, 300, 530), "背包已滿", "Bag is full")

    assert [r.translated for r in suppress_overlapping_regions([big, line])] == [
        "World of Warcraft"
    ]
    assert [r.translated for r in suppress_overlapping_regions(
        [big, line], frame_size=(1024, 640)
    )] == ["Bag is full"]


# ── frame-size-aware downscaling ─────────────────────────────────────

def test_a_locked_region_keeps_the_measured_live_budget():
    from shared_types.constants import LIVE_MAX_DIMENSION
    from xian.grounding import live_max_dimension

    assert live_max_dimension((800, 600)) == LIVE_MAX_DIMENSION
    assert live_max_dimension((291, 442)) == LIVE_MAX_DIMENSION


def test_a_whole_screen_capture_gets_a_larger_budget():
    """1024 across 2880 renders a measured 23px glyph as 8.2px.

    Too small for a ten-stroke character to stay distinct, and the failure is
    silent — boxes come back in plausible places carrying useless text — so the
    budget has to grow with the capture instead.
    """
    from xian.grounding import live_max_dimension

    assert live_max_dimension((2880, 1800)) > 1024


def test_the_live_budget_never_exceeds_the_one_shot_path():
    from shared_types.constants import QWEN_MAX_DIMENSION
    from xian.grounding import live_max_dimension

    assert live_max_dimension((7680, 4320)) == QWEN_MAX_DIMENSION


def test_grounding_sizes_the_frame_from_the_capture_it_was_given():
    """Both sizes go through one call, so the wiring is what is under test."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from PIL import Image

    from xian.grounding import ground_and_translate, live_max_dimension

    def prepared_width(source_size):
        processor = MagicMock()
        choice = MagicMock()
        choice.message.content = "[]"
        response = MagicMock()
        response.choices = [choice]
        processor.client.chat.completions.create = AsyncMock(return_value=response)
        processor.get_vision_model_name.return_value = "test-vlm"
        sizes = []
        processor.encode_image.side_effect = lambda img, **kw: sizes.append(img.size) or "B64"

        asyncio.run(ground_and_translate(
            processor, Image.new("RGB", source_size), "Chinese", "English"
        ))
        return sizes[0][0]

    assert prepared_width((800, 600)) == 800  # never upscaled
    assert prepared_width((2880, 1800)) == live_max_dimension((2880, 1800))


# ── truncated responses ──────────────────────────────────────────────

def test_a_truncated_array_keeps_the_regions_it_did_contain():
    """A whole screen can hold more regions than max_tokens has room for.

    The array pattern cannot match an unterminated array, so a response cut off
    mid-region used to discard every region before it too, and the screen came
    back blank instead of partly translated.
    """
    raw = (
        '[{"box":[0,0,100,20],"original":"一","translated":"one"},'
        '{"box":[0,30,100,50],"original":"二","translated":"two"},'
        '{"box":[0,60,100,'
    )

    regions = parse_regions(raw, 1000, 1000)

    assert [r.translated for r in regions] == ["one", "two"]


def test_a_complete_array_is_still_parsed_whole():
    raw = '[{"box":[0,0,100,20],"original":"一","translated":"one"}]'

    assert [r.translated for r in parse_regions(raw, 1000, 1000)] == ["one"]


def test_grounding_asks_for_enough_tokens_to_describe_a_screen():
    from xian.grounding import GROUNDING_MAX_TOKENS

    assert GROUNDING_MAX_TOKENS >= 4096


def test_the_prompt_rules_out_the_boxes_the_code_has_to_drop():
    """Asking is cheaper than filtering, and the filter still runs regardless."""
    from xian.grounding import build_grounding_prompt

    prompt = build_grounding_prompt("Chinese", "English").lower()

    assert "whole image" in prompt
    assert "box each line on its own" in prompt


def test_a_lone_frame_sized_box_is_dropped_on_a_whole_screen_capture():
    """"This entire screen is one line of text" is never true.

    Painting nothing costs one tick; painting the box hides the game behind a
    single flat rectangle, which is what was reported.
    """
    from xian.grounding import suppress_overlapping_regions

    whole = TextRegion((0, 0, 1700, 1050), "魔獸世界", "World of Warcraft")

    assert suppress_overlapping_regions([whole], frame_size=(1728, 1080)) == []


# ── schema variation ─────────────────────────────────────────────────

def test_the_qwen_grounding_key_is_accepted():
    """Qwen-VL is trained to emit ``bbox_2d``, whatever the prompt asks for."""
    raw = '[{"bbox_2d":[0,0,500,100],"text":"文","translation":"text"}]'

    regions = parse_regions(raw, 1000, 1000)

    assert len(regions) == 1
    assert regions[0].box == (0, 0, 500, 100)
    assert regions[0].original == "文"
    assert regions[0].translated == "text"


def test_a_box_nested_one_level_deep_is_unwrapped():
    """The same convention nests coordinates when it expects several boxes."""
    raw = '[{"box":[[326,11,656,207]],"original":"魔獸世界","translated":"World of Warcraft"}]'

    regions = parse_regions(raw, 1000, 1000)

    assert [r.box for r in regions] == [(326, 11, 656, 207)]


def test_objects_with_no_usable_box_are_reported_not_swallowed(caplog):
    """A model answering off-schema looks exactly like an empty screen.

    Zero regions from a non-empty response leaves the overlay blank while the
    loop keeps paying for calls, with nothing anywhere saying why.
    """
    import logging

    raw = '[{"label":"魔獸世界"},{"label":"時光"}]'

    with caplog.at_level(logging.WARNING):
        assert parse_regions(raw, 1000, 1000) == []

    assert any("none held a usable box" in r.message for r in caplog.records)


def test_an_empty_array_is_not_reported_as_a_schema_failure(caplog):
    """"No text on this screen" is a legitimate answer."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert parse_regions("[]", 1000, 1000) == []

    assert not caplog.records


def test_an_undecodable_response_is_reported(caplog):
    """Small models emit broken JSON, and a blank overlay is the only sign."""
    import logging

    raw = '```json\n[\n{"box":[[326, 11, 656, 207], "WORLD OF WARCRAFT"},\n'

    with caplog.at_level(logging.WARNING):
        assert parse_regions(raw, 1000, 1000) == []

    assert any("could not be decoded" in r.message for r in caplog.records)


def test_an_empty_response_is_not_reported():
    """A model that returned nothing at all is the caller's problem, not a
    decode failure, and the timeout path already covers it."""
    assert parse_regions("", 1000, 1000) == []
