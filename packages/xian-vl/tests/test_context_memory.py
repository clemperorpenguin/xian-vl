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

"""Sliding-window context and chat-history assembly."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image

from xian.context_manager import ContextManager
from xian.pipeline import VLProcessor, VLConfig


def test_add_frame_records_metadata():
    cm = ContextManager(max_frames=3)
    frame_id = cm.add_frame(
        Image.new("RGB", (4, 4)),
        window_title="Jianghu Online",
        region=(10, 20, 300, 100),
    )

    frame = cm.get_frame_by_id(frame_id)
    assert frame is not None
    assert frame.window_title == "Jianghu Online"
    assert frame.region == (10, 20, 300, 100)
    assert frame.ts > 0


def test_frame_ids_are_unique_and_evicted_frames_are_unresolvable():
    cm = ContextManager(max_frames=2)
    ids = [cm.add_frame(Image.new("RGB", (4, 4))) for _ in range(3)]

    assert len(set(ids)) == 3
    assert cm.get_frame_by_id(ids[0]) is None, "oldest frame should have slid out"
    assert cm.get_frame_by_id(ids[2]) is not None
    assert cm.get_latest_frame_id() == ids[2]


def test_user_message_binds_the_frame_present_at_the_time():
    cm = ContextManager(max_frames=3)
    first = cm.add_frame(Image.new("RGB", (4, 4)))
    cm.add_user_message("what is this?")
    second = cm.add_frame(Image.new("RGB", (4, 4)))
    cm.add_user_message("and now?")

    markers = [
        item
        for msg in cm.get_chat_history()
        for item in msg["content"]
        if item["type"] == "image"
    ]
    assert [m["frame_id"] for m in markers] == [first, second]


def test_trim_history_starts_on_a_user_turn():
    cm = ContextManager(max_frames=1)
    cm.add_frame(Image.new("RGB", (4, 4)))
    for i in range(ContextManager.MAX_CHAT_MESSAGES):
        cm.add_user_message(f"user {i}", with_image=False)
        cm.add_assistant_message(f"assistant {i}")

    history = cm.get_chat_history()
    assert len(history) <= ContextManager.MAX_CHAT_MESSAGES
    assert history[0]["role"] == "user"


def _make_processor(tmp_path):
    processor = VLProcessor(VLConfig())
    processor.wiki_dir = str(tmp_path)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    processor.engine = MagicMock()
    processor.engine.client = mock_client
    return processor, mock_client


@pytest.mark.anyio
async def test_only_the_newest_turn_carries_a_screenshot(tmp_path):
    """Replayed history must not re-attach the current frame to older turns."""
    processor, mock_client = _make_processor(tmp_path)

    old_frame = processor.context_manager.add_frame(Image.new("RGB", (8, 8), "red"))
    processor.context_manager.update_last_frame_data(
        "ORIGINAL: 前往城門\nTRANSLATED: Head to the city gate\nCONFIDENCE: 0.9", []
    )
    processor.context_manager.add_user_message("where do I go?")
    processor.context_manager.add_assistant_message("To the gate.")
    processor.context_manager.add_frame(Image.new("RGB", (8, 8), "blue"))

    reply = MagicMock()
    reply.message.tool_calls = []
    reply.message.content = "Follow the road east."
    response = MagicMock()
    response.choices = [reply]
    mock_client.chat.completions.create.return_value = response

    await processor.process_chat("what now?")

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    parts = [p for m in messages if isinstance(m["content"], list) for p in m["content"]]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 1, "exactly one screenshot should be sent per chat turn"

    placeholders = [
        p["text"] for p in parts
        if p["type"] == "text" and p["text"].startswith("[Earlier screenshot")
    ]
    assert len(placeholders) == 1
    assert "前往城門" in placeholders[0], "placeholder should carry that frame's own OCR text"
    assert processor.context_manager.get_frame_by_id(old_frame) is not None


@pytest.mark.anyio
async def test_past_frame_placeholder_survives_eviction(tmp_path):
    processor, _ = _make_processor(tmp_path)
    assert processor._describe_past_frame(9999) == "[Earlier screenshot omitted.]"
    assert processor._describe_past_frame(None) == "[Earlier screenshot omitted.]"
