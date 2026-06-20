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

import os
import pytest
from unittest.mock import AsyncMock, MagicMock
from PIL import Image
from xian.pipeline import VLProcessor, VLConfig

@pytest.mark.anyio
async def test_process_chat_symbolic_paths(tmp_path):
    # Setup VLProcessor with a temp wiki_dir
    config = VLConfig()
    processor = VLProcessor(config)
    processor.wiki_dir = str(tmp_path)
    
    # Mock AsyncOpenAI client
    mock_client = MagicMock()
    mock_create = AsyncMock()
    mock_client.chat.completions.create = mock_create
    processor.engine = MagicMock()
    processor.engine.client = mock_client
    
    # Mock context manager frame
    test_image = Image.new("RGB", (10, 10), color="blue")
    processor.context_manager.add_frame(test_image)
    
    # First turn: model calls analyze_image with a symbolic path 'screenshot'
    mock_choice_1 = MagicMock()
    mock_tool_call_1 = MagicMock()
    mock_tool_call_1.id = "call_1"
    mock_tool_call_1.function.name = "analyze_image"
    mock_tool_call_1.function.arguments = '{"image_path": "screenshot", "question": "what is this?"}'
    mock_choice_1.message.tool_calls = [mock_tool_call_1]
    mock_choice_1.message.content = None
    
    # Second turn: model returns final text
    mock_choice_2 = MagicMock()
    mock_choice_2.message.tool_calls = []
    mock_choice_2.message.content = "It is a blue image."
    
    # Setup create returns
    mock_response_1 = MagicMock()
    mock_response_1.choices = [mock_choice_1]
    mock_response_2 = MagicMock()
    mock_response_2.choices = [mock_choice_2]
    
    # For vision call inside analyze_image
    mock_vision_response = MagicMock()
    mock_vision_response.choices = [MagicMock()]
    mock_vision_response.choices[0].message.content = "A blue square."
    
    # Setup call side_effect to return mock responses in sequence
    mock_create.side_effect = [mock_response_1, mock_vision_response, mock_response_2]
    
    # Call process_chat
    res = await processor.process_chat("describe the screen")
    assert res == "It is a blue image."
    
    # Verify screenshot was saved and loaded
    media_dir = os.path.join(processor.wiki_dir, "media")
    expected_screenshot_path = os.path.join(media_dir, "current_screenshot.png")
    assert os.path.exists(expected_screenshot_path)
    
    # Clean up
    await processor.close()

@pytest.mark.anyio
async def test_process_chat_file_not_found(tmp_path):
    # Setup VLProcessor with a temp wiki_dir
    config = VLConfig()
    processor = VLProcessor(config)
    processor.wiki_dir = str(tmp_path)
    
    # Mock AsyncOpenAI client
    mock_client = MagicMock()
    mock_create = AsyncMock()
    mock_client.chat.completions.create = mock_create
    processor.engine = MagicMock()
    processor.engine.client = mock_client
    
    # First turn: model calls transcribe_audio with a non-existent file
    mock_choice_1 = MagicMock()
    mock_tool_call_1 = MagicMock()
    mock_tool_call_1.id = "call_1"
    mock_tool_call_1.function.name = "transcribe_audio"
    mock_tool_call_1.function.arguments = '{"audio_path": "non_existent_file.wav"}'
    mock_choice_1.message.tool_calls = [mock_tool_call_1]
    mock_choice_1.message.content = None
    
    # Second turn: model returns final text
    mock_choice_2 = MagicMock()
    mock_choice_2.message.tool_calls = []
    mock_choice_2.message.content = "Sorry, that file does not exist."
    
    # Setup create returns
    mock_response_1 = MagicMock()
    mock_response_1.choices = [mock_choice_1]
    mock_response_2 = MagicMock()
    mock_response_2.choices = [mock_choice_2]
    
    mock_create.side_effect = [mock_response_1, mock_response_2]
    
    # Call process_chat
    res = await processor.process_chat("transcribe my audio")
    assert res == "Sorry, that file does not exist."
    
    # Check that tool result has been passed to messages
    calls = mock_create.call_args_list
    assert len(calls) == 2
    # The second call should contain the tool response containing "File not found:"
    second_call_messages = calls[1][1]["messages"]
    tool_message = [m for m in second_call_messages if m.get("role") == "tool"][0]
    assert "File not found:" in tool_message["content"]
    
    await processor.close()
