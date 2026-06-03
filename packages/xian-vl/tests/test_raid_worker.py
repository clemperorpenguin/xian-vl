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

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from PyQt6.QtCore import QCoreApplication

@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if not app:
        app = QCoreApplication([])
    return app

@pytest.mark.anyio
async def test_raid_worker_resilience(qapp):
    # Mock processor
    processor = MagicMock()
    processor.config.model_name = "LMX-Omni-5.5B-Lite"
    processor.router.asr.return_value = "Whisper-Tiny"
    processor.get_model_name.return_value = "Qwen3.5-4B-MTP-GGUF"
    
    import sys
    from pathlib import Path
    mage_src = str(Path(__file__).parents[3] / "apps" / "mage-client" / "src")
    if mage_src not in sys.path:
        sys.path.append(mage_src)
        
    from mage.workers import RaidWorker
    
    worker = RaidWorker(processor, target_lang="English", source_lang="Chinese", save_lore=False)
    
    # Mock ContinuousAudioStreamer
    mock_streamer = MagicMock()
    async def mock_read_chunks():
        yield b"chunk1"
        yield b"chunk2"
        # stop worker to break the loop
        worker.stop()
        
    mock_streamer.read_chunks = mock_read_chunks
    mock_streamer.start = AsyncMock()
    mock_streamer.stop = AsyncMock()
    
    # Mock LemonadeClient
    mock_lemonade_client = MagicMock()
    mock_lemonade_client.__aenter__.return_value = mock_lemonade_client
    mock_transcribe = AsyncMock()
    # Let first transcription fail/timeout, and second pass
    mock_transcribe.side_effect = [
        asyncio.TimeoutError("Transcription timed out"),
        "hello world"
    ]
    mock_lemonade_client.transcribe = mock_transcribe
    mock_lemonade_client.close = AsyncMock()
    
    # Mock completions create
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"translation": "Hello World"}'
    mock_response.choices = [mock_choice]
    mock_create.return_value = mock_response
    
    processor.client.chat.completions.create = mock_create
    
    # Patch streamer and client
    with patch("mage.capture.audio.ContinuousAudioStreamer", return_value=mock_streamer), \
         patch("mage.workers.LemonadeClient", return_value=mock_lemonade_client):
         
         # Connect signals to verify
         progress_messages = []
         translations = []
         
         worker.progress.connect(progress_messages.append)
         worker.chunk_translated.connect(lambda orig, trans: translations.append((orig, trans)))
         
         # Run _run_async directly so we can await it
         await worker._run_async()
         
         # Verify that the first chunk's failure didn't crash the loop
         # and the second chunk was successfully processed.
         assert len(translations) == 1
         assert translations[0] == ("hello world", "Hello World")
         
         # Verify we had "Transcription failed" progress message
         any_fail_msg = any("Transcription failed" in msg for msg in progress_messages)
         assert any_fail_msg
