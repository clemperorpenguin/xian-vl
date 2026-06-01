"""Tests for AsyncEngine and VLProcessor caching/streaming behavior."""

from __future__ import annotations

import asyncio
import time
from PIL import Image, ImageDraw
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xian.async_engine import AsyncEngine
from xian.pipeline import VLProcessor, VLConfig


def test_async_engine_lifecycle():
    engine = AsyncEngine(base_url="http://fake-api/v1")
    engine.start()

    # Verify client is created and ready
    client = engine.client
    assert client is not None
    assert engine._loop is not None
    assert engine._loop.is_running()

    # Reconfigure
    engine.reconfigure(base_url="http://new-fake-api/v1")
    # Wait slightly for the async reconfiguration task to run
    time.sleep(0.1)
    
    # Shutdown
    engine.shutdown()
    engine.join(timeout=2.0)
    assert not engine.is_alive()


def test_vlprocessor_image_caching():
    processor = VLProcessor(VLConfig())
    try:
        # Create two identical images
        img1 = Image.new("RGB", (100, 100), color="white")
        img2 = Image.new("RGB", (100, 100), color="white")
        
        # Create a different image (structurally different)
        img3 = Image.new("RGB", (100, 100), color="white")
        draw = ImageDraw.Draw(img3)
        draw.ellipse([10, 10, 90, 90], fill="black")

        b64_1, cached_1 = processor._get_or_encode_image(img1)
        assert not cached_1
        assert b64_1 is not None

        # Same image should be cached
        b64_2, cached_2 = processor._get_or_encode_image(img2)
        assert cached_2
        assert b64_1 == b64_2

        # Different image should not be cached
        b64_3, cached_3 = processor._get_or_encode_image(img3)
        assert not cached_3
        assert b64_1 != b64_3
    finally:
        # Clean up the processor engine
        asyncio.run(processor.close())


@pytest.mark.anyio
async def test_vlprocessor_streaming_identical_image():
    processor = VLProcessor(VLConfig())
    try:
        from shared_types.models import TranslationResult
        # Pre-populate cache
        img = Image.new("RGB", (100, 100), color="white")
        cached_results = [TranslationResult(original_text="テスト", translated_text="Test", confidence=1.0)]
        
        processor._last_results = cached_results
        # Seed the last hash and b64
        import imagehash
        from shared_types.constants import IMAGE_HASH_SIZE
        phash = str(imagehash.phash(img, hash_size=IMAGE_HASH_SIZE))
        processor._last_phash = phash
        processor._last_b64 = "fake-b64"

        # Convert image to bytes
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Run stream_frame, it should yield immediately from cache without calling the API
        partials = []
        async for orig, trans, _ in processor.stream_frame(
            img_bytes, "Chinese", "English", "Game", []
        ):
            partials.append((orig, trans))

        assert len(partials) == 1
        assert partials[0] == ("テスト", "Test")
        assert processor.last_stream_results == cached_results
    finally:
        await processor.close()


def test_setup_logger_namespace_scoping():
    from xian.logging_config import setup_logger
    
    # Test standard setup
    l1 = setup_logger("my_submodule")
    assert l1.name == "xian.my_submodule"
    
    l2 = setup_logger("xian.already_scoped")
    assert l2.name == "xian.already_scoped"
    
    l3 = setup_logger(None)
    assert l3.name == "xian"


def make_dummy_wav(sampwidth=2, framerate=44100, channels=1, num_frames=100) -> bytes:
    import io
    import wave
    out_io = io.BytesIO()
    with wave.open(out_io, "wb") as w:
        w.setparams((channels, sampwidth, framerate, num_frames, "NONE", "not compressed"))
        w.writeframes(b"\x00" * (num_frames * channels * sampwidth))
    return out_io.getvalue()


def test_wav_concatenation_standard_wave():
    import sys
    from pathlib import Path
    luduan_src = str(Path(__file__).parents[3] / "apps" / "luduan-client" / "src")
    if luduan_src not in sys.path:
        sys.path.append(luduan_src)

    from luduan.audio_engine import concatenate_wavs
    
    wav1 = make_dummy_wav(num_frames=100)
    wav2 = make_dummy_wav(num_frames=200)
    
    combined = concatenate_wavs([wav1, wav2])
    assert combined.startswith(b"RIFF")
    
    import io
    import wave
    with wave.open(io.BytesIO(combined), "rb") as w:
        assert w.getnframes() == 300
        assert w.getframerate() == 44100
        assert w.getsampwidth() == 2


def test_wav_concatenation_fallback():
    import sys
    from pathlib import Path
    luduan_src = str(Path(__file__).parents[3] / "apps" / "luduan-client" / "src")
    if luduan_src not in sys.path:
        sys.path.append(luduan_src)

    from luduan.audio_engine import concatenate_wavs
    
    wav1 = b"RIFF" + b"\x00" * 40
    wav2_long = b"RIFF" + b"\x00" * 46
    
    combined = concatenate_wavs([wav1, wav2_long])
    # Fallback should strip the first 44 bytes from subsequent chunks and concatenate them
    # wav1 length is 44, wav2_long length is 50. Leaving 50 - 44 = 6 bytes from wav2_long.
    assert len(combined) == 44 + 6


@pytest.mark.anyio
async def test_vlprocessor_source_lang_and_temp():
    # Mocking completion create
    mock_create = AsyncMock()
    mock_create.return_value = MagicMock()
    # Mock return value choices structure for chat completions
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "ORIGINAL:\n测试\n\nTRANSLATED:\nTest\n\nCONFIDENCE:\n1.0"
    mock_chunk.choices[0].finish_reason = "stop"
    
    async def async_gen():
        yield mock_chunk
        
    mock_create.return_value = async_gen()

    config = VLConfig(temperature=0.8)
    processor = VLProcessor(config)
    
    # Mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    processor.engine = MagicMock()
    processor.engine.client = mock_client
    
    img = Image.new("RGB", (100, 100), color="white")
    import io
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # Force mock hash to be unique so it doesn't hit cache
    processor._last_phash = "different-hash"
    
    partials = []
    async for orig, trans, _ in processor.stream_frame(
        img_bytes, "Japanese", "English", "Game", []
    ):
        partials.append((orig, trans))
        
    # Verify the arguments passed to create
    mock_create.assert_called_once()
    kwargs = mock_create.call_args.kwargs
    assert kwargs["temperature"] == 0.8
    # Verify Japanese is in the prompt/messages
    messages = kwargs["messages"]
    system_prompt = messages[0]["content"]
    assert "Japanese" in system_prompt or "japanese" in system_prompt.lower()
    
    await processor.close()


@pytest.mark.anyio
async def test_vlprocessor_process_frame():
    mock_create = AsyncMock()
    mock_create.return_value = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "ORIGINAL:\n测试\n\nTRANSLATED:\nTest\n\nCONFIDENCE:\n1.0"
    mock_chunk.choices[0].finish_reason = "stop"
    
    async def async_gen():
        yield mock_chunk
        
    mock_create.return_value = async_gen()

    config = VLConfig(temperature=0.8)
    processor = VLProcessor(config)
    processor.engine = MagicMock()
    processor.engine.client = MagicMock()
    processor.engine.client.chat.completions.create = mock_create
    
    img = Image.new("RGB", (100, 100), color="white")
    import io
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    processor._last_phash = "different-hash-again"
    
    results = await processor.process_frame(
        img_bytes, "Chinese", "English", "Game", []
    )
    
    assert len(results) == 1
    assert results[0].original_text == "测试"
    assert results[0].translated_text == "Test"
    
    await processor.close()


@pytest.mark.anyio
async def test_vlprocessor_translate_query():
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "<think>some thought</think>translated text"
    mock_response.choices = [mock_choice]
    mock_create.return_value = mock_response

    processor = VLProcessor(VLConfig())
    processor.engine = MagicMock()
    processor.engine.client = MagicMock()
    processor.engine.client.chat.completions.create = mock_create

    result = await processor.translate_query("query", "English")
    assert result == "translated text"

    # Verify that it disables thinking block
    kwargs = mock_create.call_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    await processor.close()
