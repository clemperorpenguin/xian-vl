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
        async for orig, trans in processor.stream_frame(
            img_bytes, "Chinese", "English", "Game", []
        ):
            partials.append((orig, trans))

        assert len(partials) == 1
        assert partials[0] == ("テスト", "Test")
        assert processor.last_stream_results == cached_results
    finally:
        await processor.close()
