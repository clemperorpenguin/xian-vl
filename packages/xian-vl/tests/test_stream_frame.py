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

"""Edge cases of the streaming OCR path: empty streams and result caching."""

import io

import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock

from xian.pipeline import VLProcessor, VLConfig


def _image_bytes(color="white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _chunk(content: str, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    chunk.choices[0].delta.reasoning_content = None
    chunk.choices[0].finish_reason = finish_reason
    return chunk


def _processor_with_chunks(*chunk_lists):
    """A processor whose client replays one async chunk stream per call."""
    processor = VLProcessor(VLConfig())

    streams = list(chunk_lists)

    async def _create(**_kwargs):
        chunks = streams.pop(0)

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    mock_create = AsyncMock(side_effect=_create)
    processor.engine = MagicMock()
    processor.engine.client = MagicMock()
    processor.engine.client.chat.completions.create = mock_create
    return processor, mock_create


@pytest.mark.anyio
async def test_empty_stream_yields_no_results_without_raising():
    """A stream with zero content chunks must not blow up on the final yield.

    Regression: `orig`/`trans` were only bound inside the chunk loop, so a
    server that closed the stream immediately raised UnboundLocalError.
    """
    processor, _ = _processor_with_chunks([])
    processor._last_phash = "seed-a-miss"

    partials = []
    async for orig, trans, extra in processor.stream_frame(
        _image_bytes(), "Chinese", "English", "Game", []
    ):
        partials.append((orig, trans, extra))

    assert len(partials) == 1
    assert partials[0][0] == ""
    assert partials[0][1] == ""
    assert partials[0][2][0] == []

    await processor.close()


@pytest.mark.anyio
async def test_cached_results_are_not_reused_across_target_languages():
    """An unchanged frame with a new target language must be re-translated.

    Regression: the result cache keyed on the perceptual hash alone, so
    switching target language returned the previous language's translation.
    """
    chinese = "ORIGINAL:\n测试\n\nTRANSLATED:\nTest\n\nCONFIDENCE:\n1.0"
    spanish = "ORIGINAL:\n测试\n\nTRANSLATED:\nPrueba\n\nCONFIDENCE:\n1.0"
    processor, mock_create = _processor_with_chunks(
        [_chunk(chinese, "stop")], [_chunk(spanish, "stop")]
    )
    processor._last_phash = "seed-a-miss"
    frame = _image_bytes()

    results = await processor.process_frame(frame, "Chinese", "English", "Game", [])
    assert results[0].translated_text == "Test"

    results = await processor.process_frame(frame, "Chinese", "Spanish", "Game", [])
    assert results[0].translated_text == "Prueba"
    assert mock_create.call_count == 2

    await processor.close()


@pytest.mark.anyio
async def test_cached_results_are_reused_for_identical_settings():
    """The cache still has to work — the same frame and settings skip the model."""
    reply = "ORIGINAL:\n测试\n\nTRANSLATED:\nTest\n\nCONFIDENCE:\n1.0"
    processor, mock_create = _processor_with_chunks([_chunk(reply, "stop")])
    processor._last_phash = "seed-a-miss"
    frame = _image_bytes()

    await processor.process_frame(frame, "Chinese", "English", "Game", [])
    results = await processor.process_frame(frame, "Chinese", "English", "Game", [])

    assert results[0].translated_text == "Test"
    assert mock_create.call_count == 1

    await processor.close()
