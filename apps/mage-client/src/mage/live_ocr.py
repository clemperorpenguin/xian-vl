# MAGE — Gaming HUD for real-time screen translation.
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

"""Continuous translation of a locked region, by local OCR then translation.

The second live engine.  Where :mod:`mage.live_lens` asks one vision model to
detect, read and translate together, this reads locally with PP-OCRv5 and sends
only text to the network.

That split is what buys the architecture its speed, and not in the obvious way.
Reading is not much cheaper than grounding was; what changes is *how often the
network is called*.  A perceptual hash can only ask whether the pixels moved,
and in a game they move constantly — particle effects, an animated HUD, a
scrolling combat log — while the dialogue sits still.  With text in hand the
question becomes whether the *text* moved, and the expensive half fires on a
small fraction of ticks instead of all of them.  A repeated line costs nothing
at all, because the cache has it.

Everything around the call — the tick, the phash pre-gate, the carry, the
masking, the painting — is inherited from :class:`mage.live_common.LiveWorkerBase`.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtCore import QRect

from mage.live_common import LiveWorkerBase

logger = logging.getLogger(__name__)

__all__ = ["LiveOcrWorker", "read_blocks"]


async def read_blocks(engine, frame: Image.Image, source_lang: str, text_filter=None) -> list:
    """Read a frame into translation blocks, in reading order.

    The reading half of the OCR engine, shared by the live worker and the
    one-shot capture: detect and recognize off the event loop, drop what the
    filter says to ignore, and group what is left into blocks.
    """
    from xian.ocr.grouping import group_lines
    from xian.text_gate import is_cjk_text

    engine.set_source_language(source_lang)

    # BGR, because that is what OpenCV and the Paddle models expect; a
    # channel swap here is much cheaper than the wrong colours everywhere.
    image = np.asarray(frame.convert("RGB"))[:, :, ::-1]
    lines = await asyncio.to_thread(engine.read, np.ascontiguousarray(image))
    if text_filter is not None:
        lines = text_filter.apply(lines)
    if not lines:
        return []

    # Decided from what was read rather than from the language setting: a
    # Chinese game still has English item names in it, and "auto" tells us
    # nothing at all.
    space_delimited = not any(is_cjk_text(line.text) for line in lines)
    return group_lines(lines, space_delimited=space_delimited)


class LiveOcrWorker(LiveWorkerBase):
    """Reads a region with PP-OCRv5 and translates only what changed."""

    def __init__(
        self,
        processor,
        rect: QRect,
        *,
        engine=None,
        translator=None,
        text_filter=None,
        gate=None,
        exclude_regions: list[QRect] | None = None,
        detector_model: str | None = None,
        **kwargs,
    ):
        super().__init__(processor, rect, **kwargs)
        self._engine = engine
        self._translator = translator
        self._filter = text_filter
        self._gate = gate
        self._detector_model = detector_model
        # Screen-coordinate rectangles masked out of the frame before anything
        # looks at it.  A ticking clock or an animated minimap inside the
        # translated region would otherwise change the content hash on every
        # tick and defeat the whole gate.
        self._exclude_regions = list(exclude_regions or [])
        self._glossary_terms: dict[str, str] | None = None

    # ── lazily built collaborators ───────────────────────────────────

    def _ensure_engine(self):
        if self._engine is None:
            from xian.ocr import PaddleOcrEngine

            kwargs = {"source_language": self.source_lang}
            if self._detector_model:
                kwargs["detector_model"] = self._detector_model
            self._engine = PaddleOcrEngine(**kwargs)
        return self._engine

    def _ensure_translator(self):
        if self._translator is None:
            from xian.translate import LineTranslator

            self._translator = LineTranslator(processor=self.processor)
        return self._translator

    def _ensure_filter(self):
        if self._filter is None:
            from xian.filters import TextFilter

            self._filter = TextFilter()
        return self._filter

    def _ensure_gate(self):
        if self._gate is None:
            from xian.text_gate import SettleGate

            self._gate = SettleGate()
        return self._gate

    def reset_gate(self) -> None:
        """Force the next frame through, whatever the gate thinks.

        The escape hatch behind a manual retry: when the gate holds text the
        user can see is wrong, there has to be a way to say so.
        """
        self._ensure_gate().reset()
        self._clean_hash = None

    # ── capture ──────────────────────────────────────────────────────

    def _grab(self) -> Image.Image | None:
        """Capture, then blank the excluded rectangles.

        Masking here rather than in :meth:`_translate_frame` is deliberate: the
        change gate hashes whatever this returns, and an exclusion that only
        applied to the reader would still be firing the gate on every tick.
        """
        frame = super()._grab()
        if frame is None or not self._exclude_regions:
            return frame

        served = self._served_rect
        draw = ImageDraw.Draw(frame)
        scale = frame.width / float(served.width()) if served.width() else 1.0
        for region in self._exclude_regions:
            local = region.translated(-served.left(), -served.top())
            draw.rectangle(
                (
                    int(local.left() * scale),
                    int(local.top() * scale),
                    int(local.right() * scale),
                    int(local.bottom() * scale),
                ),
                fill=(0, 0, 0),
            )
        return frame

    # ── the pipeline ─────────────────────────────────────────────────

    async def _translate_frame(self, frame: Image.Image):
        """Read the frame, and translate it only if the text actually changed.

        Returns ``None`` when nothing needs painting, which the base class
        already handles correctly: it keeps carrying what is on screen.
        """
        from xian.grounding import TextRegion, suppress_overlapping_regions

        blocks = await read_blocks(
            self._ensure_engine(), frame, self.source_lang, self._ensure_filter()
        )
        if not blocks:
            self.nothing_read.emit()
            return None

        texts = [block.text for block in blocks]
        gate = self._ensure_gate()
        if not gate.should_translate(texts):
            if gate.settling:
                # The settle window only closes if the same text is looked at
                # again, and the perceptual-hash gate upstream skips frames
                # whose pixels have not moved -- which is every frame of a
                # static dialogue box.  Left alone, the two gates deadlock and
                # nothing is ever translated.  Clearing the hash lets the next
                # tick back through to finish settling.
                self._clean_hash = None
            return None

        translator = self._ensure_translator()
        if self._glossary_terms is None:
            self._glossary_terms = await self._glossary()
            translator.glossary = self._glossary_terms

        translations = await translator.translate_lines(texts, self.source_lang, self.target_lang)

        regions = [
            TextRegion(box=block.box, original=text, translated=translation)
            for block, text, translation in zip(blocks, texts, translations)
        ]
        # A detector stacks boxes on tightly-packed UI just as a vision model
        # does, and the overlay fills before it draws, so the same suppression
        # is still needed.
        return suppress_overlapping_regions(regions, frame_size=frame.size)
