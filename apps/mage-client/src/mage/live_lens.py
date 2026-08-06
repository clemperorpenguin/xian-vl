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

"""Continuous translation of a locked screen region.

Dialogue mode re-captures on a click; this re-captures on a timer and keeps an
in-place overlay in sync with whatever is on screen.  Three things make that
practical:

* **A change gate.** Re-running a vision model on an unchanged frame is pure
  waste, so a perceptual hash of the raw crop decides whether anything actually
  moved.  It is a distance threshold, not the exact-match cache the bubble path
  uses, because anti-aliasing and cursor blink perturb a few bits constantly.

* **Single flight.** At most one inference is outstanding.  Ticks that arrive
  while one is running are dropped rather than queued, so the overlay always
  reflects recent state instead of working through a backlog.

* **Overlay-aware capture.** Our own overlay is on screen, inside the region we
  are about to capture.  Hiding it before every capture — what dialogue mode
  does — makes it flicker continuously.  Instead the worker remembers what it
  painted and only hides when the capture stops matching its own output.
"""

from __future__ import annotations

import asyncio
import logging
import time

import imagehash
from PIL import Image
from PyQt6.QtCore import QRect, QThread, pyqtSignal

logger = logging.getLogger(__name__)

__all__ = ["LiveLensWorker", "LiveRegion", "phash_distance", "regions_signature"]

#: Perceptual-hash size for the change gate. Smaller than the bubble path's 16:
#: we want "did this line change", not "is this the identical frame".
LIVE_HASH_SIZE = 8

#: Hamming distance above which a frame counts as changed.
DEFAULT_CHANGE_THRESHOLD = 6

#: How often to look at the region, in milliseconds.
DEFAULT_INTERVAL_MS = 700

#: Consecutive capture failures tolerated before giving up.
MAX_CAPTURE_FAILURES = 5


class LiveRegion:
    """A translated line plus its background colour, in region coordinates."""

    __slots__ = ("box", "original", "translated", "fill")

    def __init__(
        self,
        box: tuple[int, int, int, int],
        original: str,
        translated: str,
        fill: tuple[int, int, int] = (0, 0, 0),
    ):
        self.box = box
        self.original = original
        self.translated = translated
        self.fill = fill


def sample_background(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """Average a thin band just outside the box to get its background colour.

    Sampling outside rather than inside avoids averaging in the very glyphs
    about to be covered, which would tint the fill toward the text colour.
    """
    left, top, right, bottom = box
    width, height = image.size
    band = max(2, (bottom - top) // 6)

    totals = [0, 0, 0]
    count = 0

    def accumulate(x: int, y: int) -> None:
        nonlocal count
        if 0 <= x < width and 0 <= y < height:
            pixel = image.getpixel((x, y))
            if isinstance(pixel, int):
                pixel = (pixel, pixel, pixel)
            totals[0] += pixel[0]
            totals[1] += pixel[1]
            totals[2] += pixel[2]
            count += 1

    for x in range(left, right, max(1, (right - left) // 32)):
        for d in range(1, band + 1):
            accumulate(x, top - d)
            accumulate(x, bottom + d)

    for y in range(top, bottom, max(1, (bottom - top) // 16)):
        for d in range(1, band + 1):
            accumulate(left - d, y)
            accumulate(right + d, y)

    if not count:
        # Nothing sampleable (box hugs the region edge); black reads as a
        # deliberate label and keeps white text legible.
        return (0, 0, 0)
    return (totals[0] // count, totals[1] // count, totals[2] // count)


def phash_distance(a, b) -> int:
    """Hamming distance between two perceptual hashes."""
    if a is None or b is None:
        return 10**6
    return abs(a - b)


def regions_signature(regions) -> tuple:
    """Identity of a render, used to skip repainting identical output."""
    return tuple((tuple(r.box), r.translated) for r in regions)


class LiveLensWorker(QThread):
    """Polls a screen region and emits translated, positioned text."""

    #: Region translations ready to paint, in region-local pixels.
    regions_ready = pyqtSignal(object, object)  # (list[LiveRegion], QRect)
    #: Ask the UI to hide the overlay so the next capture sees clean pixels.
    overlay_hide = pyqtSignal()
    status = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        processor,
        rect: QRect,
        *,
        source_lang: str = "Chinese",
        target_lang: str = "English",
        interval_ms: int = DEFAULT_INTERVAL_MS,
        change_threshold: int = DEFAULT_CHANGE_THRESHOLD,
        session_recorder=None,
    ):
        super().__init__()
        self.processor = processor
        self.rect = rect
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.interval_ms = max(200, interval_ms)
        self.change_threshold = change_threshold
        self._session_recorder = session_recorder
        self._running = True

        # What we last painted, so a capture containing our own overlay can be
        # told apart from a genuine content change.
        self._clean_hash = None
        self._painted_hash = None
        self._last_signature: tuple = ()
        self._needs_clean_frame = False

    def stop(self):
        self._running = False

    # ── capture ──────────────────────────────────────────────────────

    def _grab(self) -> Image.Image | None:
        """Capture the bound region and return it as a PIL image."""
        from mage.capture.screen import ScreenCapture
        from mage.utils.images import qimage_to_pil
        from PyQt6.QtGui import QImage

        data, already_cropped = ScreenCapture.capture_region(self.rect)
        if not data:
            return None

        image = QImage.fromData(data)
        if image.isNull():
            return None

        if not already_cropped:
            total = ScreenCapture.get_virtual_desktop_geometry()
            local = self.rect.translated(-total.left(), -total.top()).intersected(image.rect())
            if local.isEmpty():
                return None
            image = image.copy(local)

        return qimage_to_pil(image)

    # ── pipeline ─────────────────────────────────────────────────────

    async def _run_async(self):
        from xian.grounding import ground_and_translate

        self.status.emit("live_lens.status.watching")
        interval = self.interval_ms / 1000.0
        failures = 0
        inflight = False

        while self._running:
            started = time.monotonic()

            frame = await asyncio.to_thread(self._grab)
            if frame is None:
                failures += 1
                if failures >= MAX_CAPTURE_FAILURES:
                    self.error.emit("Screen capture failed repeatedly")
                    return
                await self._sleep_remainder(started, interval)
                continue
            failures = 0

            current = imagehash.phash(frame, hash_size=LIVE_HASH_SIZE)

            if inflight or not self._should_translate(current):
                await self._sleep_remainder(started, interval)
                continue

            # The frame changed, but it may still contain our own overlay from
            # the previous render. Take one clean frame before translating.
            if self._needs_clean_frame:
                self.overlay_hide.emit()
                await asyncio.sleep(0.12)  # let the compositor drop the window
                clean = await asyncio.to_thread(self._grab)
                if clean is not None:
                    frame = clean
                    current = imagehash.phash(frame, hash_size=LIVE_HASH_SIZE)

            self._clean_hash = current
            inflight = True
            try:
                regions = await ground_and_translate(
                    self.processor, frame, self.source_lang, self.target_lang,
                    glossary=await self._glossary(),
                )
            except asyncio.TimeoutError:
                logger.warning("Live lens inference timed out; skipping frame")
                regions = None
            except Exception as exc:
                logger.warning("Live lens inference failed: %s", exc)
                self.error.emit(str(exc))
                regions = None
            finally:
                inflight = False

            if regions is not None and self._publish(regions, frame):
                # The overlay now covers part of the region, so the next
                # capture will not match the clean frame. Record what the
                # screen looks like *with* our render on it, or that change
                # would be mistaken for the game moving on.
                await asyncio.sleep(0.15)
                painted = await asyncio.to_thread(self._grab)
                if painted is not None:
                    self._painted_hash = imagehash.phash(painted, hash_size=LIVE_HASH_SIZE)

            await self._sleep_remainder(started, interval)

    def _should_translate(self, current) -> bool:
        """Decide whether this capture reflects real change on screen."""
        if self._clean_hash is None:
            self._needs_clean_frame = False
            return True

        # Matches what we painted last time: the overlay is showing over
        # unchanged content. Nothing to do, and crucially no hide/flicker.
        if self._painted_hash is not None and phash_distance(current, self._painted_hash) <= self.change_threshold:
            return False

        # Matches the clean frame we translated from: also unchanged.
        if phash_distance(current, self._clean_hash) <= self.change_threshold:
            return False

        self._needs_clean_frame = self._painted_hash is not None
        return True

    def _publish(self, regions, frame: Image.Image) -> bool:
        """Emit regions if they differ from what is already on screen.

        Returns whether anything was actually emitted, so the caller knows
        whether the overlay is about to change.
        """
        live = [
            LiveRegion(r.box, r.original, r.translated, sample_background(frame, r.box))
            for r in regions
            if r.is_valid()
        ]
        signature = regions_signature(live)
        if signature == self._last_signature:
            return False
        self._last_signature = signature

        if self._session_recorder:
            for region in live:
                self._session_recorder(region.original, region.translated)

        self.regions_ready.emit(live, self.rect)
        return True

    async def _sleep_remainder(self, started: float, interval: float) -> None:
        elapsed = time.monotonic() - started
        await asyncio.sleep(max(0.05, interval - elapsed))

    async def _glossary(self) -> dict[str, str]:
        try:
            return await self.processor.load_glossary_from_wiki()
        except Exception:
            return {}

    def run(self):
        future = self.processor.engine.submit(self._run_async())
        try:
            future.result()
        except Exception as e:
            logger.error("LiveLensWorker error: %s", e)
            self.error.emit(str(e))
