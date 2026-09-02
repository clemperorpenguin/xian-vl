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

* **Masked comparison.** Our own overlay is on screen, inside the region we are
  about to capture, so a naive hash sees our translations as a change and
  re-translates forever.  Rather than hide the overlay and re-grab — which cost
  two extra captures and 270ms of compositor sleeps per frame, and flickered —
  the areas we painted are blanked in *both* frames before hashing.  The gate
  then only ever looks at pixels the game controls.
"""

from __future__ import annotations

import asyncio
import logging
import time

import imagehash
from PIL import Image
from PyQt6.QtCore import QRect, QThread, pyqtSignal

logger = logging.getLogger(__name__)

__all__ = [
    "LiveLensWorker",
    "LiveRegion",
    "phash_distance",
    "region_difference",
    "regions_signature",
]

#: Perceptual-hash size for the change gate.
#:
#: 8 (64 bits) was too coarse to do the job. Measured over 33 real game frames:
#: re-captures of one unchanged screen reach a distance of 4, while two
#: genuinely different screens start at 2 — the two classes *overlap*, so no
#: threshold separates them. A loot panel opening over a busy scene scored 2,
#: and the overlay would have kept showing the previous translation.
#:
#: At 16 (256 bits) they separate: re-captures top out at 20, distinct frames
#: start at 22. See ``test_live_lens_benchmark.py``.
LIVE_HASH_SIZE = 16

#: Fraction of the hash's bits that must differ before a frame counts as
#: changed. Expressed as a ratio rather than a bit count so it keeps its
#: meaning if the hash size moves again — 21/256 sits in the measured gap.
CHANGE_THRESHOLD_RATIO = 21 / 256

#: Hamming distance above which a frame counts as changed.
DEFAULT_CHANGE_THRESHOLD = round(LIVE_HASH_SIZE**2 * CHANGE_THRESHOLD_RATIO)

#: Pixels of slack around a painted box when masking it out of the change
#: check. The overlay draws a rounded rect with antialiased edges, so masking
#: the exact box leaves a fringe of our own paint in the comparison.
PAINT_MASK_PADDING = 2

#: Size a region crop is normalised to before two captures of it are compared.
REGION_COMPARE_SIZE = 32

#: Mean absolute intensity difference (0-255) above which a region's pixels are
#: showing different text.
#:
#: Deliberately *not* a perceptual hash. The frame gate compares two whole
#: desktops and wants tolerance; this compares one fixed rectangle across two
#: captures of the same display milliseconds apart, where the question is
#: simply whether those pixels moved. Measured over the corpus: a re-captured
#: crop differs by a median of 1.8 and a 99th percentile of 6.4, while the same
#: box holding different content differs by a median of 64. At 6 the gate
#: retains 98.6% of genuinely unchanged regions.
#:
#: Errors here are asymmetric, which is why the threshold sits near the noise
#: floor rather than midway: dropping a still-valid translation costs nothing —
#: the frame is being grounded anyway — while keeping a stale one paints the
#: wrong words over live text.
REGION_CHANGE_THRESHOLD = 6.0

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


#: How much two boxes must overlap to be treated as the same line of text.
SAME_REGION_IOU = 0.6


def _boxes_describe_the_same_text(a, b) -> bool:
    """Whether two boxes are the same line, allowing for detector jitter."""
    overlap_x = min(a[2], b[2]) - max(a[0], b[0])
    overlap_y = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_x <= 0 or overlap_y <= 0:
        return False
    intersection = overlap_x * overlap_y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return union > 0 and intersection / union >= SAME_REGION_IOU


#: Overlap above which a carried region would paint over a fresh one. The
#: overlay fills before it draws, so any real overlap loses a translation.
CARRY_CONFLICT_COVER = 0.3


def _boxes_conflict(a, b) -> bool:
    """Whether painting both boxes would have one fill cover the other's text."""
    overlap_x = min(a[2], b[2]) - max(a[0], b[0])
    overlap_y = min(a[3], b[3]) - max(a[1], b[1])
    if overlap_x <= 0 or overlap_y <= 0:
        return False
    intersection = overlap_x * overlap_y
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and intersection / smaller > CARRY_CONFLICT_COVER


def region_difference(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute intensity difference between two region fingerprints."""
    from PIL import ImageChops, ImageStat

    return ImageStat.Stat(ImageChops.difference(a, b)).mean[0]


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

    #: Region translations ready to paint, in region-local capture pixels,
    #: plus the capture-pixels-per-logical-pixel scale needed to map them into
    #: Qt's coordinate space.  The scale is measured from the frame rather than
    #: assumed from the display: grim hands back device pixels while the PyQt
    #: fallback composites in logical ones, so a fixed devicePixelRatio would
    #: be wrong for one of them on any HiDPI screen.
    regions_ready = pyqtSignal(object, object, float)  # (list[LiveRegion], QRect, scale)
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
        frame_stream=None,
    ):
        super().__init__()
        self.processor = processor
        self.rect = rect
        # Owned and started by the GUI thread; None when continuous capture is
        # unavailable, which leaves every grab on the screenshot path.
        self._frame_stream = frame_stream
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.interval_ms = max(200, interval_ms)
        self.change_threshold = change_threshold
        self._session_recorder = session_recorder
        self._running = True

        # What the last capture actually covered. A stream session captures one
        # screen, so a region reaching past its edge comes back clipped, and
        # both the coordinate scale and the overlay's geometry have to describe
        # the clipped area rather than the requested one.
        self._served_rect = rect

        # The last frame we translated, hashed with our own overlay masked out
        # so the paint we put on screen is never mistaken for the game moving.
        self._clean_hash = None
        self._painted_boxes: list[tuple[int, int, int, int]] = []
        self._last_signature: tuple = ()
        self._warned_untranslated = False

        # Translations from earlier frames, each with a hash of the pixels it
        # was read from. A box whose pixels are unchanged is still showing the
        # same text, so its translation stands without another inference —
        # which is the whole point: a tooltip is read once and then kept.
        self._retained: list[tuple[LiveRegion, object]] = []

        # The subset of those still valid for the frame being translated.
        # Painted while the call runs, so the overlay neither blanks nor keeps
        # showing translations whose text has already gone.
        self._carry: list[LiveRegion] = []

        # Background colours already sampled for the frame being published.
        # Streaming republishes the whole region list on every batch, so
        # without this each box is re-sampled once per batch — quadratic in the
        # number of regions, on the whole-screen frames that have the most.
        self._fill_frame: Image.Image | None = None
        self._fill_cache: dict[tuple[int, int, int, int], tuple[int, int, int]] = {}
        self._region_hash_cache: dict[tuple[int, int, int, int], object] = {}

    def stop(self):
        self._running = False

    # ── capture ──────────────────────────────────────────────────────

    def _grab(self) -> Image.Image | None:
        """Capture the bound region and return it as a PIL image.

        Prefers the continuous session when one is running: it hands back
        decoded pixels, where the screenshot path costs a subprocess or a
        full-desktop composite plus an encode and a decode on every tick.  It
        can legitimately have nothing yet — the portal may still be
        negotiating — so an empty result falls through rather than failing the
        tick.
        """
        from mage.capture.screen import ScreenCapture
        from mage.utils.images import qimage_to_pil
        from PyQt6.QtGui import QImage

        if self._frame_stream is not None:
            served = self._frame_stream.grab_region(self.rect)
            if served is not None and not served[0].isNull():
                frame, self._served_rect = served
                return qimage_to_pil(frame)

        # The screenshot path crops the virtual desktop itself, so it always
        # serves the whole request.
        self._served_rect = self.rect
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

    async def _translate_frame(self, frame: Image.Image):
        """Produce located, translated regions for one frame.

        One vision call does detection, reading and translation together.
        Measured against the local-OCR sidecar this replaced: ~1.0s versus
        ~2.25s for OCR plus a separate batch translation, with the first line
        on screen at ~460ms rather than after the whole chain.
        """
        glossary = await self._glossary()

        from xian.grounding import ground_and_translate

        # Paint each line as the model describes it rather than waiting for the
        # whole screen: on a busy frame the last box can be seconds behind the
        # first, and a partly-translated overlay is useful immediately.
        def publish_partial(regions):
            self._publish(regions, frame)

        return await ground_and_translate(
            self.processor, frame, self.source_lang, self.target_lang,
            glossary=glossary, on_region=publish_partial,
        )

    async def _run_async(self):
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

            current = self._masked_hash(frame)

            if inflight or not self._should_translate(current):
                await self._sleep_remainder(started, interval)
                continue

            self._clean_hash = current

            # Whatever is painted was read from an older frame. Re-check each
            # box against this one: unchanged text keeps its translation while
            # the call runs, and text that has gone stops being painted now
            # rather than seconds from now when the call returns.
            self._carry = self._surviving_regions(frame)
            self._publish([], frame)

            inflight = True
            try:
                regions = await self._translate_frame(frame)
            except asyncio.TimeoutError:
                logger.warning("Live lens inference timed out; skipping frame")
                regions = None
            except Exception as exc:
                logger.warning("Live lens inference failed: %s", exc)
                self.error.emit(str(exc))
                regions = None
            finally:
                inflight = False

            if regions is None:
                # The call failed. The carry is still valid for this frame, so
                # the overlay keeps the lines whose text has not changed
                # instead of blanking, and they survive into the next tick.
                self._remember(self._carry, frame)
            else:
                self._publish(regions, frame)
                self._carry = []
                # The boxes we mask changed with this render, so re-hash the
                # frame we just translated under the *new* mask. Comparing the
                # next capture against a hash taken with the old mask would
                # register our own repaint as a screen change.
                #
                # This is why there is no hide-and-re-grab here any more: that
                # cost two extra captures and 270ms of compositor sleeps on
                # every published frame, and made the overlay flicker.
                self._clean_hash = self._masked_hash(frame)

            await self._sleep_remainder(started, interval)

    # ── retention ────────────────────────────────────────────────────

    def _region_fingerprint(self, frame: Image.Image, box: tuple[int, int, int, int]):
        """The pixels a region was read from, normalised for comparison."""
        left, top, right, bottom = box
        if right - left < 2 or bottom - top < 2:
            return None
        if box in self._region_hash_cache:
            return self._region_hash_cache[box]

        crop = frame.crop((left, top, right, bottom))
        sampled = None
        if crop.width and crop.height:
            sampled = crop.convert("L").resize((REGION_COMPARE_SIZE, REGION_COMPARE_SIZE))
        self._region_hash_cache[box] = sampled
        return sampled

    def _surviving_regions(self, frame: Image.Image) -> list[LiveRegion]:
        """Retained translations whose text is still on screen, unchanged.

        Checked against *this* frame, so a tooltip that has closed drops out
        immediately rather than being painted over whatever replaced it.
        """
        if self._fill_frame is not frame:
            self._fill_frame = frame
            self._fill_cache = {}
            self._region_hash_cache = {}

        survivors = []
        for region, previous in self._retained:
            if previous is None:
                continue
            current = self._region_fingerprint(frame, region.box)
            if current is None:
                continue
            if region_difference(current, previous) <= REGION_CHANGE_THRESHOLD:
                survivors.append(region)
        return survivors

    def _remember(self, live: list[LiveRegion], frame: Image.Image) -> None:
        self._retained = [
            (region, self._region_fingerprint(frame, region.box)) for region in live
        ]

    def _reuse_translations(self, live: list[LiveRegion]) -> list[LiveRegion]:
        """Keep the wording a still-unchanged region already had.

        The same pixels re-read by a generative model come back worded slightly
        differently every time, and the overlay shimmers. Where the text has
        demonstrably not changed, the translation should not either.
        """
        if not self._carry:
            return live

        for region in live:
            for carried in self._carry:
                if _boxes_describe_the_same_text(region.box, carried.box):
                    region.translated = carried.translated
                    break
        return live

    def _mask_painted(self, frame: Image.Image) -> Image.Image:
        """Blank out the areas our overlay covers.

        The change gate compares captures that contain our own translations.
        Masking those areas in both the reference and the candidate leaves the
        comparison looking only at pixels the game controls.
        """
        if not self._painted_boxes:
            return frame

        from PIL import ImageDraw

        masked = frame.copy()
        draw = ImageDraw.Draw(masked)
        for left, top, right, bottom in self._painted_boxes:
            draw.rectangle(
                (
                    left - PAINT_MASK_PADDING,
                    top - PAINT_MASK_PADDING,
                    right + PAINT_MASK_PADDING,
                    bottom + PAINT_MASK_PADDING,
                ),
                fill=(0, 0, 0),
            )
        return masked

    def _masked_hash(self, frame: Image.Image):
        return imagehash.phash(self._mask_painted(frame), hash_size=LIVE_HASH_SIZE)

    def _should_translate(self, current) -> bool:
        """Decide whether this capture reflects real change on screen."""
        if self._clean_hash is None:
            return True
        # bool(): imagehash subtraction yields a numpy integer, so the bare
        # comparison would hand callers a numpy bool that fails an `is True`.
        return bool(phash_distance(current, self._clean_hash) > self.change_threshold)

    def _publish(self, regions, frame: Image.Image) -> bool:
        """Emit regions if they differ from what is already on screen.

        Returns whether anything was actually emitted, so the caller knows
        whether the overlay is about to change.
        """
        if self._fill_frame is not frame:
            self._fill_frame = frame
            self._fill_cache = {}
            self._region_hash_cache = {}

        live = []
        for region in regions:
            if not region.is_valid():
                continue
            fill = self._fill_cache.get(region.box)
            if fill is None:
                fill = sample_background(frame, region.box)
                self._fill_cache[region.box] = fill
            live.append(LiveRegion(region.box, region.original, region.translated, fill))
        self._warn_if_nothing_translated(live)
        live = self._reuse_translations(live)
        painted = self._merge_with_carry(live)

        # Recorded even when the render is unchanged: these drive the mask the
        # change gate uses, and they must describe what is currently on screen.
        self._painted_boxes = [region.box for region in painted]
        self._remember(painted, frame)

        signature = regions_signature(painted)
        if signature == self._last_signature:
            return False
        self._last_signature = signature

        if self._session_recorder:
            # Only what was just read. A carried region was recorded on the
            # frame it came from, and recording it again every tick would fill
            # the session log with one tooltip.
            for region in live:
                self._session_recorder(region.original, region.translated)

        self.regions_ready.emit(painted, self._served_rect, self._capture_scale(frame))
        return True

    def _merge_with_carry(self, live: list[LiveRegion]) -> list[LiveRegion]:
        """Add back retained regions the fresh ones do not already cover.

        A carried region that overlaps a fresh one is dropped rather than
        painted alongside it: the overlay fills before it draws, so two boxes
        over the same text means one translation is painted out.
        """
        if not self._carry:
            return live
        return live + [
            carried for carried in self._carry
            if not any(_boxes_conflict(carried.box, region.box) for region in live)
        ]

    def _warn_if_nothing_translated(self, live) -> None:
        """Say so when every line came back as its own source text.

        Translation failures fall back to the original on purpose — showing the
        source beats showing a blank box. But that makes a model which cannot
        follow the prompt indistinguishable from a working overlay: the user
        sees their own game text neatly boxed and no error anywhere. A small
        model really does do this; one tested here answered the batch prompt
        with an empty ``json`` fence, and all twelve lines fell through.

        Warned once per run, not per frame, so a partly-translatable screen
        does not spam.
        """
        if self._warned_untranslated or not live:
            return
        if any(region.translated.strip() != region.original.strip() for region in live):
            return

        self._warned_untranslated = True
        message = (
            f"None of the {len(live)} detected lines were translated — the overlay is "
            "showing the original text. The translation model may be too small to follow "
            "the prompt, or unable to read this language."
        )
        logger.warning("%s", message)
        self.error.emit(message)

    def _capture_scale(self, frame: Image.Image) -> float:
        """Capture pixels per logical pixel, measured from the frame itself."""
        logical_width = self._served_rect.width()
        if logical_width <= 0 or not frame.width:
            return 1.0
        return frame.width / float(logical_width)

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
