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

"""Translation boxes: rectangles you place once and leave there.

Not the classic lens selection, which is a one-shot gesture that translates and
forgets.  A box is a persistent object with a position, a mode and a state, and
it is the whole interaction model of the new UI: everything the leader key and
its letters used to do to a region, you now do to a box you can see.

Four modes, cycled from the box's own toolbar, no modifiers anywhere:

    Live    ticks continuously and paints in place
    Once    reads and translates on click, then holds — replaces dialogue mode
    Ignore  masked out of every other box's frame before it is read *or*
            hashed — this is RST's "exclude regions" with no settings page
    Off     present and remembered, doing nothing

Geometry, drag, click-through and multi-monitor clamping all come from
:class:`~mage.ui.overlay_base.MageOverlayWindow`, so a box's position persists
per layout preset exactly as every other overlay's does.
"""

from __future__ import annotations

import json
import logging
from enum import Enum

from PyQt6.QtCore import QObject, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QRegion
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from mage.ui.overlay_base import MageOverlayWindow
from shared_types.state import t

logger = logging.getLogger(__name__)

#: How long a capture session outlives the one-off that opened it.
ONCE_STREAM_GRACE_MS = 120_000

#: How long a one-off that waited on the share prompt keeps reading after an
#: empty first read.  The first frames after the user picks a screen can be
#: blank, or still show the prompt closing over the area; the reader re-reads
#: as soon as the frame changes, so this is only how long "no text" waits.
ONCE_PROMPT_SETTLE_MS = 2500

__all__ = ["BoxManager", "BoxMode", "BoxState", "TranslationBox", "MAX_BOXES"]

#: RST allows five saved areas and nobody has ever asked for a sixth.  The cap
#: matters here for a harder reason: every live box is a reader, and they share
#: one engine.
MAX_BOXES = 5

#: Width of the frame you can actually grab, in pixels.
#:
#: A box has to be two contradictory things: a window you can move, resize the
#: attention of, and delete; and a hole you can click *through*, because the
#: game underneath is still a game.  The resolution is that only a band around
#: the edge belongs to the box — wide enough to hit without aiming, and no
#: wider, because every pixel of it is a pixel of the game you cannot click.
#:
#: The 2px border this replaced was, on Windows, the entire interactive
#: surface of a translation box: a layered window passes input straight
#: through its fully transparent pixels, so the only thing that could be
#: hovered or dragged was the drawn line itself.  Boxes were, in the user's
#: words, unclickable, undraggable and impossible to clear.
FRAME_BAND = 10

#: Alpha of the band's fill.  Two jobs, and the second is why it cannot be
#: zero: it shows you what you can grab, and on Windows a pixel with no alpha
#: at all is a pixel the mouse never reaches.
BAND_ALPHA = 46


class BoxMode(Enum):
    LIVE = "live"
    ONCE = "once"
    IGNORE = "ignore"
    OFF = "off"

    def next(self) -> "BoxMode":
        order = list(BoxMode)
        return order[(order.index(self) + 1) % len(order)]


class BoxState(Enum):
    """What the box is doing, shown as its border colour.

    This replaces the status LEDs a HUD would otherwise need: the indicator
    lives on the thing it describes, where the user is already looking.
    """

    IDLE = "idle"
    READING = "reading"
    TRANSLATING = "translating"
    SETTLED = "settled"
    FAILED = "failed"


_MODE_COLORS = {
    BoxMode.LIVE: QColor(90, 200, 250),
    BoxMode.ONCE: QColor(160, 220, 130),
    BoxMode.IGNORE: QColor(120, 120, 130),
    BoxMode.OFF: QColor(90, 90, 100),
}

_STATE_COLORS = {
    BoxState.READING: QColor(250, 200, 90),
    BoxState.TRANSLATING: QColor(200, 140, 250),
    BoxState.SETTLED: QColor(120, 210, 140),
    BoxState.FAILED: QColor(230, 100, 100),
}


class TranslationBox(MageOverlayWindow):
    """One placed rectangle, with its mode and its state on its own border."""

    mode_changed = pyqtSignal(object)
    delete_requested = pyqtSignal(object)
    activated = pyqtSignal(object)
    moved = pyqtSignal(object)

    def __init__(self, box_id: str, rect: QRect, mode: BoxMode = BoxMode.LIVE, app=None, parent=None):
        super().__init__(box_id, app=app, parent=parent)
        self.box_id = box_id
        self.mode = mode
        self.state = BoxState.IDLE
        self.setGeometry(rect)
        # Boxes are frames around the game, not windows over it: the middle has
        # to stay clickable or the box covers the thing it is translating.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._toolbar = _BoxToolbar(self)
        self._toolbar.hide()
        self._toolbar.set_mode(mode)
        self._toolbar.mode_clicked.connect(self._cycle_mode)
        self._toolbar.delete_clicked.connect(lambda: self.delete_requested.emit(self))
        self.setMouseTracking(True)
        self._refresh_mask()

    # ── what belongs to the box, and what belongs to the game ────────

    def _refresh_mask(self) -> None:
        """Keep the middle of the box out of the window entirely.

        Masking rather than painting-nothing, because the two platforms
        disagree about what a transparent pixel means: X11 hands the click to
        whatever is on top regardless of what it drew, so without this the box
        swallows every click aimed at the game inside it.  The toolbar is
        united back in while it is showing, or the mask would clip the very
        controls it exists to reveal.
        """
        outer = self.rect()
        region = QRegion(outer)
        inner = outer.adjusted(FRAME_BAND, FRAME_BAND, -FRAME_BAND, -FRAME_BAND)
        if inner.width() > 0 and inner.height() > 0:
            region = region.subtracted(QRegion(inner))
        if self._toolbar.isVisible():
            region = region.united(QRegion(self._toolbar.geometry()))
        self.setMask(region)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_mask()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_mask()

    # ── mode and state ───────────────────────────────────────────────

    def _cycle_mode(self) -> None:
        self.mode = self.mode.next()
        self._toolbar.set_mode(self.mode)
        self.update()
        self.mode_changed.emit(self)

    def set_mode(self, mode: BoxMode) -> None:
        if mode != self.mode:
            self.mode = mode
            self._toolbar.set_mode(mode)
            self.update()
            self.mode_changed.emit(self)

    def set_state(self, state: BoxState) -> None:
        if state != self.state:
            self.state = state
            self.update()

    # ── serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        geometry = self.geometry()
        return {
            "id": self.box_id,
            "x": geometry.x(),
            "y": geometry.y(),
            "w": geometry.width(),
            "h": geometry.height(),
            "mode": self.mode.value,
        }

    # ── painting ─────────────────────────────────────────────────────

    def enterEvent(self, event):
        self._toolbar.move(4, 4)
        self._toolbar.show()
        self._refresh_mask()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._toolbar.hide()
        self._refresh_mask()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        # A dragged box is reading somewhere else now, and its stored position
        # is stale until this fires.
        super().mouseReleaseEvent(event)
        self.moved.emit(self)

    def mouseDoubleClickEvent(self, event):
        # Double-click, not single: single-click drags the box, and a gesture
        # that both moves and fires a translation is a gesture nobody trusts.
        if self.mode is BoxMode.ONCE:
            self.activated.emit(self)
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        # Keeps the base class's edit-mode highlight working underneath.
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        color = _STATE_COLORS.get(self.state, _MODE_COLORS[self.mode])

        # The grabbable band, drawn as the frame it is.  Painted rather than
        # left clear because on Windows an unpainted pixel is one the mouse
        # passes through, and a frame nobody can hit is a box nobody can move
        # or delete.
        band = QColor(color)
        band.setAlpha(BAND_ALPHA)
        outer = self.rect()
        inner = outer.adjusted(FRAME_BAND, FRAME_BAND, -FRAME_BAND, -FRAME_BAND)
        frame = QRegion(outer)
        if inner.width() > 0 and inner.height() > 0:
            frame = frame.subtracted(QRegion(inner))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setClipRegion(frame)
        painter.fillRect(outer, band)
        painter.setClipping(False)

        painter.setPen(QPen(color, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(outer.adjusted(1, 1, -2, -2), 6, 6)

        if self.mode is BoxMode.IGNORE:
            # Hatched, so an Ignore box does not read as a translation box
            # that has stopped working.
            painter.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            for offset in range(0, self.width() + self.height(), 18):
                painter.drawLine(offset, 0, 0, offset)


class _BoxToolbar(QWidget):
    """The box's own controls, shown on hover.  No modifiers, no chords."""

    mode_clicked = pyqtSignal()
    delete_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self._mode_button = QPushButton()
        self._mode_button.clicked.connect(self.mode_clicked)
        self._delete_button = QPushButton("✕")
        self._delete_button.setFixedWidth(22)
        self._delete_button.clicked.connect(self.delete_clicked)

        for button in (self._mode_button, self._delete_button):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)

        self.setStyleSheet(
            "QPushButton { background: rgba(20,20,28,220); color: #eee; border: 1px solid #555;"
            " border-radius: 4px; padding: 2px 8px; font-size: 11px; }"
            "QPushButton:hover { background: rgba(50,50,70,240); }"
        )
        self.set_mode(BoxMode.LIVE)

    def set_mode(self, mode: BoxMode) -> None:
        self._mode_button.setText(t(f"newui.box.mode.{mode.value}"))
        self.adjustSize()


class BoxManager(QObject):
    """Owns the boxes, their persistence, and the workers behind them.

    Every live box reads through **one shared OCR engine**.  The engine
    serialises its own sessions, so five boxes cost five reads spread over the
    ticks rather than five ONNX sessions competing for the same cores — which
    on a machine also running a game is the difference between a slow overlay
    and a stuttering one.  Their first ticks are staggered for the same reason.
    """

    boxes_changed = pyqtSignal()
    #: A one-off translation has finished, whatever its outcome.
    once_finished = pyqtSignal()
    #: (original, translated), emitted from the worker's thread so the orb's
    #: log is written on the GUI thread, where its widgets live.
    translated = pyqtSignal(str, str)

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.translated.connect(self._show_translation)
        self.boxes: list[TranslationBox] = []
        self._workers: dict[str, object] = {}
        self._overlays: dict[str, object] = {}
        self._streams: dict[str, object] = {}
        self._engine = None
        self._next_id = 1

    # ── the shared engine ────────────────────────────────────────────

    def engine(self):
        if self._engine is None:
            from mage.settings_keys import KEY_OCR_DETECTOR, KEY_SOURCE_LANG
            from shared_types import constants
            from xian.ocr import PaddleOcrEngine

            detector = self.app.settings.value(KEY_OCR_DETECTOR, "") or None
            self._engine = PaddleOcrEngine(
                source_language=self.app.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG),
                **({"detector_model": detector} if detector else {}),
            )
        return self._engine

    # ── the shared capture sessions ──────────────────────────────────

    def _stream_for(self, rect: QRect):
        """The continuous capture session covering a rectangle's screen.

        Opened here, on the GUI thread, because the session delivers its frames
        through a signal; the worker only ever grabs from it.  One per screen
        rather than one per box: boxes on the same monitor are readers of the
        same pixels, and a second session of one screen is the whole cost
        again for an identical frame.

        Without it every tick falls back to the screenshot path, which on
        anything but Wayland means compositing the entire virtual desktop —
        per box, per tick.  That is what made a single live box unusable.

        A session that will not start is remembered as ``None`` so it is not
        renegotiated every time a box restarts; those boxes use screenshots,
        which is slower but correct.
        """
        from mage.capture.stream import FrameStream, screen_for_rect

        screen = screen_for_rect(rect)
        if screen is None:
            return None
        name = screen.name()
        existing = self._streams.get(name)
        if existing is not None and not existing.active:
            # It started and then failed — on Wayland, most often the share
            # prompt was declined.  Asking again is the only way to recover.
            existing.stop()
            del self._streams[name]
        if name not in self._streams:
            stream = FrameStream(screen, parent=self)
            self._streams[name] = stream if stream.start() else None
        return self._streams[name]

    def _release_streams(self) -> None:
        for stream in self._streams.values():
            if stream is not None:
                stream.stop()
        self._streams.clear()

    def _release_streams_if_idle(self) -> None:
        """Close the sessions once nothing is reading through them.

        A capture session is a live compositor client; leaving one open for a
        box that has been switched to Off costs frames nobody looks at.  Boxes
        still in Live mode count even with no worker running, because
        :meth:`_sync_box` stops one only to start it again.
        """
        if self._workers or any(box.mode is BoxMode.LIVE for box in self.boxes):
            return
        self._release_streams()

    # ── the boxes ────────────────────────────────────────────────────

    def add_box(self, rect: QRect, mode: BoxMode = BoxMode.LIVE) -> TranslationBox | None:
        if len(self.boxes) >= MAX_BOXES:
            logger.info("box limit of %d reached", MAX_BOXES)
            return None

        box = TranslationBox(f"box_{self._next_id}", rect, mode, app=self.app)
        self._next_id += 1
        self._connect(box)
        self.boxes.append(box)
        box.show()

        self._sync_box(box)
        self.save()
        self.boxes_changed.emit()
        return box

    def remove_box(self, box: TranslationBox) -> None:
        # Removed before it is stopped: the capture sessions are released once
        # no box still wants them, and a box on its way out must not count.
        if box in self.boxes:
            self.boxes.remove(box)
        self._stop_box(box)
        box.close()
        box.deleteLater()
        self.save()
        self.boxes_changed.emit()

    def clear(self) -> None:
        for box in list(self.boxes):
            self.remove_box(box)

    def box(self, box_id: str) -> "TranslationBox | None":
        """The box with this id, for controls that live somewhere else."""
        for box in self.boxes:
            if box.box_id == box_id:
                return box
        return None

    def overlays(self) -> list:
        """The painted overlays, for the keep-on-top tick.

        These are the windows the translations are actually drawn in, so
        leaving them out of the promote loop meant the boxes stayed above a
        fullscreen game and the text inside them did not.
        """
        return [overlay for overlay in self._overlays.values() if overlay is not None]

    def exclude_regions(self) -> list[QRect]:
        """The Ignore boxes, for masking out of every other box's frame."""
        return [box.geometry() for box in self.boxes if box.mode is BoxMode.IGNORE]

    # ── running ──────────────────────────────────────────────────────

    def _connect(self, box: TranslationBox) -> None:
        box.delete_requested.connect(self.remove_box)
        box.mode_changed.connect(self._on_mode_changed)
        box.activated.connect(self.run_once)
        box.moved.connect(self._on_moved)

    def _on_mode_changed(self, box: TranslationBox) -> None:
        self._sync_box(box)
        self.save()
        # The orb's row names each box's mode, and a mode changed from the
        # box's own toolbar has to reach it.
        self.boxes_changed.emit()

    def _on_moved(self, box: TranslationBox) -> None:
        """A box that moved is bound to the wrong rectangle until it restarts."""
        self.save()
        if box.mode is BoxMode.LIVE:
            self._sync_box(box)

    def _sync_box(self, box: TranslationBox, *, delay_ms: int = 0) -> None:
        """Start or stop a box's worker to match its mode."""
        self._stop_box(box)
        if box.mode is not BoxMode.LIVE:
            box.set_state(BoxState.IDLE)
            return
        if delay_ms:
            QTimer.singleShot(delay_ms, lambda: self._start_box(box))
        else:
            self._start_box(box)

    def _start_box(self, box: TranslationBox) -> None:
        if box not in self.boxes or box.mode is not BoxMode.LIVE:
            return
        try:
            worker, overlay = self._build_worker(box)
        except Exception as exc:
            logger.error("could not start box %s: %s", box.box_id, exc)
            box.set_state(BoxState.FAILED)
            return
        self._workers[box.box_id] = worker
        self._overlays[box.box_id] = overlay
        box.set_state(BoxState.READING)
        worker.start()

    def _build_worker(self, box: TranslationBox):
        return self._build_worker_for_rect(box.geometry(), box=box)

    def _build_worker_for_rect(self, rect: QRect, *, box: "TranslationBox | None" = None, settle_now: bool = False):
        """A reader and an overlay for one rectangle, box or not.

        ``settle_now`` drops the settle window, which exists to stop a
        typewriter reveal being translated mid-sentence.  A one-off
        translation the user just asked for should not wait for a second look
        at text they can already see.
        """
        from mage.live_engine import resolve_live_engine
        from mage.settings_keys import (
            KEY_IGNORE_PHRASES,
            KEY_LIVE_INTERVAL_MS,
            KEY_SOURCE_LANG,
            KEY_TARGET_LANG,
            LIVE_ENGINE_OCR,
        )
        from mage.ui.inpaint_overlay import InpaintOverlay
        from shared_types import constants

        overlay = InpaintOverlay()
        overlay.bind_to_rect(rect)
        overlay.show()

        settings = self.app.settings
        common = dict(
            frame_stream=self._stream_for(rect),
            source_lang=settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG),
            target_lang=settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG),
            interval_ms=int(settings.value(KEY_LIVE_INTERVAL_MS, constants.DEFAULT_LIVE_INTERVAL_MS)),
            session_recorder=self._record,
        )

        # Whichever engine the settings ask for, and can actually have.  The
        # boxes used to build an OCR reader whatever was chosen, so a machine
        # that had never run ``scripts/export_ppocr_onnx.py`` — which is every
        # machine until someone does — had every box fail on its first read
        # while the settings screen said something else entirely.
        if resolve_live_engine(settings) == LIVE_ENGINE_OCR:
            from mage.live_ocr import LiveOcrWorker
            from mage.translation import make_translator
            from xian.filters import TextFilter
            from xian.text_gate import SettleGate

            worker = LiveOcrWorker(
                self.app.processor,
                rect,
                engine=self.engine(),
                translator=make_translator(settings, self.app.processor),
                text_filter=TextFilter.from_settings(settings.value(KEY_IGNORE_PHRASES, "")),
                exclude_regions=[r for r in self.exclude_regions() if r != rect],
                gate=SettleGate(settle_seconds=0.0) if settle_now else None,
                **common,
            )
        else:
            from mage.live_lens import LiveLensWorker

            worker = LiveLensWorker(self.app.processor, rect, **common)

        worker.regions_ready.connect(
            lambda regions, served, scale, ov=overlay, b=box: self._on_regions(ov, b, regions, served, scale)
        )
        worker.error.connect(lambda message, b=box: self._on_worker_error(b, message))
        return worker, overlay

    def _on_worker_error(self, box, message: str) -> None:
        """Say what went wrong, where the user is already looking.

        A box that could not read has only ever turned its border red, and the
        reason went to a log nobody has open — so the commonest first-run
        failure of all, weights that were never exported, looked exactly like
        a feature that does nothing.  The message these raise says what to
        run; it needs somewhere to be read.
        """
        logger.error("live box %s: %s", getattr(box, "box_id", "one-off"), message)
        if box is not None:
            box.set_state(BoxState.FAILED)
            box.setToolTip(message)
        shell = getattr(self.app, "_shell", None)
        if shell is not None:
            shell.on_error(message)

    def _record(self, original: str, translated: str) -> None:
        """Into session memory, and into the orb's log if there is one.

        Called on the async engine's thread, not the GUI thread: the session
        store is fine with that, but the orb's panel is a widget and must only
        be touched through the queued signal.
        """
        self.app.processor.record_event("inpaint", original, translated)
        self.translated.emit(original, translated)

    def _show_translation(self, original: str, translated: str) -> None:
        shell = getattr(self.app, "_shell", None)
        if shell is not None:
            shell.on_translation(original, translated)

    def _on_regions(self, overlay, box, regions, served_rect, scale) -> None:
        from mage.ui.inpaint_overlay import InpaintRegion, contrasting_text_color

        if box is not None:
            box.set_state(BoxState.SETTLED if regions else BoxState.READING)

        # Boxes arrive from the worker in capture pixels as (left, top, right,
        # bottom); Qt paints in logical coordinates and wants x/y/w/h.  The
        # ratio is measured from the frame that was actually captured, because
        # the display's devicePixelRatio is wrong for the PyQt capture path,
        # which already composites in logical pixels.
        ratio = scale if scale and scale > 0 else 1.0
        painted = []
        for region in regions:
            left, top, right, bottom = region.box
            fill = QColor(*region.fill)
            painted.append(
                InpaintRegion(
                    rect=QRect(
                        int(left / ratio),
                        int(top / ratio),
                        int((right - left) / ratio),
                        int((bottom - top) / ratio),
                    ),
                    text=region.translated,
                    fill=fill,
                    text_color=contrasting_text_color(fill),
                )
            )

        overlay.bind_to_rect(served_rect)
        overlay.set_regions(painted)
        overlay.show()
        overlay.promote()

    def _stop_box(self, box: TranslationBox) -> None:
        worker = self._workers.pop(box.box_id, None)
        if worker is not None:
            try:
                worker.stop()
                worker.requestInterruption()
                # Same reasoning as XianApp.stop_live_lens: a read already in
                # flight can outlive this wait, and dropping the last reference
                # to a running QThread aborts the process.
                if not worker.wait(2000):
                    worker.finished.connect(worker.deleteLater)
            except Exception as exc:
                logger.error("error stopping box %s: %s", box.box_id, exc)

        overlay = self._overlays.pop(box.box_id, None)
        if overlay is not None:
            overlay.close()
            overlay.deleteLater()

        self._release_streams_if_idle()

    def run_once(self, box: TranslationBox) -> None:
        """A single pass for a Once box: start it, and stop after one publish."""
        if box.mode is not BoxMode.ONCE:
            return
        self._stop_box(box)
        try:
            worker, overlay = self._build_worker(box)
        except Exception as exc:
            logger.error("could not run box %s: %s", box.box_id, exc)
            box.set_state(BoxState.FAILED)
            return
        self._workers[box.box_id] = worker
        self._overlays[box.box_id] = overlay
        box.set_state(BoxState.READING)
        worker.regions_ready.connect(lambda *_args, b=box: QTimer.singleShot(0, lambda: self._finish_once(b)))
        worker.start()

    def _finish_once(self, box: TranslationBox) -> None:
        """Stop the worker but leave its overlay painted."""
        worker = self._workers.pop(box.box_id, None)
        if worker is not None:
            worker.stop()
            worker.requestInterruption()
            if not worker.wait(2000):
                worker.finished.connect(worker.deleteLater)
        box.set_state(BoxState.SETTLED)
        self._release_streams_if_idle()

    # ── one-off ──────────────────────────────────────────────────────

    def translate_once(self, rect: QRect, *, hold_ms: int = 12000) -> None:
        """Read and translate a rectangle once, then let it fade.

        No box is left behind.  For the case a persistent box is too much
        ceremony for — a sign, an item tooltip, one line of a menu you are
        never coming back to.
        """
        key = f"once_{id(rect)}_{len(self._workers)}"
        try:
            worker, overlay = self._build_worker_for_rect(rect, settle_now=True)
        except Exception as exc:
            logger.error("one-off translation could not start: %s", exc)
            return

        self._workers[key] = worker
        self._overlays[key] = overlay
        # Asked now, before the worker starts: a session still waiting on the
        # share prompt is one whose first frame should not be trusted.
        stream = self._stream_for(rect)
        prompted = stream is not None and stream.waiting
        settling = [False]

        def finish(*_args):
            existing = self._workers.pop(key, None)
            if existing is None:
                return  # already finished by another signal
            existing.stop()
            existing.requestInterruption()
            if not existing.wait(2000):
                existing.finished.connect(existing.deleteLater)
            # Held open for a while rather than closed now: on Wayland every
            # new session asks the user to share their screen again, and a
            # one-off is rarely the last.
            QTimer.singleShot(ONCE_STREAM_GRACE_MS, self._release_streams_if_idle)
            # The overlay outlives its reader so the translation stays on
            # screen long enough to be read, then goes on its own.
            QTimer.singleShot(hold_ms, lambda: self._dismiss_once(key))
            self.once_finished.emit()

        def nothing_read():
            if key not in self._workers:
                return
            if settling[0]:
                return  # still inside the window; it decides
            if prompted:
                # Let the reader see a few more frames; a read with text
                # finishes it before the window closes.
                settling[0] = True
                QTimer.singleShot(ONCE_PROMPT_SETTLE_MS, settled)
                return
            finish()
            self._notify(t("newui.once.no_text"))

        def settled():
            if key in self._workers:
                finish()
                self._notify(t("newui.once.no_text"))

        worker.regions_ready.connect(lambda *_a: QTimer.singleShot(0, finish))
        worker.nothing_read.connect(lambda: QTimer.singleShot(0, nothing_read))
        worker.error.connect(lambda _m: QTimer.singleShot(0, finish))
        worker.start()

    def _notify(self, message: str) -> None:
        shell = getattr(self.app, "_shell", None)
        if shell is not None:
            shell.on_error(message)

    def _dismiss_once(self, key: str) -> None:
        overlay = self._overlays.pop(key, None)
        if overlay is not None:
            overlay.close()
            overlay.deleteLater()

    def start_all(self) -> None:
        """Bring every live box up, staggered so they do not all read at once."""
        live = [box for box in self.boxes if box.mode is BoxMode.LIVE]
        for index, box in enumerate(live):
            self._sync_box(box, delay_ms=index * 250)

    def stop_all(self) -> None:
        for box in self.boxes:
            self._stop_box(box)
            box.set_state(BoxState.IDLE)
        # Unconditionally, unlike the per-box release: the boxes keep their
        # modes across a stop, so "is anything still Live" would hold every
        # session open until the app exits.
        self._release_streams()

    # ── persistence ──────────────────────────────────────────────────

    def _settings_key(self) -> str:
        preset = self.app.settings.value("layout_preset", "Default")
        return f"boxes/{preset}"

    def save(self) -> None:
        payload = json.dumps([box.to_dict() for box in self.boxes])
        self.app.settings.setValue(self._settings_key(), payload)

    def load(self) -> None:
        raw = self.app.settings.value(self._settings_key(), "")
        if not raw:
            return
        try:
            entries = json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.error("stored box layout is unreadable: %s", exc)
            return

        for entry in entries[:MAX_BOXES]:
            try:
                rect = QRect(int(entry["x"]), int(entry["y"]), int(entry["w"]), int(entry["h"]))
                mode = BoxMode(entry.get("mode", BoxMode.LIVE.value))
            except (KeyError, TypeError, ValueError) as exc:
                logger.error("skipping unreadable box entry %r: %s", entry, exc)
                continue
            # Added without starting: start_all decides that, once every box
            # (including the Ignore ones) exists and can be masked out.
            box = TranslationBox(entry.get("id") or f"box_{self._next_id}", rect, mode, app=self.app)
            self._next_id = max(self._next_id + 1, len(self.boxes) + 2)
            self._connect(box)
            self.boxes.append(box)
            box.show()
        self.boxes_changed.emit()
