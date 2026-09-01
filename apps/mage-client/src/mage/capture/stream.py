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

"""A persistent capture session, for modes that need frames rather than a photo.

:mod:`mage.capture.screen` takes a *screenshot*: one frame, on demand, through
whatever the platform offers — a subprocess on Wayland, a full-desktop
composite plus a PNG encode elsewhere.  That is the right shape for the
dialogue path, which captures once when the user asks.

Continuous modes want the opposite: a session opened once, delivering frames
until it is closed.  ``QScreenCapture`` provides exactly that, and it is the
same API on every target — the XDG ScreenCast portal and PipeWire on Wayland,
XCB on X11, Windows Graphics Capture on Windows, AVFoundation on macOS.  No
subprocess, no temp file, no encode round trip per frame.

It is not always there.  It needs Qt 6.5+ and the QtMultimedia module, so a
distribution shipping an older Qt (Debian stable, for one) will not have it,
and even where the class exists the session can be refused at runtime — the
user can decline the portal prompt, or a compositor may not implement the
interface.  Both cases have to degrade to the screenshot path rather than
break live mode, so support is probed rather than assumed, and a session that
fails after starting reports itself unavailable from then on.

Ownership: the capture session is a ``QObject`` chain that delivers frames via
a signal, so it must be created and started on the GUI thread.  The live worker
runs on its own thread and only ever calls :meth:`grab`, which is guarded by a
mutex.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QMutex, QMutexLocker, QObject, QRect
from PyQt6.QtGui import QGuiApplication, QImage, QScreen

logger = logging.getLogger(__name__)

__all__ = ["FrameStream", "map_region_to_frame", "screen_for_rect", "screen_capture_supported"]


def screen_capture_supported() -> bool:
    """Whether this Qt build offers a continuous screen-capture session.

    QtMultimedia is a separate module and may be absent from a distribution's
    PyQt6 package entirely; ``QScreenCapture`` itself only arrived in Qt 6.5.
    """
    try:
        from PyQt6.QtMultimedia import QMediaCaptureSession, QScreenCapture, QVideoSink  # noqa: F401
    except ImportError:
        return False
    return True


def screen_for_rect(rect: QRect) -> QScreen | None:
    """The screen a region sits on, chosen by its centre.

    A region straddling two monitors has to come from one of them; the centre
    picks the screen showing most of it, which is the one the user was looking
    at when they drew the box.
    """
    screens = QGuiApplication.screens()
    if not screens:
        return None

    centre = rect.center()
    for screen in screens:
        if screen.geometry().contains(centre):
            return screen

    # The centre landed in a gap between monitors (or the rect is off-screen);
    # fall back to whichever screen the rect overlaps most.
    best, best_area = None, 0
    for screen in screens:
        overlap = screen.geometry().intersected(rect)
        area = overlap.width() * overlap.height()
        if area > best_area:
            best, best_area = screen, area
    return best or QGuiApplication.primaryScreen()


def map_region_to_frame(
    rect: QRect, screen_geometry: QRect, frame_size: tuple[int, int]
) -> QRect | None:
    """Map a global logical rect onto a captured frame's device pixels.

    Qt reports geometry in logical pixels while the capture session hands back
    device pixels, so on any HiDPI screen the two differ by the device pixel
    ratio.  That ratio is measured from the frame rather than read from the
    screen: a compositor is free to hand back a buffer that is neither, and the
    frame is the thing we are actually cropping.

    Returns ``None`` when the region does not intersect the frame at all.
    """
    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        return None
    if screen_geometry.width() <= 0 or screen_geometry.height() <= 0:
        return None

    scale_x = frame_width / screen_geometry.width()
    scale_y = frame_height / screen_geometry.height()

    local = rect.translated(-screen_geometry.left(), -screen_geometry.top())
    mapped = QRect(
        int(local.left() * scale_x),
        int(local.top() * scale_y),
        max(1, int(local.width() * scale_x)),
        max(1, int(local.height() * scale_y)),
    )

    clipped = mapped.intersected(QRect(0, 0, frame_width, frame_height))
    return clipped if not clipped.isEmpty() else None


class FrameStream(QObject):
    """A live capture session for one screen, polled by the translation loop.

    Create and :meth:`start` on the GUI thread; :meth:`grab` is safe to call
    from a worker.
    """

    def __init__(self, screen: QScreen | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self._screen = screen or QGuiApplication.primaryScreen()
        self._mutex = QMutex()
        self._frame: QImage | None = None
        self._failed = False
        self._capture = None
        self._session = None
        self._sink = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the session. False means the caller should use screenshots.

        Returning True only means the session was accepted, not that a frame
        has arrived — the first few :meth:`grab` calls may still come back
        empty while the portal negotiates.
        """
        if not screen_capture_supported() or self._screen is None:
            return False

        from PyQt6.QtMultimedia import QMediaCaptureSession, QScreenCapture, QVideoSink

        try:
            self._capture = QScreenCapture(self)
            self._capture.setScreen(self._screen)
            self._sink = QVideoSink(self)
            self._session = QMediaCaptureSession(self)
            self._session.setScreenCapture(self._capture)
            self._session.setVideoSink(self._sink)

            self._sink.videoFrameChanged.connect(self._on_frame)
            self._capture.errorOccurred.connect(self._on_error)
            self._capture.setActive(True)
        except Exception as exc:
            # A Qt that has the class can still refuse the session — no
            # compositor support, no portal, a headless display.
            logger.info("Continuous capture unavailable (%s); using screenshots", exc)
            self._teardown()
            return False

        logger.info("Continuous capture started on %s", self._screen.name())
        return True

    def stop(self) -> None:
        """Close the session and release the frame buffer."""
        self._teardown()
        with QMutexLocker(self._mutex):
            self._frame = None

    def _teardown(self) -> None:
        if self._capture is not None:
            try:
                self._capture.setActive(False)
            except Exception:
                pass
        self._capture = None
        self._session = None
        self._sink = None

    @property
    def active(self) -> bool:
        """Whether the session is still expected to deliver frames."""
        return self._capture is not None and not self._failed

    # ── frame plumbing (GUI thread) ──────────────────────────────────

    def _on_frame(self, frame) -> None:
        """Keep the newest frame. Runs on the GUI thread."""
        if not frame.isValid():
            return
        image = frame.toImage()
        if image.isNull():
            return
        # toImage may alias the frame's buffer, which Qt reuses; copy before
        # handing it to another thread.
        with QMutexLocker(self._mutex):
            self._frame = image.copy()

    def _on_error(self, error, message: str) -> None:
        """A session that errors is done; fall back for the rest of the run."""
        logger.info("Continuous capture failed (%s); falling back to screenshots", message)
        self._failed = True
        self._teardown()

    # ── consumption (worker thread) ──────────────────────────────────

    def grab(self, rect: QRect) -> QImage | None:
        """The newest frame cropped to a global logical rect, if there is one.

        ``None`` means "no frame yet, use the screenshot path" — during portal
        negotiation, or after the session has failed.
        """
        if self._failed:
            return None

        with QMutexLocker(self._mutex):
            frame = self._frame
        if frame is None or frame.isNull() or self._screen is None:
            return None

        region = map_region_to_frame(
            rect, self._screen.geometry(), (frame.width(), frame.height())
        )
        if region is None:
            return None
        return frame.copy(region)
