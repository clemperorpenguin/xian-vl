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

"""What a single live tick pays to get pixels.

Every other benchmark here runs against a folder of captured frames, which
says nothing about how those frames were obtained.  Capture is the one part of
the live loop that cannot be measured from a corpus — it depends entirely on
the machine, the session type and the compositor — so it gets measured here,
against the real display, on whatever platform is running the tests.

This is the number that decides whether the polling interval is achievable at
all: it is spent before any inference starts, on every tick.

Needs a real display, so it skips headless and in CI.  On Wayland, starting a
continuous session shows the desktop's screen-share prompt — that is the
feature working, not a fault, but it does mean this benchmark is interactive
the first time it runs.
"""

import os
import statistics
import sys
import time

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.capture.screen import ScreenCapture  # noqa: E402
from mage.capture.stream import FrameStream, screen_capture_supported, screen_for_rect  # noqa: E402

pytestmark = pytest.mark.benchmark

#: The interval the live loop polls on (``live_lens.DEFAULT_INTERVAL_MS``).
LIVE_TICK_MS = 700

#: Grabs per measurement. Small: a screenshot on some backends spawns a
#: process, and this is not worth twenty seconds of anyone's test run.
SAMPLES = 5

#: How long to let a capture session negotiate before giving up on it.
FIRST_FRAME_TIMEOUT_S = 5.0


@pytest.fixture(scope="session", autouse=True)
def q_app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(scope="session")
def display(q_app):
    """A real screen, or skip: capture cost is meaningless offscreen."""
    if os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen"):
        pytest.skip("headless session; capture timings would measure nothing")
    if not q_app.screens():
        pytest.skip("no screens available")
    return q_app.primaryScreen()


@pytest.fixture(scope="session")
def region(display) -> QRect:
    """A dialogue-box-sized region, the shape live mode is actually used on."""
    geometry = display.geometry()
    return QRect(geometry.left() + 100, geometry.top() + 100, 900, 500)


def _median_ms(operation, samples: int = SAMPLES) -> float | None:
    """Median wall time of *operation*, or None if it never produced anything."""
    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        result = operation()
        elapsed = (time.perf_counter() - started) * 1000
        if result is None:
            return None
        timings.append(elapsed)
    return statistics.median(timings)


def test_the_screenshot_path_costs_what_it_costs(region, record_property):
    """Measure the existing per-tick screenshot, whatever backend wins here.

    No assertion on the number: it is a property of the machine, and the point
    is to have it written down rather than guessed at.  What *is* asserted is
    that the path works at all — live mode falls back to it whenever a
    continuous session is unavailable, so a broken screenshot path means no
    live mode on those systems.
    """
    def grab():
        data, _cropped = ScreenCapture.capture_region(region)
        return data

    median = _median_ms(grab)
    if median is None:
        pytest.skip("screen capture returned nothing (permissions, or a locked session)")

    record_property("screenshot_capture_ms", round(median, 1))
    record_property("screenshot_fits_live_tick", bool(median < LIVE_TICK_MS))
    record_property("platform", sys.platform)
    record_property("session_type", os.environ.get("XDG_SESSION_TYPE", "n/a"))
    record_property("desktop", os.environ.get("XDG_CURRENT_DESKTOP", "n/a"))


@pytest.mark.skipif(not screen_capture_supported(), reason="Qt build has no QScreenCapture")
def test_a_continuous_session_is_cheaper_than_a_screenshot(region, record_property):
    """The claim behind the frame stream, measured rather than assumed.

    A screenshot re-acquires the screen every time — on some backends by
    spawning a process and round-tripping a PNG through a temp file.  A session
    is opened once and hands back decoded pixels.  If that does not actually
    win on this machine, the stream is not worth its complexity here.
    """
    stream = FrameStream(screen_for_rect(region))
    if not stream.start():
        pytest.skip("continuous capture session was refused on this system")

    try:
        # Let the portal negotiate and the first frame land.
        app = QApplication.instance()
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
        while time.monotonic() < deadline and stream.grab(region) is None:
            app.processEvents()
            time.sleep(0.05)

        if stream.grab(region) is None:
            pytest.skip("no frame arrived; the prompt may have been declined")

        stream_ms = _median_ms(lambda: stream.grab(region))
    finally:
        stream.stop()

    def screenshot():
        data, _cropped = ScreenCapture.capture_region(region)
        return data

    screenshot_ms = _median_ms(screenshot)

    record_property("stream_grab_ms", round(stream_ms, 1))
    record_property("stream_fits_live_tick", bool(stream_ms < LIVE_TICK_MS))
    if screenshot_ms is not None:
        record_property("screenshot_capture_ms", round(screenshot_ms, 1))
        record_property("speedup", round(screenshot_ms / max(0.01, stream_ms), 1))

    assert stream_ms < LIVE_TICK_MS, (
        f"a frame from the continuous session took {stream_ms:.0f}ms, which does not "
        f"fit a {LIVE_TICK_MS}ms tick even before inference"
    )
    if screenshot_ms is not None:
        assert stream_ms < screenshot_ms, (
            f"the continuous session ({stream_ms:.0f}ms) was no faster than a "
            f"screenshot ({screenshot_ms:.0f}ms); the stream is not earning its keep"
        )
