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

"""Region mapping and the fallback contract for continuous capture.

The parts that can be tested without a compositor: the logical-to-device
coordinate mapping, and the rule that anything short of a delivered frame has
to fall back to the screenshot path rather than drop the tick.
"""

import os
import sys

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.capture.stream import (  # noqa: E402
    FrameStream,
    map_region_to_frame,
    screen_capture_supported,
)


@pytest.fixture(scope="session", autouse=True)
def q_app():
    """FrameStream is a QObject; constructing one needs a live application."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


# ── coordinate mapping ───────────────────────────────────────────────

def test_a_region_maps_straight_through_at_one_to_one():
    screen = QRect(0, 0, 1920, 1080)
    mapped = map_region_to_frame(QRect(100, 200, 400, 300), screen, (1920, 1080))
    assert mapped == QRect(100, 200, 400, 300)


def test_a_region_scales_up_on_a_hidpi_frame():
    """Qt reports logical pixels; the capture session hands back device ones."""
    screen = QRect(0, 0, 1440, 900)
    mapped = map_region_to_frame(QRect(100, 200, 400, 300), screen, (2880, 1800))
    assert mapped == QRect(200, 400, 800, 600)


def test_a_region_on_a_second_monitor_is_made_screen_local():
    """Global coordinates are desktop-wide; the frame starts at that screen."""
    screen = QRect(1920, 0, 1920, 1080)
    mapped = map_region_to_frame(QRect(2020, 100, 300, 200), screen, (1920, 1080))
    assert mapped == QRect(100, 100, 300, 200)


def test_a_region_hanging_off_the_edge_is_clipped_to_the_frame():
    screen = QRect(0, 0, 1920, 1080)
    mapped = map_region_to_frame(QRect(1800, 1000, 400, 300), screen, (1920, 1080))
    assert mapped == QRect(1800, 1000, 120, 80)


def test_a_region_entirely_off_the_frame_maps_to_nothing():
    """Better no crop than a bogus one — the caller falls back."""
    screen = QRect(0, 0, 1920, 1080)
    assert map_region_to_frame(QRect(4000, 100, 200, 200), screen, (1920, 1080)) is None


@pytest.mark.parametrize("frame_size", [(0, 1080), (1920, 0)])
def test_a_degenerate_frame_maps_to_nothing(frame_size):
    assert map_region_to_frame(QRect(0, 0, 10, 10), QRect(0, 0, 1920, 1080), frame_size) is None


def test_a_degenerate_screen_geometry_maps_to_nothing():
    """Guards the scale division; an unmapped screen reports zero geometry."""
    assert map_region_to_frame(QRect(0, 0, 10, 10), QRect(0, 0, 0, 0), (1920, 1080)) is None


def test_a_sub_pixel_region_keeps_at_least_one_pixel():
    """A thin region scaled down must not round away to an empty crop."""
    mapped = map_region_to_frame(QRect(10, 10, 1, 1), QRect(0, 0, 3840, 2160), (1920, 1080))
    assert mapped is not None and mapped.width() >= 1 and mapped.height() >= 1


# ── the fallback contract ────────────────────────────────────────────

def test_support_probe_answers_without_raising():
    """Debian stable ships a Qt with no QScreenCapture; that is not an error."""
    assert isinstance(screen_capture_supported(), bool)


def test_a_stream_with_no_frame_yet_yields_nothing():
    """Portal negotiation takes a moment; those ticks use the screenshot path."""
    stream = FrameStream()
    assert stream.grab(QRect(0, 0, 100, 100)) is None


def test_a_failed_session_stops_offering_frames():
    """One error means fall back for the rest of the run, not retry every tick."""
    stream = FrameStream()
    stream._on_error(None, "compositor refused")

    assert stream.active is False
    assert stream.grab(QRect(0, 0, 100, 100)) is None


def test_starting_without_support_reports_failure(monkeypatch):
    monkeypatch.setattr("mage.capture.stream.screen_capture_supported", lambda: False)
    assert FrameStream().start() is False


def test_the_live_worker_falls_back_when_the_stream_is_empty(monkeypatch):
    """The stream is an optimisation; an empty one must not break the tick."""
    from mage.live_lens import LiveLensWorker

    worker = LiveLensWorker.__new__(LiveLensWorker)
    worker.rect = QRect(0, 0, 64, 48)
    worker._frame_stream = None

    captured = {}

    class _FakeCapture:
        @staticmethod
        def capture_region(rect):
            captured["rect"] = rect
            return None, False

    monkeypatch.setitem(
        sys.modules, "mage.capture.screen", type("m", (), {"ScreenCapture": _FakeCapture})
    )

    assert LiveLensWorker._grab(worker) is None
    assert captured["rect"] == worker.rect, "the screenshot path should still be tried"
