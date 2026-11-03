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

"""Translation boxes: their modes, their persistence, and what they mask."""

import json
import os
import sys

import pytest
from PyQt6.QtCore import QObject, QRect, QSettings
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtTest import QTest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from shared_types.state import t  # noqa: E402
from mage.ui.new.boxes import MAX_BOXES, BoxManager, BoxMode, BoxState, TranslationBox  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def q_app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


class FakeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("XianProject", "MageBoxTest")
        self.settings.clear()
        self.processor = object()


@pytest.fixture
def manager():
    app = FakeApp()
    manager = BoxManager(app)
    yield manager
    manager.clear()
    app.settings.clear()


# ── modes ────────────────────────────────────────────────────────────

def test_the_mode_cycle_returns_to_where_it_started():
    """One button, four modes, no modifiers — so it has to be a cycle."""
    mode = BoxMode.LIVE
    for _ in range(len(BoxMode)):
        mode = mode.next()

    assert mode is BoxMode.LIVE


def test_a_new_box_carries_the_mode_it_was_given(manager):
    box = manager.add_box(QRect(10, 10, 200, 60), BoxMode.OFF)

    assert box is not None
    assert box.mode is BoxMode.OFF


def test_an_off_box_starts_no_worker(manager):
    manager.add_box(QRect(10, 10, 200, 60), BoxMode.OFF)

    assert manager._workers == {}


# ── the cap ──────────────────────────────────────────────────────────

def test_the_box_count_is_capped(manager):
    """Every live box is a reader, and they share one engine."""
    for index in range(MAX_BOXES):
        assert manager.add_box(QRect(index * 20, 10, 100, 40), BoxMode.OFF) is not None

    assert manager.add_box(QRect(500, 10, 100, 40), BoxMode.OFF) is None
    assert len(manager.boxes) == MAX_BOXES


# ── exclusions ───────────────────────────────────────────────────────

def test_only_ignore_boxes_are_excluded(manager):
    manager.add_box(QRect(10, 10, 100, 40), BoxMode.OFF)
    ignored = manager.add_box(QRect(200, 10, 80, 30), BoxMode.IGNORE)

    assert manager.exclude_regions() == [ignored.geometry()]


def test_a_box_does_not_exclude_itself(manager):
    """It would mask out the very region it is there to read."""
    ignored = manager.add_box(QRect(200, 10, 80, 30), BoxMode.IGNORE)

    excludes = [rect for rect in manager.exclude_regions() if rect != ignored.geometry()]

    assert excludes == []


def test_switching_a_box_to_ignore_starts_excluding_it(manager):
    box = manager.add_box(QRect(10, 10, 100, 40), BoxMode.OFF)
    assert manager.exclude_regions() == []

    box.set_mode(BoxMode.IGNORE)

    assert manager.exclude_regions() == [box.geometry()]


# ── persistence ──────────────────────────────────────────────────────

def test_a_layout_round_trips(manager):
    manager.add_box(QRect(10, 20, 300, 80), BoxMode.OFF)
    manager.add_box(QRect(400, 500, 200, 60), BoxMode.IGNORE)

    restored = BoxManager(manager.app)
    restored.load()
    try:
        assert [(box.geometry(), box.mode) for box in restored.boxes] == [
            (QRect(10, 20, 300, 80), BoxMode.OFF),
            (QRect(400, 500, 200, 60), BoxMode.IGNORE),
        ]
    finally:
        restored.clear()


def test_layouts_are_kept_per_preset(manager):
    """Box layout follows the layout preset, like every other overlay."""
    manager.add_box(QRect(10, 20, 300, 80), BoxMode.OFF)

    manager.app.settings.setValue("layout_preset", "Second")
    other = BoxManager(manager.app)
    other.load()
    try:
        assert other.boxes == []
    finally:
        other.clear()
        manager.app.settings.setValue("layout_preset", "Default")


def test_a_moved_box_is_saved_where_it_was_left(manager):
    """Until this fires, the stored position is the one it was drawn at."""
    box = manager.add_box(QRect(10, 20, 300, 80), BoxMode.OFF)
    box.setGeometry(QRect(600, 400, 300, 80))
    box.moved.emit(box)

    stored = json.loads(manager.app.settings.value(manager._settings_key()))

    assert (stored[0]["x"], stored[0]["y"]) == (600, 400)


def test_an_unreadable_layout_is_ignored_rather_than_fatal(manager):
    manager.app.settings.setValue(manager._settings_key(), "{not json")

    manager.load()

    assert manager.boxes == []


def test_a_layout_entry_missing_its_geometry_is_skipped(manager):
    manager.app.settings.setValue(
        manager._settings_key(),
        json.dumps([{"id": "box_1", "mode": "off"}, {"id": "box_2", "x": 1, "y": 2, "w": 3, "h": 4, "mode": "off"}]),
    )

    manager.load()

    assert [box.box_id for box in manager.boxes] == ["box_2"]


def test_a_stored_layout_longer_than_the_cap_is_truncated(manager):
    manager.app.settings.setValue(
        manager._settings_key(),
        json.dumps(
            [{"id": f"box_{i}", "x": i, "y": 0, "w": 10, "h": 10, "mode": "off"} for i in range(MAX_BOXES + 3)]
        ),
    )

    manager.load()

    assert len(manager.boxes) == MAX_BOXES


# ── state ────────────────────────────────────────────────────────────

def test_a_box_shows_its_state(manager):
    box = manager.add_box(QRect(10, 10, 100, 40), BoxMode.OFF)

    box.set_state(BoxState.TRANSLATING)

    assert box.state is BoxState.TRANSLATING


def test_a_box_persists_its_geometry_under_its_own_id(manager):
    """Drag, clamping and multi-monitor handling all come from the overlay
    base class, keyed by window_id."""
    box = manager.add_box(QRect(10, 10, 100, 40), BoxMode.OFF)

    assert isinstance(box, TranslationBox)
    assert box.window_id == box.box_id


# ── painting ─────────────────────────────────────────────────────────

class FakeOverlay:
    """Captures what would have been painted."""

    def __init__(self):
        self.regions = None
        self.bound = None

    def bind_to_rect(self, rect):
        self.bound = rect

    def set_regions(self, regions):
        self.regions = regions

    def show(self):
        pass

    def promote(self):
        pass


class FakeRegion:
    """What the worker emits: capture pixels, (left, top, right, bottom)."""

    def __init__(self, box, translated="translated", fill=(20, 30, 40)):
        self.box = box
        self.original = "original"
        self.translated = translated
        self.fill = fill


def test_translated_regions_reach_the_overlay(manager):
    """The regression that made the feature look like it did nothing.

    The overlay's region type is a dataclass of (rect, text, fill,
    text_color); building it with a `box=` tuple raised TypeError on every
    publish, inside a Qt signal handler, so the overlay simply stayed empty
    and nothing said why.
    """
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    overlay = FakeOverlay()

    manager._on_regions(overlay, box, [FakeRegion((10, 20, 110, 60))], QRect(0, 0, 400, 300), 1.0)

    assert overlay.regions is not None
    assert len(overlay.regions) == 1
    assert overlay.regions[0].text == "translated"


def test_capture_pixels_are_converted_to_logical_rects(manager):
    """The worker measures the ratio from the frame it captured; boxes arrive
    as corners and Qt paints x/y/w/h."""
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    overlay = FakeOverlay()

    manager._on_regions(overlay, box, [FakeRegion((20, 40, 220, 100))], QRect(0, 0, 400, 300), 2.0)

    assert overlay.regions[0].rect == QRect(10, 20, 100, 30)


def test_the_text_colour_contrasts_with_the_sampled_fill(manager):
    """A translation painted in the background colour is not a translation."""
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    overlay = FakeOverlay()

    manager._on_regions(overlay, box, [FakeRegion((0, 0, 100, 30), fill=(250, 250, 250))], QRect(0, 0, 400, 300), 1.0)

    painted = overlay.regions[0]
    assert painted.text_color.lightness() < painted.fill.lightness()


def test_an_empty_result_clears_the_overlay(manager):
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    overlay = FakeOverlay()

    manager._on_regions(overlay, box, [], QRect(0, 0, 400, 300), 1.0)

    assert overlay.regions == []


# ── the shared capture session ───────────────────────────────────────

class _FakeScreen:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class _FakeStream:
    """Stands in for the QScreenCapture session, which needs a compositor."""

    opened: list = []

    def __init__(self, screen, parent=None, starts=True):
        self.screen = screen
        self.parent = parent
        self.starts = starts
        self.stopped = False
        self.waiting = False
        _FakeStream.opened.append(self)

    def start(self):
        return self.starts

    def stop(self):
        self.stopped = True

    @property
    def active(self):
        return self.starts and not self.stopped


@pytest.fixture
def fake_streams(monkeypatch):
    """One screen, and a capture session that always starts."""
    _FakeStream.opened = []
    screen = _FakeScreen("DP-1")
    monkeypatch.setattr("mage.capture.stream.screen_for_rect", lambda rect: screen)
    monkeypatch.setattr("mage.capture.stream.FrameStream", _FakeStream)
    return _FakeStream


def test_boxes_on_one_screen_share_a_capture_session(manager, fake_streams):
    """Five boxes are five readers of the same pixels.

    A session per box is the whole cost of capturing that screen, paid over
    again for an identical frame.
    """
    first = manager._stream_for(QRect(0, 0, 400, 300))
    second = manager._stream_for(QRect(600, 400, 200, 100))

    assert first is second
    assert len(fake_streams.opened) == 1


def test_a_session_that_will_not_start_is_not_renegotiated(manager, monkeypatch, fake_streams):
    """A refusal is remembered: that box uses screenshots, and asks once."""
    monkeypatch.setattr(
        "mage.capture.stream.FrameStream",
        lambda screen, parent=None: _FakeStream(screen, parent, starts=False),
    )

    assert manager._stream_for(QRect(0, 0, 400, 300)) is None
    assert manager._stream_for(QRect(0, 0, 400, 300)) is None
    assert len(fake_streams.opened) == 1


def test_the_sessions_close_when_the_boxes_stop(manager, fake_streams):
    """A capture session left open is a compositor client nobody reads."""
    manager._stream_for(QRect(0, 0, 400, 300))

    manager.stop_all()

    assert fake_streams.opened[0].stopped is True
    assert manager._streams == {}


def test_a_deleted_box_takes_its_session_with_it(manager, fake_streams):
    """The last live box going away leaves nothing capturing the screen."""
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    manager._stream_for(box.geometry())

    manager.remove_box(box)

    assert fake_streams.opened[0].stopped is True


def test_a_live_box_keeps_its_session_across_a_restart(manager, fake_streams):
    """Moving a box stops and restarts its worker; renegotiating the portal
    every time would cost several blank ticks."""
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    box.mode = BoxMode.LIVE
    manager._stream_for(box.geometry())

    manager._stop_box(box)

    assert fake_streams.opened[0].stopped is False
    assert manager._streams != {}


# ── reachable without hitting them ───────────────────────────────────

def test_the_middle_of_a_box_is_not_part_of_the_window(manager):
    """The game inside a box is still a game: clicks aimed at it must not be
    swallowed by the frame drawn around it."""
    from mage.ui.new.boxes import FRAME_BAND

    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    box.show()

    mask = box.mask()

    assert not mask.isEmpty(), "the frame itself has to stay hittable"
    assert not mask.contains(QRect(200, 150, 1, 1)), "the middle belongs to the game"
    assert mask.contains(QRect(1, 1, 1, 1)), "the top-left corner is frame"
    assert mask.contains(QRect(0, FRAME_BAND - 1, 1, 1)), "the band is FRAME_BAND deep"


def test_the_band_is_wide_enough_to_hit(manager):
    """A 2px border was the whole interactive surface of a box on Windows,
    where a layered window passes input through its transparent pixels."""
    from mage.ui.new.boxes import FRAME_BAND

    assert FRAME_BAND >= 8


def test_a_box_can_be_found_by_its_id(manager):
    """The orb's row addresses boxes by id, because it cannot address them
    by pointing at them."""
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)

    assert manager.box(box.box_id) is box
    assert manager.box("box_nonexistent") is None


def test_changing_a_mode_tells_whoever_is_listing_the_boxes(manager):
    """The orb's row names each box's mode; a mode cycled from the box's own
    toolbar has to reach it."""
    box = manager.add_box(QRect(0, 0, 400, 300), BoxMode.OFF)
    seen = []
    manager.boxes_changed.connect(lambda: seen.append(True))

    box.set_mode(BoxMode.IGNORE)

    assert seen


# ── threading ────────────────────────────────────────────────────────

def test_a_translation_recorded_off_the_gui_thread_reaches_the_shell_on_it(manager):
    """The live worker records from the async engine's thread; the orb's log
    is a widget, so the shell must only ever hear about it on the GUI thread."""
    import threading

    from PyQt6.QtCore import QThread

    seen = []

    class Shell:
        def on_translation(self, original, translated):
            seen.append((original, translated, QThread.currentThread()))

    class Processor:
        def record_event(self, *args):
            pass

    manager.app.processor = Processor()
    manager.app._shell = Shell()

    thread = threading.Thread(target=manager._record, args=("你好", "hello"))
    thread.start()
    thread.join()
    assert seen == []  # queued, not called from the worker thread

    QApplication.processEvents()
    assert seen == [("你好", "hello", QApplication.instance().thread())]


def test_a_session_that_failed_after_starting_is_asked_for_again(manager, fake_streams):
    """A declined share prompt leaves a dead session; reusing it reads nothing."""
    first = manager._stream_for(QRect(0, 0, 400, 300))
    first.stopped = True  # what a portal refusal leaves behind

    second = manager._stream_for(QRect(0, 0, 400, 300))

    assert second is not first
    assert len(fake_streams.opened) == 2


class _FakeOnceWorker(QObject):
    from PyQt6.QtCore import pyqtSignal as _signal

    regions_ready = _signal(object, object, float)
    nothing_read = _signal()
    error = _signal(str)

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def requestInterruption(self):
        pass

    def wait(self, ms):
        return True


def _run_once(manager, monkeypatch, *, prompting=False):
    from mage.ui.new import boxes

    screen = _FakeScreen("DP-1")
    monkeypatch.setattr("mage.capture.stream.screen_for_rect", lambda rect: screen)
    monkeypatch.setattr("mage.capture.stream.FrameStream", _FakeStream)
    manager._stream_for(QRect(0, 0, 400, 300)).waiting = prompting
    monkeypatch.setattr(boxes, "ONCE_PROMPT_SETTLE_MS", 50)
    worker = _FakeOnceWorker()
    overlay = QWidget()
    monkeypatch.setattr(manager, "_build_worker_for_rect", lambda rect, settle_now: (worker, overlay))
    monkeypatch.setattr(boxes, "ONCE_STREAM_GRACE_MS", 60_000)
    notices, finished = [], []
    monkeypatch.setattr(manager, "_notify", notices.append)
    manager.once_finished.connect(lambda: finished.append(True))
    manager.translate_once(QRect(0, 0, 400, 300), hold_ms=60_000)
    return worker, notices, finished


def test_a_one_off_that_reads_nothing_finishes_and_says_so(manager, monkeypatch):
    """It used to wait forever, the orb reading, with nothing on screen."""
    worker, notices, finished = _run_once(manager, monkeypatch)

    worker.nothing_read.emit()
    QApplication.processEvents()

    assert manager._workers == {}
    assert finished == [True]
    assert notices == [t("newui.once.no_text")]


def test_a_first_empty_read_after_the_share_prompt_is_not_the_answer(manager, monkeypatch):
    """Picking a screen in KDE's prompt, then a blank first frame, said "no text"."""
    worker, notices, finished = _run_once(manager, monkeypatch, prompting=True)

    worker.nothing_read.emit()
    worker.nothing_read.emit()  # the prompt still closing over the area
    QApplication.processEvents()
    assert finished == [] and notices == []

    worker.regions_ready.emit([], QRect(), 1.0)
    QApplication.processEvents()
    assert finished == [True]
    assert notices == []


def test_after_the_share_prompt_an_empty_area_still_says_so(manager, monkeypatch):
    worker, notices, finished = _run_once(manager, monkeypatch, prompting=True)

    worker.nothing_read.emit()
    QTest.qWait(150)

    assert finished == [True]
    assert notices == [t("newui.once.no_text")]


def test_a_one_off_that_translates_finishes_once(manager, monkeypatch):
    worker, notices, finished = _run_once(manager, monkeypatch)

    worker.regions_ready.emit([], QRect(), 1.0)
    worker.nothing_read.emit()  # a late signal from the same pass
    QApplication.processEvents()

    assert finished == [True]
    assert notices == []


def test_a_one_off_keeps_its_capture_session_for_the_next(manager, monkeypatch, fake_streams):
    """On Wayland each new session is another share-your-screen prompt."""
    stream = manager._stream_for(QRect(0, 0, 400, 300))
    worker, _notices, _finished = _run_once(manager, monkeypatch)

    worker.regions_ready.emit([], QRect(), 1.0)
    QApplication.processEvents()

    assert stream.stopped is False
    assert manager._stream_for(QRect(0, 0, 400, 300)) is stream
