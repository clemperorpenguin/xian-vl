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

"""Region capture geometry and the Qt/PIL buffer hand-off.

The screenshot subprocesses are mocked, so these run headless in CI.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6.QtCore import QBuffer, QIODevice, QRect  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402

from mage.capture.screen import ScreenCapture  # noqa: E402
from mage.utils.images import qimage_to_pil  # noqa: E402


def _png_bytes(width: int, height: int, rgb=(200, 30, 30)) -> bytes:
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(int(0xFF000000 | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.buffer())


class _Completed:
    def __init__(self, stdout: bytes, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


def test_capture_region_passes_grim_geometry():
    rect = QRect(120, 45, 640, 200)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _Completed(_png_bytes(640, 200))

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
            patch("mage.capture.screen.sys.platform", "linux"), \
            patch("mage.capture.screen.subprocess.run", fake_run):
        data, already_cropped = ScreenCapture.capture_region(rect)

    assert already_cropped is True
    assert captured["cmd"][:3] == ["grim", "-g", "120,45 640x200"]
    assert QImage.fromData(data).size().width() == 640


def test_capture_region_falls_back_to_full_desktop_when_grim_missing():
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("grim")

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
            patch("mage.capture.screen.sys.platform", "linux"), \
            patch("mage.capture.screen.subprocess.run", fake_run), \
            patch.object(ScreenCapture, "capture_screen", return_value=_png_bytes(1920, 1080)):
        data, already_cropped = ScreenCapture.capture_region(QRect(0, 0, 100, 100))

    assert already_cropped is False, "caller still has to crop the full-desktop frame"
    assert QImage.fromData(data).size().width() == 1920


def test_capture_region_rejects_empty_rect():
    assert ScreenCapture.capture_region(QRect()) == (None, False)


def test_an_all_black_region_grab_is_kept():
    """A dark HUD panel is legitimately black. Rejecting it would drop every
    frame back to a full-desktop capture and lose the fast path."""
    def fake_run(cmd, **kwargs):
        return _Completed(_png_bytes(320, 80, (0, 0, 0)))

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
            patch("mage.capture.screen.sys.platform", "linux"), \
            patch("mage.capture.screen.subprocess.run", fake_run):
        data, already_cropped = ScreenCapture.capture_region(QRect(0, 0, 320, 80))

    assert already_cropped is True
    assert data is not None


def test_a_failed_grim_run_still_falls_back():
    """grim reports failure with a non-zero exit code, not a black frame."""
    def fake_run(cmd, **kwargs):
        return _Completed(b"", returncode=1)

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
            patch("mage.capture.screen.sys.platform", "linux"), \
            patch("mage.capture.screen.subprocess.run", fake_run), \
            patch.object(ScreenCapture, "capture_screen", return_value=_png_bytes(800, 600)):
        _data, already_cropped = ScreenCapture.capture_region(QRect(0, 0, 320, 80))

    assert already_cropped is False


@pytest.mark.parametrize("size", [(3, 3), (1, 40), (200, 120)])
def test_small_regions_are_not_misjudged_as_empty(size):
    """Region grabs can be narrower than the 5x5 sample grid."""
    assert ScreenCapture._is_image_empty(_png_bytes(*size)) is False


def test_all_black_capture_is_still_rejected():
    assert ScreenCapture._is_image_empty(_png_bytes(64, 64, (0, 0, 0))) is True


@pytest.mark.parametrize("width", [16, 17, 19])
def test_qimage_to_pil_handles_scanline_padding(width):
    """Qt pads scanlines to 4 bytes; odd widths must not shear the image."""
    source = QImage(width, 5, QImage.Format.Format_RGB32)
    source.fill(int(0xFF3C64C8))

    pil = qimage_to_pil(source)

    assert pil.size == (width, 5)
    assert pil.mode == "RGB"
    assert pil.getpixel((width - 1, 4)) == (0x3C, 0x64, 0xC8)


# ── pixels, not PNG ──────────────────────────────────────────────────

def test_the_region_image_path_skips_the_png_round_trip():
    """The live loop wants pixels, and pays three codec passes to get them
    through the encoded path — per region, per tick."""
    encoded = {"calls": 0}

    def fake_capture_screen():
        encoded["calls"] += 1
        return _png_bytes(1920, 1080)

    with patch.object(ScreenCapture, "_capture_pyqt_region", return_value=None), \
            patch.object(ScreenCapture, "_capture_pyqt_image", return_value=QImage(1920, 1080, QImage.Format.Format_RGB32)), \
            patch.object(ScreenCapture, "capture_screen", fake_capture_screen):
        image, already_cropped = ScreenCapture.capture_region_image(QRect(0, 0, 100, 100))

    assert already_cropped is False
    assert image.width() == 1920
    assert encoded["calls"] == 0, "nothing on this path should encode a PNG"


def test_a_region_on_one_screen_is_grabbed_without_the_whole_desktop():
    """The expensive part of a tick is compositing a desktop to keep a crop
    of it; QScreen.grabWindow takes the rectangle directly."""
    asked = {}

    class _FakePixmap:
        """A QPixmap needs a QGuiApplication; only these two calls are made."""

        def __init__(self, width, height):
            self._image = QImage(width, height, QImage.Format.Format_RGB32)

        def isNull(self):
            return False

        def toImage(self):
            return self._image

    class _FakeScreen:
        @staticmethod
        def geometry():
            return QRect(0, 0, 1920, 1080)

        @staticmethod
        def grabWindow(window, x, y, width, height):
            asked["args"] = (window, x, y, width, height)
            return _FakePixmap(width, height)

    with patch("mage.capture.screen.QGuiApplication.screens", return_value=[_FakeScreen()]), \
            patch.object(ScreenCapture, "_capture_pyqt_image", side_effect=AssertionError("composited anyway")):
        image, already_cropped = ScreenCapture.capture_region_image(QRect(120, 45, 640, 200))

    assert already_cropped is True
    assert asked["args"] == (0, 120, 45, 640, 200)
    assert image.width() == 640


def test_a_region_grab_is_offset_by_its_own_screen():
    """grabWindow's offset is relative to the screen, not the desktop."""
    asked = {}

    class _FakeScreen:
        def __init__(self, geometry):
            self._geometry = geometry

        def geometry(self):
            return self._geometry

        def grabWindow(self, window, x, y, width, height):
            asked["args"] = (x, y)
            image = QImage(width, height, QImage.Format.Format_RGB32)
            return type("_P", (), {"isNull": lambda self: False, "toImage": lambda self: image})()

    screens = [_FakeScreen(QRect(0, 0, 1920, 1080)), _FakeScreen(QRect(1920, 0, 1920, 1080))]
    with patch("mage.capture.screen.QGuiApplication.screens", return_value=screens):
        ScreenCapture.capture_region_image(QRect(2020, 45, 640, 200))

    assert asked["args"] == (100, 45)


def test_a_region_spanning_two_screens_falls_back_to_the_desktop():
    class _FakeScreen:
        def __init__(self, geometry):
            self._geometry = geometry

        def geometry(self):
            return self._geometry

        def grabWindow(self, *args):
            raise AssertionError("no single screen holds this region")

    screens = [_FakeScreen(QRect(0, 0, 1920, 1080)), _FakeScreen(QRect(1920, 0, 1920, 1080))]
    with patch("mage.capture.screen.QGuiApplication.screens", return_value=screens), \
            patch.object(ScreenCapture, "_capture_pyqt_image", return_value=QImage(3840, 1080, QImage.Format.Format_RGB32)):
        image, already_cropped = ScreenCapture.capture_region_image(QRect(1800, 45, 640, 200))

    assert already_cropped is False
    assert image.width() == 3840


def test_the_region_image_path_still_prefers_grim():
    def fake_run(cmd, **kwargs):
        return _Completed(_png_bytes(640, 200))

    with patch.dict(os.environ, {"XDG_SESSION_TYPE": "wayland"}), \
            patch("mage.capture.screen.sys.platform", "linux"), \
            patch("mage.capture.screen.subprocess.run", fake_run):
        image, already_cropped = ScreenCapture.capture_region_image(QRect(120, 45, 640, 200))

    assert already_cropped is True
    assert image.width() == 640


def test_the_region_image_path_rejects_an_empty_rect():
    assert ScreenCapture.capture_region_image(QRect()) == (None, False)


def _scaled_desktop_png(logical: QRect, scale: float, mark: QRect) -> bytes:
    """A desktop screenshot in device pixels, as Spectacle returns one on a
    scaled display, with ``mark`` (logical coordinates) painted green."""
    from PyQt6.QtGui import QColor

    image = QImage(round(logical.width() * scale), round(logical.height() * scale), QImage.Format.Format_RGB32)
    image.fill(QColor(200, 30, 30))
    for y in range(round(mark.top() * scale), round((mark.bottom() + 1) * scale)):
        for x in range(round(mark.left() * scale), round((mark.right() + 1) * scale)):
            image.setPixelColor(x, y, QColor(30, 200, 30))
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(buffer.buffer())


def test_a_dialogue_crop_on_a_scaled_desktop_maps_to_device_pixels():
    from mage.workers import CaptureWorker

    desktop = QRect(0, 0, 1800, 1125)
    mark = QRect(300, 200, 100, 50)
    png = _scaled_desktop_png(desktop, 1.6, mark)

    worker = CaptureWorker("dialogue", [mark], desktop)
    results = []
    worker.capture_done.connect(lambda img, rect: results.append(img))
    worker.error.connect(lambda msg: pytest.fail(msg))
    with patch.object(ScreenCapture, "capture_region", return_value=(png, False)):
        worker.run()

    (img,) = results
    assert img.size == (160, 80)
    colours = img.getcolors()
    assert colours == [(160 * 80, (30, 200, 30))]
