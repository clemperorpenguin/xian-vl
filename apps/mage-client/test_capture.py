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
