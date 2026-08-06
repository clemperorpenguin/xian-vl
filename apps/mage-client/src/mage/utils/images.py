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

"""Conversions between Qt image buffers and PIL.

Capture backends hand back a QImage; the vision pipeline wants a PIL image.
Going through an encoded intermediate (PNG or JPEG) costs an encode plus a
decode per frame, which is dead weight in continuous modes — these helpers
copy the pixel buffer across directly instead.
"""

from __future__ import annotations

from PIL import Image
from PyQt6.QtGui import QImage

__all__ = ["qimage_to_pil"]


def qimage_to_pil(image: QImage) -> Image.Image:
    """Convert a QImage to a PIL RGB image without an encode round trip."""
    if image.isNull():
        raise ValueError("Cannot convert a null QImage")

    converted = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = converted.width(), converted.height()

    ptr = converted.constBits()
    ptr.setsize(converted.sizeInBytes())

    # Qt pads each scanline to a 4-byte boundary, so hand the stride to PIL
    # rather than assuming the rows are contiguous.
    return Image.frombuffer(
        "RGB", (width, height), bytes(ptr), "raw", "RGB", converted.bytesPerLine(), 1
    )
