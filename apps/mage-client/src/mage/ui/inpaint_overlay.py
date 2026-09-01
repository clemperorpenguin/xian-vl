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

"""In-place translation overlay — text replaced where it sits, not in a bubble.

Each detected line is covered with its own sampled background colour and the
translation is drawn fitted into the same box, the way Google Lens and DeepL
present camera translations.  The fill is flat, not generative: on a game HUD
the background behind a line is nearly uniform, and a flat fill is instant and
never hallucinates.

The window is click-through and never takes focus, so the game underneath keeps
receiving every mouse and key event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PyQt6.QtCore import Qt, QRect, QRectF
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from mage.utils.window_binder import (
    set_above_state_x11,
    set_bypass_compositor_hint_x11,
    set_overlay_window_type_x11,
)

logger = logging.getLogger(__name__)

__all__ = ["InpaintRegion", "InpaintOverlay"]

#: Below this relative luminance the fill counts as dark and text goes white.
LIGHT_FILL_LUMINANCE = 0.55

#: Fraction of the box height used as the starting font size.
FONT_HEIGHT_RATIO = 0.78

#: Fraction of the box width the text must fit inside.
TEXT_WIDTH_RATIO = 0.96

MIN_FONT_POINT_SIZE = 6.0


@dataclass
class InpaintRegion:
    """One translated line, positioned in the overlay's local coordinates."""

    rect: QRect
    text: str
    fill: QColor
    text_color: QColor


class InpaintOverlay(QWidget):
    """Click-through window that paints translations over the original text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._regions: list[InpaintRegion] = []

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # ── content ──────────────────────────────────────────────────────

    def set_regions(self, regions: list[InpaintRegion]) -> None:
        """Replace what is painted. An empty list clears the overlay."""
        self._regions = regions
        self.update()

    def clear_regions(self) -> None:
        self.set_regions([])

    @property
    def regions(self) -> list[InpaintRegion]:
        return list(self._regions)

    def bind_to_rect(self, rect: QRect) -> None:
        """Cover a region of the desktop, in logical (Qt global) coordinates."""
        self.setGeometry(rect)

    # ── painting ─────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._regions:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        for region in self._regions:
            if not region.text.strip() or region.rect.isEmpty():
                continue
            painter.setBrush(region.fill)
            # A slight radius reads as a deliberate label rather than a patch.
            painter.drawRoundedRect(region.rect, 3, 3)
            _draw_fitted_text(painter, region)

        painter.end()

    def showEvent(self, event):
        super().showEvent(event)
        set_bypass_compositor_hint_x11(self.winId())
        set_above_state_x11(self.winId())
        set_overlay_window_type_x11(self.winId())

    def promote(self):
        """Re-assert stacking above a fullscreen game without taking focus."""
        if not self.isVisible():
            return
        self.raise_()
        set_above_state_x11(self.winId())
        set_overlay_window_type_x11(self.winId())


def _draw_fitted_text(painter: QPainter, region: InpaintRegion) -> None:
    """Draw the translation at the largest size that fits its box."""
    rect = QRectF(region.rect)
    font = QFont(painter.font())

    size = max(MIN_FONT_POINT_SIZE, rect.height() * FONT_HEIGHT_RATIO)
    max_width = rect.width() * TEXT_WIDTH_RATIO
    font.setPixelSize(max(1, int(size)))

    metrics = QFontMetricsF(font)
    while size > MIN_FONT_POINT_SIZE and metrics.horizontalAdvance(region.text) > max_width:
        size -= 1.0
        font.setPixelSize(max(1, int(size)))
        metrics = QFontMetricsF(font)

    painter.setFont(font)
    painter.setPen(QPen(region.text_color))

    # Below a readable size, wrap inside the box instead of shrinking further:
    # two legible lines beat one unreadable one.
    if metrics.horizontalAdvance(region.text) > max_width:
        painter.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
            region.text,
        )
    else:
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, region.text)

    painter.setPen(Qt.PenStyle.NoPen)


def contrasting_text_color(fill: QColor) -> QColor:
    """Black on light fills, white on dark ones."""
    luminance = (0.299 * fill.red() + 0.587 * fill.green() + 0.114 * fill.blue()) / 255.0
    return QColor(0, 0, 0) if luminance > LIGHT_FILL_LUMINANCE else QColor(255, 255, 255)
