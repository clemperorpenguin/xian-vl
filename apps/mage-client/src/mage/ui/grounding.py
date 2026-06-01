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

import logging
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QPen

logger = logging.getLogger(__name__)

class GroundingHighlight(QWidget):
    """Draws a highly visible, glowing bounding box over the target element."""
    def __init__(self, rect: QRect, duration_ms: int = 3000):
        super().__init__()
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Add some padding for the glow effect
        padding = 20
        self.setGeometry(rect.adjusted(-padding, -padding, padding, padding))
        self.inner_rect = QRect(padding, padding, rect.width(), rect.height())
        
        # Animation states
        self.opacity = 1.0
        
        # Calculate fade step: we want opacity to reach ~0 right as duration_ms expires.
        # Timer fires every 50ms, so total ticks = duration_ms / 50.
        tick_interval_ms = 50
        total_ticks = max(1, duration_ms // tick_interval_ms)
        self._fade_step = 1.0 / total_ticks
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._fade_out)
        self.timer.start(tick_interval_ms)
        
        # Will self-destruct after duration_ms
        QTimer.singleShot(duration_ms, self.close)
        
        self.show()
        
    def _fade_out(self):
        self.opacity = max(0.0, self.opacity - self._fade_step)
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw glow
        glow_color = QColor(255, 50, 50, int(150 * self.opacity))
        for i in range(1, 6):
            pen = QPen(glow_color, i * 2)
            painter.setPen(pen)
            painter.drawRect(self.inner_rect.adjusted(-i, -i, i, i))
            
        # Draw solid border
        solid_color = QColor(255, 0, 0, int(255 * self.opacity))
        painter.setPen(QPen(solid_color, 3))
        painter.drawRect(self.inner_rect)
