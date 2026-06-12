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
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen
from mage.utils.window_binder import set_bypass_compositor_hint_x11, set_above_state_x11

logger = logging.getLogger(__name__)


class MageOverlayWindow(QWidget):
    """Base class for all draggable, always-on-top, and layout-persistent overlay windows."""

    # Global "hide all overlays" mode (driven by the toggle gesture / tray).
    # While active, any overlay that tries to become visible is suppressed and
    # queued here instead, then restored when the mode is turned off. This is
    # the single chokepoint that makes a translation finishing *while hidden*
    # queue up and reappear on un-hide, rather than popping over the game.
    _overlays_suppressed = False
    _suppressed_windows: list = []

    @classmethod
    def hide_all_overlays(cls, overlays):
        """Enter hide-all mode and hide every currently-visible overlay.

        Hidden windows are queued for restoration; any overlay shown later
        (e.g. a new result bubble) is captured by ``setVisible`` while the mode
        is active and joins the same queue.
        """
        cls._overlays_suppressed = True
        for w in overlays:
            if w.isVisible():
                if w not in cls._suppressed_windows:
                    cls._suppressed_windows.append(w)
                # super(): bypass our own gate so it actually hides.
                QWidget.setVisible(w, False)

    @classmethod
    def show_all_overlays(cls):
        """Leave hide-all mode and restore every queued overlay."""
        cls._overlays_suppressed = False
        queued = cls._suppressed_windows
        cls._suppressed_windows = []
        for w in queued:
            try:
                from PyQt6 import sip
                if sip.isdeleted(w):
                    continue
            except Exception:
                pass
            QWidget.setVisible(w, True)
            if hasattr(w, "promote"):
                w.promote()

    def setVisible(self, visible):
        # When the global hide-all mode is active, swallow show requests and
        # remember the window so it can be restored on un-hide.
        if visible and MageOverlayWindow._overlays_suppressed:
            if self not in MageOverlayWindow._suppressed_windows:
                MageOverlayWindow._suppressed_windows.append(self)
            QWidget.setVisible(self, False)
            return
        super().setVisible(visible)

    def __init__(self, window_id: str, app=None, parent=None):
        super().__init__(parent)
        self.window_id = window_id
        self.app = app or parent
        self._drag_position = QPoint()
        self._is_dragging = False
        self._system_moving = False
        self._click_through = False
        self._edit_mode_active = False

        # Apply default overlay flags
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Restore saved coordinates if available
        self.restore_geometry()

    def restore_geometry(self):
        """Restores the saved coordinates for this window based on the active layout preset."""
        if not hasattr(self.app, "settings") or not self.app.settings:
            return
        preset = self.app.settings.value("layout_preset", "Default")
        key = f"layout/{preset}/{self.window_id}"
        geo_val = self.app.settings.value(key)
        if geo_val:
            try:
                if isinstance(geo_val, QRect):
                    self.setGeometry(geo_val)
                elif isinstance(geo_val, (list, tuple)) and len(geo_val) == 4:
                    self.setGeometry(QRect(int(geo_val[0]), int(geo_val[1]), int(geo_val[2]), int(geo_val[3])))
            except Exception as e:
                logger.error("Failed to restore geometry for %s under preset %s: %s", self.window_id, preset, e)

    def save_geometry(self):
        """Saves the current coordinates of this window to settings under the active layout preset."""
        if not hasattr(self.app, "settings") or not self.app.settings:
            return
        preset = self.app.settings.value("layout_preset", "Default")
        key = f"layout/{preset}/{self.window_id}"
        geo = self.geometry()
        self.app.settings.setValue(key, [geo.x(), geo.y(), geo.width(), geo.height()])

    def set_click_through(self, click_through: bool):
        """Toggles window click-through state (WA_TransparentForMouseEvents)."""
        self._click_through = click_through
        # If in edit mode, we must accept input so we remain draggable
        if not self._edit_mode_active:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, click_through)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def set_edit_mode(self, active: bool):
        """Toggles the UI Layout Edit Mode for this window."""
        self._edit_mode_active = active
        if active:
            # Disable click-through in edit mode so we can be dragged
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            # Remove focus avoidance flags during edit mode so it interacts better
            flags = self.windowFlags() & ~Qt.WindowType.WindowDoesNotAcceptFocus
            self.setWindowFlags(flags)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self._click_through)
            flags = self.windowFlags() | Qt.WindowType.WindowDoesNotAcceptFocus
            self.setWindowFlags(flags)

        if self.isVisible():
            self.show()
        self.update()  # Repaint to show/hide dashed edit border

    def promote(self):
        """Force this overlay above the active window without taking focus.

        Re-asserts the stays-on-top flag (some WMs drop it), restacks via
        raise_(), and re-applies the EWMH _NET_WM_STATE_ABOVE hint. Never
        activates the window, so focus stays with the game.
        """
        if not self.isVisible():
            return
        if not (self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint):
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
            # WA_ShowWithoutActivating keeps this from stealing focus on re-show.
            self.show()
        self.raise_()
        set_above_state_x11(self.winId())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and (self._edit_mode_active or not self._click_through):
            handle = self.windowHandle()
            if handle and handle.startSystemMove():
                self._system_moving = True
                event.accept()
                return
            # Fallback for platforms where startSystemMove() is unavailable
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._is_dragging = True
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() == Qt.MouseButton.LeftButton:
            new_pos = event.globalPosition().toPoint() - self._drag_position
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._system_moving:
                self._system_moving = False
                self.save_geometry()
                event.accept()
                return
            if self._is_dragging:
                self._is_dragging = False
                self.save_geometry()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.save_geometry()

    def paintEvent(self, event):
        super().paintEvent(event)
        # In edit mode, draw a custom dashed highlight border on top of the widget
        if self._edit_mode_active:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Teal/Accent highlights
            pen = QPen(QColor(78, 205, 196), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
            painter.fillRect(self.rect(), QColor(78, 205, 196, 30))

    def showEvent(self, event):
        super().showEvent(event)
        set_bypass_compositor_hint_x11(self.winId())
        set_above_state_x11(self.winId())
