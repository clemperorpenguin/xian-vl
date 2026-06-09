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
from typing import ClassVar

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QRect, QPoint, QBuffer, QIODevice, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QMouseEvent, QPixmap, QImage, QGuiApplication
from mage.capture.screen import ScreenCapture
from mage.ui.theme import accent_hex, accent_hover_hex, accent_qcolor
from mage.utils.window_binder import set_bypass_compositor_hint_x11
from shared_types.state import t

logger = logging.getLogger(__name__)

class ActionBarWidget(QWidget):
    action_triggered = pyqtSignal(str, QRect) # action name, selected rect
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(30, 30, 30, 220);
                border-radius: 8px;
                border: 1px solid {accent_hex()};
            }}
            QPushButton {{
                background-color: #333;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {accent_hex()};
            }}
        """)
        
        btn_translate = QPushButton(t("lens.button.translate"))
        btn_translate.clicked.connect(lambda: self._on_click("translate"))
        
        btn_dialogue = QPushButton(t("lens.button.dialogue"))
        btn_dialogue.clicked.connect(lambda: self._on_click("dialogue"))
        
        btn_chat = QPushButton(t("lens.button.chat_context"))
        btn_chat.clicked.connect(lambda: self._on_click("chat"))
        
        layout.addWidget(btn_translate)
        layout.addWidget(btn_dialogue)
        layout.addWidget(btn_chat)
        
        self.selected_rect = QRect()
        
    def _on_click(self, action: str):
        self.action_triggered.emit(action, self.selected_rect)

class LensOverlayWindow(QWidget):
    closed = pyqtSignal()
    action_requested = pyqtSignal(str, QRect, bytes) # action, rect, cropped_image

    # Persists across instances so the next overlay can recall it
    _last_rect: ClassVar[QRect | None] = None
    
    def __init__(self, previous_rect: QRect | None = None):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Capture screen
        self.full_image_data = ScreenCapture.capture_screen()
        if self.full_image_data:
            img = QImage.fromData(self.full_image_data)
            self.pixmap = QPixmap.fromImage(img)
        else:
            self.pixmap = QPixmap()
            
        self.total_geo = ScreenCapture.get_virtual_desktop_geometry()
        self.setGeometry(self.total_geo)
        
        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.selecting = False
        self.selected_rect = QRect()
        
        self.action_bar = ActionBarWidget(self)
        self.action_bar.hide()
        self.action_bar.action_triggered.connect(self._handle_action)

        # Pre-populate with previous selection if provided
        self._previous_rect = previous_rect
        
    def showFullScreen(self):
        super().showFullScreen()
        # After the window is shown, apply the previous selection if set
        if self._previous_rect is not None and not self._previous_rect.isEmpty():
            self.selected_rect = QRect(self._previous_rect)
            self.start_pos = self.selected_rect.topLeft()
            self.current_pos = self.selected_rect.bottomRight()
            self._show_action_bar()
            self.update()

    def _handle_action(self, action: str, rect: QRect):
        logger.info("Lens action requested: %s on %s", action, rect)
        # Remember this selection for next time
        LensOverlayWindow._last_rect = QRect(rect)
        
        # Crop the image
        if not self.pixmap.isNull():
            # Ensure rect is within bounds
            safe_rect = rect.translated(-self.total_geo.left(), -self.total_geo.top())
            safe_rect = safe_rect.intersected(self.pixmap.rect())
            
            cropped_pixmap = self.pixmap.copy(safe_rect)
            
            # Save to bytes
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            cropped_pixmap.save(buffer, "PNG")
            cropped_data = bytes(buffer.buffer())
            
            self.action_requested.emit(action, rect, cropped_data)
        self.close()
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
            self.selecting = True
            self.action_bar.hide()
            self.selected_rect = QRect()
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.close()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.selecting:
            self.current_pos = event.globalPosition().toPoint()
            self.update()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False
            self.current_pos = event.globalPosition().toPoint()
            self.selected_rect = QRect(self.start_pos, self.current_pos).normalized()
            
            if self.selected_rect.width() > 10 and self.selected_rect.height() > 10:
                self._show_action_bar()
            else:
                self.selected_rect = QRect()
            self.update()
            
    def _show_action_bar(self):
        self.action_bar.selected_rect = self.selected_rect
        self.action_bar.adjustSize()
        
        # Position below the rect
        bar_x = self.selected_rect.center().x() - (self.action_bar.width() // 2)
        bar_y = self.selected_rect.bottom() + 10
        
        # Clamp to screen
        if bar_x < self.total_geo.left(): bar_x = self.total_geo.left()
        if bar_x + self.action_bar.width() > self.total_geo.right():
            bar_x = self.total_geo.right() - self.action_bar.width()
            
        if bar_y + self.action_bar.height() > self.total_geo.bottom():
            bar_y = self.selected_rect.top() - self.action_bar.height() - 10
            
        self.action_bar.move(bar_x - self.total_geo.left(), bar_y - self.total_geo.top())
        self.action_bar.show()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)
            
        # Dimming overlay
        dim_color = QColor(0, 0, 0, 150)
        
        if self.selecting or not self.selected_rect.isEmpty():
            rect = QRect(self.start_pos, self.current_pos).normalized() if self.selecting else self.selected_rect
            # Translate to window coordinates
            rect.translate(-self.total_geo.left(), -self.total_geo.top())
            
            # Draw 4 rects around the selection to dim the rest
            win_rect = self.rect()
            painter.fillRect(0, 0, win_rect.width(), rect.top(), dim_color)
            painter.fillRect(0, rect.bottom() + 1, win_rect.width(), win_rect.height() - rect.bottom() - 1, dim_color)
            painter.fillRect(0, rect.top(), rect.left(), rect.height() + 1, dim_color)
            painter.fillRect(rect.right() + 1, rect.top(), win_rect.width() - rect.right() - 1, rect.height() + 1, dim_color)
            
            # Draw border around selection
            painter.setPen(QPen(accent_qcolor(), 2))
            painter.drawRect(rect)
        else:
            painter.fillRect(self.rect(), dim_color)
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            
    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        set_bypass_compositor_hint_x11(self.winId())

class CinematicLensOverlay(QWidget):
    """Overlay for selecting multiple regions for Cinematic Mode."""
    closed = pyqtSignal()
    confirmed = pyqtSignal(list) # list of QRect

    _last_rects: ClassVar[list[QRect]] = []
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.full_image_data = ScreenCapture.capture_screen()
        if self.full_image_data:
            img = QImage.fromData(self.full_image_data)
            self.pixmap = QPixmap.fromImage(img)
        else:
            self.pixmap = QPixmap()
            
        self.total_geo = ScreenCapture.get_virtual_desktop_geometry()
        self.setGeometry(self.total_geo)
        
        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.selecting = False
        
        # Load previous rects
        self.rects = [QRect(r) for r in CinematicLensOverlay._last_rects]
        
        # Action Bar (Confirm / Clear)
        self.action_bar = QWidget(self)
        self.action_bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QHBoxLayout(self.action_bar)
        self.action_bar.setStyleSheet(f"""
            QWidget {{ background-color: rgba(30, 30, 30, 220); border-radius: 8px; border: 1px solid {accent_hex()}; }}
            QPushButton {{ background-color: #333; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {accent_hex()}; }}
        """)
        btn_confirm = QPushButton(t("lens.button.confirm"))
        btn_confirm.clicked.connect(self._on_confirm)
        btn_clear = QPushButton(t("lens.button.clear_all"))
        btn_clear.clicked.connect(self._on_clear)
        layout.addWidget(btn_confirm)
        layout.addWidget(btn_clear)
        
        self.action_bar.adjustSize()
        # Position at top center
        self.action_bar.move((self.total_geo.width() - self.action_bar.width()) // 2, 20)

    def _on_confirm(self):
        CinematicLensOverlay._last_rects = [QRect(r) for r in self.rects]
        self.confirmed.emit(self.rects)
        self.close()

    def _on_clear(self):
        self.rects.clear()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if clicking on action bar
            if self.action_bar.geometry().contains(event.pos()):
                super().mousePressEvent(event)
                return
            self.start_pos = event.globalPosition().toPoint()
            self.selecting = True
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.close()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.selecting:
            self.current_pos = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.selecting:
            self.selecting = False
            self.current_pos = event.globalPosition().toPoint()
            rect = QRect(self.start_pos, self.current_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.rects.append(rect)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)
            
        dim_color = QColor(0, 0, 0, 150)
        painter.fillRect(self.rect(), dim_color)
        
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        for r in self.rects:
            rect = QRect(r)
            rect.translate(-self.total_geo.left(), -self.total_geo.top())
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            
        if self.selecting:
            rect = QRect(self.start_pos, self.current_pos).normalized()
            rect.translate(-self.total_geo.left(), -self.total_geo.top())
            painter.fillRect(rect, Qt.GlobalColor.transparent)
            
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(accent_qcolor(), 2))
        for r in self.rects:
            rect = QRect(r)
            rect.translate(-self.total_geo.left(), -self.total_geo.top())
            painter.drawRect(rect)
            
        if self.selecting:
            rect = QRect(self.start_pos, self.current_pos).normalized()
            rect.translate(-self.total_geo.left(), -self.total_geo.top())
            painter.drawRect(rect)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self._on_confirm()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        set_bypass_compositor_hint_x11(self.winId())
