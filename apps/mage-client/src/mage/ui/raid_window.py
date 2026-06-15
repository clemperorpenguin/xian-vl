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
import datetime
from html import escape as html_escape
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QSizePolicy, QAbstractButton
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize, QPropertyAnimation, QRectF, pyqtProperty, QTimer
from PyQt6.QtGui import QGuiApplication, QFont, QCursor, QPainter, QColor, QBrush, QPen, QRadialGradient

from mage.ui.theme import accent_hex, accent_hover_hex, accent_qcolor
from mage.ui.overlay_base import MageOverlayWindow
from shared_types.state import t

logger = logging.getLogger(__name__)


class ToggleSwitch(QAbstractButton):
    """A premium slide toggle button drawn using QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._thumb_position = 3.0
        self._anim = QPropertyAnimation(self, b"thumb_position", self)
        self._anim.setDuration(120)

    @pyqtProperty(float)
    def thumb_position(self) -> float:
        return self._thumb_position

    @thumb_position.setter
    def thumb_position(self, pos: float):
        self._thumb_position = pos
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(42, 22)

    def nextCheckState(self):
        super().nextCheckState()
        end_pos = 23.0 if self.isChecked() else 3.0
        self._anim.stop()
        self._anim.setStartValue(self._thumb_position)
        self._anim.setEndValue(end_pos)
        self._anim.start()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        self._thumb_position = 23.0 if checked else 3.0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background track
        track_color = accent_qcolor(180) if self.isChecked() else QColor("#3A3A3C")
        p.setBrush(QBrush(track_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, self.width(), self.height(), self.height() / 2, self.height() / 2)
        
        # Draw thumb
        thumb_color = QColor("#FFFFFF")
        p.setBrush(QBrush(thumb_color))
        r = self.height() - 6
        p.drawEllipse(QRectF(self._thumb_position, 3.0, r, r))


class StatusDot(QWidget):
    """A pulsing LED-like dot showing background recording/processing states."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._alpha = 255
        self._growing = False
        self._color = QColor("#757575")  # Muted grey default (idle)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_pulse)
        self._timer.start(50)

    def set_state(self, state: str):
        """Set dot color state: 'idle', 'listening', 'processing', 'error'."""
        if state == "listening":
            self._color = QColor("#00E676")  # Neon green (actively recording stream)
        elif state == "processing":
            self._color = QColor("#FFD600")  # Neon yellow/amber (transcribing/translating)
        elif state == "error":
            self._color = QColor("#FF1744")  # Red
        else:
            self._color = QColor("#757575")  # Muted grey
        self.update()

    def _update_pulse(self):
        if self._growing:
            self._alpha += 15
            if self._alpha >= 255:
                self._alpha = 255
                self._growing = False
        else:
            self._alpha -= 15
            if self._alpha <= 90:
                self._alpha = 90
                self._growing = True
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        grad = QRadialGradient(7.0, 7.0, 7.0)
        c_outer = QColor(self._color)
        c_outer.setAlpha(0)
        c_inner = QColor(self._color)
        c_inner.setAlpha(self._alpha)
        
        grad.setColorAt(0.0, c_inner)
        grad.setColorAt(0.5, c_inner)
        grad.setColorAt(1.0, c_outer)
        
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, 14, 14)


class RaidWindow(MageOverlayWindow):
    """Frameless, draggable log window for Raid Mode translations."""

    audio_toggled = pyqtSignal(bool)
    stop_requested = pyqtSignal()
    add_to_notes_requested = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(window_id="raid_window", app=parent, parent=parent)
        self.settings = settings
        self._entries: list[tuple[str, str, str]] = []

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Transparent container with border glow
        self.root_frame = QWidget()
        self.root_frame.setObjectName("RaidWindowRoot")
        self._update_style()
        layout.addWidget(self.root_frame)

        # Inner components layout
        inner_layout = QVBoxLayout(self.root_frame)
        inner_layout.setContentsMargins(16, 12, 16, 12)
        inner_layout.setSpacing(10)

        # Header Row
        header = QHBoxLayout()
        header.setSpacing(8)

        self.status_dot = StatusDot()
        header.addWidget(self.status_dot)

        self.title_label = QLabel(t("raid.window.title"))
        self.title_label.setFont(QFont("sans-serif", 11, QFont.Weight.Bold))
        self.title_label.setStyleSheet("color: #FFFFFF;")
        header.addWidget(self.title_label)

        header.addStretch()

        self.status_text = QLabel(t("raid.window.status.idle"))
        self.status_text.setFont(QFont("sans-serif", 10))
        self.status_text.setStyleSheet("color: #888888;")
        header.addWidget(self.status_text)

        close_btn = QPushButton("✕")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #888; border: none; font-size: 14px; font-weight: bold; padding: 2px 6px;
            }
            QPushButton:hover { color: #FFF; }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)

        inner_layout.addLayout(header)

        # Transcript Area (Read-only styled scroll feed)
        self.transcript_area = QTextEdit()
        self.transcript_area.setReadOnly(True)
        self.transcript_area.setPlaceholderText("Live translations will stream here...")
        self.transcript_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba(10, 10, 10, 100);
                color: #E0E0E0;
                border: 1px solid #2D2D2D;
                border-radius: 0px;
                padding: 10px;
                font-family: sans-serif;
                font-size: 13px;
            }}
            QScrollBar:vertical {{
                border: none; background: transparent; width: 6px; margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: #555; min-height: 20px; border-radius: 0px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {accent_hex()};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        inner_layout.addWidget(self.transcript_area)

        # Bottom Controls
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        # Toggle Switch label & switch
        audio_label = QLabel(t("raid.window.audio_switch"))
        audio_label.setFont(QFont("sans-serif", 10))
        audio_label.setStyleSheet("color: #B0BEC5;")
        bottom.addWidget(audio_label)

        self.audio_switch = ToggleSwitch()
        live_voice_raid = self.settings.value("live_voice_raid", "false")
        self.audio_switch.setChecked(live_voice_raid == "true" or live_voice_raid is True)
        self.audio_switch.toggled.connect(self._on_audio_toggled)
        bottom.addWidget(self.audio_switch)

        bottom.addStretch()

        # Action Buttons
        btn_style = f"""
            QPushButton {{
                background-color: #2D2D2D; color: #EEE; border: 1px solid #444;
                border-radius: 0px; padding: 4px 10px; font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {accent_hover_hex()}; border: 1px solid {accent_hex()}; }}
        """

        self.add_note_btn = QPushButton(t("raid.window.add_note"))
        self.add_note_btn.setStyleSheet(btn_style)
        self.add_note_btn.clicked.connect(self._on_add_to_notes_clicked)
        bottom.addWidget(self.add_note_btn)

        self.clear_btn = QPushButton(t("raid.window.clear"))
        self.clear_btn.setStyleSheet(btn_style)
        self.clear_btn.clicked.connect(self.clear_log)
        bottom.addWidget(self.clear_btn)

        self.stop_btn = QPushButton("🛑 Stop")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #3d1c1c; color: #ff8a8a; border: 1px solid #732a2a;
                border-radius: 0px; padding: 4px 10px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5c2424; border: 1px solid #ff5252; color: #FFF; }
        """)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        bottom.addWidget(self.stop_btn)

        inner_layout.addLayout(bottom)

        # Sizing Defaults
        self.resize(450, 320)

    def _update_style(self):
        border = accent_hex()
        self.root_frame.setStyleSheet(f"""
            #RaidWindowRoot {{
                background-color: rgba(22, 22, 26, 235);
                border: 1px solid {border};
                border-radius: 0px;
            }}
        """)

    def set_status(self, text: str, state: str = "idle"):
        """Update window status text and LED dot state."""
        self.status_text.setText(text)
        self.status_dot.set_state(state)

    def append_translation(self, original: str, translation: str):
        """Append translated phrase to the scroll feed with rich styling."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._entries.append((timestamp, original, translation))
        accent = accent_hex()
        # original/translation are untrusted (transcribed audio + model output);
        # escape them so markup like <img src="file:///..."> can't be rendered
        # by the QTextEdit. The surrounding styling is static/trusted.
        safe_original = html_escape(original)
        safe_translation = html_escape(translation)
        html = f"""
        <div style="margin-bottom: 10px; line-height: 1.35;">
            <div style="color: #78909C; font-size: 11px;">
                <span style="font-weight: bold; color: {accent};">[{timestamp}]</span> {safe_original}
            </div>
            <div style="color: #FFFFFF; font-size: 13px; font-weight: 500; margin-top: 2px; padding-left: 4px;">
                ➔ {safe_translation}
            </div>
        </div>
        """
        self.transcript_area.append(html)
        # Scroll to bottom
        scrollbar = self.transcript_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        self.transcript_area.clear()
        self._entries.clear()

    def _on_add_to_notes_clicked(self):
        if self._entries:
            self.add_to_notes_requested.emit()

    def _on_audio_toggled(self, checked: bool):
        self.settings.setValue("live_voice_raid", "true" if checked else "false")
        self.audio_toggled.emit(checked)
        logger.info("Raid Window: Translated Audio toggled to %s", checked)

    def _on_stop_clicked(self):
        self.stop_requested.emit()
        self.close()

    def set_opacity(self, value: int):
        self.setWindowOpacity(value / 100)

    def set_text_size(self, px: int):
        self.transcript_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: rgba(10, 10, 10, 100);
                color: #E0E0E0;
                border: 1px solid #2D2D2D;
                border-radius: 0px;
                padding: 10px;
                font-family: sans-serif;
                font-size: {px}px;
            }}
            QScrollBar:vertical {{
                border: none; background: transparent; width: 6px; margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: #555; min-height: 20px; border-radius: 0px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {accent_hex()};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

    # --- Window Placement & Geometry Persistence ------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        self._update_style()  # pick up runtime accent color changes
        
        # Restore saved location
        preset = self.app.settings.value("layout_preset", "Default")
        key = f"layout/{preset}/raid_window"
        if not self.app.settings.contains(key):
            screen = QGuiApplication.screenAt(QCursor.pos())
            if not screen:
                screen = QGuiApplication.primaryScreen()
            if screen:
                sg = screen.availableGeometry()
                # Default to bottom right area
                self.setGeometry(sg.right() - 480, sg.bottom() - 360, 450, 320)

    def closeEvent(self, event):
        self.save_geometry()
        super().closeEvent(event)
