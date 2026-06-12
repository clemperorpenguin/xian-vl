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

"""Lightweight floating widget to display a translation result near the Lens selection."""

import logging
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QApplication
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication, QCursor
from mage.ui.theme import accent_hex
from mage.ui.overlay_base import MageOverlayWindow
from shared_types.state import t

logger = logging.getLogger(__name__)


class ResultBubble(MageOverlayWindow):
    """Semi-transparent floating panel that shows translated text.

    Positioned just below *anchor_rect* (in global screen coordinates).
    Auto-fades after *auto_close_ms* milliseconds.  Clicking the body
    copies the text to the clipboard.
    """

    continue_requested = pyqtSignal()
    speak_source_requested = pyqtSignal()
    speak_target_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    add_to_notes_requested = pyqtSignal()

    def __init__(self, text: str, original_text: str = "",
                 anchor_rect: QRect = None, auto_close_ms: int = 30000,
                 border_color: str | None = None,
                 truncated: bool = False,
                 show_stop: bool = False,
                 confidence: float | None = None,
                 show_add_note: bool = False,
                 parent=None):
        if parent is None:
            for widget in QApplication.topLevelWidgets():
                if widget.__class__.__name__ == "XianApp" or hasattr(widget, "settings"):
                    parent = widget
                    break
        super().__init__(window_id="result_bubble", app=parent, parent=parent)

        self._text = text
        self._original = original_text
        self._anchor_rect = anchor_rect
        self._orig_label = None
        self._continue_count = 0
        self.continuation_messages = None

        border = border_color if border_color else accent_hex()

        _app = parent
        _text_size = 13
        _opacity = 85
        if _app and hasattr(_app, "settings") and _app.settings:
            _text_size = int(_app.settings.value("overlay_text_size", 13))
            _opacity = int(_app.settings.value("overlay_opacity", 85))

        # --- layout -----------------------------------------------------------
        root = QWidget(self)
        root.setObjectName("BubbleRoot")
        root.setStyleSheet(f"""
            #BubbleRoot {{
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid {border};
                border-radius: 0px;
            }}
            QLabel {{
                color: #eee;
                font-size: {_text_size}px;
                padding: 2px;
            }}
            QPushButton {{
                background: transparent;
                color: #aaa;
                border: none;
                font-size: {_text_size - 2}px;
                padding: 2px 6px;
            }}
            QPushButton:hover {{ color: #fff; }}
        """)
        self.setWindowOpacity(_opacity / 100)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header row with confidence badge and close button
        header = QHBoxLayout()
        self._title_label = QLabel()
        header.addWidget(self._title_label)
        header.addStretch()

        self._conf_badge = QLabel()
        header.addWidget(self._conf_badge)

        self._apply_confidence_display(confidence, border_color)

        close_btn = QPushButton("✕")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton { background: transparent; color: #fff; border: none; font-size: 16px; font-weight: bold; padding: 0px 4px; }
            QPushButton:hover { background: #c0392b; color: #fff; }
        """)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        layout.addLayout(header)

        # Original text (collapsed by default)
        self._orig_label = QLabel(original_text)
        # Model-generated text is never trusted as markup: PlainText stops Qt
        # from interpreting stray tags (e.g. <img src="http://..."> would
        # otherwise trigger an outbound fetch when the bubble is shown).
        self._orig_label.setTextFormat(Qt.TextFormat.PlainText)
        self._orig_label.setWordWrap(True)
        self._orig_label.setStyleSheet(f"color: #bbb; font-size: {_text_size - 1}px;")
        self._orig_label.setMaximumHeight(60)
        self._orig_label.setVisible(bool(original_text))
        layout.addWidget(self._orig_label)

        # Translation
        self._trans_label = QLabel(text)
        self._trans_label.setTextFormat(Qt.TextFormat.PlainText)
        self._trans_label.setWordWrap(True)
        self._trans_label.setFont(QFont("sans-serif", _text_size))
        self._trans_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._trans_label)

        # Footer
        footer = QHBoxLayout()
        copy_btn = QPushButton(t("bubble.button.copy"))
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_to_clipboard)
        footer.addWidget(copy_btn)

        self._continue_btn = QPushButton(t("bubble.button.continue"))
        self._continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._continue_btn.setStyleSheet(
            "color: #4ecdc4; font-weight: bold; font-size: 11px;"
        )
        self._continue_btn.clicked.connect(self._on_continue_clicked)
        self._continue_btn.setVisible(truncated)
        footer.addWidget(self._continue_btn)

        self._stop_btn = QPushButton(t("bubble.button.stop"))
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setStyleSheet(
            "color: #ff5555; font-weight: bold; font-size: 11px;"
        )
        self._stop_btn.clicked.connect(self.stop_requested.emit)
        self._stop_btn.setVisible(show_stop)
        footer.addWidget(self._stop_btn)

        self._speak_src_btn = QPushButton(t("bubble.button.speak_src"))
        self._speak_src_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._speak_src_btn.clicked.connect(self._on_speak_src_clicked)
        self._speak_src_btn.setEnabled(bool(original_text))
        footer.addWidget(self._speak_src_btn)

        self._speak_tgt_btn = QPushButton(t("bubble.button.speak_tgt"))
        self._speak_tgt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._speak_tgt_btn.clicked.connect(self._on_speak_tgt_clicked)
        footer.addWidget(self._speak_tgt_btn)

        self._add_note_btn = QPushButton(t("bubble.button.add_note"))
        self._add_note_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_note_btn.clicked.connect(self.add_to_notes_requested.emit)
        self._add_note_btn.setVisible(show_add_note)
        footer.addWidget(self._add_note_btn)

        footer.addStretch()
        layout.addLayout(footer)

        # --- sizing & positioning --------------------------------------------
        self.adjustSize()
        # Clamp width
        max_w = 500
        if self.width() > max_w:
            self.setFixedWidth(max_w)
            self.adjustSize()

        preset = self.app.settings.value("layout_preset", "Default") if (self.app and hasattr(self.app, "settings") and self.app.settings) else "Default"
        key = f"layout/{preset}/result_bubble"
        has_saved = False
        if self.app and hasattr(self.app, "settings") and self.app.settings:
            has_saved = self.app.settings.contains(key)
            
        if has_saved:
            self.restore_geometry()
        elif anchor_rect and not anchor_rect.isEmpty():
            self._position_near(anchor_rect)

        # Auto-close timer
        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, self.close)

        self.show()
        self.promote()

    # ------------------------------------------------------------------
    def _position_near(self, rect: QRect):
        """Place the bubble just below *rect*, clamped to screen edges."""
        screen = QGuiApplication.screenAt(QCursor.pos())
        if not screen:
            screen = QGuiApplication.primaryScreen()
            
        if not screen:
            return
        sg = screen.availableGeometry()

        x = rect.center().x() - self.width() // 2
        y = rect.bottom() + 10

        # Clamp
        if x < sg.left():
            x = sg.left() + 8
        if x + self.width() > sg.right():
            x = sg.right() - self.width() - 8
        if y + self.height() > sg.bottom():
            y = rect.top() - self.height() - 10
        if y < sg.top():
            y = sg.top() + 8

        self.move(x, y)

    def _copy_to_clipboard(self):
        cb = QApplication.clipboard()
        if cb:
            cb.setText(self._text)
            logger.info("Translation copied to clipboard")

    def update_text(self, text: str, original_text: str = "", border_color: str | None = None, truncated: bool = False, show_stop: bool = False, confidence: float | None = None, show_add_note: bool = False) -> None:
        """Update displayed translation text in-place."""
        self._text = text
        self._trans_label.setText(text)
        
        self._original = original_text
        if self._orig_label:
            self._orig_label.setText(original_text)
            self._orig_label.setVisible(bool(original_text))
            
        if hasattr(self, "_speak_src_btn"):
            self._speak_src_btn.setEnabled(bool(original_text))
            
        if border_color:
            border = border_color
            root = self.findChild(QWidget, "BubbleRoot")
            if root:
                root.setStyleSheet(f"""
                    #BubbleRoot {{
                        background-color: rgba(20, 20, 20, 230);
                        border: 1px solid {border};
                        border-radius: 0px;
                    }}
                    QLabel {{
                        color: #eee;
                        font-size: 13px;
                        padding: 2px;
                    }}
                    QPushButton {{
                        background: transparent;
                        color: #888;
                        border: none;
                        font-size: 11px;
                        padding: 2px 6px;
                    }}
                    QPushButton:hover {{ color: #fff; }}
                """)
        else:
            border = accent_hex()
            root = self.findChild(QWidget, "BubbleRoot")
            if root:
                root.setStyleSheet(f"""
                    #BubbleRoot {{
                        background-color: rgba(20, 20, 20, 230);
                        border: 1px solid {border};
                        border-radius: 0px;
                    }}
                    QLabel {{
                        color: #eee;
                        font-size: 13px;
                        padding: 2px;
                    }}
                    QPushButton {{
                        background: transparent;
                        color: #888;
                        border: none;
                        font-size: 11px;
                        padding: 2px 6px;
                    }}
                    QPushButton:hover {{ color: #fff; }}
                """)
                
        self._apply_confidence_display(confidence, border_color)

        if hasattr(self, "_continue_btn"):
            self._continue_btn.setVisible(truncated)
            if truncated:
                self._continue_btn.setText(t("bubble.button.continue"))
                self._continue_btn.setEnabled(True)

        if hasattr(self, "_stop_btn"):
            self._stop_btn.setVisible(show_stop)

        if hasattr(self, "_add_note_btn") and show_add_note:
            self._add_note_btn.setVisible(True)
        
        self.adjustSize()
        # Clamp width
        max_w = 500
        if self.width() > max_w:
            self.setFixedWidth(max_w)
            self.adjustSize()
        preset = self.app.settings.value("layout_preset", "Default") if (self.app and hasattr(self.app, "settings") and self.app.settings) else "Default"
        key = f"layout/{preset}/result_bubble"
        has_saved = False
        if self.app and hasattr(self.app, "settings") and self.app.settings:
            has_saved = self.app.settings.contains(key)
            
        if not has_saved and self._anchor_rect and not self._anchor_rect.isEmpty():
            self._position_near(self._anchor_rect)

        self.promote()

    def _on_continue_clicked(self):
        self._continue_btn.setText(t("bubble.status.continuing"))
        self._continue_btn.setEnabled(False)
        self.continue_requested.emit()

    def show_continue_button(self, visible: bool = True) -> None:
        """Show or hide the continue button (e.g. after continuation completes)."""
        self._continue_btn.setVisible(visible)
        if visible:
            self._continue_btn.setText(t("bubble.button.continue"))
            self._continue_btn.setEnabled(True)

    def _apply_confidence_display(self, confidence: float | None, border_color: str | None):
        """Update the title (speculative vs normal) and the confidence % badge."""
        speculative = border_color is not None
        self._title_label.setText(
            t("bubble.title.speculative") if speculative else t("bubble.title.translation")
        )
        title_color = border_color if border_color else accent_hex()
        self._title_label.setStyleSheet(
            f"color: {title_color}; font-weight: bold; font-size: 11px;"
        )

        if confidence is None:
            self._conf_badge.setVisible(False)
            return

        pct = max(0, min(100, round(confidence * 100)))
        if confidence >= 0.85:
            badge_color = "#4ecdc4"
        elif confidence >= 0.70:
            badge_color = "#e5a93c"
        else:
            badge_color = "#ff5555"
        self._conf_badge.setText(f"{pct}%")
        self._conf_badge.setStyleSheet(
            f"color: {badge_color}; font-weight: bold; font-size: 11px; "
            f"border: 1px solid {badge_color}; padding: 0px 4px;"
        )
        self._conf_badge.setVisible(True)

    def mark_note_saved(self):
        """Flash the Add-to-Notes button to confirm the note was saved."""
        self._add_note_btn.setText(t("bubble.button.added"))
        self._add_note_btn.setEnabled(False)
        QTimer.singleShot(1500, self._restore_note_btn)

    def _restore_note_btn(self):
        self._add_note_btn.setText(t("bubble.button.add_note"))
        self._add_note_btn.setEnabled(True)

    def set_opacity(self, value: int):
        self.setWindowOpacity(value / 100)

    def set_text_size(self, px: int):
        self._trans_label.setFont(QFont("sans-serif", px))
        self._orig_label.setStyleSheet(f"color: #bbb; font-size: {px - 1}px;")
        self.adjustSize()

    def _on_speak_src_clicked(self):
        self.speak_source_requested.emit()

    def _on_speak_tgt_clicked(self):
        self.speak_target_requested.emit()


