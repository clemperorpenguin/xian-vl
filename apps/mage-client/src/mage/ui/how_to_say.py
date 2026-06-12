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
import os
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QTextEdit, QHBoxLayout, QPushButton, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QGuiApplication, QFont, QKeyEvent, QCursor

from mage.ui.theme import accent_hex, accent_hover_hex
from mage.ui.overlay_base import MageOverlayWindow
from shared_types.state import t

logger = logging.getLogger(__name__)

class HowToSayDialog(MageOverlayWindow):
    """An interactive dialog for translating text, modifying it, and saving it."""
    
    translation_requested = pyqtSignal(str)
    dialog_hidden = pyqtSignal()

    def __init__(self, app, parent=None):
        super().__init__(window_id="how_to_say", app=app, parent=parent)
        
        self.target_lang = "English"
        self.source_lang = "Chinese"
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.bg_frame = QFrame()
        self.bg_frame.setObjectName("HowToSayBg")
        self.bg_frame.setStyleSheet(f"""
            #HowToSayBg {{
                background-color: rgba(20, 20, 20, 240);
                border: 1px solid {accent_hex()};
                border-radius: 0px;
            }}
        """)
        layout.addWidget(self.bg_frame)

        inner_layout = QVBoxLayout(self.bg_frame)
        inner_layout.setContentsMargins(24, 20, 24, 20)
        inner_layout.setSpacing(12)

        # Title
        self.title_label = QLabel(t("chat.title.translate"))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("sans-serif", 12, QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {accent_hex()};")
        inner_layout.addWidget(self.title_label)

        # Input field
        self.input_label = QLabel(t("chat.label.original"))
        self.input_label.setStyleSheet("color: #888; font-size: 11px;")
        inner_layout.addWidget(self.input_label)
        
        self.input_field = QTextEdit()
        self.input_field.setMinimumWidth(400)
        self.input_field.setMaximumHeight(80)
        self.input_field.setPlaceholderText(t("chat.placeholder.type_message"))
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #444;
                border-radius: 0px;
                padding: 6px;
                font-size: 13px;
            }}
            QTextEdit:focus {{ border: 1px solid {accent_hex()}; }}
        """)
        inner_layout.addWidget(self.input_field)

        # Output field
        self.output_label = QLabel(t("chat.label.translation"))
        self.output_label.setStyleSheet("color: #888; font-size: 11px;")
        inner_layout.addWidget(self.output_label)
        
        self.output_field = QTextEdit()
        self.output_field.setMinimumWidth(400)
        self.output_field.setMaximumHeight(80)
        self.output_field.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #444;
                border-radius: 0px;
                padding: 6px;
                font-size: 14px;
            }}
            QTextEdit:focus {{ border: 1px solid {accent_hex()}; }}
        """)
        inner_layout.addWidget(self.output_field)

        # Status Label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #4ecdc4; font-size: 11px;")
        inner_layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        btn_style = f"""
            QPushButton {{
                background-color: #333; color: #EEE; border: 1px solid #555;
                border-radius: 0px; padding: 6px 12px; font-size: 12px;
            }}
            QPushButton:hover {{ background-color: {accent_hover_hex()}; border: 1px solid {accent_hex()}; }}
            QPushButton:disabled {{ background-color: #222; color: #555; border: 1px solid #333; }}
        """

        self.btn_translate = QPushButton(t("chat.button.translate"))
        self.btn_translate.setStyleSheet(btn_style)
        self.btn_translate.clicked.connect(self._on_translate)
        
        self.btn_regenerate = QPushButton(t("chat.button.regenerate"))
        self.btn_regenerate.setStyleSheet(btn_style)
        self.btn_regenerate.clicked.connect(self._on_translate)
        self.btn_regenerate.setEnabled(False)
        
        self.btn_copy = QPushButton(t("chat.button.copy"))
        self.btn_copy.setStyleSheet(btn_style)
        self.btn_copy.clicked.connect(self._on_copy)
        self.btn_copy.setEnabled(False)
        
        self.btn_save = QPushButton(t("chat.button.save"))
        self.btn_save.setStyleSheet(btn_style)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save.setEnabled(False)

        btn_layout.addWidget(self.btn_translate)
        btn_layout.addWidget(self.btn_regenerate)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_copy)

        inner_layout.addLayout(btn_layout)

    def _on_translate(self):
        text = self.input_field.toPlainText().strip()
        if text:
            self.status_label.setText(t("chat.status.translating"))
            self.status_label.setStyleSheet("color: #4ecdc4;")
            self.btn_translate.setEnabled(False)
            self.btn_regenerate.setEnabled(False)
            self.btn_copy.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.output_field.clear()
            self.translation_requested.emit(text)

    def _on_copy(self):
        text = self.output_field.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.status_label.setText(t("chat.status.copied"))
            self.status_label.setStyleSheet("color: #4ecdc4;")

    def _on_save(self):
        original = self.input_field.toPlainText().strip()
        translated = self.output_field.toPlainText().strip()
        if original and translated:
            try:
                import pathlib
                save_dir = pathlib.Path.home() / ".local" / "share" / "xian-vl"
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / "saved_translations.txt"
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(f"Original ({self.target_lang}): {original}\nTranslation ({self.source_lang}): {translated}\n---\n")
                self.status_label.setText(t("chat.status.saved").format(path=save_path))
                self.status_label.setStyleSheet("color: #4ecdc4;")
            except Exception as e:
                self.status_label.setText(t("chat.status.error").format(error=e))
                self.status_label.setStyleSheet("color: #e74c3c;")

    def set_opacity(self, value: int):
        self.setWindowOpacity(value / 100)

    def set_text_size(self, px: int):
        self.input_field.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A2A; color: #FFFFFF;
                border: 1px solid #444; border-radius: 0px;
                padding: 6px; font-size: {px}px;
            }}
            QTextEdit:focus {{ border: 1px solid {accent_hex()}; }}
        """)
        self.output_field.setStyleSheet(f"""
            QTextEdit {{
                background-color: #2A2A2A; color: #FFFFFF;
                border: 1px solid #444; border-radius: 0px;
                padding: 6px; font-size: {px}px;
            }}
            QTextEdit:focus {{ border: 1px solid {accent_hex()}; }}
        """)

    def set_result(self, translated_text: str):
        self.output_field.setPlainText(translated_text)
        self.status_label.setText(t("chat.status.complete"))
        self.status_label.setStyleSheet("color: #4ecdc4;")
        self._on_copy() # Auto-copy by default
        self.btn_translate.setEnabled(True)
        self.btn_regenerate.setEnabled(True)
        self.btn_copy.setEnabled(True)
        self.btn_save.setEnabled(True)

    def set_error(self, error_msg: str):
        self.status_label.setText(t("chat.status.error").format(error=error_msg))
        self.status_label.setStyleSheet("color: #e74c3c;")
        self.btn_translate.setEnabled(True)
        self.btn_regenerate.setEnabled(True)

    def show_centered(self, target_lang: str, source_lang: str):
        self.target_lang = target_lang
        self.source_lang = source_lang
        
        self.title_label.setText(t("chat.title.translate") + f" ({target_lang} → {source_lang})")
        self.input_label.setText(t("chat.label.original") + f" ({target_lang}):")
        self.output_label.setText(t("chat.label.translation") + f" ({source_lang}):")
        
        self.input_field.clear()
        self.output_field.clear()
        self.status_label.setText("")
        
        self.btn_translate.setEnabled(True)
        self.btn_regenerate.setEnabled(False)
        self.btn_copy.setEnabled(False)
        self.btn_save.setEnabled(False)
        
        self.adjustSize()
        self.setFixedSize(self.size())
        
        preset = self.app.settings.value("layout_preset", "Default")
        key = f"layout/{preset}/how_to_say"
        if self.app.settings.contains(key):
            self.restore_geometry()
        else:
            screen = QGuiApplication.screenAt(QCursor.pos())
            if not screen:
                screen = QGuiApplication.primaryScreen()
                
            if screen:
                geo = screen.geometry()
                x = geo.center().x() - self.width() // 2
                y = geo.center().y() - self.height() // 2
                self.move(x, y)
        
        self.show()
        self.activateWindow()
        self.input_field.setFocus()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        elif event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
            # Allow shift+enter for newlines in QTextEdit
            super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Enter submits
            if self.input_field.hasFocus() and self.btn_translate.isEnabled():
                self._on_translate()
                event.accept()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.dialog_hidden.emit()


