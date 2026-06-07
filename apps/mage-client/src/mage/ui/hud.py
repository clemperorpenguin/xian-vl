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

import os
import json
import logging
from PyQt6.QtWidgets import (
    QWidget, QDialog, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLineEdit, QTextEdit, QMessageBox,
    QSystemTrayIcon, QInputDialog
)
from PyQt6.QtCore import (
    Qt, QRect, QPoint, QTimer, pyqtSignal, QStandardPaths,
    QBuffer, QIODevice
)
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QMouseEvent, QPixmap, QImage,
    QGuiApplication, QFont, QCursor
)

from mage.ui.theme import accent_hex, accent_hover_hex, accent_qcolor
from mage.utils.window_binder import set_bypass_compositor_hint_x11
from mage.capture.screen import ScreenCapture
from mage.workers import InferenceWorker
from shared_types.state import t

logger = logging.getLogger(__name__)


def get_hud_presets_dir() -> str:
    """Get path to the HUD presets folder in AppData, creating it if needed."""
    path = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    presets_dir = os.path.join(path, "hud_presets")
    os.makedirs(presets_dir, exist_ok=True)
    return presets_dir


def get_pinyin_for_text(text: str, dictionary) -> str:
    """Convert Chinese text to Pinyin using LocalDictionary (CC-CEDICT)."""
    if not dictionary:
        return ""
    pinyin_list = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace() or char in ",.?!;:，。？！；：":
            pinyin_list.append(char)
            i += 1
            continue
            
        matched = False
        # Try to find the longest matching word from this index (max 4 chars)
        for length in range(min(4, len(text) - i), 0, -1):
            sub = text[i:i+length]
            entries = dictionary.lookup(sub)
            if entries:
                pinyin_list.append(entries[0][1])
                i += length
                matched = True
                break
        
        if not matched:
            pinyin_list.append(char)
            i += 1
            
    return " ".join(pinyin_list)


class HudPresetSelectionDialog(QDialog):
    """Allows loading an existing preset or starting a new preset configuration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("hud.preset.dialog.title"))
        self.setFixedSize(400, 200)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        prompt = QLabel(t("hud.preset.dialog.prompt"))
        prompt.setWordWrap(True)
        prompt.setStyleSheet("color: #ccc; font-size: 13px;")
        layout.addWidget(prompt)
        
        # Preset selection row
        self.preset_combo = QComboBox()
        self._populate_presets()
        layout.addWidget(self.preset_combo)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        
        self.load_btn = QPushButton(t("hud.preset.dialog.button.load"))
        self.load_btn.clicked.connect(self._on_load)
        self.load_btn.setEnabled(self.preset_combo.count() > 0)
        
        self.create_btn = QPushButton(t("hud.preset.dialog.button.create"))
        self.create_btn.clicked.connect(self._on_create)
        
        cancel_btn = QPushButton(t("hud.preset.dialog.button.cancel"))
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.create_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        # Styling
        self.setStyleSheet(f"""
            QDialog {{ background: #1e1e1e; color: #eee; }}
            QLabel {{ color: #ccc; }}
            QComboBox {{
                background: #2a2a2a; color: #eee; border: 1px solid #555;
                border-radius: 4px; padding: 6px;
            }}
            QPushButton {{
                background: #333; color: white; border: none;
                padding: 8px 14px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #444; }}
            QPushButton#LoadBtn {{
                background: {accent_hex()};
            }}
            QPushButton#LoadBtn:hover {{
                background: {accent_hover_hex()};
            }}
        """)
        self.load_btn.setObjectName("LoadBtn")
        self.selected_preset_path = None
        self.action = None  # "load" or "create"

    def _populate_presets(self):
        self.preset_combo.clear()
        presets_dir = get_hud_presets_dir()
        for filename in os.listdir(presets_dir):
            if filename.endswith(".json"):
                preset_name, _ = os.path.splitext(filename)
                self.preset_combo.addItem(preset_name, os.path.join(presets_dir, filename))

    def _on_load(self):
        self.action = "load"
        self.selected_preset_path = self.preset_combo.currentData()
        self.accept()

    def _on_create(self):
        self.action = "create"
        self.accept()


class HudSetupOverlay(QWidget):
    """Full-screen screen freeze transparent overlay to select:
       1. Hover trigger region (invisible button)
       2. Translation tooltip display location
       3. Original text box (OCR source)
    """
    setup_completed = pyqtSignal(QRect, QRect, QRect, bytes)
    closed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Frozen screen capture
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
        
        # Steps: "hover_trigger", "tooltip_display", "ocr_source"
        self.step = "hover_trigger"
        
        # Saved rectangles
        self.hover_rect = QRect()
        self.display_rect = QRect()
        self.ocr_rect = QRect()
        
        # Header banner for instruction
        self.banner = QLabel(self)
        self.banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner.setStyleSheet(f"""
            background-color: rgba(20, 20, 20, 230);
            color: #eee;
            border: 1px solid {accent_hex()};
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            padding: 8px 16px;
        """)
        self._update_banner_text()
        
    def _update_banner_text(self):
        if self.step == "hover_trigger":
            self.banner.setText(t("hud.setup.overlay.step1"))
        elif self.step == "ocr_source":
            self.banner.setText(t("hud.setup.overlay.step2"))
        elif self.step == "tooltip_display":
            self.banner.setText(t("hud.setup.overlay.step3"))
        self.banner.adjustSize()
        self.banner.move((self.width() - self.banner.width()) // 2, 40)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.globalPosition().toPoint()
            self.current_pos = self.start_pos
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
            
            # Skip tiny clicks
            if rect.width() > 5 and rect.height() > 5:
                if self.step == "hover_trigger":
                    self.hover_rect = rect
                    self.start_ocr_capture_countdown()
                elif self.step == "ocr_source":
                    self.ocr_rect = rect
                    self.step = "tooltip_display"
                    self._update_banner_text()
                elif self.step == "tooltip_display":
                    self.display_rect = rect
                    self._finalize_selection()
            self.update()

    def start_ocr_capture_countdown(self):
        self.hide() # hide selection overlay to let mouse trigger game tooltip
        self.countdown_val = 2
        
        self.countdown_widget = QWidget()
        self.countdown_widget.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.countdown_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        root = QWidget(self.countdown_widget)
        root.setStyleSheet(f"""
            background-color: rgba(20, 20, 20, 230);
            color: #eee;
            border: 2px solid {accent_hex()};
            border-radius: 8px;
        """)
        
        lbl = QLabel(t("hud.countdown.label").format(secs=self.countdown_val), root)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; padding: 16px; color: #fff;")
        
        layout = QVBoxLayout(self.countdown_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(root)
        
        self.countdown_widget.adjustSize()
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.countdown_widget.move((geo.width() - self.countdown_widget.width()) // 2, 80)
            
        self.countdown_widget.show()
        
        self.countdown_timer = QTimer(self)
        
        def tick():
            self.countdown_val -= 1
            if self.countdown_val > 0:
                lbl.setText(t("hud.countdown.label").format(secs=self.countdown_val))
                self.countdown_widget.adjustSize()
            else:
                self.countdown_timer.stop()
                self.countdown_widget.close()
                self._capture_screen_for_ocr()
                
        self.countdown_timer.timeout.connect(tick)
        self.countdown_timer.start(1000)

    def _capture_screen_for_ocr(self):
        # Capture screen now that the tooltip is shown
        self.full_image_data = ScreenCapture.capture_screen()
        if self.full_image_data:
            img = QImage.fromData(self.full_image_data)
            self.pixmap = QPixmap.fromImage(img)
        else:
            self.pixmap = QPixmap()
            
        # Re-show selection overlay to select the tooltip box
        self.step = "ocr_source"
        self._update_banner_text()
        self.showFullScreen()
        self.update()

    def _finalize_selection(self):
        # Crop the OCR region
        cropped_data = b""
        if not self.pixmap.isNull() and not self.ocr_rect.isEmpty():
            safe_rect = self.ocr_rect.translated(-self.total_geo.left(), -self.total_geo.top())
            safe_rect = safe_rect.intersected(self.pixmap.rect())
            cropped = self.pixmap.copy(safe_rect)
            
            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            cropped.save(buf, "PNG")
            cropped_data = bytes(buf.buffer())
            
        self.setup_completed.emit(self.hover_rect, self.display_rect, self.ocr_rect, cropped_data)
        self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap.isNull():
            painter.drawPixmap(0, 0, self.pixmap)
            
        # Draw dimming
        dim_color = QColor(0, 0, 0, 150)
        painter.fillRect(self.rect(), dim_color)
        
        # Clear selected areas
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        for r in (self.hover_rect, self.display_rect, self.ocr_rect):
            if not r.isEmpty():
                painter.fillRect(r.translated(-self.total_geo.left(), -self.total_geo.top()), Qt.GlobalColor.transparent)
                
        # Clear currently selecting rect
        if self.selecting:
            curr_rect = QRect(self.start_pos, self.current_pos).normalized()
            painter.fillRect(curr_rect.translated(-self.total_geo.left(), -self.total_geo.top()), Qt.GlobalColor.transparent)
            
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        
        # Draw completed rects with borders
        # Hover trigger: Yellow
        if not self.hover_rect.isEmpty():
            painter.setPen(QPen(QColor(230, 219, 116), 2))
            painter.drawRect(self.hover_rect.translated(-self.total_geo.left(), -self.total_geo.top()))
            
        # Display tooltip location: Green
        if not self.display_rect.isEmpty():
            painter.setPen(QPen(QColor(166, 226, 46), 2))
            painter.drawRect(self.display_rect.translated(-self.total_geo.left(), -self.total_geo.top()))
            
        # OCR source: Cyan/Accent
        if not self.ocr_rect.isEmpty():
            painter.setPen(QPen(accent_qcolor(), 2))
            painter.drawRect(self.ocr_rect.translated(-self.total_geo.left(), -self.total_geo.top()))
            
        # Current selection
        if self.selecting:
            curr_rect = QRect(self.start_pos, self.current_pos).normalized()
            painter.setPen(QPen(Qt.GlobalColor.white, 2, Qt.PenStyle.DashLine))
            painter.drawRect(curr_rect.translated(-self.total_geo.left(), -self.total_geo.top()))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        set_bypass_compositor_hint_x11(self.winId())


class HudTranslationConfirmDialog(QDialog):
    """Shows the OCR results and lets the user verify or rewrite the translation."""
    
    def __init__(self, image_data: bytes, original: str, translation: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("hud.confirm.dialog.title"))
        self.setFixedSize(500, 420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)
        
        # Display image preview
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setFixedHeight(80)
        self.img_label.setStyleSheet("background-color: #111; border: 1px solid #444; border-radius: 4px;")
        if image_data:
            pix = QPixmap()
            pix.loadFromData(image_data)
            self.img_label.setPixmap(pix.scaled(self.img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(self.img_label)
        
        # Original Text
        layout.addWidget(QLabel(t("hud.confirm.dialog.label.original")))
        self.orig_edit = QLineEdit(original)
        layout.addWidget(self.orig_edit)
        
        # Translation
        layout.addWidget(QLabel(t("hud.confirm.dialog.label.translated")))
        self.trans_edit = QTextEdit(translation)
        layout.addWidget(self.trans_edit)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.confirm_btn = QPushButton(t("hud.confirm.dialog.button.confirm"))
        self.confirm_btn.clicked.connect(self.accept)
        self.confirm_btn.setObjectName("ConfirmBtn")
        
        self.recapture_btn = QPushButton(t("hud.confirm.dialog.button.recapture"))
        self.recapture_btn.clicked.connect(self._on_recapture)
        
        self.cancel_btn = QPushButton(t("hud.confirm.dialog.button.cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.confirm_btn)
        btn_layout.addWidget(self.recapture_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # Styling
        self.setStyleSheet(f"""
            QDialog {{ background: #1e1e1e; color: #eee; }}
            QLabel {{ color: #ccc; font-weight: bold; font-size: 11px; }}
            QLineEdit, QTextEdit {{
                background: #2a2a2a; color: #eee; border: 1px solid #555;
                border-radius: 4px; padding: 6px;
            }}
            QPushButton {{
                background: #333; color: white; border: none;
                padding: 8px 14px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #444; }}
            QPushButton#ConfirmBtn {{
                background: {accent_hex()};
            }}
            QPushButton#ConfirmBtn:hover {{
                background: {accent_hover_hex()};
            }}
        """)
        self.recapture_requested = False

    def _on_recapture(self):
        self.recapture_requested = True
        self.reject()

    def get_data(self) -> tuple[str, str]:
        return self.orig_edit.text().strip(), self.trans_edit.toPlainText().strip()


class HudSetupControlDialog(QDialog):
    """Small persistent window showing how many buttons are configured, allowing preset saving."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("hud.setup.dialog.title"))
        self.setFixedSize(300, 180)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowDoesNotAcceptFocus |
            Qt.WindowType.Tool
        )
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #ddd;")
        layout.addWidget(self.status_label)
        
        self.add_btn = QPushButton(t("hud.setup.dialog.button.add"))
        self.add_btn.setObjectName("AddBtn")
        
        self.save_btn = QPushButton(t("hud.setup.dialog.button.save"))
        self.save_btn.setObjectName("SaveBtn")
        
        self.update_count(0)
        
        cancel_btn = QPushButton(t("hud.preset.dialog.button.cancel"))
        cancel_btn.clicked.connect(self.reject)
        
        layout.addWidget(self.add_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(cancel_btn)
        
        self.setStyleSheet(f"""
            QDialog {{ background: #1e1e1e; border: 1px solid {accent_hex()}; border-radius: 8px; }}
            QLabel {{ color: #ccc; }}
            QPushButton {{
                background: #333; color: white; border: none;
                padding: 8px 12px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #444; }}
            QPushButton#AddBtn {{
                background: {accent_hex()};
            }}
            QPushButton#AddBtn:hover {{
                background: {accent_hover_hex()};
            }}
            QPushButton#SaveBtn {{
                background: #2e7d32;
            }}
            QPushButton#SaveBtn:hover {{
                background: #388e3c;
            }}
        """)
        
    def update_count(self, count: int):
        self.status_label.setText(t("hud.setup.dialog.prompt").format(count=count))
        self.save_btn.setEnabled(count > 0)


class HudTooltip(QWidget):
    """The floating text tooltip displayed statically when a trigger region is hovered."""
    
    def __init__(self, rect: QRect, text: str, original: str = "", pinyin: str = "", parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.WindowDoesNotAcceptFocus |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        root = QWidget(self)
        root.setObjectName("TooltipRoot")
        root.setStyleSheet(f"""
            #TooltipRoot {{
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid {accent_hex()};
                border-radius: 6px;
            }}
            QLabel {{
                color: #eee;
            }}
        """)
        
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        
        # Original text if requested
        if original:
            orig_lbl = QLabel(original)
            orig_lbl.setWordWrap(True)
            orig_lbl.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
            layout.addWidget(orig_lbl)
            
        # Pinyin if requested
        if pinyin:
            pin_lbl = QLabel(pinyin)
            pin_lbl.setWordWrap(True)
            pin_lbl.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(pin_lbl)
            
        # Translated text
        trans_lbl = QLabel(text)
        trans_lbl.setWordWrap(True)
        trans_lbl.setFont(QFont("sans-serif", 12, QFont.Weight.Bold))
        layout.addWidget(trans_lbl)
        
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)
        
        # Layout sizing
        self.setGeometry(rect)
        self.adjustSize()
        self.show()
        
    def showEvent(self, event):
        super().showEvent(event)
        set_bypass_compositor_hint_x11(self.winId())


class HudManager(QWidget):
    """Orchestrates HUD preset creation, loading, deactivation, and cursor-polling loop."""
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.active_preset = None
        self.hovered_button = None
        self.tooltip_widget = None
        
        # The 100ms mouse hover polling loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_hover)
        
        # Configure state
        self.setup_buttons = []
        self.control_dialog = None
        self.setup_overlay = None

    def show_hud_presets(self):
        """HUD activation/deactivation handler triggered by hotkey or OSD."""
        if self.timer.isActive():
            self.deactivate()
            return
            
        dialog = HudPresetSelectionDialog(self.app)
        self.app._apply_transient_parent(dialog)
        if dialog.exec():
            if dialog.action == "load":
                self.load_preset(dialog.selected_preset_path)
            elif dialog.action == "create":
                self.start_preset_creation()

    def start_preset_creation(self):
        """Starts a session to configure a new HUD preset."""
        self.setup_buttons = []
        
        self.control_dialog = HudSetupControlDialog(self.app)
        self.app._apply_transient_parent(self.control_dialog)
        self.control_dialog.add_btn.clicked.connect(self._trigger_selection_overlay)
        self.control_dialog.save_btn.clicked.connect(self._save_preset_prompt)
        
        self.control_dialog.show()
        self._trigger_selection_overlay()

    def _trigger_selection_overlay(self):
        if self.control_dialog:
            self.control_dialog.hide()
            
        self.setup_overlay = HudSetupOverlay()
        self.setup_overlay.setup_completed.connect(self._on_setup_overlay_completed)
        self.setup_overlay.closed.connect(self._on_setup_overlay_closed)
        self.setup_overlay.showFullScreen()

    def _on_setup_overlay_closed(self):
        if self.control_dialog:
            self.control_dialog.show()

    def _on_setup_overlay_completed(self, hover_rect: QRect, display_rect: QRect, ocr_rect: QRect, cropped_data: bytes):
        # Crop was done. Spawn background thread inference to translate it
        if not cropped_data:
            QMessageBox.warning(self.control_dialog, "Error", "No text region captured.")
            if self.control_dialog:
                self.control_dialog.show()
            return
            
        # Spawn a worker to get OCR & Translation
        worker = InferenceWorker(
            self.app.processor,
            image_data=cropped_data,
            source_lang="Chinese",
            target_lang="English",
            mode="Game",
            styles=[],  # Standard translation defaults
            action="translate",
            anchor_rect=ocr_rect
        )
        
        def on_done(results, action):
            orig_combined = "\n".join(r.original_text for r in results if r.original_text)
            trans_combined = "\n".join(r.translated_text for r in results if r.translated_text)
            
            # Show confirmation dialog
            confirm = HudTranslationConfirmDialog(cropped_data, orig_combined, trans_combined, self.app)
            self.app._apply_transient_parent(confirm)
            if confirm.exec():
                final_orig, final_trans = confirm.get_data()
                
                # Precompute Pinyin now so we don't do lookups on every single frame hover
                final_pinyin = get_pinyin_for_text(final_orig, self.app.dictionary)
                
                btn_cfg = {
                    "hover_rect": [hover_rect.x(), hover_rect.y(), hover_rect.width(), hover_rect.height()],
                    "display_rect": [display_rect.x(), display_rect.y(), display_rect.width(), display_rect.height()],
                    "original_text": final_orig,
                    "translated_text": final_trans,
                    "pinyin": final_pinyin
                }
                self.setup_buttons.append(btn_cfg)
                
                if self.control_dialog:
                    self.control_dialog.update_count(len(self.setup_buttons))
                    self.control_dialog.show()
            else:
                if confirm.recapture_requested:
                    self._trigger_selection_overlay()
                elif self.control_dialog:
                    self.control_dialog.show()
                    
        def on_error(msg):
            QMessageBox.critical(self.control_dialog, "Inference Error", f"Translation failed:\n{msg}")
            if self.control_dialog:
                self.control_dialog.show()
                
        worker.translation_done.connect(on_done)
        worker.error.connect(on_error)
        worker.start()
        # Keep worker ref to prevent GC
        self._worker_ref = worker

    def _save_preset_prompt(self):
        # Save dialog prompt
        presets_dir = get_hud_presets_dir()
        name, ok = QInputDialog.getText(
            self.control_dialog,
            t("hud.setup.dialog.save_prompt.title"),
            t("hud.setup.dialog.save_prompt.label")
        )
        if ok and name.strip():
            preset_name = name.strip()
            preset_file = os.path.join(presets_dir, f"{preset_name}.json")
            
            preset_data = {
                "name": preset_name,
                "buttons": self.setup_buttons
            }
            try:
                with open(preset_file, "w", encoding="utf-8") as f:
                    json.dump(preset_data, f, ensure_ascii=False, indent=2)
                
                self.control_dialog.accept()
                self.load_preset(preset_file)
            except Exception as e:
                QMessageBox.critical(self.control_dialog, "Error", f"Failed to save preset:\n{e}")

    def load_preset(self, path: str):
        """Load JSON preset data and begin hover tracking loop."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.active_preset = json.load(f)
            
            self.timer.start(100)
            logger.info("HUD Mode Active, loaded preset: %s", self.active_preset.get("name"))
            
            # Show a tray message
            if hasattr(self.app, "tray") and self.app.tray:
                self.app.tray.showMessage(
                    t("hud.preset.dialog.title"),
                    t("hud.status.activated").format(name=self.active_preset.get("name")),
                    QSystemTrayIcon.MessageIcon.Information if hasattr(self.app, "tray") else None,
                    3000
                )
        except Exception as e:
            logger.error("Failed to load HUD preset %s: %s", path, e)
            QMessageBox.critical(self.app, "Error", f"Failed to load preset:\n{e}")

    def deactivate(self):
        """Stops the hover polling loop and destroys active tooltips."""
        self.timer.stop()
        self.hide_tooltip()
        self.hovered_button = None
        self.active_preset = None
        
        logger.info("HUD Mode Disabled")
        if hasattr(self.app, "tray") and self.app.tray:
            self.app.tray.showMessage(
                t("hud.preset.dialog.title"),
                t("hud.status.deactivated"),
                QSystemTrayIcon.MessageIcon.Information if hasattr(self.app, "tray") else None,
                3000
            )

    def _check_hover(self):
        if not self.active_preset:
            return
            
        pos = QCursor.pos()
        matched_btn = None
        
        # Check if cursor is in any of the button trigger areas
        for btn in self.active_preset.get("buttons", []):
            trigger_rect = QRect(*btn["hover_rect"])
            if trigger_rect.contains(pos):
                matched_btn = btn
                break
                
        if matched_btn:
            if self.hovered_button != matched_btn:
                self.hide_tooltip()
                self.hovered_button = matched_btn
                self.show_tooltip(matched_btn)
        else:
            self.hide_tooltip()
            self.hovered_button = None

    def show_tooltip(self, button_cfg: dict):
        """Displays the translation tooltip on the screen."""
        display_rect = QRect(*button_cfg["display_rect"])
        
        # Settings configurations
        show_original = self.app.settings.value("hud_show_original", "true") == "true"
        show_pinyin = self.app.settings.value("hud_show_pinyin", "false") == "true"
        
        orig_text = button_cfg.get("original_text", "") if show_original else ""
        pinyin_text = button_cfg.get("pinyin", "") if (show_pinyin and show_original) else ""
        trans_text = button_cfg.get("translated_text", "")
        
        self.tooltip_widget = HudTooltip(
            rect=display_rect,
            text=trans_text,
            original=orig_text,
            pinyin=pinyin_text,
            parent=None
        )
        self.app._apply_transient_parent(self.tooltip_widget)

    def hide_tooltip(self):
        """Hides/destroys the current tooltip window."""
        if self.tooltip_widget:
            try:
                self.tooltip_widget.close()
            except Exception:
                pass
            self.tooltip_widget = None
