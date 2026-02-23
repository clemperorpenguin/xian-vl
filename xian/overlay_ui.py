from typing import List
import logging
import os
from PyQt6 import sip
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QPlainTextEdit,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QSlider,
    QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, QSettings, QObject, pyqtSignal, QSize
from PyQt6.QtGui import QPainter, QColor, QFont, QGuiApplication, QMouseEvent, QPaintEvent, QFontMetrics, QRegion, QIcon
from .models import TranslationResult, TranslationMode

logger = logging.getLogger(__name__)


class _LogEmitter(QObject):
    message = pyqtSignal(str)


class OverlayLogHandler(logging.Handler):
    def __init__(self, emitter: _LogEmitter):
        super().__init__()
        self._emitter = emitter

    def emit(self, record):
        try:
            msg = self.format(record)
            self._emitter.message.emit(msg)
        except Exception:
            # Never let logging take down the UI
            pass


class OverlayControlPanel(QWidget):
    """Small always-on-top panel to monitor logs and control overlay."""

    request_clear = pyqtSignal()
    request_hide_overlay = pyqtSignal()
    request_show_overlay = pyqtSignal()
    request_start = pyqtSignal()
    request_stop = pyqtSignal()
    request_reset_settings = pyqtSignal()
    settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        # Parent the control panel to the overlay window for unification.
        # This allows it to be part of the same coordinate system and window.
        self.setWindowFlags(Qt.WindowType.Widget | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._overlay_visible = True
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.settings = QSettings("Xian", "VideoGameTranslator")

        root = QWidget(self)
        root.setObjectName("PanelRoot")
        self.root_widget = root

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header
        header = QHBoxLayout()
        logo_label = QLabel("Xian")
        logo_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        header.addWidget(logo_label)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa; font-size: 11px; margin-left: 10px;")
        header.addWidget(self.status_label)

        header.addStretch()

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self.request_start.emit)
        header.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("StopBtn")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.clicked.connect(self.request_stop.emit)
        self.stop_btn.hide()
        header.addWidget(self.stop_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setCheckable(True)
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.clicked.connect(self._toggle_settings)
        header.addWidget(self.settings_btn)

        layout.addLayout(header)

        # Main Stack
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Page 1: Logs
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 5, 0, 0)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setStyleSheet(
            "color: #ddd; background: rgba(0,0,0,80); border: 1px solid rgba(255,255,255,20); font-family: monospace; font-size: 10px;"
        )
        log_layout.addWidget(self.log_view)

        log_footer = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Logs")
        self.clear_btn.setStyleSheet("font-size: 10px;")
        self.clear_btn.clicked.connect(self.log_view.clear)
        log_footer.addWidget(self.clear_btn)

        self.clear_trans_btn = QPushButton("Clear Bubbles")
        self.clear_trans_btn.setStyleSheet("font-size: 10px;")
        self.clear_trans_btn.clicked.connect(self.request_clear.emit)
        log_footer.addWidget(self.clear_trans_btn)

        self.toggle_overlay_btn = QPushButton("Hide Overlay")
        self.toggle_overlay_btn.setStyleSheet("font-size: 10px;")
        self.toggle_overlay_btn.clicked.connect(self._toggle_overlay)
        log_footer.addWidget(self.toggle_overlay_btn)

        log_footer.addStretch()
        log_layout.addLayout(log_footer)

        self.stack.addWidget(log_container)

        # Page 2: Settings
        settings_container = QWidget()
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setStyleSheet("background: transparent; border: none;")

        settings_content = QWidget()
        settings_content.setStyleSheet("background: transparent;")
        settings_layout = QFormLayout(settings_content)
        settings_layout.setSpacing(12)
        settings_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Full Screen", "Region Selection"])
        settings_layout.addRow("Mode:", self.mode_combo)

        # Language
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["auto", "Japanese", "Korean", "Chinese", "Spanish", "French", "English"])
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["English", "Japanese", "Korean", "Chinese", "Spanish", "French"])

        settings_layout.addRow("Source:", self.source_lang_combo)
        settings_layout.addRow("Target:", self.target_lang_combo)

        # Model
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Qwen3-VL-8B-Thinking (Auto-select)",
            "Qwen3-VL-4B-Thinking",
            "Qwen3-VL-8B-Thinking",
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-8B-Instruct",
            "TranslateGemma-12B (High Quality)",
            "TranslateGemma-4B (Lower Resource)",
        ])
        self.model_combo.setEditable(True)
        self.model_combo.setToolTip("Vision-Language model for unified OCR and translation")
        settings_layout.addRow("Model:", self.model_combo)

        # Timing
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(500, 10000)
        self.interval_spin.setSingleStep(500)
        self.interval_spin.setSuffix(" ms")
        settings_layout.addRow("Interval:", self.interval_spin)

        # Appearance
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(30, 100)
        settings_layout.addRow("Opacity:", self.opacity_slider)

        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 50)
        self.margin_spin.setSuffix(" px")
        settings_layout.addRow("Redact Margin:", self.margin_spin)

        self.debug_check = QCheckBox("Enable Debug Mode")
        settings_layout.addRow("", self.debug_check)

        # Readability options
        self.combine_check = QCheckBox("Combine nearby lines into paragraphs")
        settings_layout.addRow("", self.combine_check)

        self.show_full_check = QCheckBox("Show full text (expanded) by default")
        settings_layout.addRow("", self.show_full_check)

        # Link/anchor visualization
        self.show_link_check = QCheckBox("Show link to source when bubble is clicked")
        settings_layout.addRow("", self.show_link_check)

        self.reset_btn = QPushButton("Reset All Settings")
        self.reset_btn.setStyleSheet("background-color: rgba(183, 28, 28, 150); font-size: 11px; margin-top: 10px;")
        self.reset_btn.clicked.connect(self.request_reset_settings)
        settings_layout.addRow("", self.reset_btn)

        settings_scroll.setWidget(settings_content)
        sv = QVBoxLayout(settings_container)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.addWidget(settings_scroll)
        self.stack.addWidget(settings_container)

        self.resize(520, 300)
        self._move_to_default_position()
        self._load_panel_settings()
        self.apply_opacity(self.opacity_slider.value())
        # Default to the Settings view so the app opens directly into overlay settings.
        self.settings_btn.setChecked(True)
        self._toggle_settings(True)
        self._connect_internal_signals()

        # Wire python logging -> Qt
        self._emitter = _LogEmitter()
        self._emitter.message.connect(self._append_log)
        self._handler = OverlayLogHandler(self._emitter)
        self._handler.setLevel(logging.DEBUG)
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        )
        logging.getLogger().addHandler(self._handler)

    def show_settings_view(self):
        """Switch to the Settings view and ensure the button state matches."""
        self.settings_btn.setChecked(True)
        self._toggle_settings(True)

    def _connect_internal_signals(self):
        self.mode_combo.currentIndexChanged.connect(self._save_panel_settings)
        self.source_lang_combo.currentIndexChanged.connect(self._save_panel_settings)
        self.target_lang_combo.currentIndexChanged.connect(self._save_panel_settings)
        self.model_combo.currentTextChanged.connect(self._save_panel_settings)
        self.interval_spin.valueChanged.connect(self._save_panel_settings)
        self.opacity_slider.valueChanged.connect(self._save_panel_settings)
        self.margin_spin.valueChanged.connect(self._save_panel_settings)
        self.debug_check.toggled.connect(self._save_panel_settings)
        self.combine_check.toggled.connect(self._save_panel_settings)
        self.show_full_check.toggled.connect(self._save_panel_settings)
        self.show_link_check.toggled.connect(self._save_panel_settings)

    def _toggle_settings(self, checked):
        self.stack.setCurrentIndex(1 if checked else 0)
        self.settings_btn.setText("Logs" if checked else "Settings")

    def _load_panel_settings(self):
        s = self.settings
        self.mode_combo.setCurrentText(
            "Full Screen" if s.value("translation_mode", "full_screen") == "full_screen" else "Region Selection")
        self.source_lang_combo.setCurrentText(s.value("source_lang", "auto"))
        self.target_lang_combo.setCurrentText(s.value("target_lang", "English"))
        self.model_combo.setCurrentText(s.value("model_name", "Qwen3-VL-8B-Thinking (Auto-select)"))
        self.interval_spin.setValue(int(s.value("interval", 2000)))
        self.opacity_slider.setValue(int(s.value("opacity", 80)))
        self.margin_spin.setValue(int(s.value("redaction_margin", 15)))
        self.debug_check.setChecked(s.value("debug_mode", "false") == "true")
        self.combine_check.setChecked(s.value("combine_paragraphs", "true") == "true")
        self.show_full_check.setChecked(s.value("show_full_text", "true") == "true")
        self.show_link_check.setChecked(s.value("show_link_on_click", "false") == "true")

    def _save_panel_settings(self):
        self.apply_opacity(self.opacity_slider.value())
        s = self.settings
        s.setValue("translation_mode",
                   "full_screen" if self.mode_combo.currentText() == "Full Screen" else "region_select")
        s.setValue("source_lang", self.source_lang_combo.currentText())
        s.setValue("target_lang", self.target_lang_combo.currentText())
        s.setValue("model_name", self.model_combo.currentText())
        s.setValue("interval", self.interval_spin.value())
        s.setValue("opacity", self.opacity_slider.value())
        s.setValue("redaction_margin", self.margin_spin.value())
        s.setValue("debug_mode", "true" if self.debug_check.isChecked() else "false")
        s.setValue("combine_paragraphs", "true" if self.combine_check.isChecked() else "false")
        s.setValue("show_full_text", "true" if self.show_full_check.isChecked() else "false")
        s.setValue("show_link_on_click", "true" if self.show_link_check.isChecked() else "false")
        self.settings_changed.emit()

    def set_running(self, running: bool):
        self.start_btn.setVisible(not running)
        self.stop_btn.setVisible(running)
        if not running:
            self.status_label.setText("Ready")

    def _move_to_default_position(self):
        screen = QGuiApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        self.move(geo.left() + 20, geo.top() + 20)

    def _append_log(self, line: str):
        self.log_view.appendPlainText(line)

    def set_overlay_visible(self, visible: bool):
        self._overlay_visible = visible
        self.toggle_overlay_btn.setText("Hide Overlay" if visible else "Show Overlay")

    def set_stats(self, bubble_count: int):
        self.status_label.setText(f"Overlay Panel — {bubble_count} bubbles")

    def _toggle_overlay(self):
        if self._overlay_visible:
            self.request_hide_overlay.emit()
            self.set_overlay_visible(False)
        else:
            self.request_show_overlay.emit()
            self.set_overlay_visible(True)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.position().toPoint()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            new_pos = self.pos() + event.position().toPoint() - self.drag_start_pos
            self.move(new_pos)

            # Update mask of the parent overlay window
            parent = self.parentWidget()
            if parent and hasattr(parent, 'update_mask_during_drag'):
                parent.update_mask_during_drag()

            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.dragging = False
        event.accept()

    def closeEvent(self, event):
        try:
            logging.getLogger().removeHandler(self._handler)
        except Exception:
            pass
        super().closeEvent(event)

    def apply_opacity(self, opacity: int):
        """Apply opacity to the control panel using style-based alpha (Wayland-safe)."""
        clamped = max(30, min(100, int(opacity)))
        panel_alpha = max(60, min(240, int(clamped * 2.4)))
        border_alpha = max(30, min(180, int(panel_alpha * 0.35)))
        button_alpha = max(80, min(230, int(clamped * 2.2)))
        hover_alpha = min(255, button_alpha + 20)
        combo_alpha = max(80, min(230, int(clamped * 2.0)))

        self.root_widget.setStyleSheet(
            f"""
            QWidget#PanelRoot {{
                background-color: rgba(20, 20, 20, {panel_alpha});
                border: 1px solid rgba(255, 255, 255, {border_alpha});
                border-radius: 12px;
            }}
            QLabel {{ color: #eee; }}
            QPushButton {{
                background-color: rgba(60, 60, 60, {button_alpha});
                color: white;
                border-radius: 4px;
                padding: 4px 8px;
                border: 1px solid rgba(255, 255, 255, 30);
            }}
            QPushButton:hover {{
                background-color: rgba(80, 80, 80, {hover_alpha});
            }}
            QPushButton#StartBtn {{ background-color: rgba(46, 125, 50, {button_alpha}); }}
            QPushButton#StartBtn:hover {{ background-color: rgba(56, 142, 60, {hover_alpha}); }}
            QPushButton#StopBtn {{ background-color: rgba(198, 40, 40, {button_alpha}); }}
            QPushButton#StopBtn:hover {{ background-color: rgba(211, 47, 47, {hover_alpha}); }}
            QComboBox, QSpinBox {{
                background-color: rgba(40, 40, 40, {combo_alpha});
                color: white;
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 4px;
                padding: 2px;
            }}
        """
        )


class OverlayWindow(QWidget):
    """Full-screen transparent container for bubbles to fix Wayland positioning"""

    def __init__(self):
        super().__init__()
        # On some Wayland compositors, keeping a window always-on-top requires
        # both WindowStaysOnTopHint and Tool flags, plus periodic raise_ calls.
        flags = (
                Qt.WindowType.Window
                | Qt.WindowType.Tool
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.NoDropShadowWindowHint
        )

        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Cover the entire virtual desktop to fix Wayland positioning issues
        total_geo = QRect()
        for screen in QGuiApplication.screens():
            total_geo = total_geo.united(screen.geometry())

        self.setGeometry(total_geo)
        # Initial mask is empty so it's click-through
        self.setMask(QRegion())
        self.show()

    def paintEvent(self, event: QPaintEvent):
        """Transparent background, plus optional link rendering between bubbles and source rects."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(event.rect(), Qt.GlobalColor.transparent)

        # Determine if link visualization is enabled via the control panel
        show_links_enabled = False
        try:
            panels = self.findChildren(OverlayControlPanel)
            if panels:
                panel = panels[0]
                show_links_enabled = panel.show_link_check.isChecked()
        except Exception:
            show_links_enabled = False

        if show_links_enabled:
            try:
                # Draw connectors for bubbles that requested it
                bubbles = self.findChildren(TranslationBubble)
                for b in bubbles:
                    if sip.isdeleted(b) or not b.isVisible():
                        continue
                    if not getattr(b, "show_link", False):
                        continue

                    # Bubble anchor: center of its geometry
                    b_rect = b.geometry()
                    bubble_center = b_rect.center()

                    # Source rect in overlay coordinates
                    r = b.result
                    src_rect = QRect(int(r.x), int(r.y), int(r.width), int(r.height))

                    # Draw highlighted source rectangle
                    highlight_color = QColor(76, 175, 80, 140)  # greenish
                    border_color = QColor(76, 175, 80, 220)
                    painter.setBrush(highlight_color)
                    painter.setPen(border_color)
                    painter.drawRoundedRect(src_rect, 6, 6)

                    # Draw a connector line from bubble center to source rect center
                    src_center = src_rect.center()
                    pen = painter.pen()
                    pen.setWidth(3)
                    pen.setColor(border_color)
                    painter.setPen(pen)
                    painter.drawLine(bubble_center, src_center)

                    # Subtle glow by overdrawing with alpha
                    glow_pen = painter.pen()
                    glow_pen.setWidth(7)
                    glow_pen.setColor(QColor(76, 175, 80, 80))
                    painter.setPen(glow_pen)
                    painter.drawLine(bubble_center, src_center)

                    # Reset pen for next iteration
                    painter.setPen(Qt.PenStyle.NoPen)
            except Exception:
                pass

        painter.end()

    def update_mask_during_drag(self):
        """Recalculate mask from all children during a drag operation"""
        mask = QRegion()
        # Base mask: all visible child widgets (bubbles + control panel)
        for child in self.findChildren(QWidget):
            if child.isVisible() and not child.isWindow():
                mask += child.geometry()

        # If link visualization is enabled, include source rects and a thin band along connectors
        show_links_enabled = False
        try:
            panels = self.findChildren(OverlayControlPanel)
            if panels:
                show_links_enabled = panels[0].show_link_check.isChecked()
        except Exception:
            show_links_enabled = False

        if show_links_enabled:
            try:
                bubbles = self.findChildren(TranslationBubble)
                for b in bubbles:
                    if sip.isdeleted(b) or not b.isVisible():
                        continue
                    if not getattr(b, "show_link", False):
                        continue
                    r = b.result
                    src_rect = QRect(int(r.x), int(r.y), int(r.width), int(r.height))
                    mask += src_rect

                    # Add a thin series of rectangles along the line from bubble center to src center
                    b_center = b.geometry().center()
                    s_center = src_rect.center()
                    dx = s_center.x() - b_center.x()
                    dy = s_center.y() - b_center.y()
                    steps = max(1, int((abs(dx) + abs(dy)) / 30))
                    for i in range(1, steps):
                        t = i / steps
                        x = int(b_center.x() + dx * t)
                        y = int(b_center.y() + dy * t)
                        mask += QRect(x - 4, y - 4, 8, 8)
            except Exception:
                pass
        self.setMask(mask)
        self.update()


class TranslationBubble(QWidget):
    """Translation bubble, now a child of OverlayWindow for reliable positioning"""

    def __init__(self, result: TranslationResult, opacity: int, parent_overlay: QWidget = None,
                 default_expanded: bool = False):
        super().__init__(parent_overlay)
        self.result = result
        self.opacity = opacity
        self.dragging = False
        self.expanded = bool(default_expanded)
        self.drag_start_pos = QPoint()
        self.press_pos = QPoint()

        # Since it's a child widget, we don't need all the window flags
        # But we still want it to look like a bubble
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.setup_ui()
        # Respect default expansion state on creation
        self.stack.setCurrentIndex(1 if self.expanded else 0)
        self.update_geometry()
        self.show_link = False  # toggled when clicked if feature enabled

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 20, 10, 10)

        # Close button
        self.close_btn = QPushButton("×", self)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(200, 0, 0, 180);
                color: white;
                border-radius: 10px;
                font-weight: bold;
                border: none;
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 0, 0, 220);
            }}
        """)
        self.close_btn.clicked.connect(self.deleteLater)

        self.stack = QStackedWidget()

        # Collapsed view
        self.collapsed_label = QLabel()
        self.collapsed_label.setWordWrap(True)
        self.collapsed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.collapsed_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent;")

        # Expanded view
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")

        self.expanded_label = QLabel()
        self.expanded_label.setWordWrap(True)
        self.expanded_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.expanded_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent;")

        self.scroll_area.setWidget(self.expanded_label)

        self.stack.addWidget(self.collapsed_label)
        self.stack.addWidget(self.scroll_area)

        layout.addWidget(self.stack)
        self._update_text_displays()

    def _get_truncated_text(self, text, word_limit=8):
        words = text.split()
        if len(words) <= word_limit:
            return text
        return " ".join(words[:word_limit]) + "..."

    def _update_text_displays(self):
        full_text = self.result.translated_text
        truncated = self._get_truncated_text(full_text)
        self.collapsed_label.setText(truncated)
        self.expanded_label.setText(full_text)

    def update_geometry(self):
        # Calculate size based on text
        font = QFont("Arial", 12, QFont.Weight.Bold)
        metrics = QFontMetrics(font)

        padding = 20
        if not self.expanded:
            text = self.collapsed_label.text()
            # `TranslationResult` coordinates/sizes may be floats.
            # Qt geometry APIs require ints.
            measure_width = max(150.0, float(self.result.width) + padding * 2)
            if measure_width > 350:
                measure_width = 350.0

            measure_width_i = int(round(measure_width))
            content_width_i = max(1, measure_width_i - padding * 2)

            text_rect = metrics.boundingRect(
                QRect(0, 0, content_width_i, 1000),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                text)

            box_width = int(text_rect.width() + padding * 2)
            box_height = int(text_rect.height() + padding * 2 + 10)
        else:
            # Expanded mode size
            box_width = 400
            box_height = 250

        # Apply min sizes
        box_width = max(box_width, 100)
        box_height = max(box_height, 40)

        self.setFixedSize(box_width, box_height)

        # Parent geometry (OverlayWindow covers screen(s))
        if self.parentWidget():
            parent_geo = self.parentWidget().geometry()
        else:
            parent_geo = QGuiApplication.primaryScreen().geometry()

        # Center the bubble over the original text coordinates
        # result.x/y are relative to the captured image.
        # If capture was full desktop, and OverlayWindow covers full desktop, then
        # they are already in the correct coordinate space relative to OverlayWindow.
        target_x = self.result.x + (self.result.width - box_width) // 2
        target_y = self.result.y + (self.result.height - box_height) // 2

        # Constraint to parent bounds
        x = max(parent_geo.left() + 10, min(target_x, parent_geo.right() - box_width - 10))
        y = max(parent_geo.top() + 10, min(target_y, parent_geo.bottom() - box_height - 10))

        # When moving a child widget, it's relative to the parent's (0,0).
        # Since OverlayWindow covers the whole virtual desktop, we need to adjust
        # by the parent's top-left if it's not (0,0).
        rel_x = x - parent_geo.left()
        rel_y = y - parent_geo.top()

        self.move(int(rel_x), int(rel_y))

    def toggle_expansion(self):
        self.expanded = not self.expanded
        self.stack.setCurrentIndex(1 if self.expanded else 0)
        self.update_geometry()

        # Update mask because size changed
        parent = self.parentWidget()
        if parent and hasattr(parent, 'update_mask_during_drag'):
            parent.update_mask_during_drag()

    def update_content(self, result: TranslationResult):
        """Update bubble with new translation result"""
        if self.result.translated_text != result.translated_text:
            self.result = result
            self._update_text_displays()
            self.update_geometry()
            self._pulse()
        else:
            # Just update coordinates if they changed significantly
            old_pos = QPoint(int(self.result.x), int(self.result.y))
            new_pos = QPoint(int(result.x), int(result.y))
            if (old_pos - new_pos).manhattanLength() > 5:
                self.result = result
                self.update_geometry()

    def _pulse(self):
        """Briefly highlight the bubble when updated"""
        target = self.expanded_label if self.expanded else self.collapsed_label
        target.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 15px; background: transparent;")
        QTimer.singleShot(500, self._reset_style)

    def _reset_style(self):
        if not sip.isdeleted(self):
            style = "color: white; font-weight: bold; font-size: 14px; background: transparent;"
            self.collapsed_label.setStyleSheet(style)
            self.expanded_label.setStyleSheet(style)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        # Match the control panel look: slightly lighter by default + subtle border.
        # Opacity setting still influences the alpha, but we clamp so bubbles don't get overly dark.
        opacity_alpha = int(self.opacity * 2.55)
        bg_alpha = max(80, min(170, opacity_alpha))

        radius = 10
        # Draw subtle shadow
        painter.setBrush(QColor(0, 0, 0, min(200, bg_alpha + 20)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect.translated(2, 2), radius, radius)

        # Draw background
        painter.setBrush(QColor(0, 0, 0, bg_alpha))
        painter.setPen(QColor(255, 255, 255, 60))
        painter.drawRoundedRect(rect, radius, radius)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.position().toPoint()
            self.press_pos = event.globalPosition().toPoint()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.deleteLater()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.dragging:
            new_pos = self.pos() + event.position().toPoint() - self.drag_start_pos
            self.move(new_pos)

            parent = self.parentWidget()
            if parent and hasattr(parent, 'update_mask_during_drag'):
                parent.update_mask_during_drag()

            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.dragging:
            # Check for click vs drag
            curr_pos = event.globalPosition().toPoint()
            if (curr_pos - self.press_pos).manhattanLength() < 5:
                # Toggle expansion and link visualization on click
                self.toggle_expansion()
                self.show_link = not self.show_link
                try:
                    parent = self.parentWidget()
                    if parent:
                        parent.update()  # trigger overlay repaint for link drawing
                except Exception:
                    pass
        self.dragging = False
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.close_btn.move(self.width() - 25, 5)


class TranslationOverlay(QObject):
    """Manager for TranslationBubble widgets using a full-screen container"""

    def __init__(self, parent_window=None):
        super().__init__(parent_window)
        self.bubbles = []
        self.parent_window = parent_window
        self.overlay_window = OverlayWindow()

        # Parent the control panel to the overlay window for unification.
        self.control_panel = OverlayControlPanel(self.overlay_window)

        # Persisted settings
        self.settings = QSettings("Xian", "VideoGameTranslator")
        self.opacity = int(self.settings.value("opacity", 80))
        self._supports_window_opacity = self._detect_window_opacity_support()

        self.control_panel.request_clear.connect(self.clear_translations)
        self.control_panel.request_hide_overlay.connect(self.hide_overlay_window)
        self.control_panel.request_show_overlay.connect(self.show_overlay_window)

        # Ensure overlay window follows main window lifecycle
        if parent_window:
            parent_window.destroyed.connect(self.overlay_window.deleteLater)
            parent_window.destroyed.connect(self.control_panel.deleteLater)

        # Apply initial opacity to all overlay pieces
        self.set_opacity(self.opacity)

        # Keep overlay above other windows by periodically re-raising.
        # This helps on Wayland/KWin when other windows steal the topmost layer.
        self._keep_on_top_timer = QTimer(self)
        self._keep_on_top_timer.setInterval(2000)
        self._keep_on_top_timer.timeout.connect(self._ensure_on_top)
        self._keep_on_top_timer.start()

    def _ensure_on_top(self):
        try:
            if self.overlay_window.isVisible():
                self.overlay_window.raise_()
            if self.control_panel.isVisible():
                self.control_panel.raise_()
        except Exception:
            pass

    def set_opacity(self, opacity: int):
        """Apply a unified opacity value across overlay window, control panel, and bubbles."""
        self.opacity = int(opacity)
        clamped = max(0, min(100, self.opacity))
        alpha = clamped / 100.0

        if self._supports_window_opacity:
            try:
                self.overlay_window.setWindowOpacity(alpha)
            except Exception:
                pass
            try:
                self.control_panel.setWindowOpacity(alpha)
            except Exception:
                pass

        try:
            self.control_panel.apply_opacity(clamped)
        except Exception:
            pass

        for bubble in self.bubbles:
            if not sip.isdeleted(bubble):
                bubble.opacity = self.opacity
                bubble.update()

    def _detect_window_opacity_support(self) -> bool:
        """Determine whether the current platform backend supports window opacity APIs."""
        try:
            platform_name = (QGuiApplication.platformName() or "").lower()
        except Exception:
            platform_name = ""

        platform_env = os.environ.get("QT_QPA_PLATFORM", "").lower()

        if "wayland" in platform_name or "wayland" in platform_env:
            return False
        return True

    def hide_overlay_window(self):
        """Hide only the overlay window (keep control panel visible)."""
        self.overlay_window.hide()
        try:
            self.control_panel.set_overlay_visible(False)
        except Exception:
            pass

    def show_overlay_window(self):
        """Show only the overlay window (control panel remains visible)."""
        self.overlay_window.show()
        self.overlay_window.raise_()
        try:
            self.control_panel.set_overlay_visible(True)
        except Exception:
            pass

    def hide(self):
        """Hide overlay and control panel (used when stopping translation)."""
        self.overlay_window.hide()
        self.control_panel.hide()

    def show(self):
        """Show overlay and control panel (used when starting translation)."""
        self.overlay_window.show()
        self.overlay_window.raise_()
        self.control_panel.show()
        try:
            self.control_panel.raise_()
        except Exception:
            pass
        try:
            self.control_panel.set_overlay_visible(True)
        except Exception:
            pass
        # Ensure input mask accounts for visible UI even before bubbles are drawn
        self._update_mask()

    def update_translations(self, translations: List[TranslationResult], updated_area: QRect = None):
        """Add new translations as bubbles with smart merging and grouping"""
        # Clean up any deleted objects first
        self.bubbles = [b for b in self.bubbles if not sip.isdeleted(b)]

        if not translations:
            # Do not auto-clear; keep existing bubbles until user clears them.
            logger.debug("No translations provided; preserving existing overlay contents")
            return

        # Limit the number of input translations to prevent O(N^2) hangs
        if len(translations) > 50:
            logger.warning(f"Too many translations received ({len(translations)}), limiting to 50")
            translations = translations[:50]

        logger.info(
            f"Updating overlay with {len(translations)} results" + (f" in area {updated_area}" if updated_area else ""))
        opacity = self.opacity

        # Track which bubbles were matched/created in this update
        matched_bubble_ids = set()

        # 1. Pre-process: Merge very close results from the API itself
        merged_results = []
        sorted_results = sorted(translations, key=lambda r: (r.y, r.x))

        for res in sorted_results:
            found_group = False
            for existing in merged_results:
                y_diff = abs(existing.y - res.y)
                x_diff = res.x - (existing.x + existing.width)

                if y_diff < 20:
                    if res.translated_text.strip() == existing.translated_text.strip() and abs(x_diff) < 50:
                        found_group = True
                        break

                    if -20 < x_diff < 40:
                        if res.translated_text.strip().lower() not in existing.translated_text.lower():
                            existing.translated_text += " " + res.translated_text

                        new_right = max(existing.x + existing.width, res.x + res.width)
                        existing.width = new_right - existing.x
                        existing.height = max(existing.height, res.height)
                        existing.y = min(existing.y, res.y)
                        found_group = True
                        break

            if not found_group:
                from dataclasses import replace
                merged_results.append(replace(res))

        # Optional: combine nearby lines into paragraph clusters for readability
        combine_mode = False
        default_expanded = False
        try:
            if self.control_panel and not sip.isdeleted(self.control_panel):
                combine_mode = self.control_panel.combine_check.isChecked()
                default_expanded = self.control_panel.show_full_check.isChecked()
        except Exception:
            pass

        clustered_results: List[TranslationResult]
        if combine_mode:
            # Cluster by vertical proximity and horizontal overlap
            from dataclasses import replace
            clusters = []  # each: dict(rect: QRect, items: List[TranslationResult])
            for res in merged_results:
                placed = False
                for cl in clusters:
                    rect: QRect = cl["rect"]
                    # proximity thresholds
                    vert_close = abs(res.y - (rect.y() + rect.height() / 2)) < 28 or (
                                rect.top() - 30 <= res.y <= rect.bottom() + 30)
                    # horizontal overlap check
                    res_rect = QRect(int(res.x), int(res.y), int(res.width), int(res.height))
                    overlap = rect.intersects(res_rect) or (
                                res_rect.left() <= rect.right() + 20 and res_rect.right() >= rect.left() - 20)
                    if vert_close and overlap:
                        # add to cluster
                        cl["items"].append(res)
                        rect = rect.united(res_rect)
                        cl["rect"] = rect
                        placed = True
                        break
                if not placed:
                    clusters.append(
                        {"rect": QRect(int(res.x), int(res.y), int(res.width), int(res.height)), "items": [res]})

            # For each cluster, order items top-to-bottom, left-to-right, and combine text
            clustered_results = []
            for cl in clusters:
                items = sorted(cl["items"], key=lambda r: (int(r.y), int(r.x)))
                rect = cl["rect"]
                texts = []
                for it in items:
                    t = (it.translated_text or "").strip()
                    if not t:
                        continue
                    texts.append(t)
                # Join with newlines to keep independence clear
                combined_text = "\n".join(texts)
                if not combined_text:
                    continue
                # Build a representative TranslationResult
                base = replace(items[0])
                base.translated_text = combined_text
                base.x = float(rect.x())
                base.y = float(rect.y())
                base.width = float(rect.width())
                base.height = float(rect.height())
                clustered_results.append(base)
            # Sort clusters
            clustered_results.sort(key=lambda r: (int(r.y), int(r.x)))
        else:
            clustered_results = merged_results

        # 2. Update existing bubbles or create new ones
        for result in clustered_results:
            best_match = None
            highest_score = 0.0
            append_below_target = None

            result_text_norm = result.translated_text.strip().lower()
            new_source_rect = QRect(int(result.x), int(result.y), int(result.width), int(result.height))

            for bubble in self.bubbles:
                if sip.isdeleted(bubble): continue

                score = 0.0
                ex = bubble.result
                ex_text_norm = ex.translated_text.strip().lower()
                ex_source_rect = QRect(int(ex.x), int(ex.y), int(ex.width), int(ex.height))

                # Detect "append beneath" case: new text appears just below existing bubble's source
                try:
                    vert_gap = new_source_rect.top() - ex_source_rect.bottom()
                    horiz_overlap = min(ex_source_rect.right(), new_source_rect.right()) - max(ex_source_rect.left(),
                                                                                               new_source_rect.left())
                    min_overlap = min(ex_source_rect.width(), new_source_rect.width()) * 0.3
                    if 0 <= vert_gap <= 36 and horiz_overlap >= min_overlap:
                        append_below_target = bubble
                except Exception:
                    pass

                iou = 0.0
                if ex_source_rect.intersects(new_source_rect):
                    inter = ex_source_rect.intersected(new_source_rect)
                    union = ex_source_rect.united(new_source_rect)
                    iou = (inter.width() * inter.height()) / (union.width() * union.height())

                if ex_text_norm == result_text_norm or ex_text_norm in result_text_norm or result_text_norm in ex_text_norm:
                    dist = abs(ex.x - result.x) + abs(ex.y - result.y)
                    if dist < 500:
                        score = 0.7 + (1.0 - min(1.0, dist / 500)) * 0.3

                score = max(score, iou)

                c_dist = (ex_source_rect.center() - new_source_rect.center()).manhattanLength()
                if c_dist < 100:
                    score = max(score, (1.0 - c_dist / 100) * 0.6)

                if score > highest_score:
                    highest_score = score
                    best_match = bubble

            # If we detected a likely line continuation beneath an existing bubble, append text
            if append_below_target and result.translated_text.strip():
                try:
                    from dataclasses import replace
                    base = replace(append_below_target.result)
                    # Append a new line with the new translated text
                    if base.translated_text.endswith("\n"):
                        base.translated_text = base.translated_text + result.translated_text.strip()
                    else:
                        base.translated_text = base.translated_text + "\n" + result.translated_text.strip()
                    # Expand source rect to include the new area below
                    union_rect = QRect(int(base.x), int(base.y), int(base.width), int(base.height)).united(
                        new_source_rect)
                    base.x = float(union_rect.x())
                    base.y = float(union_rect.y())
                    base.width = float(union_rect.width())
                    base.height = float(union_rect.height())
                    append_below_target.update_content(base)
                    matched_bubble_ids.add(id(append_below_target))
                    continue
                except Exception:
                    pass

            if best_match and highest_score > 0.4:
                try:
                    best_match.update_content(result)
                    matched_bubble_ids.add(id(best_match))
                    continue
                except (RuntimeError, AttributeError):
                    pass

            try:
                bubble = TranslationBubble(result, opacity, self.overlay_window, default_expanded=default_expanded)
                if not sip.isdeleted(bubble):
                    self.bubbles.append(bubble)
                    matched_bubble_ids.add(id(bubble))
                    bubble.destroyed.connect(self._remove_bubble)

                    if self.parent_window and not sip.isdeleted(self.parent_window):
                        if self.parent_window.hide_overlay_checkbox.isChecked():
                            bubble.hide()
                        else:
                            bubble.show()
                    else:
                        bubble.show()

                    try:
                        bubble.raise_()
                    except (RuntimeError, AttributeError):
                        pass
            except (RuntimeError, AttributeError) as e:
                logger.error(f"Failed to create or show bubble: {e}")
                continue

        # 3. If an updated_area was provided, remove unmatched bubbles in that area
        if updated_area:
            for bubble in self.bubbles[:]:
                if sip.isdeleted(bubble): continue
                if id(bubble) not in matched_bubble_ids:
                    # Check if bubble's original text area is within the updated_area
                    r = bubble.result
                    bubble_source_rect = QRect(int(r.x), int(r.y), int(r.width), int(r.height))

                    # If the bubble's source area overlaps significantly with the updated area, remove it.
                    # We use intersection or center check. Intersection is safer.
                    # We also add a small margin to the updated_area to handle floating point issues or minor shifts.
                    margin_area = updated_area.adjusted(-5, -5, 5, 5)
                    if margin_area.intersects(bubble_source_rect) or margin_area.contains(bubble_source_rect.center()):
                        bubble.close()

        # 4. Limit total number of bubbles to prevent performance issues/crashes
        MAX_BUBBLES = 50
        if len(self.bubbles) > MAX_BUBBLES:
            # Sort by age (oldest first - bubbles are appended, so early ones are older)
            # Actually, bubbles might be updated, but new ones are at the end.
            # Let's just remove the oldest ones that weren't just matched.
            num_to_remove = len(self.bubbles) - MAX_BUBBLES
            removed_count = 0
            for i in range(len(self.bubbles)):
                bubble = self.bubbles[i]
                if id(bubble) not in matched_bubble_ids:
                    bubble.close()
                    removed_count += 1
                    if removed_count >= num_to_remove:
                        break

        # 5. Resolve overlaps between bubbles and the control panel so text stays readable
        self._resolve_overlaps()

        self._update_mask()
        try:
            self.control_panel.set_stats(len(self.bubbles))
        except Exception:
            pass

    def _update_mask(self):
        """Update overlay window mask to allow click-through outside bubbles and control panel"""
        if sip.isdeleted(self.overlay_window):
            return

        mask = QRegion()
        for bubble in self.bubbles:
            if not sip.isdeleted(bubble) and bubble.isVisible():
                # We use the bubble's geometry which is relative to the overlay_window
                mask += bubble.geometry()

        # Include control panel in mask
        if not sip.isdeleted(self.control_panel) and self.control_panel.isVisible():
            mask += self.control_panel.geometry()

        self.overlay_window.setMask(mask)
        self.overlay_window.update()

    def _resolve_overlaps(self, margin: int = 8):
        """Nudge bubbles so they don't overlap each other or the control panel.

        Operates in the overlay's coordinate space; bubbles/control panel are children
        of the overlay window, so their geometries are already relative to it.
        """
        if sip.isdeleted(self.overlay_window):
            return

        work_area = self.overlay_window.rect()
        placed: List[QRect] = []

        # Treat the control panel as an obstacle if it's visible
        try:
            if not sip.isdeleted(self.control_panel) and self.control_panel.isVisible():
                placed.append(self.control_panel.geometry())
        except Exception:
            pass

        bubbles = [b for b in self.bubbles if not sip.isdeleted(b) and b.isVisible()]
        bubbles.sort(key=lambda b: (b.y(), b.x()))

        for bubble in bubbles:
            rect = bubble.geometry()
            attempt = 0

            # Iteratively nudge down/right to avoid intersections
            while attempt < 80:
                collision = False
                for obstacle in placed:
                    if rect.intersects(obstacle):
                        # Move bubble just below the obstacle with a margin
                        rect.moveTop(obstacle.bottom() + margin)
                        collision = True
                        break

                # Wrap if we ran off the bottom; shift right and reset to top margin
                if rect.bottom() > work_area.bottom():
                    rect.moveTop(work_area.top() + margin)
                    rect.moveLeft(rect.left() + rect.width() + margin)
                    collision = True

                # Clamp horizontally inside the work area
                if rect.right() > work_area.right():
                    rect.moveLeft(max(work_area.left() + margin, work_area.right() - rect.width() - margin))
                if rect.left() < work_area.left():
                    rect.moveLeft(work_area.left() + margin)

                if not collision:
                    break

                attempt += 1

            # Final clamp to ensure on-screen
            rect.moveLeft(max(work_area.left() + margin, min(rect.left(), work_area.right() - rect.width() - margin)))
            rect.moveTop(max(work_area.top() + margin, min(rect.top(), work_area.bottom() - rect.height() - margin)))

            try:
                bubble.move(rect.topLeft())
            except Exception:
                continue

            placed.append(rect)

    def _remove_bubble(self, qobj):
        """Handle bubble destruction safely"""
        for bubble in self.bubbles[:]:
            if bubble is qobj or sip.isdeleted(bubble):
                try:
                    self.bubbles.remove(bubble)
                except (ValueError, RuntimeError):
                    pass
        self._update_mask()

    def clear_translations(self):
        """Clear all active translation bubbles"""
        logger.info("Clearing all translations")
        to_close = [b for b in self.bubbles if not sip.isdeleted(b)]
        self.bubbles = []
        for bubble in to_close:
            try:
                bubble.close()
            except:
                pass
        self._update_mask()
        try:
            self.control_panel.set_stats(0)
        except Exception:
            pass

    def get_redaction_geometries(self) -> List[QRect]:
        """Return geometries to redact from the capture (bubbles + control panel).
        Returns global screen coordinates.
        """
        active_geoms = self.get_bubble_geometries()

        try:
            if not sip.isdeleted(self.control_panel) and self.control_panel.isVisible():
                # Map control panel geometry to global coordinates
                top_left = self.control_panel.mapToGlobal(QPoint(0, 0))
                active_geoms.append(QRect(top_left, self.control_panel.size()))
        except Exception as e:
            logger.debug(f"Failed to get control panel geometry for redaction: {e}")

        return active_geoms

    def get_bubble_geometries(self) -> List[QRect]:
        """Return list of current bubble geometries and original source geometries for redaction.
        Returns global screen coordinates.
        """
        active_geoms = []
        for b in self.bubbles:
            if not sip.isdeleted(b):
                try:
                    # 1. Current bubble geometry (global coords)
                    top_left = b.mapToGlobal(QPoint(0, 0))
                    active_geoms.append(QRect(top_left, b.size()))

                    # 2. Original source text geometry
                    # These were already stored in image-relative coords during OCR.
                    # We need to translate them to global coords.
                    # Assuming the capture was the full virtual desktop:
                    r = b.result

                    # We'll use the overlay_window's origin to map back to global if needed,
                    # but actually r.x/y are relative to the capture.
                    # If we assume capture was at total_geo.topLeft(), then:
                    if not sip.isdeleted(self.overlay_window):
                        origin = self.overlay_window.geometry().topLeft()
                        active_geoms.append(QRect(
                            int(r.x + origin.x()),
                            int(r.y + origin.y()),
                            int(r.width),
                            int(r.height)
                        ))
                except (RuntimeError, AttributeError):
                    pass

        self.bubbles = [b for b in self.bubbles if not sip.isdeleted(b)]
        return active_geoms
