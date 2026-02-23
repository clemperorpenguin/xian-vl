import json
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QCheckBox, QGroupBox, QSlider, QTabWidget,
    QListWidget, QListWidgetItem, QFormLayout, QLineEdit, QSystemTrayIcon, QMenu,
    QApplication
)
from PyQt6.QtCore import Qt, QTimer, QRect, QSettings, pyqtSlot
from PyQt6.QtGui import QIcon, QShortcut, QKeySequence

from .qwen_pipeline import VLProcessor, VLConfig
from .qwen_translation_workers import QwenTranslationWorker, QwenTranslatorStatusWorker, QwenModelWarmupWorker
from .overlay_ui import TranslationOverlay
from .region_selector import RegionSelector
from .models import TranslationMode, TranslationRegion

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("xian.png"))
        self.qwen_processor = VLProcessor()
        self.translation_worker = QwenTranslationWorker(self.qwen_processor)
        self.translator_status_worker = QwenTranslatorStatusWorker(self.qwen_processor)
        self.model_warmup_worker = QwenModelWarmupWorker(self.qwen_processor)
        self.translation_overlay = TranslationOverlay(self)
        self.region_selector = None
        self.regions = []
        self.settings = QSettings("Xian", "VideoGameTranslator")

        # Debounce timer for API status checks
        self.api_check_timer = QTimer()
        self.api_check_timer.setSingleShot(True)
        self.api_check_timer.setInterval(1000)  # 1 second debounce
        self.api_check_timer.timeout.connect(self._do_api_status_check)

        self.setup_ui()
        self.setup_tray_icon()
        self.connect_signals()
        self.load_settings()

        # Launch directly into the overlay control panel settings and keep the legacy window hidden by default.
        self.show_overlay_settings_panel()
        self.hide()

        # Check API availability
        self.check_api_status()

        self._pending_translation_start = None

    def setup_ui(self):
        self.setWindowTitle("Xian - Unified Translation Interface")
        self.setMinimumSize(600, 500)
        self.resize(650, 550)
        
        # Modern dark theme for the main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget#CentralWidget {
                background-color: #121212;
            }
            QGroupBox {
                color: #4CAF50;
                font-weight: bold;
                border: 1px solid #333;
                border-radius: 8px;
                margin-top: 1.5ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #333;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                border: 1px solid #444;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton#StartBtn {
                background-color: #2e7d32;
                font-weight: bold;
            }
            QPushButton#StopBtn {
                background-color: #c62828;
                font-weight: bold;
            }
            QComboBox, QSpinBox, QLineEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                background: #121212;
            }
            QTabBar::tab {
                background: #1e1e1e;
                color: #888;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #333;
                color: white;
            }
            QListWidget {
                background-color: #1e1e1e;
                color: #ccc;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)

        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QHBoxLayout()
        title_label = QLabel("Xian")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #4CAF50;")
        header.addWidget(title_label)
        header.addStretch()
        
        self.header_status = QLabel("Ready")
        self.header_status.setStyleSheet("color: #888;")
        header.addWidget(self.header_status)
        layout.addLayout(header)

        # Create tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # General tab
        general_tab = self._create_general_tab()
        self.tabs.addTab(general_tab, "General")

        # Regions tab
        regions_tab = self._create_regions_tab()
        self.tabs.addTab(regions_tab, "Regions")

        # Settings tab
        settings_tab = self._create_settings_tab()
        self.tabs.addTab(settings_tab, "Advanced")

        # Control buttons
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Translation")
        self.start_button.setObjectName("StartBtn")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.setMinimumHeight(40)
        
        self.stop_button = QPushButton("Stop Translation")
        self.stop_button.setObjectName("StopBtn")
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        
        layout.addLayout(controls_layout)

    def setup_tray_icon(self):
        """Initialize system tray icon and menu"""
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(QIcon("xian.png"))
        
        tray_menu = QMenu()
        
        self.tray_show_action = tray_menu.addAction("Show Overlay Settings")
        self.tray_show_action.triggered.connect(self.show_overlay_settings_panel)
        
        tray_menu.addSeparator()
        
        self.tray_toggle_action = tray_menu.addAction("Start Translation")
        self.tray_toggle_action.triggered.connect(self.toggle_translation)
        
        self.tray_clear_action = tray_menu.addAction("Clear Translations")
        self.tray_clear_action.triggered.connect(self.clear_all_translations)
        
        tray_menu.addSeparator()
        
        self.tray_quit_action = tray_menu.addAction("Quit")
        self.tray_quit_action.triggered.connect(QApplication.instance().quit)
        
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_overlay_settings_panel()

    def show_and_activate(self):
        # Legacy entry point retained for compatibility; prefer overlay panel.
        self.show_overlay_settings_panel()
        self.hide()

    def show_overlay_settings_panel(self):
        """Bring the overlay control panel to the front and open Settings view."""
        try:
            self.translation_overlay.show()
            self.translation_overlay.control_panel.show_settings_view()
            self.translation_overlay.control_panel.raise_()
        except Exception:
            pass

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Translation mode
        mode_group = QGroupBox("Translation Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.full_screen_radio = QCheckBox("Full Screen Analysis")
        self.region_select_radio = QCheckBox("Region Selection")
        self.full_screen_radio.setChecked(True)

        mode_layout.addWidget(self.full_screen_radio)
        mode_layout.addWidget(self.region_select_radio)

        layout.addWidget(mode_group)

        # Language settings
        lang_group = QGroupBox("Languages")
        lang_layout = QFormLayout(lang_group)

        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["auto", "Japanese", "Korean", "Chinese", "Spanish", "French"])

        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["English", "Japanese", "Korean", "Chinese", "Spanish", "French"])

        lang_layout.addRow("Source Language:", self.source_lang_combo)
        lang_layout.addRow("Target Language:", self.target_lang_combo)

        layout.addWidget(lang_group)

        # Timing settings
        timing_group = QGroupBox("Timing")
        timing_layout = QFormLayout(timing_group)

        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(500, 10000)
        self.interval_spinbox.setValue(2000)
        self.interval_spinbox.setSuffix(" ms")

        timing_layout.addRow("Update Interval:", self.interval_spinbox)

        layout.addWidget(timing_group)

        # UI behavior settings
        ui_group = QGroupBox("UI Behavior")
        ui_layout = QVBoxLayout(ui_group)
        self.minimize_on_start_checkbox = QCheckBox("Minimize to Tray on Start")
        self.minimize_on_start_checkbox.setChecked(True)
        ui_layout.addWidget(self.minimize_on_start_checkbox)
        layout.addWidget(ui_group)

        # Clear translations button
        self.clear_translations_button = QPushButton("Clear All Translations (Ctrl+L)")
        self.clear_translations_button.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
        layout.addWidget(self.clear_translations_button)

        self.hide_overlay_checkbox = QCheckBox("Hide All Translations (Ctrl+H)")
        self.hide_overlay_checkbox.setToolTip("Temporarily hide all translation bubbles from the screen")
        layout.addWidget(self.hide_overlay_checkbox)

        # Log area
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_list = QListWidget()
        self.log_list.setMinimumHeight(150)
        self.log_list.setStyleSheet("font-size: 11px; color: #333;")
        log_layout.addWidget(self.log_list)
        layout.addWidget(log_group)

        layout.addStretch()
        return widget

    def _create_regions_tab(self) -> QWidget:
        """Create region management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Region list
        self.regions_list = QListWidget()
        layout.addWidget(QLabel("Translation Regions:"))
        layout.addWidget(self.regions_list)

        # Region controls
        controls_layout = QHBoxLayout()

        self.add_region_button = QPushButton("Add Region")
        self.remove_region_button = QPushButton("Remove Region")
        self.test_region_button = QPushButton("Test Region")

        controls_layout.addWidget(self.add_region_button)
        controls_layout.addWidget(self.remove_region_button)
        controls_layout.addWidget(self.test_region_button)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        return widget

    def _create_settings_tab(self) -> QWidget:
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Translator settings
        translator_group = QGroupBox("Translator Settings")
        translator_layout = QFormLayout(translator_group)

        self.api_model_edit = QComboBox()
        self.api_model_edit.setEditable(True)
        # Suggested defaults (editable) - Updated for both Qwen3-VL and TranslateGemma
        self.api_model_edit.addItem("Qwen3-VL-8B-Thinking (Auto-select)")
        self.api_model_edit.addItem("Qwen3-VL-4B-Thinking")
        self.api_model_edit.addItem("Qwen3-VL-8B-Thinking")
        self.api_model_edit.addItem("Qwen3-VL-4B-Instruct")
        self.api_model_edit.addItem("Qwen3-VL-8B-Instruct")
        self.api_model_edit.addItem("TranslateGemma-12B (High Quality)")
        self.api_model_edit.addItem("TranslateGemma-4B (Lower Resource)")
        self.api_model_edit.setToolTip("Vision-Language model (Qwen3-VL or TranslateGemma)")
        self.api_status_label = QLabel("Checking...")

        translator_layout.addRow("Model:", self.api_model_edit)
        translator_layout.addRow("Status:", self.api_status_label)

        layout.addWidget(translator_group)

        # Qwen3-VL specific settings
        qwen_group = QGroupBox("Qwen3-VL Settings")
        qwen_layout = QFormLayout(qwen_group)

        # Thinking mode toggle
        self.thinking_mode_checkbox = QCheckBox("Enable Thinking Mode (Chain of Thought)")
        self.thinking_mode_checkbox.setToolTip("Enable Chain of Thought reasoning for complex layouts (slower but more accurate)")
        qwen_layout.addRow(self.thinking_mode_checkbox)

        # Max tokens slider
        self.max_tokens_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_tokens_slider.setRange(256, 4096)
        self.max_tokens_slider.setValue(1024)
        self.max_tokens_slider.setToolTip("Maximum tokens for the model response")
        max_tokens_label = QLabel(f"{self.max_tokens_slider.value()}")
        self.max_tokens_slider.valueChanged.connect(lambda v: max_tokens_label.setText(str(v)))
        max_tokens_hbox = QHBoxLayout()
        max_tokens_hbox.addWidget(self.max_tokens_slider)
        max_tokens_hbox.addWidget(max_tokens_label)
        qwen_layout.addRow("Max Tokens:", max_tokens_hbox)

        # Model size override
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["Auto-detect", "4B", "8B"])
        self.model_size_combo.setToolTip("Override automatic model size selection based on VRAM")
        qwen_layout.addRow("Model Size Override:", self.model_size_combo)

        layout.addWidget(qwen_group)

        # Overlay settings
        overlay_group = QGroupBox("Overlay")
        overlay_layout = QFormLayout(overlay_group)

        self.overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity_slider.setRange(50, 100)
        self.overlay_opacity_slider.setValue(80)

        overlay_layout.addRow("Opacity:", self.overlay_opacity_slider)

        self.debug_mode_checkbox = QCheckBox("Enable Debug Mode")
        overlay_layout.addRow(self.debug_mode_checkbox)

        self.redaction_margin_spin = QSpinBox()
        self.redaction_margin_spin.setRange(0, 100)
        self.redaction_margin_spin.setValue(15)
        self.redaction_margin_spin.setSuffix(" px")
        self.redaction_margin_spin.setToolTip("Margin for redacting existing translations.")
        overlay_layout.addRow("Redaction Margin:", self.redaction_margin_spin)

        layout.addWidget(overlay_group)

        # Clear settings button
        self.reset_button = QPushButton("Reset All Settings")
        self.reset_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        layout.addWidget(self.reset_button)

        layout.addStretch()
        return widget

    def connect_signals(self):
        """Connect UI signals"""
        self.start_button.clicked.connect(self.start_translation)
        self.stop_button.clicked.connect(self.stop_translation)
        self.add_region_button.clicked.connect(self.add_region)
        self.remove_region_button.clicked.connect(self.remove_region)
        self.test_region_button.clicked.connect(self.test_region)
        self.reset_button.clicked.connect(self.reset_settings)
        self.clear_translations_button.clicked.connect(self.clear_all_translations)
        self.hide_overlay_checkbox.toggled.connect(self.toggle_overlay_visibility)

        # Shortcuts
        self.clear_shortcut = QShortcut(QKeySequence("Ctrl+L"), self)
        self.clear_shortcut.activated.connect(self.clear_all_translations)
        
        self.hide_shortcut = QShortcut(QKeySequence("Ctrl+H"), self)
        self.hide_shortcut.activated.connect(lambda: self.hide_overlay_checkbox.setChecked(not self.hide_overlay_checkbox.isChecked()))

        self.start_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.start_shortcut.activated.connect(self.start_translation)
        
        self.stop_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self.stop_shortcut.activated.connect(self.stop_translation)

        self.full_screen_radio.toggled.connect(self.on_mode_changed)
        self.region_select_radio.toggled.connect(self.on_mode_changed)

        self.api_model_edit.editTextChanged.connect(self.check_api_status)
        # Keep overlay panel and translator in sync when main model changes
        self.api_model_edit.currentTextChanged.connect(self._on_main_model_changed)

        # Keep Source/Target language in sync (Main → Overlay panel)
        try:
            self.source_lang_combo.currentTextChanged.connect(self._on_main_source_changed)
            self.target_lang_combo.currentTextChanged.connect(self._on_main_target_changed)
        except Exception:
            # In case widgets are not yet available in some init paths
            pass

        self.debug_mode_checkbox.toggled.connect(self.save_settings)
        self.overlay_opacity_slider.valueChanged.connect(self.save_settings)
        self.redaction_margin_spin.valueChanged.connect(self.save_settings)
        self.minimize_on_start_checkbox.toggled.connect(self.save_settings)
        
        # Connect Qwen3-VL specific settings
        self.thinking_mode_checkbox.toggled.connect(self._on_thinking_mode_changed)
        self.max_tokens_slider.valueChanged.connect(self._on_max_tokens_changed)
        self.model_size_combo.currentTextChanged.connect(self._on_model_size_changed)

        self.translator_status_worker.status_changed.connect(self._on_api_status_changed)

        self.translation_worker.status_update.connect(
            self.header_status.setText
        )
        self.translation_worker.status_update.connect(
            self.add_log
        )
        self.translation_worker.translation_ready.connect(
            self.translation_overlay.update_translations
        )
        self.translation_worker.request_hide_overlay.connect(
            self.on_translation_worker_capture_prepare
        )
        self.translation_worker.request_show_overlay.connect(
            self.translation_overlay.show
        )

        self.translation_overlay.control_panel.request_start.connect(self.start_translation)
        self.translation_overlay.control_panel.request_stop.connect(self.stop_translation)
        self.translation_overlay.control_panel.request_reset_settings.connect(self.reset_settings)
        self.translation_overlay.control_panel.settings_changed.connect(self._sync_settings_from_panel)

        self.model_warmup_worker.warmup_finished.connect(self._on_model_warmup_finished)

    def _on_thinking_mode_changed(self, checked: bool):
        """Handle thinking mode toggle change"""
        self.qwen_processor.config.thinking_mode = checked
        self.save_settings()

    def _on_max_tokens_changed(self, value: int):
        """Handle max tokens slider change"""
        self.qwen_processor.config.max_tokens = value
        self.save_settings()

    def _on_model_size_changed(self, text: str):
        """Handle model size combo change"""
        if text == "4B":
            self.qwen_processor.config.model_size = "4b"
        elif text == "8B":
            self.qwen_processor.config.model_size = "8b"
        else:  # Auto-detect
            self.qwen_processor.config.model_size = "auto"
        self.save_settings()

    def _on_main_model_changed(self, text: str):
        """Propagate model selection from main settings -> overlay panel + translator, and persist."""
        try:
            panel = self.translation_overlay.control_panel
            panel.blockSignals(True)
            panel.model_combo.setCurrentText(text)
        except Exception:
            pass
        finally:
            try:
                panel.blockSignals(False)
            except Exception:
                pass

        # Update processor and save
        # For QwenVLProcessor, model selection is handled differently
        self.save_settings()

    def _on_main_source_changed(self, text: str):
        """Mirror main Source language selection to overlay panel and persist."""
        try:
            panel = self.translation_overlay.control_panel
            panel.blockSignals(True)
            panel.source_lang_combo.setCurrentText(text)
        except Exception:
            pass
        finally:
            try:
                panel.blockSignals(False)
            except Exception:
                pass
        # Save settings
        self.save_settings()

    def _on_main_target_changed(self, text: str):
        """Mirror main Target language selection to overlay panel and persist."""
        try:
            panel = self.translation_overlay.control_panel
            panel.blockSignals(True)
            panel.target_lang_combo.setCurrentText(text)
        except Exception:
            pass
        finally:
            try:
                panel.blockSignals(False)
            except Exception:
                pass
        # Save settings
        self.save_settings()

    def _sync_settings_from_panel(self):
        """Update worker and internal state when settings are changed in the overlay panel.
        IMPORTANT: Do NOT call load_settings() here, as it overwrites the panel's current
        selections with persisted values and makes it impossible to change models from the panel.
        """
        panel = self.translation_overlay.control_panel

        # Determine mode
        mode = TranslationMode.FULL_SCREEN if panel.mode_combo.currentText() == "Full Screen" else TranslationMode.REGION_SELECT

        # Push panel settings to running worker (if active)
        if self.translation_worker.running:
            self.translation_worker.set_config(
                mode=mode,
                regions=self.regions,
                source_lang=panel.source_lang_combo.currentText(),
                target_lang=panel.target_lang_combo.currentText(),
                interval=panel.interval_spin.value(),
                redaction_margin=panel.margin_spin.value(),
            )

        # Sync model selection from panel -> main + processor
        new_model = panel.model_combo.currentText()
        if new_model:
            try:
                self.api_model_edit.blockSignals(True)
                self.api_model_edit.setCurrentText(new_model)
            finally:
                self.api_model_edit.blockSignals(False)
            # For QwenVLProcessor, model selection is handled differently

        # Apply unified opacity instantly across overlay components
        self.translation_overlay.set_opacity(panel.opacity_slider.value())

        # Mirror other settings from panel -> main widgets
        try:
            self.source_lang_combo.blockSignals(True)
            self.target_lang_combo.blockSignals(True)
            self.interval_spinbox.blockSignals(True)
            self.overlay_opacity_slider.blockSignals(True)
            self.redaction_margin_spin.blockSignals(True)

            self.source_lang_combo.setCurrentText(panel.source_lang_combo.currentText())
            self.target_lang_combo.setCurrentText(panel.target_lang_combo.currentText())
            self.interval_spinbox.setValue(panel.interval_spin.value())
            self.overlay_opacity_slider.setValue(panel.opacity_slider.value())
            self.redaction_margin_spin.setValue(panel.margin_spin.value())
        finally:
            try:
                self.source_lang_combo.blockSignals(False)
                self.target_lang_combo.blockSignals(False)
                self.interval_spinbox.blockSignals(False)
                self.overlay_opacity_slider.blockSignals(False)
                self.redaction_margin_spin.blockSignals(False)
            except Exception:
                pass

        # Persist new settings
        self.save_settings()

        # Optionally refresh API status to reflect new model selection
        self.check_api_status()

    def add_log(self, message):
        """Add message to activity log"""
        self.log_list.insertItem(0, message)
        if self.log_list.count() > 50:
            self.log_list.takeItem(self.log_list.count() - 1)

    def on_translation_worker_capture_prepare(self):
        """Handle worker preparing for capture: update geometries (used to hide)"""
        # Update worker with latest bubble geometries for redaction
        geoms = self.translation_overlay.get_redaction_geometries()
        self.translation_worker.set_active_geometries(geoms)
        
        # We NO LONGER hide the overlay to prevent flickering/visibility issues
        # Redaction handles removing existing bubbles from the screenshot.

    def check_api_status(self):
        """Start the API status check process with debouncing"""
        self.api_status_label.setText("Checking...")
        self.api_status_label.setStyleSheet("color: #888")
        self.api_check_timer.start()

    def _do_api_status_check(self):
        """Perform the actual Qwen processor status check in a background thread"""
        # For QwenVLProcessor, model selection is handled differently
        # We'll pass the model size setting to the processor
        model_text = self.api_model_edit.currentText()
        
        if self.translator_status_worker.isRunning():
            self.translator_status_worker.terminate()
            self.translator_status_worker.wait()

        self.translator_status_worker.start()

    def _on_api_status_changed(self, is_available: bool, models: list):
        """Handle the result of the API status check"""
        if is_available:
            self.api_status_label.setText("✓ Connected")
            self.api_status_label.setStyleSheet("color: #4CAF50")
            
            # Update models list if we got any
            if models:
                current_model = self.api_model_edit.currentText()
                self.api_model_edit.blockSignals(True)
                self.api_model_edit.clear()
                self.api_model_edit.addItems(models)
                self.api_model_edit.setCurrentText(current_model)
                self.api_model_edit.blockSignals(False)
        else:
            self.api_status_label.setText("✗ Disconnected")
            self.api_status_label.setStyleSheet("color: #f44336")

    def toggle_overlay_visibility(self, visible):
        """Toggle translation overlay visibility"""
        if visible:
            self.translation_overlay.hide_overlay_window()
            self.header_status.setText("Translations Hidden")
        else:
            self.translation_overlay.show_overlay_window()
            self.header_status.setText("Translations Visible")

    def on_mode_changed(self):
        """Handle translation mode change"""
        sender = self.sender()
        if not sender.isChecked():
            # If everything is unchecked, re-check full screen
            if not self.full_screen_radio.isChecked() and \
               not self.region_select_radio.isChecked():
                self.full_screen_radio.setChecked(True)
            return

        # Ensure only one mode is selected
        if sender == self.full_screen_radio:
            self.region_select_radio.setChecked(False)
        elif sender == self.region_select_radio:
            self.full_screen_radio.setChecked(False)
        
        self.save_settings()

    def add_region(self):
        """Add new translation region"""
        self.region_selector = RegionSelector()
        self.region_selector.region_selected.connect(self.on_region_selected)
        self.region_selector.show()

    def on_region_selected(self, rect: QRect):
        """Handle new region selection"""
        region = TranslationRegion(
            rect.x(), rect.y(), rect.width(), rect.height(),
            f"Region {len(self.regions) + 1}"
        )
        self.regions.append(region)
        self.update_regions_list()

    def remove_region(self):
        """Remove selected region"""
        current_row = self.regions_list.currentRow()
        if 0 <= current_row < len(self.regions):
            del self.regions[current_row]
            self.update_regions_list()

    def test_region(self):
        """Test translation on selected region"""
        current_row = self.regions_list.currentRow()
        if 0 <= current_row < len(self.regions):
            region = self.regions[current_row]
            # TODO: Implement region testing
            logger.info(f"Testing region: {region.name}")

    def update_regions_list(self):
        """Update regions list display"""
        self.regions_list.clear()
        for region in self.regions:
            item_text = f"{region.name} ({region.x}, {region.y}, {region.width}x{region.height})"
            item = QListWidgetItem(item_text)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if region.enabled else Qt.CheckState.Unchecked)
            self.regions_list.addItem(item)

    def clear_all_translations(self):
        """Clear all translations and reset hashes to force new analysis"""
        self.translation_overlay.clear_translations()
        self.translation_worker.clear_hashes()
        self.header_status.setText("Translations Cleared")

    def toggle_translation(self):
        """Toggle translation process start/stop"""
        if self.translation_worker.running:
            self.stop_translation()
        else:
            self.start_translation()

    def start_translation(self):
        """Start translation process"""
        # For QwenVLProcessor, model selection is handled differently
        model_selection = self.translation_overlay.control_panel.model_combo.currentText()

        if self.model_warmup_worker.isRunning():
            self.header_status.setText("Model is still loading...")
            self.translation_overlay.control_panel.status_label.setText("Model loading...")
            return

        # Configure worker (deferred until model warmup completes)
        if self.full_screen_radio.isChecked():
            mode = TranslationMode.FULL_SCREEN
        else:
            mode = TranslationMode.REGION_SELECT

        # Source/Target are taken from the overlay panel; ensure they mirror main settings
        src_lang = self.translation_overlay.control_panel.source_lang_combo.currentText()
        tgt_lang = self.translation_overlay.control_panel.target_lang_combo.currentText()

        # Log current selections for diagnostics
        try:
            logger.debug(
                "Starting translation with model=%s, source=%s, target=%s",
                model_selection, src_lang, tgt_lang
            )
        except Exception:
            pass

        # For VLProcessor, we don't need hint_source_lang/hint_target_lang
        # Set the model in the processor based on the selection
        self.qwen_processor.config.model_name = model_selection
        
        # Set model size for Qwen models, or mark as TranslateGemma
        if "translategemma" in model_selection.lower():
            # For TranslateGemma, we don't use model_size, just the model name
            pass
        elif "4b" in model_selection.lower():
            self.qwen_processor.config.model_size = "4b"
        elif "8b" in model_selection.lower():
            self.qwen_processor.config.model_size = "8b"
        else:
            self.qwen_processor.config.model_size = "auto"

        # Update thinking mode based on settings
        self.qwen_processor.config.thinking_mode = "thinking" in model_selection.lower()

        self._pending_translation_start = {
            "mode": mode,
            "regions": self.regions,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "interval": self.translation_overlay.control_panel.interval_spin.value(),
            "redaction_margin": self.translation_overlay.control_panel.margin_spin.value(),
            "minimize": self.minimize_on_start_checkbox.isChecked(),
        }

        # Warmup model before starting OCR/translation loop
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.header_status.setText("Loading Qwen3-VL model...")
        self.translation_overlay.control_panel.status_label.setText("Loading model...")
        self.model_warmup_worker.start()

    def _on_model_warmup_finished(self, ok: bool, error: str):
        if not ok:
            self._pending_translation_start = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.translation_overlay.control_panel.set_running(False)
            self.tray_toggle_action.setText("Start Translation")
            detailed = error or "Qwen3-VL model failed to load"
            msg = f"Model load failed: {detailed}" if detailed else "Model load failed"
            self.header_status.setText(msg)
            self.translation_overlay.control_panel.status_label.setText("Load failed")
            return

        cfg = self._pending_translation_start
        self._pending_translation_start = None
        if not cfg:
            # Start was cancelled
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.translation_overlay.control_panel.set_running(False)
            self.tray_toggle_action.setText("Start Translation")
            self.header_status.setText("Ready")
            return

        self.translation_worker.set_config(
            mode=cfg["mode"],
            regions=cfg["regions"],
            source_lang=cfg["source_lang"],
            target_lang=cfg["target_lang"],
            interval=cfg["interval"],
            redaction_margin=cfg["redaction_margin"],
        )

        self.translation_worker.clear_hashes()
        self.translation_worker.start_translation()
        self.translation_overlay.show()
        self.translation_overlay.control_panel.set_running(True)
        self.translation_overlay.control_panel.status_label.setText("Translating")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.tray_toggle_action.setText("Stop Translation")
        self.header_status.setText("Translating...")

        if cfg.get("minimize"):
            self.hide()

    def stop_translation(self):
        """Stop translation process"""
        if self.model_warmup_worker.isRunning():
            # Best-effort cancel; avoids starting OCR while the model is still loading
            self._pending_translation_start = None
            self.model_warmup_worker.terminate()
            self.model_warmup_worker.wait(1000)

        self.translation_worker.stop_translation()
        self.translation_overlay.control_panel.set_running(False)

        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.tray_toggle_action.setText("Start Translation")
        self.header_status.setText("Ready")

    def reset_settings(self):
        """Reset all settings to default values"""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self, 'Reset Settings',
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.settings.clear()
            # Reload settings will fall back to defaults since they are cleared
            self.load_settings()
            # Additional UI cleanup that load_settings might not fully cover
            self.regions = []
            self.update_regions_list()
            self.check_api_status()
            self.header_status.setText("Settings Reset")

    def load_settings(self):
        """Load application settings"""
        self.api_model_edit.setCurrentText(self.settings.value("api_model", "Qwen3-VL-8B-Thinking (Auto-select)"))
        self.source_lang_combo.setCurrentText(self.settings.value("source_lang", "auto"))
        self.target_lang_combo.setCurrentText(self.settings.value("target_lang", "English"))
        self.interval_spinbox.setValue(int(self.settings.value("interval", 2000)))
        self.overlay_opacity_slider.setValue(int(self.settings.value("opacity", 80)))
        self.redaction_margin_spin.setValue(int(self.settings.value("redaction_margin", 15)))
        self.debug_mode_checkbox.setChecked(self.settings.value("debug_mode", "false") == "true")
        self.minimize_on_start_checkbox.setChecked(self.settings.value("minimize_on_start", "true") == "true")

        # Load Qwen3-VL specific settings
        self.thinking_mode_checkbox.setChecked(self.settings.value("thinking_mode", "true") == "true")
        self.max_tokens_slider.setValue(int(self.settings.value("max_tokens", 1024)))
        self.model_size_combo.setCurrentText(self.settings.value("model_size_override", "Auto-detect"))

        # Load mode
        mode_str = self.settings.value("translation_mode", "full_screen")
        self.full_screen_radio.setChecked(mode_str == "full_screen")
        self.region_select_radio.setChecked(mode_str != "full_screen")

        # Sync with panel if it's already initialized
        try:
            panel = self.translation_overlay.control_panel
            panel.blockSignals(True)
            panel.source_lang_combo.setCurrentText(self.source_lang_combo.currentText())
            panel.target_lang_combo.setCurrentText(self.target_lang_combo.currentText())
            panel.model_combo.setCurrentText(self.api_model_edit.currentText())
            panel.interval_spin.setValue(self.interval_spinbox.value())
            panel.opacity_slider.setValue(self.overlay_opacity_slider.value())
            panel.margin_spin.setValue(self.redaction_margin_spin.value())
            panel.blockSignals(False)
        except Exception:
            pass

        # Apply opacity immediately to the live overlay components
        try:
            self.translation_overlay.set_opacity(self.overlay_opacity_slider.value())
        except Exception:
            pass

        # Sync VLProcessor object
        # For VLProcessor, model selection is handled differently
        model_text = self.api_model_edit.currentText()
        self.qwen_processor.config.model_name = model_text
        
        # Set model size for Qwen models, or mark as TranslateGemma
        if "translategemma" in model_text.lower():
            # For TranslateGemma, we don't use model_size, just the model name
            pass
        elif "4b" in model_text.lower():
            self.qwen_processor.config.model_size = "4b"
        elif "8b" in model_text.lower():
            self.qwen_processor.config.model_size = "8b"
        else:
            self.qwen_processor.config.model_size = "auto"

        # Update thinking mode based on settings
        self.qwen_processor.config.thinking_mode = "thinking" in model_text.lower() or self.thinking_mode_checkbox.isChecked()
        self.qwen_processor.config.max_tokens = self.max_tokens_slider.value()

        # Apply model size override if set
        override_size = self.model_size_combo.currentText()
        if override_size == "4B":
            self.qwen_processor.config.model_size = "4b"
        elif override_size == "8B":
            self.qwen_processor.config.model_size = "8b"

        # Load regions
        regions_json = self.settings.value("regions", "")
        if regions_json:
            try:
                regions_data = json.loads(regions_json)
                self.regions = [TranslationRegion(**r) for r in regions_data]
                self.update_regions_list()
            except Exception as e:
                logger.error(f"Error loading regions: {e}")

    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("api_model", self.api_model_edit.currentText())
        self.settings.setValue("source_lang", self.source_lang_combo.currentText())
        self.settings.setValue("target_lang", self.target_lang_combo.currentText())
        self.settings.setValue("interval", self.interval_spinbox.value())
        self.settings.setValue("opacity", self.overlay_opacity_slider.value())
        self.settings.setValue("redaction_margin", self.redaction_margin_spin.value())
        self.settings.setValue("debug_mode", "true" if self.debug_mode_checkbox.isChecked() else "false")
        self.settings.setValue("minimize_on_start", "true" if self.minimize_on_start_checkbox.isChecked() else "false")
        
        # Save mode
        if self.full_screen_radio.isChecked():
            mode_str = "full_screen"
        else:
            mode_str = "region_select"
        self.settings.setValue("translation_mode", mode_str)

        # Save Qwen3-VL specific settings
        self.settings.setValue("thinking_mode", "true" if self.thinking_mode_checkbox.isChecked() else "false")
        self.settings.setValue("max_tokens", self.max_tokens_slider.value())
        self.settings.setValue("model_size_override", self.model_size_combo.currentText())

        # Save regions
        regions_data = [
            {
                "x": r.x,
                "y": r.y,
                "width": r.width,
                "height": r.height,
                "name": r.name,
                "enabled": r.enabled
            }
            for r in self.regions
        ]
        self.settings.setValue("regions", json.dumps(regions_data))

    def closeEvent(self, event):
        """Handle application close"""
        self.stop_translation()
        self.save_settings()
        event.accept()
