"""Tray-resident application controller for Xian-VL.

Owns the system tray icon, hotkey listener, VLProcessor, and orchestrates
the Lens overlay and Chat sidebar.  No main window is shown — the tray
icon is the primary entry point.
"""

import logging
import os

from PyQt6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QDialog, QFormLayout,
    QLineEdit, QComboBox, QSpinBox, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QWidget, QCheckBox
)
from PyQt6.QtCore import Qt, QSettings, QRect, QTimer
from PyQt6.QtGui import QIcon, QAction

from mage.ui.theme import accent_hex, accent_hover_hex

from xian.pipeline import VLProcessor, VLConfig
from mage.workers import InferenceWorker, StatusWorker, ModelPullWorker
from mage.ui.lens import LensOverlayWindow
from mage.ui.chat_sidebar import ChatSidebar
from mage.ui.result_bubble import ResultBubble
from mage.capture.hotkeys import create_hotkey_listener
from xian.dictionary import LocalDictionary
from mage.ui.command_osd import CommandOSD
from shared_types import constants
from shared_types.enums import SourceLanguage, TargetLanguage, TranslationMode
from mage.settings_keys import *

logger = logging.getLogger(__name__)

ORGANIZATION = constants.ORGANIZATION_NAME
APP_NAME = constants.APPLICATION_NAME


class SettingsDialog(QDialog):
    """Small modal dialog for configuring the Lemonade backend."""

    def __init__(self, settings: QSettings, models: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Xian — Settings")
        self.setMinimumWidth(400)
        self.settings = settings

        layout = QFormLayout(self)

        # Server URL
        self.url_edit = QLineEdit()
        self.url_edit.setText(settings.value(KEY_API_URL, constants.DEFAULT_API_URL))
        layout.addRow("Server URL:", self.url_edit)

        # Model
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        if models:
            self.model_combo.addItems(models)
        self.model_combo.setCurrentText(settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL))
        layout.addRow("Model:", self.model_combo)

        # Source language
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems([e.value for e in SourceLanguage])
        self.source_lang_combo.setCurrentText(settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG))
        layout.addRow("Source Language:", self.source_lang_combo)

        # Target language
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([e.value for e in TargetLanguage])
        self.lang_combo.setCurrentText(settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG))
        layout.addRow("Target Language:", self.lang_combo)

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([e.value for e in TranslationMode])
        self.mode_combo.setCurrentText(settings.value(KEY_MODE, constants.DEFAULT_MODE))
        layout.addRow("Mode:", self.mode_combo)

        # Leader Key
        self.leader_combo = QComboBox()
        self.leader_combo.addItems(["Shift+Space", "Ctrl+Space", "Alt+Space", "Super+Space"])
        self.leader_combo.setCurrentText(settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY))
        layout.addRow("Leader Key:", self.leader_combo)

        # Styles
        style_layout = QVBoxLayout()
        self.style_checkboxes = {}
        saved_styles = settings.value(KEY_STYLES, constants.DEFAULT_STYLES)
        if isinstance(saved_styles, str):
            saved_styles = [s.strip() for s in saved_styles.split(",") if s.strip()]
        elif not isinstance(saved_styles, list):
            saved_styles = []
        for style in ["Romance", "Wuxia", "Visual Novel", "MMORPG"]:
            cb = QCheckBox(style)
            if style in saved_styles:
                cb.setChecked(True)
            self.style_checkboxes[style] = cb
            style_layout.addWidget(cb)
        layout.addRow("Styles:", style_layout)

        # Max tokens
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(256, 32768)
        self.tokens_spin.setValue(int(settings.value("max_tokens", constants.DEFAULT_MAX_TOKENS)))
        layout.addRow("Max Tokens:", self.tokens_spin)

        # GPU Memory Utilization
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["Default", "0.5", "0.75"])
        self.gpu_combo.setCurrentText(settings.value("gpu_memory_utilization", constants.DEFAULT_GPU_MEMORY_UTILIZATION))
        layout.addRow("GPU Memory Utilization:", self.gpu_combo)

        # Buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addRow(btn_row)

        self.setStyleSheet(f"""
            QDialog {{ background: #1e1e1e; color: #eee; }}
            QLabel, QCheckBox {{ color: #ccc; }}
            QLineEdit, QComboBox, QSpinBox {{
                background: #2a2a2a; color: #eee; border: 1px solid #555;
                border-radius: 4px; padding: 4px;
            }}
            QPushButton {{
                background: {accent_hex()}; color: white; border: none;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {accent_hover_hex()}; }}
        """)

    def _save(self):
        self.settings.setValue(KEY_API_URL, self.url_edit.text())
        self.settings.setValue(KEY_API_MODEL, self.model_combo.currentText())
        self.settings.setValue(KEY_SOURCE_LANG, self.source_lang_combo.currentText())
        self.settings.setValue(KEY_TARGET_LANG, self.lang_combo.currentText())
        self.settings.setValue(KEY_MODE, self.mode_combo.currentText())
        selected_styles = [s for s, cb in self.style_checkboxes.items() if cb.isChecked()]
        self.settings.setValue(KEY_STYLES, selected_styles)
        self.settings.setValue(KEY_MAX_TOKENS, self.tokens_spin.value())
        self.settings.setValue(KEY_LEADER_KEY, self.leader_combo.currentText())
        self.settings.setValue(KEY_GPU_UTIL, self.gpu_combo.currentText())
        self.accept()


class XianApp(QWidget):
    """Tray-resident application controller.

    Inherits QWidget so it participates in the Qt object tree (preventing
    garbage collection), but is never shown as a window.
    """

    def __init__(self):
        super().__init__()
        self.hide()  # never shown

        self.settings = QSettings(ORGANIZATION, APP_NAME)
        self._available_models: list = []

        # --- Core objects ---
        self.processor = VLProcessor(VLConfig(
            model_name=self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL),
            api_url=self.settings.value(KEY_API_URL, constants.DEFAULT_API_URL),
            max_tokens=int(self.settings.value(KEY_MAX_TOKENS, constants.DEFAULT_MAX_TOKENS)),
        ))
        self.dictionary = LocalDictionary()

        # --- Hotkeys ---
        self.hotkey_listener = create_hotkey_listener()
        
        # Load and set initial leader key
        initial_leader = self.settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
        if hasattr(self.hotkey_listener, "set_leader_key"):
            self.hotkey_listener.set_leader_key(initial_leader)
            
        self.hotkey_listener.trigger_lens.connect(self.show_lens)
        self.hotkey_listener.trigger_chat.connect(self.toggle_chat)
        self.hotkey_listener.trigger_settings.connect(self._open_settings)
        self.hotkey_listener.command_mode_started.connect(self._on_command_mode_started)
        self.hotkey_listener.start()

        # --- Command OSD ---
        self.osd = CommandOSD()
        self.osd.initialize_settings(
            source=self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG),
            target=self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG),
            model=self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
        )
        self.osd.setting_changed.connect(self._on_osd_setting_changed)
        
        self.osd_timer = QTimer(self)
        self.osd_timer.setSingleShot(True)
        self.osd_timer.timeout.connect(self.osd.hide)

        # --- Chat sidebar (created once, toggled) ---
        self.chat_sidebar = ChatSidebar(self.processor)

        # Lens window reference (created on demand)
        self._lens: LensOverlayWindow | None = None
        # Active inference workers (prevent GC)
        self._workers: list = []
        # Active result bubbles (prevent GC)
        self._bubbles: list = []
        # Guard against repeated pull attempts
        self._model_pull_attempted: bool = False

        # --- System tray ---
        self._setup_tray()

        # --- Initial health check ---
        self._run_health_check()

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------
    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        # Try to load icon, fall back to a theme icon
        icon = QIcon("xian.png")
        if icon.isNull():
            icon = QIcon.fromTheme("applications-graphics")
        self.tray.setIcon(icon)
        self.tray.setToolTip("Xian-VL — Lens & Chat Assistant")

        menu = QMenu()
        capture_action = menu.addAction("📸 Capture (Leader+C)")
        capture_action.triggered.connect(self.show_lens)

        chat_action = menu.addAction("💬 Chat (Leader+A)")
        chat_action.triggered.connect(self.toggle_chat)

        menu.addSeparator()

        settings_action = menu.addAction("⚙ Settings… (Leader+S)")
        settings_action.triggered.connect(self._open_settings)

        menu.addSeparator()

        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)

        self.tray.setContextMenu(menu)
        self.tray.show()

        logger.info("System tray icon initialised")

    def _on_command_mode_started(self):
        self.osd.show_centered()
        self.osd_timer.start(15000)
        
    def _on_osd_setting_changed(self, key: str, value: str):
        """Handle quick-settings updates from the OSD."""
        logger.info(f"OSD updated {key} to {value}")
        self.settings.setValue(key, value)
        if key == KEY_API_MODEL:
            self.processor.config.model_name = value
            self.processor.client = None
            self._run_health_check()

    # ------------------------------------------------------------------
    # Lens
    # ------------------------------------------------------------------
    def show_lens(self):
        """Capture the screen and open the Lens overlay."""
        self.osd.hide()
        self.osd_timer.stop()
        logger.info("Opening Lens overlay")
        # Close any existing lens
        if self._lens is not None:
            try:
                self._lens.close()
            except Exception:
                logger.debug("Error closing previous lens overlay", exc_info=True)

        self._lens = LensOverlayWindow(
            previous_rect=LensOverlayWindow._last_rect,
        )
        self._lens.action_requested.connect(self._on_lens_action)
        self._lens.closed.connect(self._on_lens_closed)
        self._lens.showFullScreen()

    def _on_lens_closed(self):
        self._lens = None

    def _on_lens_action(self, action: str, rect: QRect, image_data: bytes):
        """Handle an action from the Lens action bar."""
        logger.info(f"Lens action: {action}, rect: {rect}")

        if action == "chat":
            # Push image into chat context and open sidebar
            self.chat_sidebar.add_image_context(image_data)
            if not self.chat_sidebar.isVisible():
                self.chat_sidebar.show()
            self.chat_sidebar.raise_()
            return

        if action == "dictionary":
            # Run a quick translate, then look up in dictionary
            self._run_inference(image_data, action, rect)
            return

        if action in ("translate", "explain"):
            self._run_inference(image_data, action, rect)
            return

        logger.warning(f"Unknown lens action: {action}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_inference(self, image_data: bytes, action: str, anchor_rect: QRect):
        """Spawn an InferenceWorker for the given image crop."""
        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        mode = self.settings.value(KEY_MODE, constants.DEFAULT_MODE)
        
        saved_styles = self.settings.value(KEY_STYLES, constants.DEFAULT_STYLES)
        if isinstance(saved_styles, str):
            styles = [s.strip() for s in saved_styles.split(",") if s.strip()]
        elif isinstance(saved_styles, list):
            styles = saved_styles
        else:
            styles = []

        worker = InferenceWorker(
            self.processor,
            image_data=image_data,
            source_lang=source_lang,
            target_lang=target_lang,
            mode=mode,
            styles=styles,
            action=action,
            anchor_rect=anchor_rect,
        )

        worker.translation_done.connect(
            lambda results, act, w=worker: self._on_inference_done(results, act, w)
        )
        worker.error.connect(
            lambda msg, w=worker: self._on_inference_error(msg, w)
        )
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))

        self._workers.append(worker)
        worker.start()
        logger.info(f"InferenceWorker started for action={action}")

    def _on_inference_done(self, results, action, worker):
        """Handle completed inference."""
        anchor = worker.anchor_rect

        if not results:
            logger.info("Inference returned no results")
            bubble = ResultBubble(
                "No text detected in the selected region.",
                anchor_rect=anchor,
                auto_close_ms=5000,
            )
            self._bubbles = [b for b in self._bubbles if b.isVisible()]
            self._bubbles.append(bubble)
            return

        # Combine all results into a single display
        originals = []
        translations = []
        for r in results:
            if r.original_text:
                originals.append(r.original_text)
            if r.translated_text:
                translations.append(r.translated_text)

        original_combined = "\n".join(originals)
        translation_combined = "\n".join(translations)

        if action == "dictionary" and original_combined:
            # Look up each character/word in the dictionary
            entries = self.dictionary.lookup(original_combined)
            if entries:
                dict_text = "\n".join(
                    f"{trad} ({pinyin}): {english}"
                    for trad, pinyin, english in entries[:10]
                )
                translation_combined = dict_text

        bubble = ResultBubble(
            translation_combined,
            original_text=original_combined,
            anchor_rect=anchor,
        )
        self._bubbles = [b for b in self._bubbles if b.isVisible()]
        self._bubbles.append(bubble)

    def _on_inference_error(self, msg, worker):
        anchor = worker.anchor_rect
        bubble = ResultBubble(
            f"⚠ Error: {msg}",
            anchor_rect=anchor,
            auto_close_ms=8000,
        )
        self._bubbles = [b for b in self._bubbles if b.isVisible()]
        self._bubbles.append(bubble)

    def _cleanup_worker(self, worker):
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def toggle_chat(self):
        """Toggle the chat sidebar visibility."""
        self.osd.hide()
        self.osd_timer.stop()
        if self.chat_sidebar.isVisible():
            self.chat_sidebar.hide()
        else:
            self.chat_sidebar.show()
            self.chat_sidebar.raise_()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def _run_health_check(self):
        api_url = self.settings.value(KEY_API_URL, constants.DEFAULT_API_URL)
        self._status_worker = StatusWorker(api_url)
        self._status_worker.status_changed.connect(self._on_health_result)
        self._status_worker.start()

    def _on_health_result(self, available: bool, models: list):
        if available:
            logger.info(f"Lemonade server connected. Models: {models}")
            self._available_models = models
            self.osd.update_models(models)
            
            # Ensure the default model is pulled
            target_model = self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
            if target_model not in models and not self._model_pull_attempted:
                logger.info(f"Model '{target_model}' not found on server, pulling...")
                self._model_pull_attempted = True
                self._pull_model(target_model)
        else:
            logger.warning("Lemonade server not reachable")
            self._available_models = []

    def _pull_model(self, model_name: str):
        """Download a model via the Lemonade /v1/pull endpoint."""
        api_url = self.settings.value(KEY_API_URL, constants.DEFAULT_API_URL)
        gpu_util = self.settings.value(KEY_GPU_UTIL, constants.DEFAULT_GPU_MEMORY_UTILIZATION)
        self._pull_worker = ModelPullWorker(api_url, model_name, gpu_util)
        self._pull_worker.pull_done.connect(self._on_pull_done)
        self._pull_worker.start()

    def _on_pull_done(self, success: bool, message: str):
        if success:
            logger.info(f"Model pull complete: {message}")
            # Re-run health check to refresh the models list
            self._run_health_check()
        else:
            logger.warning(f"Model pull failed: {message}")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _open_settings(self):
        self.osd.hide()
        self.osd_timer.stop()
        dlg = SettingsDialog(self.settings, self._available_models)
        if dlg.exec():
            # Apply changed settings to processor
            self.processor.config.api_url = self.settings.value(KEY_API_URL, constants.DEFAULT_API_URL)
            self.processor.config.model_name = self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
            self.processor.config.max_tokens = int(self.settings.value(KEY_MAX_TOKENS, constants.DEFAULT_MAX_TOKENS))
            # Force re-init on next inference
            self.processor.client = None
            # Re-run health check
            self._run_health_check()
            
            # Apply new leader key
            new_leader = self.settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
            if hasattr(self.hotkey_listener, "set_leader_key"):
                self.hotkey_listener.set_leader_key(new_leader)
                
            logger.info("Settings updated and applied")
