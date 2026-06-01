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
    QHBoxLayout, QWidget, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings, QRect, QTimer, QBuffer, QIODevice
from PyQt6.QtGui import QIcon, QAction, QImage, QPixmap

from mage.ui.theme import accent_hex, accent_hover_hex

from xian.pipeline import VLProcessor, VLConfig
from mage.workers import InferenceWorker, StatusWorker, ModelPullWorker, CinematicWorker, PrewarmWorker, ContinueWorker, ChatTranslationWorker, RaidWorker
from mage.ui.lens import LensOverlayWindow, CinematicLensOverlay
from mage.ui.chat_sidebar import ChatSidebar
from mage.ui.how_to_say import HowToSayDialog
from mage.ui.result_bubble import ResultBubble
from mage.capture.hotkeys import create_hotkey_listener
from mage.capture.mouse import create_mouse_listener
from mage.capture.screen import ScreenCapture
from mage.capture.audio import play_audio_async
from xian.dictionary import LocalDictionary
from xian.lemonade_url import normalize_lemonade_api_base_url, should_warn_http_to_non_loopback
from mage.ui.command_osd import CommandOSD
from shared_types import constants
from shared_types.enums import SourceLanguage, TargetLanguage, TranslationMode, TranslationStyle
from mage.settings_keys import (
    KEY_API_URL, KEY_API_MODEL, KEY_SOURCE_LANG, KEY_TARGET_LANG,
    KEY_MODE, KEY_STYLES, KEY_MAX_TOKENS, KEY_LEADER_KEY,
    KEY_GPU_UTIL, KEY_DIALOGUE_DELAY, KEY_CINEMATIC_TRIGGER,
    KEY_AUTO_CONTINUE, KEY_AUTO_SPEAK,
)

logger = logging.getLogger(__name__)

ORGANIZATION = constants.ORGANIZATION_NAME
APP_NAME = constants.APPLICATION_NAME
MAX_AUTO_CONTINUES = 5


def _normalized_api_url_from_settings(settings: QSettings) -> str:
    return normalize_lemonade_api_base_url(str(settings.value(KEY_API_URL, constants.DEFAULT_API_URL)))


def _parse_styles(settings: QSettings) -> list[str]:
    """Parse the styles QSettings value into a clean list of strings."""
    raw = settings.value(KEY_STYLES, constants.DEFAULT_STYLES)
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return raw if isinstance(raw, list) else []


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
        self.url_edit.setText(_normalized_api_url_from_settings(settings))
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
        self.leader_combo.addItems(["Double-Tap Shift", "Double-Tap Ctrl", "Double-Tap Alt", "Double-Tap Super"])
        leader_val = settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
        if leader_val == "Shift+Space": leader_val = "Double-Tap Shift"
        idx = self.leader_combo.findText(leader_val)
        if idx >= 0:
            self.leader_combo.setCurrentIndex(idx)
        layout.addRow("Leader Key:", self.leader_combo)

        # Styles
        style_layout = QVBoxLayout()
        self.style_checkboxes = {}
        saved_styles = _parse_styles(settings)
        for style in TranslationStyle:
            cb = QCheckBox(style.value)
            if style.value in saved_styles:
                cb.setChecked(True)
            self.style_checkboxes[style.value] = cb
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

        # Dialogue Delay
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(100, 10000)
        self.delay_spin.setSingleStep(100)
        self.delay_spin.setValue(int(settings.value(KEY_DIALOGUE_DELAY, 1500)))
        layout.addRow("Dialogue Delay (ms):", self.delay_spin)

        # Auto-continue truncated translations
        self.auto_continue_cb = QCheckBox("Automatically continue truncated translations")
        auto_val = settings.value(KEY_AUTO_CONTINUE, "false")
        self.auto_continue_cb.setChecked(auto_val == "true" or auto_val is True)
        layout.addRow(self.auto_continue_cb)

        # Auto-speak translations
        self.auto_speak_cb = QCheckBox("Auto-speak translations (Text-to-Speech)")
        speak_val = settings.value(KEY_AUTO_SPEAK, "false")
        self.auto_speak_cb.setChecked(speak_val == "true" or speak_val is True)
        layout.addRow(self.auto_speak_cb)

        # Live Voice Translation (Raid Mode)
        self.live_voice_raid_cb = QCheckBox("Live Voice Translation (Raid Mode)")
        lv_raid_val = settings.value("live_voice_raid", "false")
        self.live_voice_raid_cb.setChecked(lv_raid_val == "true" or lv_raid_val is True)
        layout.addRow(self.live_voice_raid_cb)
        
        # Save Raid Notes to LORE
        self.live_raid_lore_save_cb = QCheckBox("Save Raid Notes to LORE")
        lr_lore_val = settings.value("live_raid_lore_save", "false")
        self.live_raid_lore_save_cb.setChecked(lr_lore_val == "true" or lr_lore_val is True)
        layout.addRow(self.live_raid_lore_save_cb)

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
        normalized = normalize_lemonade_api_base_url(self.url_edit.text().strip())
        if should_warn_http_to_non_loopback(normalized):
            choice = QMessageBox.warning(
                self,
                "HTTP to remote server",
                "You configured Lemonade over HTTP to a host that is not loopback. "
                "Traffic can be read or modified on the network path. Lemonade does not "
                "provide HTTPS itself; use a VPN, SSH tunnel, or a TLS reverse proxy if "
                "this endpoint is reachable beyond a single trusted machine.\n\n"
                "Choose Save to continue or Cancel to go back.",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
        self.url_edit.setText(normalized)
        self.settings.setValue(KEY_API_URL, normalized)
        self.settings.setValue(KEY_API_MODEL, self.model_combo.currentText())
        self.settings.setValue(KEY_SOURCE_LANG, self.source_lang_combo.currentText())
        self.settings.setValue(KEY_TARGET_LANG, self.lang_combo.currentText())
        self.settings.setValue(KEY_MODE, self.mode_combo.currentText())
        selected_styles = [s for s, cb in self.style_checkboxes.items() if cb.isChecked()]
        self.settings.setValue(KEY_STYLES, selected_styles)
        self.settings.setValue(KEY_MAX_TOKENS, self.tokens_spin.value())
        self.settings.setValue(KEY_LEADER_KEY, self.leader_combo.currentText())
        self.settings.setValue(KEY_GPU_UTIL, self.gpu_combo.currentText())
        self.settings.setValue(KEY_DIALOGUE_DELAY, self.delay_spin.value())
        self.settings.setValue(KEY_AUTO_CONTINUE, "true" if self.auto_continue_cb.isChecked() else "false")
        self.settings.setValue(KEY_AUTO_SPEAK, "true" if self.auto_speak_cb.isChecked() else "false")
        self.settings.setValue("live_voice_raid", "true" if self.live_voice_raid_cb.isChecked() else "false")
        self.settings.setValue("live_raid_lore_save", "true" if self.live_raid_lore_save_cb.isChecked() else "false")
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
            api_url=_normalized_api_url_from_settings(self.settings),
            max_tokens=int(self.settings.value(KEY_MAX_TOKENS, constants.DEFAULT_MAX_TOKENS)),
        ))
        self.dictionary = LocalDictionary()

        # Pre-warm target model in VRAM
        self._prewarm_worker = PrewarmWorker(self.processor)
        self._prewarm_worker.start()

        # --- Hotkeys ---
        self.hotkey_listener = create_hotkey_listener()
        
        # Load and set initial leader key
        initial_leader = self.settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
        if hasattr(self.hotkey_listener, "set_leader_key"):
            self.hotkey_listener.set_leader_key(initial_leader)
            
        self.hotkey_listener.trigger_lens.connect(self.show_lens)
        self.hotkey_listener.trigger_chat.connect(self.toggle_chat)
        self.hotkey_listener.trigger_settings.connect(self._open_settings)
        self.hotkey_listener.trigger_dialogue_mode.connect(self.toggle_dialogue_mode)
        self.hotkey_listener.trigger_cinematic_mode.connect(self.toggle_cinematic_mode)
        self.hotkey_listener.trigger_how_to_say.connect(self.show_how_to_say)
        self.hotkey_listener.trigger_raid_mode.connect(self.start_raid_mode)
        self.hotkey_listener.cinematic_capture.connect(self._on_cinematic_trigger)
        self.hotkey_listener.command_mode_started.connect(self._on_command_mode_started)
        if hasattr(self.hotkey_listener, "command_mode_cancelled"):
            self.hotkey_listener.command_mode_cancelled.connect(self.hide_osd)
        self.hotkey_listener.start()

        # --- Cinematic Mode ---
        self.cinematic_mode_active = False
        self.cinematic_bubble = None

        # --- Raid Mode ---
        self.raid_bubble = None

        # --- Dialogue Mode ---
        self.dialogue_mode_active = False
        self.dialogue_timer = QTimer(self)
        self.dialogue_timer.setSingleShot(True)
        self.dialogue_timer.timeout.connect(self.capture_dialogue)
        self.mouse_listener = create_mouse_listener()
        self.mouse_listener.left_click.connect(self.on_dialogue_click)
        self.dialogue_bubble = None

        # --- Command OSD ---
        self.osd = CommandOSD()
        self.osd.initialize_settings(
            source=self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG),
            target=self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG),
            model=self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
        )
        self.osd.setting_changed.connect(self._on_osd_setting_changed)
        self.osd.command_triggered.connect(self._on_osd_command)
        self.osd.osd_hidden.connect(self._on_osd_hidden)
        
        self.osd_timer = QTimer(self)
        self.osd_timer.setSingleShot(True)
        self.osd_timer.timeout.connect(self.hide_osd)

        # --- Chat sidebar (created once, toggled) ---
        self.chat_sidebar = ChatSidebar(self.processor)

        # --- How to Say Dialog ---
        self.how_to_say_dialog = HowToSayDialog()
        self.how_to_say_dialog.translation_requested.connect(self._on_how_to_say_submit)
        self.how_to_say_dialog.dialog_hidden.connect(self._on_osd_hidden)

        # Lens window reference (created on demand)
        self._lens: LensOverlayWindow | None = None
        # Active inference workers (prevent GC)
        self._workers: list = []
        # Active result bubbles (prevent GC)
        self._bubbles: list = []
        # Map active InferenceWorker to its temporary/loading ResultBubble
        self._active_bubbles: dict[InferenceWorker, ResultBubble] = {}
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
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "xian.png")
        if not os.path.exists(icon_path):
            # Fall back to checking root of workspace/repo
            icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))), "xian.png")
        if not os.path.exists(icon_path):
            icon_path = "xian.png"
        icon = QIcon(icon_path)
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

    def hide_osd(self):
        self.osd.hide()
        self.osd_timer.stop()

    def _on_osd_hidden(self):
        if hasattr(self.hotkey_listener, "cancel_command_mode"):
            self.hotkey_listener.cancel_command_mode()

    def _on_command_mode_started(self):
        self.osd.show_centered()
        self.osd_timer.start(15000)
        
    def _on_osd_setting_changed(self, key: str, value: str):
        """Handle quick-settings updates from the OSD."""
        logger.info("OSD updated %s to %s", key, value)
        self.settings.setValue(key, value)
        if key == KEY_API_MODEL:
            self.processor.config.model_name = value
            self.processor.engine.reconfigure(
                base_url=self.processor.config.api_url
            )
            self._run_health_check()
            self._prewarm_worker = PrewarmWorker(self.processor)
            self._prewarm_worker.start()

    def _on_osd_command(self, key: str):
        """Handle option buttons clicked in the OSD."""
        logger.info("OSD command triggered: %s", key)
        self.hide_osd()
        
        if key == "C":
            self.show_lens()
        elif key == "A":
            self.toggle_chat()
        elif key == "O":
            self.toggle_dialogue_mode()
        elif key == "M":
            self.toggle_cinematic_mode()
        elif key == "T":
            self.show_how_to_say()
        elif key == "R":
            self.start_raid_mode()
        elif key == "S":
            self._open_settings()

    # ------------------------------------------------------------------
    # Lens
    # ------------------------------------------------------------------
    def show_lens(self):
        """Capture the screen and open the Lens overlay."""
        self.hide_osd()
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
        logger.info("Lens action: %s, rect: %s", action, rect)

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

        logger.warning("Unknown lens action: %s", action)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_inference(self, image_data: bytes, action: str, anchor_rect: QRect):
        """Spawn an InferenceWorker for the given image crop."""
        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        mode = self.settings.value(KEY_MODE, constants.DEFAULT_MODE)
        styles = _parse_styles(self.settings)

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

        worker.thinking.connect(
            lambda w=worker: self._on_inference_thinking(w)
        )
        worker.translation_partial.connect(
            lambda text, act, w=worker: self._on_inference_partial(text, act, w)
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
        logger.info("InferenceWorker started for action=%s", action)

    def _add_bubble(self, bubble: ResultBubble):
        """Clean up old bubbles, limit maximum tracked bubbles, and append new one."""
        self._bubbles = [b for b in self._bubbles if b.isVisible()]
        if len(self._bubbles) >= 20:  # MAX_BUBBLES
            # Close the oldest visible bubbles to avoid memory leak and clutter
            for old in self._bubbles[:-19]:
                old.close()
            self._bubbles = self._bubbles[-19:]
        self._bubbles.append(bubble)
        self._setup_bubble_connections(bubble)

    def _replace_persistent_bubble(self, attr_name: str, text: str, original_text: str = "", anchor: QRect = QRect(), border_color: str | None = None, truncated: bool = False) -> ResultBubble:
        """Helper to close and recreate a persistent dialogue or cinematic bubble."""
        old_bubble = getattr(self, attr_name, None)
        if old_bubble is not None:
            try:
                old_bubble.close()
            except Exception:
                pass
        
        bubble = ResultBubble(
            text,
            original_text=original_text,
            anchor_rect=anchor,
            auto_close_ms=0,
            border_color=border_color,
            truncated=truncated,
        )
        setattr(self, attr_name, bubble)
        if truncated:
            bubble.continue_requested.connect(
                lambda b=bubble: self._on_continue_requested(b)
            )
        self._setup_bubble_connections(bubble)
        return bubble

    def _on_inference_thinking(self, worker):
        anchor = worker.anchor_rect
        bubble = ResultBubble(
            "Translating...",
            anchor_rect=anchor,
            auto_close_ms=0,
        )
        self._active_bubbles[worker] = bubble
        self._add_bubble(bubble)

    def _on_inference_partial(self, text, action, worker):
        bubble = self._active_bubbles.get(worker)
        if bubble and bubble.isVisible():
            bubble.update_text(text)

    def _on_inference_done(self, results, action, worker):
        """Handle completed inference."""
        anchor = worker.anchor_rect
        bubble = self._active_bubbles.pop(worker, None)
        if bubble:
            bubble.close()

        if not results:
            logger.info("Inference returned no results")
            bubble = ResultBubble(
                "No text detected in the selected region.",
                anchor_rect=anchor,
                auto_close_ms=5000,
            )
            self._add_bubble(bubble)
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

        # Determine if speculative (confidence < 0.70)
        min_confidence = min((r.confidence for r in results), default=1.0)
        is_speculative = min_confidence < 0.70
        border_color = "#e5a93c" if is_speculative else None

        # Detect truncation
        any_truncated = any(getattr(r, 'truncated', False) for r in results)

        if action == "cinematic":
            target_bubble = self._replace_persistent_bubble(
                "cinematic_bubble",
                translation_combined,
                original_text=original_combined,
                anchor=anchor,
                border_color=border_color,
                truncated=any_truncated,
            )
        elif self.dialogue_mode_active:
            target_bubble = self._replace_persistent_bubble(
                "dialogue_bubble",
                translation_combined,
                original_text=original_combined,
                anchor=anchor,
                border_color=border_color,
                truncated=any_truncated,
            )
        else:
            bubble = ResultBubble(
                translation_combined,
                original_text=original_combined,
                anchor_rect=anchor,
                border_color=border_color,
                truncated=any_truncated,
            )
            target_bubble = bubble
            if any_truncated:
                bubble.continue_requested.connect(
                    lambda b=bubble: self._on_continue_requested(b)
                )
            self._add_bubble(bubble)

        # Store continuation context on target_bubble
        if any_truncated and target_bubble:
            target_bubble.continuation_messages = getattr(worker, "continuation_messages", None)
            target_bubble.continuation_partial = getattr(worker, "continuation_context_partial", None)

        # Auto-continue if enabled
        if any_truncated:
            auto_continue = self.settings.value(KEY_AUTO_CONTINUE, "false")
            if auto_continue == "true" or auto_continue is True:
                if target_bubble:
                    count = getattr(target_bubble, "_continue_count", 0)
                    if count < MAX_AUTO_CONTINUES:
                        target_bubble._continue_count = count + 1
                        logger.info("Auto-continuing truncated translation (%d/%d)", count + 1, MAX_AUTO_CONTINUES)
                        self._on_continue_requested(target_bubble)
                    else:
                        logger.warning("Max auto-continue limit reached")

        # Store captured audio bytes on target_bubble if available
        if target_bubble:
            target_bubble.captured_audio_bytes = getattr(worker, "audio_bytes", None)

        # Auto-speak if in Cinematic Mode, or if Auto-speak setting is checked
        auto_speak = self.settings.value(KEY_AUTO_SPEAK, "false")
        if action == "cinematic" or auto_speak == "true" or auto_speak is True:
            self._speak_text(translation_combined, source=False, voice_ref_bytes=getattr(worker, "audio_bytes", None))

    def _on_inference_error(self, msg, worker):
        anchor = worker.anchor_rect
        bubble = self._active_bubbles.pop(worker, None)
        if bubble:
            bubble.close()
        bubble = ResultBubble(
            f"⚠ Error: {msg}",
            anchor_rect=anchor,
            auto_close_ms=8000,
        )
        self._add_bubble(bubble)

    def _cleanup_worker(self, worker):
        try:
            self._workers.remove(worker)
        except ValueError:
            pass

    def _setup_bubble_connections(self, bubble):
        bubble.speak_source_requested.connect(
            lambda b=bubble: self._on_speak_requested(b, source=True)
        )
        bubble.speak_target_requested.connect(
            lambda b=bubble: self._on_speak_requested(b, source=False)
        )

    def _on_speak_requested(self, bubble, source: bool):
        text = bubble._original if source else bubble._text
        if not text:
            return
        voice_ref_bytes = getattr(bubble, "captured_audio_bytes", None)
        self._speak_text(text, source=source, voice_ref_bytes=voice_ref_bytes)

    def _speak_text(self, text: str, source: bool, voice_ref_bytes: bytes | None = None):
        """Synthesize text and play it back, optionally using voice cloning if voice_ref_bytes is provided."""
        if not text:
            return

        async def _synthesize_and_play():
            import tempfile
            from pathlib import Path
            from xian.lemonade_client import LemonadeClient
            
            base_url = os.environ.get("LEMONADE_API_URL", self.processor.config.api_url)
            base_url_no_v1 = base_url.removesuffix("/v1")
            
            # Base voice defaults to English
            voice_param = "af_heart"
            lang = self.settings.value(KEY_SOURCE_LANG if source else KEY_TARGET_LANG, constants.DEFAULT_SOURCE_LANG)
            if lang == "Chinese":
                voice_param = "zf_xiaoxiao"
            elif lang == "Japanese":
                voice_param = "jf_alpha"
                
            temp_wav_path = None
            if voice_ref_bytes:
                # Voice cloning by file path only works if the server is running locally
                is_local = "localhost" in base_url or "127.0.0.1" in base_url
                if is_local:
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(voice_ref_bytes)
                            temp_wav_path = tmp.name
                        voice_param = temp_wav_path
                    except Exception as e:
                        logger.warning("Failed to write voice cloning reference audio: %s", e)
                else:
                    logger.warning("Voice cloning requires Lemonade to be on localhost. Falling back to default voice.")
            
            try:
                client = LemonadeClient(base_url=base_url_no_v1)
                
                # Discover downloaded TTS model via router
                active_model = self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
                if not self.processor.router.tts(active_model):
                    await self.processor.router.discover_async()
                tts_model = self.processor.router.tts(active_model)
                if not tts_model:
                    raise ValueError("No TTS model available on the server.")
                
                audio_bytes = await client.tts(text, voice=voice_param, model=tts_model)
                await client.close()
                play_audio_async(audio_bytes)
            except Exception as e:
                logger.error("TTS synthesis failed: %s", e)
            finally:
                if temp_wav_path:
                    try:
                        Path(temp_wav_path).unlink(missing_ok=True)
                    except Exception:
                        pass

        self.processor.engine.submit(_synthesize_and_play())


    def _on_continue_requested(self, bubble: ResultBubble):
        """User clicked Continue on a truncated result."""
        messages = getattr(bubble, "continuation_messages", None)
        partial = getattr(bubble, "continuation_partial", None)
        mode = self.settings.value(KEY_MODE, constants.DEFAULT_MODE)

        if not messages or not partial:
            logger.warning("No continuation context available")
            return

        worker = ContinueWorker(
            self.processor,
            messages=messages,
            partial_output=partial,
            mode=mode,
        )

        worker.continuation_partial.connect(
            lambda text, b=bubble: b.update_text(text) if b.isVisible() else None
        )
        worker.continuation_done.connect(
            lambda text, still_truncated, b=bubble, w=worker: self._on_continue_done(
                b, text, still_truncated, w
            )
        )
        worker.error.connect(
            lambda msg, b=bubble: b.update_text(f"⚠ Continue failed: {msg}") if b.isVisible() else None
        )
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))

        self._workers.append(worker)
        worker.start()

    def _on_continue_done(self, bubble, text, still_truncated, worker):
        """Handle completed continuation."""
        if not bubble.isVisible():
            return

        # Store updated continuation context back on the bubble
        if still_truncated:
            bubble.continuation_messages = getattr(worker, "continuation_messages", None)
            bubble.continuation_partial = getattr(worker, "continuation_context_partial", None)
        else:
            bubble.continuation_messages = None
            bubble.continuation_partial = None

        # Re-parse the combined output to extract a proper translation
        orig, trans, conf = self.processor.parse_response(text)
        if trans:
            bubble.update_text(trans, original_text=orig)
        else:
            bubble.update_text(text)
        bubble.show_continue_button(still_truncated)

        # Auto-continue again if enabled and still truncated
        if still_truncated:
            auto_continue = self.settings.value(KEY_AUTO_CONTINUE, "false")
            if auto_continue == "true" or auto_continue is True:
                count = getattr(bubble, "_continue_count", 0)
                if count < MAX_AUTO_CONTINUES:
                    bubble._continue_count = count + 1
                    logger.info("Auto-continuing again (still truncated) (%d/%d)", count + 1, MAX_AUTO_CONTINUES)
                    self._on_continue_requested(bubble)
                else:
                    logger.warning("Max auto-continue limit reached")

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def toggle_chat(self):
        """Toggle the chat sidebar visibility."""
        self.hide_osd()
        if self.chat_sidebar.isVisible():
            self.chat_sidebar.hide()
        else:
            self.chat_sidebar.show()
            self.chat_sidebar.raise_()

    # ------------------------------------------------------------------
    # How to say
    # ------------------------------------------------------------------
    def show_how_to_say(self):
        self.hide_osd()
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        self.how_to_say_dialog.show_centered(target_lang=target_lang, source_lang=source_lang)

    def _on_how_to_say_submit(self, text: str):
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        
        worker = ChatTranslationWorker(self.processor, text, target_lang, source_lang)
        worker.translation_done.connect(self._on_how_to_say_done)
        worker.error.connect(self._on_how_to_say_error)
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))
        
        self._workers.append(worker)
        worker.start()

    def _on_how_to_say_done(self, translated_text: str):
        self.how_to_say_dialog.set_result(translated_text)

    def _on_how_to_say_error(self, msg: str):
        self.how_to_say_dialog.set_error(msg)

    # ------------------------------------------------------------------
    # Dialogue Mode
    # ------------------------------------------------------------------
    def toggle_dialogue_mode(self):
        self.hide_osd()

        if self.dialogue_mode_active:
            self.dialogue_mode_active = False
            self.mouse_listener.stop()
            self.dialogue_timer.stop()
            if self.dialogue_bubble:
                self.dialogue_bubble.close()
                self.dialogue_bubble = None
            
            bubble = ResultBubble("Dialogue Mode Disabled", auto_close_ms=3000)
            self._add_bubble(bubble)
            logger.info("Dialogue Mode Deactivated")
        else:
            if LensOverlayWindow._last_rect is None or LensOverlayWindow._last_rect.isEmpty():
                bubble = ResultBubble("Please use Lens (Leader+C) to select a dialogue area first.", auto_close_ms=5000)
                self._add_bubble(bubble)
                return

            self.dialogue_mode_active = True
            self.mouse_listener.start()
            
            bubble = ResultBubble("Dialogue Mode ON", anchor_rect=LensOverlayWindow._last_rect, auto_close_ms=3000)
            self._add_bubble(bubble)
            logger.info("Dialogue Mode Activated")

    def on_dialogue_click(self):
        if self.dialogue_mode_active:
            delay = int(self.settings.value(KEY_DIALOGUE_DELAY, 1500))
            self.dialogue_timer.start(delay)

    def capture_dialogue(self):
        if not self.dialogue_mode_active or LensOverlayWindow._last_rect is None or LensOverlayWindow._last_rect.isEmpty():
            return

        data = ScreenCapture.capture_screen()
        if not data:
            return

        img = QImage.fromData(data)
        pixmap = QPixmap.fromImage(img)
        rect = LensOverlayWindow._last_rect
        total_geo = ScreenCapture.get_virtual_desktop_geometry()
        
        safe_rect = rect.translated(-total_geo.left(), -total_geo.top())
        safe_rect = safe_rect.intersected(pixmap.rect())
        
        cropped = pixmap.copy(safe_rect)
        
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        cropped.save(buffer, "JPG", 85)
        cropped_data = bytes(buffer.buffer())
        
        self._run_inference(cropped_data, "translate", rect)

    # ------------------------------------------------------------------
    # Cinematic Mode
    # ------------------------------------------------------------------
    def toggle_cinematic_mode(self):
        self.hide_osd()
        
        if self.cinematic_mode_active:
            self.cinematic_mode_active = False
            self.hotkey_listener.cinematic_mode_active = False
            if self.cinematic_bubble:
                self.cinematic_bubble.close()
                self.cinematic_bubble = None
            bubble = ResultBubble("Cinematic Mode Disabled", auto_close_ms=3000)
            self._add_bubble(bubble)
        else:
            if not CinematicLensOverlay._last_rects:
                # Open Cinematic Lens for selection
                self._show_cinematic_lens()
            else:
                self._activate_cinematic_mode()

    def _activate_cinematic_mode(self):
        self.cinematic_mode_active = True
        self.hotkey_listener.cinematic_mode_active = True
        bubble = ResultBubble("Cinematic Mode ON (Press ` to capture)", auto_close_ms=4000)
        self._add_bubble(bubble)

    def _show_cinematic_lens(self):
        if self._lens is not None:
            self._lens.close()
        self._lens = CinematicLensOverlay()
        self._lens.confirmed.connect(self._on_cinematic_lens_confirmed)
        self._lens.closed.connect(self._on_lens_closed)
        self._lens.showFullScreen()
        
    def _on_cinematic_lens_confirmed(self, rects):
        if rects:
            self._activate_cinematic_mode()
        else:
            bubble = ResultBubble("Cinematic Mode requires at least one region.", auto_close_ms=4000)
            self._add_bubble(bubble)

    def _on_cinematic_trigger(self):
        if not self.cinematic_mode_active or not CinematicLensOverlay._last_rects:
            return

        # Show capturing indicator
        self.cinematic_bubble = self._replace_persistent_bubble("cinematic_bubble", "Recording audio and capturing screen...")

        data = ScreenCapture.capture_screen()
        if not data:
            return

        img = QImage.fromData(data)
        pixmap = QPixmap.fromImage(img)
        rects = CinematicLensOverlay._last_rects
        total_geo = ScreenCapture.get_virtual_desktop_geometry()
        
        # Composite rects
        total_height = sum(r.height() for r in rects)
        max_width = max(r.width() for r in rects)
        
        composite = QPixmap(max_width, total_height)
        composite.fill(Qt.GlobalColor.black)
        
        from PyQt6.QtGui import QPainter  # delayed import: only needed when cinematic is triggered
        painter = QPainter(composite)
        y_offset = 0
        for r in rects:
            safe_rect = r.translated(-total_geo.left(), -total_geo.top())
            safe_rect = safe_rect.intersected(pixmap.rect())
            cropped = pixmap.copy(safe_rect)
            painter.drawPixmap(0, y_offset, cropped)
            y_offset += r.height()
        painter.end()

        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        composite.save(buffer, "JPG", 85)
        composite_data = bytes(buffer.buffer())

        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        styles = _parse_styles(self.settings)

        anchor_rect = rects[-1] if rects else QRect()

        worker = CinematicWorker(
            self.processor,
            image_data=composite_data,
            target_lang=target_lang,
            styles=styles,
            anchor_rect=anchor_rect,
            source_lang=source_lang,
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
        logger.info("CinematicWorker started")

    # ------------------------------------------------------------------
    # Raid Mode
    # ------------------------------------------------------------------
    def start_raid_mode(self):
        self.hide_osd()
        logger.info("Triggered Raid Mode")
        self.raid_bubble = self._replace_persistent_bubble("raid_bubble", "Recording raid leader...")

        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)

        save_lore = self.settings.value("live_raid_lore_save", "false")

        worker = RaidWorker(
            self.processor,
            target_lang=target_lang,
            source_lang=source_lang,
            save_lore=(save_lore == "true" or save_lore is True)
        )

        worker.chunk_translated.connect(
            lambda transcript, translation, w=worker: self._on_chunk_translated(transcript, translation, w)
        )
        worker.error.connect(
            lambda msg, w=worker: self._on_raid_error(msg, w)
        )
        worker.progress.connect(
            lambda text, w=worker: self._on_raid_progress(text, w)
        )
        worker.finished.connect(lambda w=worker: self._cleanup_worker(w))

        self._workers.append(worker)
        worker.start()
        logger.info("RaidWorker started")

    def _on_raid_progress(self, text, worker):
        if hasattr(self, "raid_bubble") and self.raid_bubble and self.raid_bubble.isVisible():
            self.raid_bubble.update_text(text)

    def _on_chunk_translated(self, transcript, translation, worker):
        logger.info("Raid chunk translated: %s -> %s", transcript, translation)
        
        if hasattr(self, "raid_bubble") and self.raid_bubble and self.raid_bubble.isVisible():
            current_text = getattr(self.raid_bubble, "_raid_text_buffer", [])
            current_text.append(translation)
            if len(current_text) > 5:
                current_text = current_text[-5:]
            self.raid_bubble._raid_text_buffer = current_text
            
            display_text = " ".join(current_text)
            self.raid_bubble.update_text(display_text)
            # Update original text dynamically too
            if hasattr(self.raid_bubble, "original_text"):
                self.raid_bubble.original_text = transcript
        else:
            self.raid_bubble = self._replace_persistent_bubble("raid_bubble", translation, original_text=transcript)
            self.raid_bubble._raid_text_buffer = [translation]

        # Raid Mode optionally plays audio (translating and cloning voices)
        live_voice_raid = self.settings.value("live_voice_raid", "false")
        if live_voice_raid == "true" or live_voice_raid is True:
            self._speak_text(translation, source=False)

    def _on_raid_error(self, msg, worker):
        logger.error("Raid Mode error: %s", msg)
        if hasattr(self, "raid_bubble") and self.raid_bubble and self.raid_bubble.isVisible():
            self.raid_bubble.update_text(f"⚠ Error: {msg}")
        else:
            bubble = ResultBubble(
                f"⚠ Raid Mode Error: {msg}",
                auto_close_ms=8000,
            )
            self._add_bubble(bubble)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def _run_health_check(self):
        api_url = _normalized_api_url_from_settings(self.settings)
        self._status_worker = StatusWorker(api_url)
        self._status_worker.status_changed.connect(self._on_health_result)
        self._status_worker.start()

    def _on_health_result(self, available: bool, models: list, raw_models: list = None):
        if available:
            logger.info("Lemonade server connected. Models: %s", models)
            self._available_models = models
            self.osd.update_models(models)
            
            # Update the central router
            self.processor.router.update_with_models(raw_models or [])
            
            # Prefer Omni model as default if detected
            target_model = self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
            if self.processor.router.omni_detected:
                omni_id = self.processor.router.omni_model_id
                if omni_id and (target_model in (None, "", constants.DEFAULT_MODEL, "omni-router", "default")):
                    logger.info("Omni model '%s' detected, setting as default.", omni_id)
                    self.settings.setValue(KEY_API_MODEL, omni_id)
                    self.processor.config.model_name = omni_id
                    target_model = omni_id

            # Ensure the default model is pulled
            all_downloaded = [m.get("id") for m in (raw_models or []) if m.get("downloaded", True)]
            if target_model not in all_downloaded and not self._model_pull_attempted:
                logger.info("Model '%s' not found on server, pulling...", target_model)
                self._model_pull_attempted = True
                self._pull_model(target_model)
        else:
            logger.warning("Lemonade server not reachable")
            self._available_models = []

    def _pull_model(self, model_name: str):
        """Download a model via the Lemonade /v1/pull endpoint."""
        api_url = _normalized_api_url_from_settings(self.settings)
        gpu_util = self.settings.value(KEY_GPU_UTIL, constants.DEFAULT_GPU_MEMORY_UTILIZATION)
        self._pull_worker = ModelPullWorker(api_url, model_name, gpu_util)
        self._pull_worker.pull_done.connect(self._on_pull_done)
        self._pull_worker.start()

    def _on_pull_done(self, success: bool, message: str):
        if success:
            logger.info("Model pull complete: %s", message)
            # Re-run health check to refresh the models list
            self._run_health_check()
        else:
            logger.warning("Model pull failed: %s", message)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def _open_settings(self):
        self.hide_osd()
        dlg = SettingsDialog(self.settings, self._available_models)
        if dlg.exec():
            # Apply changed settings to processor
            self.processor.config.api_url = _normalized_api_url_from_settings(self.settings)
            self.processor.config.model_name = self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
            self.processor.config.max_tokens = int(self.settings.value(KEY_MAX_TOKENS, constants.DEFAULT_MAX_TOKENS))
            self.processor.engine.reconfigure(
                base_url=self.processor.config.api_url
            )
            from xian.omni_router import OmniModelRouter
            self.processor.router = OmniModelRouter(self.processor.config.api_url)
            # Re-run health check
            self._run_health_check()
            
            # Pre-warm target model in VRAM
            self._prewarm_worker = PrewarmWorker(self.processor)
            self._prewarm_worker.start()
            
            # Apply new leader key
            new_leader = self.settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
            if hasattr(self.hotkey_listener, "set_leader_key"):
                self.hotkey_listener.set_leader_key(new_leader)
                
            logger.info("Settings updated and applied")
