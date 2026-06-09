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
    QHBoxLayout, QWidget, QCheckBox, QMessageBox, QInputDialog
)
from PyQt6.QtCore import Qt, QSettings, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QAction, QImage, QPixmap, QCursor

from mage.ui.theme import accent_hex, accent_hover_hex

from xian.pipeline import VLProcessor, VLConfig
from mage.workers import InferenceWorker, StatusWorker, ModelPullWorker, CinematicWorker, PrewarmWorker, ContinueWorker, ChatTranslationWorker, RaidWorker
from mage.ui.lens import LensOverlayWindow, CinematicLensOverlay
from mage.ui.chat_sidebar import ChatSidebar
from mage.ui.how_to_say import HowToSayDialog
from mage.ui.raid_window import RaidWindow
from mage.ui.result_bubble import ResultBubble
from mage.capture.hotkeys import create_hotkey_listener
from mage.capture.mouse import create_mouse_listener
from mage.capture.screen import ScreenCapture
from mage.capture.audio import play_audio_async
from xian.dictionary import LocalDictionary
from xian.lemonade_url import normalize_lemonade_api_base_url, should_warn_http_to_non_loopback
from mage.ui.command_osd import CommandOSD
from mage.ui.hud import HudManager
from shared_types import constants
from shared_types.enums import SourceLanguage, TargetLanguage, TranslationMode, TranslationStyle
from mage.settings_keys import (
    KEY_API_URL, KEY_API_MODEL, KEY_SOURCE_LANG, KEY_TARGET_LANG,
    KEY_MODE, KEY_STYLES, KEY_MAX_TOKENS, KEY_LEADER_KEY,
    KEY_GPU_UTIL, KEY_DIALOGUE_DELAY,
    KEY_AUTO_CONTINUE, KEY_AUTO_SPEAK, KEY_TARGET_WINDOW_TITLE, KEY_UI_LANG
)
from mage.utils.window_binder import WindowBinder
from shared_types.state import state, t

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
    
    layout_edit_requested = pyqtSignal()

    def __init__(self, settings: QSettings, models: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("settings.dialog.title"))
        self.setMinimumWidth(400)
        self.settings = settings

        layout = QFormLayout(self)

        # Server URL
        self.url_edit = QLineEdit()
        self.url_edit.setText(_normalized_api_url_from_settings(settings))
        layout.addRow(t("settings.label.server_url"), self.url_edit)

        # Model
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        if models:
            self.model_combo.addItems(models)
        self.model_combo.setCurrentText(settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL))
        layout.addRow(t("settings.label.model"), self.model_combo)

        # Source language
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems([e.value for e in SourceLanguage])
        self.source_lang_combo.setCurrentText(settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG))
        layout.addRow(t("settings.label.source_language"), self.source_lang_combo)

        # Target language
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([e.value for e in TargetLanguage])
        self.lang_combo.setCurrentText(settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG))
        layout.addRow(t("settings.label.target_language"), self.lang_combo)

        # Mode
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([e.value for e in TranslationMode])
        self.mode_combo.setCurrentText(settings.value(KEY_MODE, constants.DEFAULT_MODE))
        layout.addRow(t("settings.label.mode"), self.mode_combo)

        # Leader Key
        self.leader_combo = QComboBox()
        self.leader_combo.addItem(t("leader.double_shift"), "Double-Tap Shift")
        self.leader_combo.addItem(t("leader.double_ctrl"), "Double-Tap Ctrl")
        self.leader_combo.addItem(t("leader.double_alt"), "Double-Tap Alt")
        self.leader_combo.addItem(t("leader.double_super"), "Double-Tap Super")
        leader_val = settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
        if leader_val == "Shift+Space": leader_val = "Double-Tap Shift"
        idx = self.leader_combo.findData(leader_val)
        if idx >= 0:
            self.leader_combo.setCurrentIndex(idx)
        layout.addRow(t("settings.label.leader_key"), self.leader_combo)

        # UI Language
        self.ui_lang_combo = QComboBox()
        self.ui_lang_combo.addItems(["en", "zh", "ja", "ko", "ru", "es", "ar", "hi", "vi"])
        self.ui_lang_combo.setCurrentText(settings.value(KEY_UI_LANG, "en"))
        layout.addRow(t("settings.label.ui_language"), self.ui_lang_combo)

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
        layout.addRow(t("settings.label.styles"), style_layout)

        # Max tokens
        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(256, 32768)
        self.tokens_spin.setValue(int(settings.value(KEY_MAX_TOKENS, constants.DEFAULT_MAX_TOKENS)))
        layout.addRow(t("settings.label.max_tokens"), self.tokens_spin)

        # GPU Memory Utilization
        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["Default", "0.5", "0.75"])
        self.gpu_combo.setCurrentText(settings.value(KEY_GPU_UTIL, constants.DEFAULT_GPU_MEMORY_UTILIZATION))
        layout.addRow(t("settings.label.gpu_memory_utilization"), self.gpu_combo)

        # Dialogue Delay
        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(100, 10000)
        self.delay_spin.setSingleStep(100)
        self.delay_spin.setValue(int(settings.value(KEY_DIALOGUE_DELAY, 1000)))
        layout.addRow(t("settings.label.dialogue_delay"), self.delay_spin)

        # Auto-continue truncated translations
        self.auto_continue_cb = QCheckBox(t("settings.checkbox.auto_continue"))
        auto_val = settings.value(KEY_AUTO_CONTINUE, "false")
        self.auto_continue_cb.setChecked(auto_val == "true" or auto_val is True)
        layout.addRow(self.auto_continue_cb)

        # Auto-speak translations
        self.auto_speak_cb = QCheckBox(t("settings.checkbox.auto_speak"))
        speak_val = settings.value(KEY_AUTO_SPEAK, "false")
        self.auto_speak_cb.setChecked(speak_val == "true" or speak_val is True)
        layout.addRow(self.auto_speak_cb)

        # Live Voice Translation (Raid Mode)
        self.live_voice_raid_cb = QCheckBox(t("settings.checkbox.live_voice_raid"))
        lv_raid_val = settings.value("live_voice_raid", "false")
        self.live_voice_raid_cb.setChecked(lv_raid_val == "true" or lv_raid_val is True)
        layout.addRow(self.live_voice_raid_cb)
        
        # Save Raid Notes to LORE
        self.live_raid_lore_save_cb = QCheckBox(t("settings.checkbox.live_raid_lore_save"))
        lr_lore_val = settings.value("live_raid_lore_save", "false")
        self.live_raid_lore_save_cb.setChecked(lr_lore_val == "true" or lr_lore_val is True)
        layout.addRow(self.live_raid_lore_save_cb)

        # HUD original and pinyin settings
        self.hud_show_original_cb = QCheckBox(t("settings.checkbox.hud_show_original"))
        hud_orig_val = settings.value("hud_show_original", "true")
        self.hud_show_original_cb.setChecked(hud_orig_val == "true" or hud_orig_val is True)
        layout.addRow(self.hud_show_original_cb)

        self.hud_show_pinyin_cb = QCheckBox(t("settings.checkbox.hud_show_pinyin"))
        hud_pin_val = settings.value("hud_show_pinyin", "false")
        self.hud_show_pinyin_cb.setChecked(hud_pin_val == "true" or hud_pin_val is True)
        layout.addRow(self.hud_show_pinyin_cb)

        # Target Window Title
        self.target_window_combo = QComboBox()
        self.target_window_combo.setEditable(True)
        self.target_window_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.target_window_combo.addItem(t("settings.option.none_overlay"), "")
        try:
            titles = WindowBinder.get_active_window_titles()
            for title in titles:
                self.target_window_combo.addItem(title, title)
        except Exception as e:
            logger.error("Could not fetch active window titles: %s", e)
            
        current_title = settings.value(KEY_TARGET_WINDOW_TITLE, "")
        if current_title:
            idx = self.target_window_combo.findData(current_title)
            if idx >= 0:
                self.target_window_combo.setCurrentIndex(idx)
            else:
                self.target_window_combo.addItem(current_title, current_title)
                self.target_window_combo.setCurrentText(current_title)
        else:
            self.target_window_combo.setCurrentIndex(0)
        layout.addRow(t("settings.label.target_window_title"), self.target_window_combo)

        # Layout Presets
        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        presets = self.settings.value("layout_presets_list", ["Default"])
        if not isinstance(presets, list):
            presets = ["Default"]
        self.preset_combo.addItems(presets)
        
        current_preset = self.settings.value("layout_preset", "Default")
        idx = self.preset_combo.findText(current_preset)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)


            
        self.add_preset_btn = QPushButton("+")
        self.add_preset_btn.setFixedWidth(30)
        self.add_preset_btn.clicked.connect(self._add_preset)
        
        self.del_preset_btn = QPushButton("-")
        self.del_preset_btn.setFixedWidth(30)
        self.del_preset_btn.clicked.connect(self._del_preset)
        
        self.edit_layout_btn = QPushButton(t("settings.button.edit_layout"))
        self.edit_layout_btn.clicked.connect(self._on_edit_layout)
        
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.add_preset_btn)
        preset_layout.addWidget(self.del_preset_btn)
        preset_layout.addWidget(self.edit_layout_btn)
        layout.addRow(t("settings.label.layout_preset"), preset_layout)

        # Developer Options Checkbox
        self.dev_options_cb = QCheckBox(t("settings.checkbox.dev_options"))
        dev_options_val = settings.value("developer_options", "false")
        self.dev_options_cb.setChecked(dev_options_val == "true" or dev_options_val is True)
        layout.addRow(self.dev_options_cb)
        self.dev_options_cb.toggled.connect(self._update_dev_visibility)
        self._update_dev_visibility(self.dev_options_cb.isChecked())

        # Buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton(t("settings.button.save"))
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton(t("settings.button.cancel"))
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

    def _add_preset(self):
        name, ok = QInputDialog.getText(
            self,
            t("settings.prompt.layout_preset.new"),
            t("settings.prompt.layout_preset.new")
        )
        if ok:
            name = name.strip()
            if not name:
                QMessageBox.critical(self, "Error", t("settings.error.layout_preset.invalid"))
                return
            presets = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
            if name in presets:
                QMessageBox.critical(self, "Error", t("settings.error.layout_preset.exists"))
                return
            
            presets.append(name)
            self.settings.setValue("layout_presets_list", presets)
            self.preset_combo.addItem(name)
            self.preset_combo.setCurrentText(name)

    def _del_preset(self):
        current = self.preset_combo.currentText()
        if current == "Default":
            QMessageBox.critical(self, "Error", "The Default preset cannot be deleted.")
            return
            
        choice = QMessageBox.question(
            self,
            "Delete Preset",
            f"Are you sure you want to delete the layout preset '{current}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if choice == QMessageBox.StandardButton.Yes:
            presets = [self.preset_combo.itemText(i) for i in range(self.preset_combo.count())]
            presets.remove(current)
            self.settings.setValue("layout_presets_list", presets)
            
            self.settings.remove(f"layout/{current}")
            
            idx = self.preset_combo.findText(current)
            if idx >= 0:
                self.preset_combo.removeItem(idx)
            self.preset_combo.setCurrentText("Default")

    def _save(self):
        normalized = normalize_lemonade_api_base_url(self.url_edit.text().strip())
        if should_warn_http_to_non_loopback(normalized):
            choice = QMessageBox.warning(
                self,
                t("settings.warn.http_remote.title"),
                t("settings.warn.http_remote.body"),
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if choice == QMessageBox.StandardButton.Cancel:
                return
        self.url_edit.setText(normalized)
        self.settings.setValue(KEY_API_URL, normalized)
        self.settings.setValue(KEY_API_MODEL, self.model_combo.currentText())
        self.settings.setValue("layout_preset", self.preset_combo.currentText())
        self.settings.setValue(KEY_SOURCE_LANG, self.source_lang_combo.currentText())
        self.settings.setValue(KEY_TARGET_LANG, self.lang_combo.currentText())
        self.settings.setValue(KEY_MODE, self.mode_combo.currentText())
        ui_lang = self.ui_lang_combo.currentText()
        self.settings.setValue(KEY_UI_LANG, ui_lang)
        state.load_locale(ui_lang)
        selected_styles = [s for s, cb in self.style_checkboxes.items() if cb.isChecked()]
        self.settings.setValue(KEY_STYLES, selected_styles)
        self.settings.setValue(KEY_MAX_TOKENS, self.tokens_spin.value())
        self.settings.setValue(KEY_LEADER_KEY, self.leader_combo.currentData())
        self.settings.setValue(KEY_GPU_UTIL, self.gpu_combo.currentText())
        self.settings.setValue(KEY_DIALOGUE_DELAY, self.delay_spin.value())
        self.settings.setValue(KEY_AUTO_CONTINUE, "true" if self.auto_continue_cb.isChecked() else "false")
        self.settings.setValue(KEY_AUTO_SPEAK, "true" if self.auto_speak_cb.isChecked() else "false")
        self.settings.setValue("live_voice_raid", "true" if self.live_voice_raid_cb.isChecked() else "false")
        self.settings.setValue("live_raid_lore_save", "true" if self.live_raid_lore_save_cb.isChecked() else "false")
        self.settings.setValue("hud_show_original", "true" if self.hud_show_original_cb.isChecked() else "false")
        self.settings.setValue("hud_show_pinyin", "true" if self.hud_show_pinyin_cb.isChecked() else "false")
        target_val = self.target_window_combo.currentText().strip()
        if self.target_window_combo.currentIndex() == 0 or target_val == t("settings.option.none_overlay") or target_val == "None (Standard Overlay Mode)":
            target_val = ""
        self.settings.setValue(KEY_TARGET_WINDOW_TITLE, target_val)
        self.settings.setValue("developer_options", "true" if self.dev_options_cb.isChecked() else "false")
        self.accept()

    def _on_edit_layout(self):
        self.layout_edit_requested.emit()
        self.accept()

    def _update_dev_visibility(self, checked):
        self.live_voice_raid_cb.setVisible(checked)
        self.live_raid_lore_save_cb.setVisible(checked)
        self.hud_show_original_cb.setVisible(checked)
        self.hud_show_pinyin_cb.setVisible(checked)


class XianApp(QWidget):
    """Tray-resident application controller.

    Inherits QWidget so it participates in the Qt object tree (preventing
    garbage collection), but is never shown as a window.
    """

    def __init__(self):
        super().__init__()
        self.hide()  # never shown

        self.settings = QSettings(ORGANIZATION, APP_NAME)
        # Initialize translation state
        state.load_locale(self.settings.value(KEY_UI_LANG, "en"))

        # Migrate deprecated MAGE model setting
        if self.settings.value(KEY_API_MODEL) == "MAGE":
            self.settings.setValue(KEY_API_MODEL, constants.DEFAULT_MODEL)
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
        self._prewarm_worker.status_changed.connect(self._on_prewarm_status)
        self._prewarm_worker.start()

        # --- Layout Edit State ---
        self.layout_edit_mode_active = False

        # --- Hotkeys ---
        self.hotkey_listener = create_hotkey_listener()
        
        # Load and set initial leader key
        initial_leader = self.settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
        if hasattr(self.hotkey_listener, "set_leader_key"):
            self.hotkey_listener.set_leader_key(initial_leader)
            
        self.hotkey_listener.trigger_lens.connect(self.show_lens)
        self.hotkey_listener.trigger_chat.connect(self.toggle_chat)
        self.hotkey_listener.trigger_settings.connect(self._open_settings)

        self.hotkey_listener.trigger_cinematic_mode.connect(self.toggle_cinematic_mode)
        self.hotkey_listener.trigger_how_to_say.connect(self.show_how_to_say)
        self.hotkey_listener.trigger_raid_mode.connect(self.start_raid_mode)
        self.hotkey_listener.cinematic_capture.connect(self._on_cinematic_trigger)
        self.hotkey_listener.command_mode_started.connect(self._on_command_mode_started)
        if hasattr(self.hotkey_listener, "command_mode_cancelled"):
            self.hotkey_listener.command_mode_cancelled.connect(self.hide_osd)

        # --- HUD Manager ---
        self.hud_manager = HudManager(self)
        self.hotkey_listener.trigger_hud.connect(self.show_hud_presets)

        self.hotkey_listener.start()

        # --- Cinematic Mode ---
        self.cinematic_mode_active = False
        self.cinematic_bubble = None

        # --- Raid Mode ---
        self.raid_bubble = None
        self.raid_window = None
        self._raid_worker = None

        # --- Dialogue Mode ---
        self.dialogue_mode_active = False
        self.dialogue_timer = QTimer(self)
        self.dialogue_timer.setSingleShot(True)
        self.dialogue_timer.timeout.connect(self.capture_dialogue)
        self.mouse_listener = create_mouse_listener()
        self.mouse_listener.left_click.connect(self.on_dialogue_click)
        self.dialogue_bubble = None

        # --- Command OSD ---
        self.osd = CommandOSD(self)
        self.osd.initialize_settings(
            source=self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG),
            target=self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG),
            model=self.settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL)
        )
        dev_val = self.settings.value("developer_options", "false")
        self.osd.set_developer_options_visible(dev_val == "true" or dev_val is True)
        self.osd.setting_changed.connect(self._on_osd_setting_changed)
        self.osd.command_triggered.connect(self._on_osd_command)
        self.osd.osd_hidden.connect(self._on_osd_hidden)
        
        self.osd_timer = QTimer(self)
        self.osd_timer.setSingleShot(True)
        self.osd_timer.timeout.connect(self.hide_osd)

        # --- Chat sidebar (created once, toggled) ---
        self.chat_sidebar = ChatSidebar(self.processor, parent=self)

        # --- How to Say Dialog ---
        self.how_to_say_dialog = HowToSayDialog(self)
        self.how_to_say_dialog.translation_requested.connect(self._on_how_to_say_submit)
        self.how_to_say_dialog.dialog_hidden.connect(self._on_osd_hidden)

        # Lens window reference (created on demand)
        self._lens: LensOverlayWindow | None = None
        # Active inference workers (prevent GC)
        self._workers: list = []
        self._status_worker = None
        self._pull_worker = None
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

        # --- Target Window Binding ---
        self.target_binder = None
        self._target_last_geometry = None
        self._target_was_active = False
        self._target_was_minimized = False
        self.window_tracking_timer = QTimer(self)
        self.window_tracking_timer.timeout.connect(self._track_target_window)
        self.window_tracking_timer.start(100) # Check every 100ms
        self._setup_window_binder()

    # ------------------------------------------------------------------
    # System tray
    # ------------------------------------------------------------------
    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        # Try to load icon, fall back to a theme icon
        from mage.resources import get_resource_path
        icon_path = get_resource_path("xian.png")
        icon = QIcon(icon_path)
        if icon.isNull():
            icon = QIcon.fromTheme("applications-graphics")
        self.tray.setIcon(icon)
        self.tray.setToolTip(t("tray.tooltip"))

        menu = QMenu()
        capture_action = menu.addAction(t("tray.menu.capture"))
        capture_action.triggered.connect(self.show_lens)

        chat_action = menu.addAction(t("tray.menu.chat"))
        chat_action.triggered.connect(self.toggle_chat)

        menu.addSeparator()

        settings_action = menu.addAction(t("tray.menu.settings"))
        settings_action.triggered.connect(self._open_settings)

        menu.addSeparator()

        about_action = menu.addAction(t("tray.menu.about"))
        about_action.triggered.connect(self.show_about_dialog)

        menu.addSeparator()

        quit_action = menu.addAction(t("tray.menu.quit"))
        quit_action.triggered.connect(QApplication.quit)

        self.tray.setContextMenu(menu)
        self.tray.show()

        logger.info("System tray icon initialised")

    def _on_prewarm_status(self, msg: str):
        logger.info("[Prewarm Status] %s", msg)
        if hasattr(self, "tray") and self.tray:
            self.tray.showMessage(
                t("tray.message.setup_title"),
                msg,
                QSystemTrayIcon.MessageIcon.Information,
                5000
            )

    def show_about_dialog(self):
        from mage.resources import get_resource_path
        
        msg = QMessageBox(self)
        msg.setWindowTitle(t("about.dialog.title"))
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(t("about.dialog.text"))
        
        # Try to load and set the icon
        icon_path = get_resource_path("xian.png")
        if os.path.exists(icon_path):
            msg.setWindowIcon(QIcon(icon_path))
            
        msg.exec()

    def hide_osd(self):
        self.osd.hide()
        self.osd_timer.stop()

    def _on_osd_hidden(self):
        if hasattr(self.hotkey_listener, "cancel_command_mode"):
            self.hotkey_listener.cancel_command_mode()

    def _on_command_mode_started(self):
        dev_val = self.settings.value("developer_options", "false")
        self.osd.set_developer_options_visible(dev_val == "true" or dev_val is True)
        self.osd.show_centered()
        if self.target_binder:
            self._apply_transient_parent(self.osd)
            geom = self.target_binder.get_geometry()
            if geom:
                tx, ty, tw, th = geom
                self.osd.move(
                    tx + (tw - self.osd.width()) // 2,
                    ty + (th - self.osd.height()) // 2
                )
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
            self._safe_stop_worker("_prewarm_worker")
            self._prewarm_worker = PrewarmWorker(self.processor)
            self._prewarm_worker.status_changed.connect(self._on_prewarm_status)
            self._prewarm_worker.start()

    def _on_osd_command(self, key: str):
        """Handle option buttons clicked in the OSD."""
        logger.info("OSD command triggered: %s", key)
        self.hide_osd()
        
        if key == "C":
            self.show_lens()
        elif key == "A":
            self.toggle_chat()

        elif key == "M":
            self.toggle_cinematic_mode()
        elif key == "T":
            self.show_how_to_say()
        elif key == "R":
            self.start_raid_mode()
        elif key == "H":
            self.show_hud_presets()

        elif key == "S":
            self._open_settings()

    # ------------------------------------------------------------------
    # Lens
    # ------------------------------------------------------------------
    def show_hud_presets(self):
        """Close OSD and open the HUD preset dialog."""
        self.hide_osd()
        dev_val = self.settings.value("developer_options", "false")
        if not (dev_val == "true" or dev_val is True):
            logger.info("HUD mode bypassed: developer options disabled")
            return
        self.hud_manager.show_hud_presets()

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
            previous_rect=LensOverlayWindow._last_rect
        )
        self._lens.action_requested.connect(self._on_lens_action)
        self._lens.closed.connect(self._on_lens_closed)
        self._lens.showFullScreen()

    def _on_lens_closed(self):
        self._lens = None

    def _on_lens_action(self, action: str, rect: QRect, image_data: bytes):
        """Handle an action from the Lens action bar."""
        logger.info("Lens action: %s, rect: %s", action, rect)

        if action == "dialogue":
            LensOverlayWindow._last_rect = rect
            self.dialogue_mode_active = True
            self.mouse_listener.start()
            
            # Instantly translate the selected dialogue region using the clean crop
            self._run_inference(image_data, "translate", rect)
            
            bubble = ResultBubble(t("dialogue.status.activated"), anchor_rect=rect, auto_close_ms=3000)
            self._add_bubble(bubble)
            logger.info("Dialogue Mode Activated via Lens")
            return

        if action == "chat":
            # Push image into chat context and open sidebar
            self.chat_sidebar.add_image_context(image_data)
            if not self.chat_sidebar.isVisible():
                self.chat_sidebar.show()
                if self.target_binder:
                    self._apply_transient_parent(self.chat_sidebar)
                    geom = self.target_binder.get_geometry()
                    if geom:
                        tx, ty, tw, th = geom
                        self.chat_sidebar.setGeometry(
                            tx + tw - self.chat_sidebar.width(),
                            ty,
                            self.chat_sidebar.width(),
                            th
                        )
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
        if hasattr(self, "layout_edit_mode_active") and self.layout_edit_mode_active:
            bubble.set_edit_mode(True)

    def _replace_persistent_bubble(self, attr_name: str, text: str, original_text: str = "", anchor: QRect = QRect(), border_color: str | None = None, truncated: bool = False, auto_close_ms: int = 0, show_stop: bool = False) -> ResultBubble:
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
            auto_close_ms=auto_close_ms,
            border_color=border_color,
            truncated=truncated,
            show_stop=show_stop,
        )
        setattr(self, attr_name, bubble)
        if truncated:
            bubble.continue_requested.connect(
                lambda b=bubble: self._on_continue_requested(b)
            )
        self._setup_bubble_connections(bubble)
        if hasattr(self, "layout_edit_mode_active") and self.layout_edit_mode_active:
            bubble.set_edit_mode(True)
        return bubble

    def _on_inference_thinking(self, worker):
        anchor = worker.anchor_rect
        
        # If in dialogue mode and we already have a dialogue bubble, reuse it
        if self.dialogue_mode_active and self.dialogue_bubble and self.dialogue_bubble.isVisible():
            bubble = self.dialogue_bubble
            bubble.update_text(t("bubble.status.translating"), original_text="", show_stop=True)
            # Re-position if anchor shifted
            if anchor and not anchor.isEmpty():
                bubble._anchor_rect = anchor
                bubble._position_near(anchor)
        # If in cinematic mode and we already have a cinematic bubble, reuse it
        elif self.cinematic_mode_active and self.cinematic_bubble and self.cinematic_bubble.isVisible():
            bubble = self.cinematic_bubble
            bubble.update_text(t("bubble.status.translating"), original_text="")
            if anchor and not anchor.isEmpty():
                bubble._anchor_rect = anchor
                bubble._position_near(anchor)
        else:
            # Create a new bubble
            show_stop = self.dialogue_mode_active
            bubble = ResultBubble(
                t("bubble.status.translating"),
                anchor_rect=anchor,
                auto_close_ms=30000 if self.dialogue_mode_active else 0,
                show_stop=show_stop,
            )
            if self.dialogue_mode_active:
                self.dialogue_bubble = bubble
            elif worker.action == "cinematic":
                self.cinematic_bubble = bubble
            else:
                self._add_bubble(bubble)
                
        self._active_bubbles[worker] = bubble

    def _on_inference_partial(self, text, action, worker):
        bubble = self._active_bubbles.get(worker)
        if bubble and bubble.isVisible():
            bubble.update_text(text)

    def _on_inference_done(self, results, action, worker):
        """Handle completed inference."""
        anchor = worker.anchor_rect
        bubble = self._active_bubbles.pop(worker, None)

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

        # If a thinking bubble is active, reuse it and update in-place
        if bubble and bubble.isVisible():
            show_stop = (bubble == self.dialogue_bubble)
            bubble.update_text(
                translation_combined if results else "No text detected in the selected region.",
                original_text=original_combined,
                border_color=border_color,
                truncated=any_truncated,
                show_stop=show_stop
            )
            target_bubble = bubble
            
            if any_truncated:
                try:
                    bubble.continue_requested.disconnect()
                except Exception:
                    pass
                bubble.continue_requested.connect(
                    lambda b=bubble: self._on_continue_requested(b)
                )
        else:
            # Fallback if bubble was closed or not found
            if action == "cinematic":
                target_bubble = self._replace_persistent_bubble(
                    "cinematic_bubble",
                    translation_combined if results else "No text detected.",
                    original_text=original_combined,
                    anchor=anchor,
                    border_color=border_color,
                    truncated=any_truncated,
                )
            elif self.dialogue_mode_active:
                target_bubble = self._replace_persistent_bubble(
                    "dialogue_bubble",
                    translation_combined if results else "No text detected.",
                    original_text=original_combined,
                    anchor=anchor,
                    border_color=border_color,
                    truncated=any_truncated,
                    auto_close_ms=30000,
                    show_stop=True,
                )
            else:
                bubble = ResultBubble(
                    translation_combined if results else "No text detected in the selected region.",
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

    def _safe_stop_worker(self, attr_name: str):
        worker = getattr(self, attr_name, None)
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.quit()
                    worker.wait(2000)
            except Exception as e:
                logger.error("Error stopping worker %s: %s", attr_name, e)
            setattr(self, attr_name, None)

    def _setup_bubble_connections(self, bubble):
        self._apply_transient_parent(bubble)
        bubble.speak_source_requested.connect(
            lambda b=bubble: self._on_speak_requested(b, source=True)
        )
        bubble.speak_target_requested.connect(
            lambda b=bubble: self._on_speak_requested(b, source=False)
        )
        bubble.stop_requested.connect(self.disable_dialogue_mode)

    def disable_dialogue_mode(self):
        if self.dialogue_mode_active:
            self.dialogue_mode_active = False
            self.mouse_listener.stop()
            self.dialogue_timer.stop()
            if getattr(self, "dialogue_bubble", None):
                self.dialogue_bubble.close()
                self.dialogue_bubble = None
            
            bubble = ResultBubble(t("dialogue.status.deactivated"), auto_close_ms=3000)
            self._add_bubble(bubble)
            logger.info("Dialogue Mode Deactivated")

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
            if self.target_binder:
                self._apply_transient_parent(self.chat_sidebar)
                geom = self.target_binder.get_geometry()
                if geom:
                    tx, ty, tw, th = geom
                    self.chat_sidebar.setGeometry(
                        tx + tw - self.chat_sidebar.width(),
                        ty,
                        self.chat_sidebar.width(),
                        th
                    )
            self.chat_sidebar.raise_()

    # ------------------------------------------------------------------
    # How to say
    # ------------------------------------------------------------------
    def show_how_to_say(self):
        self.hide_osd()
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        self.how_to_say_dialog.show_centered(target_lang=target_lang, source_lang=source_lang)
        if self.target_binder:
            self._apply_transient_parent(self.how_to_say_dialog)
            geom = self.target_binder.get_geometry()
            if geom:
                tx, ty, tw, th = geom
                self.how_to_say_dialog.move(
                    tx + (tw - self.how_to_say_dialog.width()) // 2,
                    ty + (th - self.how_to_say_dialog.height()) // 2
                )

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



    def clear_active_bubbles(self):
        """Close and clear all translation bubbles to avoid capture and layering issues."""
        if hasattr(self, "_bubbles") and self._bubbles:
            for bubble in self._bubbles:
                try:
                    bubble.close()
                except Exception:
                    pass
            self._bubbles.clear()

        if hasattr(self, "_active_bubbles") and self._active_bubbles:
            for worker, bubble in list(self._active_bubbles.items()):
                try:
                    bubble.close()
                except Exception:
                    pass
            self._active_bubbles.clear()

        if hasattr(self, "dialogue_bubble") and self.dialogue_bubble:
            try:
                self.dialogue_bubble.close()
            except Exception:
                pass
            self.dialogue_bubble = None

        if hasattr(self, "cinematic_bubble") and self.cinematic_bubble:
            try:
                self.cinematic_bubble.close()
            except Exception:
                pass
            self.cinematic_bubble = None

        if hasattr(self, "raid_bubble") and self.raid_bubble:
            try:
                self.raid_bubble.close()
            except Exception:
                pass
            self.raid_bubble = None

        if hasattr(self, "raid_window") and self.raid_window:
            try:
                self.raid_window.close()
            except Exception:
                pass
            self.raid_window = None

    def on_dialogue_click(self):
        if self.dialogue_mode_active:
            if self._is_click_inside_mage(QCursor.pos()):
                logger.debug("Dialogue click ignored: clicked inside MAGE UI element.")
                return
            
            # Immediately close existing bubbles so they are not captured and don't cause layering conflicts on Wayland
            self.clear_active_bubbles()
            
            delay = int(self.settings.value(KEY_DIALOGUE_DELAY, 1000))
            self.dialogue_timer.start(delay)

    def _is_click_inside_mage(self, pos) -> bool:
        if hasattr(self, "_bubbles") and self._bubbles:
            for bubble in self._bubbles:
                try:
                    if bubble.isVisible() and bubble.geometry().contains(pos):
                        return True
                except Exception:
                    pass
        if hasattr(self, "dialogue_bubble") and self.dialogue_bubble:
            try:
                if self.dialogue_bubble.isVisible() and self.dialogue_bubble.geometry().contains(pos):
                    return True
            except Exception:
                pass
        if hasattr(self, "cinematic_bubble") and self.cinematic_bubble:
            try:
                if self.cinematic_bubble.isVisible() and self.cinematic_bubble.geometry().contains(pos):
                    return True
            except Exception:
                pass
        if hasattr(self, "raid_window") and self.raid_window:
            try:
                if self.raid_window.isVisible() and self.raid_window.geometry().contains(pos):
                    return True
            except Exception:
                pass
        if hasattr(self, "chat_sidebar") and self.chat_sidebar:
            try:
                if self.chat_sidebar.isVisible() and self.chat_sidebar.geometry().contains(pos):
                    return True
            except Exception:
                pass
        if hasattr(self, "osd") and self.osd:
            try:
                if self.osd.isVisible() and self.osd.geometry().contains(pos):
                    return True
            except Exception:
                pass
        if hasattr(self, "how_to_say_dialog") and self.how_to_say_dialog:
            try:
                if self.how_to_say_dialog.isVisible() and self.how_to_say_dialog.geometry().contains(pos):
                    return True
            except Exception:
                pass
        if hasattr(self, "_lens") and self._lens:
            try:
                if self._lens.isVisible() and self._lens.geometry().contains(pos):
                    return True
            except Exception:
                pass
        return False

    def capture_dialogue(self):
        if not self.dialogue_mode_active or LensOverlayWindow._last_rect is None or LensOverlayWindow._last_rect.isEmpty():
            return

        if hasattr(self, "_dialogue_capture_worker") and self._dialogue_capture_worker and self._dialogue_capture_worker.isRunning():
            return

        rect = LensOverlayWindow._last_rect
        total_geo = ScreenCapture.get_virtual_desktop_geometry()

        from mage.workers import CaptureWorker
        self._dialogue_capture_worker = CaptureWorker("dialogue", [rect], total_geo)
        
        def on_done(cropped_data, anchor_rect):
            self._run_inference(cropped_data, "translate", anchor_rect)
            self._cleanup_worker(self._dialogue_capture_worker)
            self._dialogue_capture_worker = None

        def on_error(err):
            logger.warning("Dialogue capture error: %s", err)
            self._cleanup_worker(self._dialogue_capture_worker)
            self._dialogue_capture_worker = None

        self._dialogue_capture_worker.capture_done.connect(on_done)
        self._dialogue_capture_worker.error.connect(on_error)
        self._workers.append(self._dialogue_capture_worker)
        self._dialogue_capture_worker.start()

    # ------------------------------------------------------------------
    # Cinematic Mode
    # ------------------------------------------------------------------
    def toggle_cinematic_mode(self):
        self.hide_osd()
        dev_val = self.settings.value("developer_options", "false")
        if not (dev_val == "true" or dev_val is True):
            logger.info("Cinematic mode bypassed: developer options disabled")
            return
            
        if self.cinematic_mode_active:
            self.cinematic_mode_active = False
            self.hotkey_listener.cinematic_mode_active = False
            if self.cinematic_bubble:
                self.cinematic_bubble.close()
                self.cinematic_bubble = None
            bubble = ResultBubble(t("cinematic.status.deactivated"), auto_close_ms=3000)
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
        bubble = ResultBubble(t("cinematic.status.activated"), auto_close_ms=4000)
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
            bubble = ResultBubble(t("cinematic.status.requires_region"), auto_close_ms=4000)
            self._add_bubble(bubble)

    def _on_cinematic_trigger(self):
        if not self.cinematic_mode_active or not CinematicLensOverlay._last_rects:
            return

        if hasattr(self, "_cinematic_capture_worker") and self._cinematic_capture_worker and self._cinematic_capture_worker.isRunning():
            return

        # Show capturing indicator
        self.cinematic_bubble = self._replace_persistent_bubble("cinematic_bubble", "Recording audio and capturing screen...")

        rects = CinematicLensOverlay._last_rects
        total_geo = ScreenCapture.get_virtual_desktop_geometry()

        from mage.workers import CaptureWorker
        self._cinematic_capture_worker = CaptureWorker("cinematic", rects, total_geo)

        def on_done(composite_data, anchor_rect):
            source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
            target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
            styles = _parse_styles(self.settings)

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

            self._cleanup_worker(self._cinematic_capture_worker)
            self._cinematic_capture_worker = None

        def on_error(err):
            logger.warning("Cinematic capture error: %s", err)
            if self.cinematic_bubble and self._is_valid_widget(self.cinematic_bubble):
                self.cinematic_bubble.update_text(f"⚠ Capture Error: {err}")
            self._cleanup_worker(self._cinematic_capture_worker)
            self._cinematic_capture_worker = None

        self._cinematic_capture_worker.capture_done.connect(on_done)
        self._cinematic_capture_worker.error.connect(on_error)
        self._workers.append(self._cinematic_capture_worker)
        self._cinematic_capture_worker.start()

    # ------------------------------------------------------------------
    # Raid Mode
    # ------------------------------------------------------------------
    def start_raid_mode(self):
        self.hide_osd()
        dev_val = self.settings.value("developer_options", "false")
        if not (dev_val == "true" or dev_val is True):
            logger.info("Raid mode bypassed: developer options disabled")
            return
            
        # If already running, toggle OFF
        if hasattr(self, "_raid_worker") and self._raid_worker is not None:
            self.stop_raid_mode()
            return

        logger.info("Triggered Raid Mode")
        
        # Instantiate/show the RaidWindow
        if not hasattr(self, "raid_window") or not self.raid_window:
            self.raid_window = RaidWindow(self.settings, parent=self)
            self.raid_window.audio_toggled.connect(self._on_raid_audio_toggled)
            self.raid_window.stop_requested.connect(self.stop_raid_mode)
            self._apply_transient_parent(self.raid_window)

        self.raid_window.clear_log()
        self.raid_window.set_status(t("raid.window.status.listening"), "listening")
        self.raid_window.show()
        self.raid_window.raise_()

        source_lang = self.settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG)
        target_lang = self.settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG)
        save_lore = self.settings.value("live_raid_lore_save", "false")
        live_voice_raid = self.settings.value("live_voice_raid", "false")

        worker = RaidWorker(
            self.processor,
            target_lang=target_lang,
            source_lang=source_lang,
            save_lore=(save_lore == "true" or save_lore is True),
            audio_enabled=(live_voice_raid == "true" or live_voice_raid is True)
        )

        worker.chunk_translated.connect(
            lambda transcript, translation, audio_bytes, w=worker: self._on_chunk_translated(transcript, translation, audio_bytes, w)
        )
        worker.error.connect(
            lambda msg, w=worker: self._on_raid_error(msg, w)
        )
        worker.progress.connect(
            lambda text, w=worker: self._on_raid_progress(text, w)
        )
        worker.finished.connect(lambda w=worker: self._cleanup_raid_worker(w))

        self._raid_worker = worker
        self._workers.append(worker)
        worker.start()
        logger.info("RaidWorker started")

    def stop_raid_mode(self):
        if hasattr(self, "_raid_worker") and self._raid_worker is not None:
            logger.info("Stopping Raid Mode")
            self._raid_worker.stop()
            self._raid_worker = None
        if hasattr(self, "raid_window") and self.raid_window:
            self.raid_window.close()

    def _cleanup_raid_worker(self, worker):
        self._cleanup_worker(worker)
        if hasattr(self, "_raid_worker") and self._raid_worker == worker:
            self._raid_worker = None
        if hasattr(self, "raid_window") and self.raid_window:
            self.raid_window.set_status(t("raid.window.status.idle"), "idle")

    def _on_raid_audio_toggled(self, checked):
        if hasattr(self, "_raid_worker") and self._raid_worker is not None:
            self._raid_worker.set_audio_enabled(checked)

    def _on_raid_progress(self, text, worker):
        if hasattr(self, "raid_window") and self.raid_window and self.raid_window.isVisible():
            state = "listening"
            if "transcrib" in text.lower():
                state = "processing"
                text = t("raid.window.status.processing")
            elif "translat" in text.lower():
                state = "processing"
                text = t("raid.window.status.processing")
            elif "speech" in text.lower():
                state = "listening"
                text = t("raid.window.status.listening")
            elif "synthesiz" in text.lower():
                state = "processing"
                text = t("raid.window.status.processing")
            self.raid_window.set_status(text, state)

    def _on_chunk_translated(self, transcript, translation, audio_bytes, worker):
        logger.info("Raid chunk translated: %s -> %s", transcript, translation)
        
        if hasattr(self, "raid_window") and self.raid_window and self.raid_window.isVisible():
            self.raid_window.append_translation(transcript, translation)

        # Sequential background TTS synthesis was executed inside RaidWorker.
        # Play the received WAV bytes directly in the background thread.
        live_voice_raid = self.settings.value("live_voice_raid", "false")
        if (live_voice_raid == "true" or live_voice_raid is True) and audio_bytes:
            from mage.capture.audio import play_audio_async
            play_audio_async(audio_bytes)

    def _on_raid_error(self, msg, worker):
        logger.error("Raid Mode error: %s", msg)
        if hasattr(self, "raid_window") and self.raid_window and self.raid_window.isVisible():
            self.raid_window.set_status(f"⚠ Error: {msg}", "error")
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
        self._safe_stop_worker("_status_worker")
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
            if target_model == "MAGE":
                target_model = constants.DEFAULT_MODEL
                self.settings.setValue(KEY_API_MODEL, target_model)

            if self.processor.router.omni_detected:
                omni_id = self.processor.router.omni_model_id
                if omni_id == "MAGE":
                    omni_id = constants.DEFAULT_MODEL
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
            
            # Set the active model on the router to rebuild modality mappings
            self.processor.router.active_model = target_model
        else:
            logger.warning("Lemonade server not reachable")
            self._available_models = []

    def _pull_model(self, model_name: str):
        """Download a model via the Lemonade /v1/pull endpoint."""
        self._safe_stop_worker("_pull_worker")
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
        
        def _handle_layout_edit():
            self.toggle_layout_edit_mode()
            
        dlg.layout_edit_requested.connect(_handle_layout_edit)
        
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
            self._safe_stop_worker("_prewarm_worker")
            self._prewarm_worker = PrewarmWorker(self.processor)
            self._prewarm_worker.status_changed.connect(self._on_prewarm_status)
            self._prewarm_worker.start()
            
            # Apply new leader key
            new_leader = self.settings.value(KEY_LEADER_KEY, constants.DEFAULT_LEADER_KEY)
            if hasattr(self.hotkey_listener, "set_leader_key"):
                self.hotkey_listener.set_leader_key(new_leader)
                
            self._setup_window_binder()
            dev_val = self.settings.value("developer_options", "false")
            self.osd.set_developer_options_visible(dev_val == "true" or dev_val is True)
            
            # Restore geometries according to layout preset
            self.osd.restore_geometry()
            self.chat_sidebar.restore_geometry()
            self.how_to_say_dialog.restore_geometry()
            if hasattr(self, "raid_window") and self.raid_window:
                self.raid_window.restore_geometry()
                
            logger.info("Settings updated and applied")

    def toggle_layout_edit_mode(self):
        """Toggle UI Layout Edit Mode for all overlays."""
        if not hasattr(self, "layout_edit_mode_active"):
            self.layout_edit_mode_active = False
            
        self.layout_edit_mode_active = not self.layout_edit_mode_active
        active = self.layout_edit_mode_active
        logger.info("Layout Edit Mode toggled: %s", active)
        
        # OSD: Make sure it's visible so it can be moved
        if active:
            if not self.osd.isVisible():
                self.osd.show_centered()
        else:
            self.osd.hide()
            
        self.osd.set_edit_mode(active)
        self.chat_sidebar.set_edit_mode(active)
        self.how_to_say_dialog.set_edit_mode(active)
        
        if hasattr(self, "raid_window") and self.raid_window:
            self.raid_window.set_edit_mode(active)
            
        # Dialogue bubble and cinematic bubble
        if hasattr(self, "dialogue_bubble") and self.dialogue_bubble:
            self.dialogue_bubble.set_edit_mode(active)
        if hasattr(self, "cinematic_bubble") and self.cinematic_bubble:
            self.cinematic_bubble.set_edit_mode(active)
            
        # All active bubbles
        for bubble in self._bubbles:
            if bubble.isVisible():
                bubble.set_edit_mode(active)
                
        # All active HUD tooltips
        if hasattr(self, "hud_manager") and self.hud_manager:
            for widget in self.hud_manager.tooltip_widgets:
                if widget.isVisible():
                    widget.set_edit_mode(active)

    def _is_valid_widget(self, w):
        if w is None:
            return False
        try:
            from PyQt6 import sip
            return not sip.isdeleted(w)
        except Exception:
            try:
                import sip
                return not sip.isdeleted(w)
            except Exception:
                return False

    def _setup_window_binder(self):
        """Initializes or updates the WindowBinder based on the current settings."""
        if hasattr(self, "target_binder") and self.target_binder:
            try:
                self.target_binder.close()
            except Exception:
                pass
            self.target_binder = None

        target_title = self.settings.value(KEY_TARGET_WINDOW_TITLE, "")
        if target_title:
            logger.info("Setting up window binding for target title: '%s'", target_title)
            self.target_binder = WindowBinder(target_title)
            self._target_last_geometry = None
            self._target_was_active = False
            self._target_was_minimized = False
            
            # Apply transient parent to already created widgets
            self._apply_transient_parent(self.osd)
            self._apply_transient_parent(self.chat_sidebar)
            self._apply_transient_parent(self.how_to_say_dialog)
        else:
            self.target_binder = None

    def _apply_transient_parent(self, widget):
        if not self.target_binder or not self._is_valid_widget(widget):
            return
        native_id = self.target_binder.get_native_id()
        if native_id:
            try:
                from PyQt6.QtGui import QWindow
                widget.winId()
                handle = widget.windowHandle()
                if handle:
                    foreign_window = QWindow.fromWinId(native_id)
                    if foreign_window:
                        handle.setTransientParent(foreign_window)
            except Exception as e:
                logger.debug("Failed to set transient parent for widget %s: %s", widget, e)

    def _track_target_window(self):
        if not self.target_binder:
            return

        try:
            exists = self.target_binder.exists()
            if not exists:
                return

            geom = self.target_binder.get_geometry()
            # If we cannot retrieve geometry (e.g. Wayland), fall back to floating mode
            if not geom:
                return

            is_minimized = self.target_binder.is_minimized()
            is_active = self.target_binder.is_active()

            # Check if focus is on any of our own overlays
            our_window_active = False
            active_win = QApplication.activeWindow()
            if active_win and self._is_valid_widget(active_win):
                our_window_active = (
                    active_win == self.osd or
                    active_win == self.chat_sidebar or
                    active_win == self.how_to_say_dialog or
                    any(active_win == b for b in self._bubbles if self._is_valid_widget(b)) or
                    any(active_win == b for b in self._active_bubbles.values() if self._is_valid_widget(b)) or
                    active_win == self.cinematic_bubble or
                    active_win == self.raid_window or
                    active_win == self.dialogue_bubble
                )

            target_should_be_visible = not is_minimized and (is_active or our_window_active)

            # Get list of all overlay windows
            raw_overlays = []
            if self.osd:
                raw_overlays.append(self.osd)
            if self.chat_sidebar:
                raw_overlays.append(self.chat_sidebar)
            if self.how_to_say_dialog:
                raw_overlays.append(self.how_to_say_dialog)

            raw_overlays.extend(self._bubbles)
            raw_overlays.extend(self._active_bubbles.values())

            if self.cinematic_bubble:
                raw_overlays.append(self.cinematic_bubble)
            if self.raid_window:
                raw_overlays.append(self.raid_window)
            if self.dialogue_bubble:
                raw_overlays.append(self.dialogue_bubble)

            overlays = []
            for w in raw_overlays:
                if self._is_valid_widget(w):
                    overlays.append(w)

            # Update visibility based on target active state
            for overlay in overlays:
                if not target_should_be_visible:
                    # Hide overlay if visible and not already marked
                    if overlay.isVisible():
                        overlay.setVisible(False)
                        overlay._temp_hidden_by_binder = True
                else:
                    # Restore visibility if temporarily hidden by binder
                    if getattr(overlay, "_temp_hidden_by_binder", False):
                        overlay.setVisible(True)
                        overlay._temp_hidden_by_binder = False

            if is_minimized or not target_should_be_visible:
                return

            # Geometry tracking and overlay tracking / translation
            if self._target_last_geometry is None:
                self._align_overlays_to_target(geom)
            elif geom != self._target_last_geometry:
                old_x, old_y, old_w, old_h = self._target_last_geometry
                new_x, new_y, new_w, new_h = geom
                dx = new_x - old_x
                dy = new_y - old_y
                self._translate_overlays_by_delta(dx, dy, new_w - old_w, new_h - old_h, geom)

            self._target_last_geometry = geom

        except Exception as e:
            logger.error("Error in _track_target_window: %s", e)

    def _align_overlays_to_target(self, geom):
        """Align active overlays to the target window geometry initially."""
        tx, ty, tw, th = geom

        if self._is_valid_widget(self.osd) and self.osd.isVisible():
            self.osd.move(
                tx + (tw - self.osd.width()) // 2,
                ty + (th - self.osd.height()) // 2
            )

        if self._is_valid_widget(self.how_to_say_dialog) and self.how_to_say_dialog.isVisible():
            self.how_to_say_dialog.move(
                tx + (tw - self.how_to_say_dialog.width()) // 2,
                ty + (th - self.how_to_say_dialog.height()) // 2
            )

        if self._is_valid_widget(self.chat_sidebar) and self.chat_sidebar.isVisible():
            self.chat_sidebar.setGeometry(
                tx + tw - self.chat_sidebar.width(),
                ty,
                self.chat_sidebar.width(),
                th
            )

    def _translate_overlays_by_delta(self, dx, dy, dw, dh, target_geom):
        """Translate or resize overlays based on target window movement delta."""
        tx, ty, tw, th = target_geom

        bubbles = []
        for b in self._bubbles:
            if self._is_valid_widget(b):
                bubbles.append(b)
        for b in self._active_bubbles.values():
            if self._is_valid_widget(b):
                bubbles.append(b)
        for b in [self.cinematic_bubble, self.raid_bubble, self.dialogue_bubble, self.raid_window]:
            if self._is_valid_widget(b):
                bubbles.append(b)

        for bubble in bubbles:
            if bubble.isVisible():
                bubble.move(bubble.x() + dx, bubble.y() + dy)

        if self._is_valid_widget(self.osd) and self.osd.isVisible():
            self.osd.move(
                tx + (tw - self.osd.width()) // 2,
                ty + (th - self.osd.height()) // 2
            )

        if self._is_valid_widget(self.how_to_say_dialog) and self.how_to_say_dialog.isVisible():
            self.how_to_say_dialog.move(
                tx + (tw - self.how_to_say_dialog.width()) // 2,
                ty + (th - self.how_to_say_dialog.height()) // 2
            )

        if self._is_valid_widget(self.chat_sidebar) and self.chat_sidebar.isVisible():
            self.chat_sidebar.setGeometry(
                tx + tw - self.chat_sidebar.width(),
                ty,
                self.chat_sidebar.width(),
                th
            )

    def closeEvent(self, event):
        if hasattr(self, "target_binder") and self.target_binder:
            try:
                self.target_binder.close()
            except Exception:
                pass
        if hasattr(self, "hotkey_listener") and self.hotkey_listener:
            try:
                self.hotkey_listener.stop()
            except Exception as e:
                logger.error("Error stopping hotkey listener on close: %s", e)
        if hasattr(self, "mouse_listener") and self.mouse_listener:
            try:
                self.mouse_listener.stop()
            except Exception as e:
                logger.error("Error stopping mouse listener on close: %s", e)
        super().closeEvent(event)
