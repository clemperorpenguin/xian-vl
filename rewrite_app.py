import re

with open('apps/mage-client/src/mage/app.py', 'r') as f:
    content = f.read()

# Replace imports
content = content.replace(
    '    QHBoxLayout, QWidget, QCheckBox, QMessageBox, QInputDialog\n',
    '    QHBoxLayout, QWidget, QCheckBox, QMessageBox, QInputDialog, QTabWidget\n'
)

# Find the start of SettingsDialog.__init__ and the end
start_marker = '    def __init__(self, settings: QSettings, models: list, parent=None):'
end_marker = '    def _add_preset(self):'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers")
    exit(1)

new_init = '''    def __init__(self, settings: QSettings, models: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("settings.dialog.title"))
        self.setMinimumWidth(450)
        self.settings = settings

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab: General
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)

        self.ui_lang_combo = QComboBox()
        self.ui_lang_combo.addItems(["en", "zh", "ja", "ko", "ru", "es", "ar", "hi", "vi"])
        self.ui_lang_combo.setCurrentText(settings.value(KEY_UI_LANG, "en"))
        general_layout.addRow(t("settings.label.ui_language"), self.ui_lang_combo)

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
        general_layout.addRow(t("settings.label.target_window_title"), self.target_window_combo)

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
        general_layout.addRow(t("settings.label.layout_preset"), preset_layout)

        self.tabs.addTab(general_tab, "General")

        # Tab: Backend
        backend_tab = QWidget()
        backend_layout = QFormLayout(backend_tab)

        self.url_edit = QLineEdit()
        self.url_edit.setText(_normalized_api_url_from_settings(settings))
        backend_layout.addRow(t("settings.label.server_url"), self.url_edit)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        if models:
            self.model_combo.addItems(models)
        self.model_combo.setCurrentText(settings.value(KEY_API_MODEL, constants.DEFAULT_MODEL))
        backend_layout.addRow(t("settings.label.model"), self.model_combo)

        self.tokens_spin = QSpinBox()
        self.tokens_spin.setRange(256, 32768)
        self.tokens_spin.setValue(int(settings.value(KEY_MAX_TOKENS, constants.DEFAULT_MAX_TOKENS)))
        backend_layout.addRow(t("settings.label.max_tokens"), self.tokens_spin)

        self.gpu_combo = QComboBox()
        self.gpu_combo.addItems(["Default", "0.5", "0.75"])
        self.gpu_combo.setCurrentText(settings.value(KEY_GPU_UTIL, constants.DEFAULT_GPU_MEMORY_UTILIZATION))
        backend_layout.addRow(t("settings.label.gpu_memory_utilization"), self.gpu_combo)

        self.tabs.addTab(backend_tab, "Backend")

        # Tab: Translation
        trans_tab = QWidget()
        trans_layout = QFormLayout(trans_tab)

        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems([e.value for e in SourceLanguage])
        self.source_lang_combo.setCurrentText(settings.value(KEY_SOURCE_LANG, constants.DEFAULT_SOURCE_LANG))
        trans_layout.addRow(t("settings.label.source_language"), self.source_lang_combo)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems([e.value for e in TargetLanguage])
        self.lang_combo.setCurrentText(settings.value(KEY_TARGET_LANG, constants.DEFAULT_TARGET_LANG))
        trans_layout.addRow(t("settings.label.target_language"), self.lang_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([e.value for e in TranslationMode])
        self.mode_combo.setCurrentText(settings.value(KEY_MODE, constants.DEFAULT_MODE))
        trans_layout.addRow(t("settings.label.mode"), self.mode_combo)

        style_layout = QVBoxLayout()
        self.style_checkboxes = {}
        saved_styles = _parse_styles(settings)
        for style in TranslationStyle:
            cb = QCheckBox(style.value)
            if style.value in saved_styles:
                cb.setChecked(True)
            self.style_checkboxes[style.value] = cb
            style_layout.addWidget(cb)
        trans_layout.addRow(t("settings.label.styles"), style_layout)

        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(100, 10000)
        self.delay_spin.setSingleStep(100)
        self.delay_spin.setValue(int(settings.value(KEY_DIALOGUE_DELAY, 1000)))
        trans_layout.addRow(t("settings.label.dialogue_delay"), self.delay_spin)

        self.tabs.addTab(trans_tab, "Translation")

        # Tab: Features
        features_tab = QWidget()
        features_layout = QFormLayout(features_tab)

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
        features_layout.addRow(t("settings.label.leader_key"), self.leader_combo)

        self.auto_continue_cb = QCheckBox(t("settings.checkbox.auto_continue"))
        auto_val = settings.value(KEY_AUTO_CONTINUE, "false")
        self.auto_continue_cb.setChecked(auto_val == "true" or auto_val is True)
        features_layout.addRow(self.auto_continue_cb)

        self.auto_speak_cb = QCheckBox(t("settings.checkbox.auto_speak"))
        speak_val = settings.value(KEY_AUTO_SPEAK, "false")
        self.auto_speak_cb.setChecked(speak_val == "true" or speak_val is True)
        features_layout.addRow(self.auto_speak_cb)

        self.live_voice_raid_cb = QCheckBox(t("settings.checkbox.live_voice_raid"))
        lv_raid_val = settings.value("live_voice_raid", "false")
        self.live_voice_raid_cb.setChecked(lv_raid_val == "true" or lv_raid_val is True)
        features_layout.addRow(self.live_voice_raid_cb)
        
        self.live_raid_lore_save_cb = QCheckBox(t("settings.checkbox.live_raid_lore_save"))
        lr_lore_val = settings.value("live_raid_lore_save", "false")
        self.live_raid_lore_save_cb.setChecked(lr_lore_val == "true" or lr_lore_val is True)
        features_layout.addRow(self.live_raid_lore_save_cb)

        self.hud_show_original_cb = QCheckBox(t("settings.checkbox.hud_show_original"))
        hud_orig_val = settings.value("hud_show_original", "true")
        self.hud_show_original_cb.setChecked(hud_orig_val == "true" or hud_orig_val is True)
        features_layout.addRow(self.hud_show_original_cb)

        self.hud_show_pinyin_cb = QCheckBox(t("settings.checkbox.hud_show_pinyin"))
        hud_pin_val = settings.value("hud_show_pinyin", "false")
        self.hud_show_pinyin_cb.setChecked(hud_pin_val == "true" or hud_pin_val is True)
        features_layout.addRow(self.hud_show_pinyin_cb)

        self.dev_options_cb = QCheckBox(t("settings.checkbox.dev_options"))
        dev_options_val = settings.value("developer_options", "false")
        self.dev_options_cb.setChecked(dev_options_val == "true" or dev_options_val is True)
        features_layout.addRow(self.dev_options_cb)
        self.dev_options_cb.toggled.connect(self._update_dev_visibility)
        self._update_dev_visibility(self.dev_options_cb.isChecked())

        self.tabs.addTab(features_tab, "Features")

        # Buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton(t("settings.button.save"))
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton(t("settings.button.cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        main_layout.addLayout(btn_row)

        self.setStyleSheet("""
            QDialog { background: #1e1e1e; color: #eee; }
            QLabel, QCheckBox { color: #ccc; }
            QLineEdit, QComboBox, QSpinBox {
                background: #2a2a2a; color: #eee; border: 1px solid #555;
                border-radius: 4px; padding: 4px;
            }
            QPushButton {
                background: %s; color: white; border: none;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background: %s; }
            QTabWidget::pane { border: 1px solid #555; background: #1e1e1e; }
            QTabBar::tab { background: #2a2a2a; color: #ccc; padding: 8px 16px; border: 1px solid #555; }
            QTabBar::tab:selected { background: %s; color: white; }
        """ % (accent_hex(), accent_hover_hex(), accent_hex()))

'''

content = content[:start_idx] + new_init + content[end_idx:]

# Remove unnecessary comments
comments_to_remove = [
    "        # Initialize translation state\n",
    "        # Migrate deprecated MAGE model setting\n",
    "        # --- Core objects ---\n",
    "        # Pre-warm target model in VRAM\n",
    "        # --- Layout Edit State ---\n",
    "        # --- Hotkeys ---\n",
    "        # Load and set initial leader key\n",
    "        # --- HUD Manager ---\n",
    "        # --- Cinematic Mode ---\n",
    "        # --- Raid Mode ---\n",
    "        # --- Dialogue Mode ---\n",
    "        # --- Command OSD ---\n",
    "        # --- Chat sidebar (created once, toggled) ---\n",
    "        # --- How to Say Dialog ---\n",
    "        # Lens window reference (created on demand)\n",
    "        # Active inference workers (prevent GC)\n",
    "        # Active result bubbles (prevent GC)\n",
    "        # Map active InferenceWorker to its temporary/loading ResultBubble\n",
    "        # Guard against repeated pull attempts\n",
    "        # --- System tray ---\n",
    "        # --- Initial health check ---\n",
    "        # --- Target Window Binding ---\n",
    "    # ------------------------------------------------------------------\n",
    "    # System tray\n",
    "    # ------------------------------------------------------------------\n",
    "    # ------------------------------------------------------------------\n",
    "    # Lens\n",
    "    # ------------------------------------------------------------------\n",
    "    # ------------------------------------------------------------------\n",
    "    # Inference\n",
    "    # ------------------------------------------------------------------\n"
]

for c in comments_to_remove:
    content = content.replace(c, "")

with open('apps/mage-client/src/mage/app.py', 'w') as f:
    f.write(content)

