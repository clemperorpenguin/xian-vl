import logging
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QComboBox
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QFont

from mage.ui.theme import accent_hex
from shared_types.enums import SourceLanguage, TargetLanguage

logger = logging.getLogger(__name__)

class CommandOSD(QWidget):
    setting_changed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.SplashScreen |
            Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.bg_frame = QFrame()
        self.bg_frame.setObjectName("OsdBg")
        self.bg_frame.setStyleSheet(f"""
            #OsdBg {{
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid {accent_hex()};
                border-radius: 12px;
            }}
        """)
        layout.addWidget(self.bg_frame)

        inner_layout = QVBoxLayout(self.bg_frame)
        inner_layout.setContentsMargins(32, 24, 32, 24)
        inner_layout.setSpacing(20)

        # Title
        title = QLabel("COMMAND MODE")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont("sans-serif", 13, QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2.0)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {accent_hex()};")
        inner_layout.addWidget(title)

        # Options Container
        options_layout = QHBoxLayout()
        options_layout.setSpacing(32)

        # We create a helper to build the options
        options_layout.addWidget(self._create_option("C", "Capture"))
        options_layout.addWidget(self._create_option("A", "Chat"))
        options_layout.addWidget(self._create_option("S", "Settings"))

        inner_layout.addLayout(options_layout)

        # Settings Container
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 16, 0, 0)
        settings_layout.setSpacing(12)

        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems([e.value for e in SourceLanguage])
        self.source_lang_combo.currentTextChanged.connect(
            lambda v: self.setting_changed.emit("source_lang", v)
        )

        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems([e.value for e in TargetLanguage])
        self.target_lang_combo.currentTextChanged.connect(
            lambda v: self.setting_changed.emit("target_lang", v)
        )

        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(
            lambda v: self.setting_changed.emit("api_model", v)
        )

        for cb in (self.source_lang_combo, self.target_lang_combo, self.model_combo):
            cb.setStyleSheet(f"""
                QComboBox {{
                    background-color: #2A2A2A;
                    color: #DDD;
                    border: 1px solid #444;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-size: 12px;
                }}
                QComboBox::drop-down {{
                    border: none;
                }}
            """)

        lbl_style = "color: #999; font-size: 12px; font-weight: bold;"
        sl = QLabel("Src:")
        sl.setStyleSheet(lbl_style)
        tl = QLabel("Tgt:")
        tl.setStyleSheet(lbl_style)
        ml = QLabel("Model:")
        ml.setStyleSheet(lbl_style)

        settings_layout.addWidget(sl)
        settings_layout.addWidget(self.source_lang_combo)
        settings_layout.addWidget(tl)
        settings_layout.addWidget(self.target_lang_combo)
        settings_layout.addWidget(ml)
        settings_layout.addWidget(self.model_combo)

        inner_layout.addLayout(settings_layout)

    def _create_option(self, key: str, label_text: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon placeholder (for user to replace later)
        icon_lbl = QLabel()
        icon_lbl.setFixedSize(64, 64)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(255, 255, 255, 10);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 30);
                color: {accent_hex()};
            }}
        """)
        icon_lbl.setText(key)
        icon_lbl.setFont(QFont("sans-serif", 24, QFont.Weight.Bold))

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #DDD; font-size: 14px; font-weight: bold;")

        layout.addWidget(icon_lbl)
        layout.addWidget(label)

        return container

    def update_models(self, models: list):
        self.model_combo.blockSignals(True)
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if current in models:
            self.model_combo.setCurrentText(current)
        self.model_combo.blockSignals(False)

    def initialize_settings(self, source: str, target: str, model: str):
        self.source_lang_combo.blockSignals(True)
        self.source_lang_combo.setCurrentText(source)
        self.source_lang_combo.blockSignals(False)

        self.target_lang_combo.blockSignals(True)
        self.target_lang_combo.setCurrentText(target)
        self.target_lang_combo.blockSignals(False)

        self.model_combo.blockSignals(True)
        items = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        if model not in items:
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(model)
        self.model_combo.blockSignals(False)

    def show_centered(self):
        """Adjusts the size and moves the OSD to the absolute center of the screen."""
        self.adjustSize()
        self.setFixedSize(self.size())
        
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            x = geo.center().x() - self.width() // 2
            y = geo.center().y() - self.height() // 2
            self.move(x, y)
        self.show()
