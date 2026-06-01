"""Lightweight floating widget to display a translation result near the Lens selection."""

import logging
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QApplication
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication, QCursor
from mage.ui.theme import accent_hex

logger = logging.getLogger(__name__)


class ResultBubble(QWidget):
    """Semi-transparent floating panel that shows translated text.

    Positioned just below *anchor_rect* (in global screen coordinates).
    Auto-fades after *auto_close_ms* milliseconds.  Clicking the body
    copies the text to the clipboard.
    """

    continue_requested = pyqtSignal()
    speak_source_requested = pyqtSignal()
    speak_target_requested = pyqtSignal()

    def __init__(self, text: str, original_text: str = "",
                 anchor_rect: QRect = None, auto_close_ms: int = 30000,
                 border_color: str | None = None,
                 truncated: bool = False,
                 parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._text = text
        self._original = original_text
        self._anchor_rect = anchor_rect
        self._orig_label = None
        self._drag_position = QPoint()

        border = border_color if border_color else accent_hex()

        # --- layout -----------------------------------------------------------
        root = QWidget(self)
        root.setObjectName("BubbleRoot")
        root.setStyleSheet(f"""
            #BubbleRoot {{
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid {border};
                border-radius: 8px;
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Header row with close button
        header = QHBoxLayout()
        display_title = "⚠️ Xian — Speculative" if border_color else "Xian — Translation"
        title = QLabel(display_title)
        title_color = border_color if border_color else accent_hex()
        title.setStyleSheet(f"color: {title_color}; font-weight: bold; font-size: 11px;")
        header.addWidget(title)
        header.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        layout.addLayout(header)

        # Original text (collapsed by default)
        if original_text:
            self._orig_label = QLabel(original_text)
            self._orig_label.setWordWrap(True)
            self._orig_label.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
            self._orig_label.setMaximumHeight(60)
            layout.addWidget(self._orig_label)

        # Translation
        self._trans_label = QLabel(text)
        self._trans_label.setWordWrap(True)
        self._trans_label.setFont(QFont("sans-serif", 13))
        self._trans_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._trans_label)

        # Footer
        footer = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_to_clipboard)
        footer.addWidget(copy_btn)

        self._continue_btn = QPushButton("▶ Continue")
        self._continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._continue_btn.setStyleSheet(
            "color: #4ecdc4; font-weight: bold; font-size: 11px;"
        )
        self._continue_btn.clicked.connect(self._on_continue_clicked)
        self._continue_btn.setVisible(truncated)
        footer.addWidget(self._continue_btn)

        self._speak_src_btn = QPushButton("🔊 Speak (Src)")
        self._speak_src_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._speak_src_btn.clicked.connect(self._on_speak_src_clicked)
        self._speak_src_btn.setEnabled(bool(original_text))
        footer.addWidget(self._speak_src_btn)

        self._speak_tgt_btn = QPushButton("🔊 Speak (Tgt)")
        self._speak_tgt_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._speak_tgt_btn.clicked.connect(self._on_speak_tgt_clicked)
        footer.addWidget(self._speak_tgt_btn)

        footer.addStretch()
        layout.addLayout(footer)

        # --- sizing & positioning --------------------------------------------
        self.adjustSize()
        # Clamp width
        max_w = 500
        if self.width() > max_w:
            self.setFixedWidth(max_w)
            self.adjustSize()

        if anchor_rect and not anchor_rect.isEmpty():
            self._position_near(anchor_rect)

        # Auto-close timer
        if auto_close_ms > 0:
            QTimer.singleShot(auto_close_ms, self.close)

        self.show()

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

    def update_text(self, text: str, original_text: str = "") -> None:
        """Update displayed translation text in-place."""
        self._text = text
        self._trans_label.setText(text)
        if original_text:
            self._original = original_text
            if self._orig_label:
                self._orig_label.setText(original_text)
                self._orig_label.show()
            if hasattr(self, "_speak_src_btn"):
                self._speak_src_btn.setEnabled(bool(original_text))
        
        self.adjustSize()
        # Clamp width
        max_w = 500
        if self.width() > max_w:
            self.setFixedWidth(max_w)
            self.adjustSize()
        if self._anchor_rect and not self._anchor_rect.isEmpty():
            self._position_near(self._anchor_rect)

    def _on_continue_clicked(self):
        self._continue_btn.setText("⏳ Continuing...")
        self._continue_btn.setEnabled(False)
        self.continue_requested.emit()

    def show_continue_button(self, visible: bool = True) -> None:
        """Show or hide the continue button (e.g. after continuation completes)."""
        self._continue_btn.setVisible(visible)
        if visible:
            self._continue_btn.setText("▶ Continue")
            self._continue_btn.setEnabled(True)

    def _on_speak_src_clicked(self):
        self.speak_source_requested.emit()

    def _on_speak_tgt_clicked(self):
        self.speak_target_requested.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
        else:
            super().mouseMoveEvent(event)
