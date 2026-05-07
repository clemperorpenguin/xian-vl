"""Lightweight floating widget to display a translation result near the Lens selection."""

import logging
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QApplication
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from PyQt6.QtGui import QFont, QGuiApplication
from .theme import accent_hex

logger = logging.getLogger(__name__)


class ResultBubble(QWidget):
    """Semi-transparent floating panel that shows translated text.

    Positioned just below *anchor_rect* (in global screen coordinates).
    Auto-fades after *auto_close_ms* milliseconds.  Clicking the body
    copies the text to the clipboard.
    """

    def __init__(self, text: str, original_text: str = "",
                 anchor_rect: QRect = None, auto_close_ms: int = 30000,
                 parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._text = text
        self._original = original_text

        # --- layout -----------------------------------------------------------
        root = QWidget(self)
        root.setObjectName("BubbleRoot")
        root.setStyleSheet(f"""
            #BubbleRoot {{
                background-color: rgba(20, 20, 20, 230);
                border: 1px solid {accent_hex()};
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
        title = QLabel("Xian — Translation")
        title.setStyleSheet(f"color: {accent_hex()}; font-weight: bold; font-size: 11px;")
        header.addWidget(title)
        header.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        layout.addLayout(header)

        # Original text (collapsed by default)
        if original_text:
            orig_label = QLabel(original_text)
            orig_label.setWordWrap(True)
            orig_label.setStyleSheet("color: #999; font-size: 11px; font-style: italic;")
            orig_label.setMaximumHeight(60)
            layout.addWidget(orig_label)

        # Translation
        trans_label = QLabel(text)
        trans_label.setWordWrap(True)
        trans_label.setFont(QFont("sans-serif", 13))
        trans_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(trans_label)

        # Footer
        footer = QHBoxLayout()
        copy_btn = QPushButton("📋 Copy")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self._copy_to_clipboard)
        footer.addWidget(copy_btn)
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
