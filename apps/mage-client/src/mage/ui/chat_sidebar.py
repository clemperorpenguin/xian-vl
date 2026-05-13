import logging
import asyncio
import re
import io
from PIL import Image
from html import escape as html_escape
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QHBoxLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRect
from PyQt6.QtGui import QGuiApplication
from mage.ui.grounding import GroundingHighlight
from mage.ui.theme import accent_hex, accent_hover_hex

logger = logging.getLogger(__name__)

class ChatWorker(QThread):
    result_ready = pyqtSignal(str)
    
    def __init__(self, processor, message):
        super().__init__()
        self.processor = processor
        self.message = message
        
    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.processor.process_chat(self.message))
            self.result_ready.emit(response)
        except Exception as e:
            logger.error("Chat worker error: %s", e)
            self.result_ready.emit(f"Error: {e}")
        finally:
            loop.close()

class ChatSidebar(QWidget):
    def __init__(self, processor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self._highlight = None  # reference to active GroundingHighlight
        self.worker = None  # reference to active ChatWorker
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.setFixedWidth(350)
        
        # Position on the right side of the screen
        primary = QGuiApplication.primaryScreen()
        if primary is None:
            self.setGeometry(0, 0, 350, 600)
        else:
            screen = primary.geometry()
            self.setGeometry(screen.right() - 350, screen.top(), 350, screen.height())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Style
        self.setStyleSheet(f"""
            QWidget {{
                background-color: rgba(20, 20, 20, 240);
                color: white;
                border-left: 1px solid {accent_hex()};
            }}
            QTextEdit {{
                background-color: transparent;
                border: none;
                font-size: 14px;
            }}
            QLineEdit {{
                background-color: #333;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }}
            QPushButton {{
                background-color: {accent_hex()};
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {accent_hover_hex()};
            }}
        """)
        
        # History
        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        layout.addWidget(self.history_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask the assistant...")
        self.input_field.returnPressed.connect(self._send_message)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_message)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet("background-color: #d32f2f;")
        self.close_btn.clicked.connect(self.hide)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        input_layout.addWidget(self.close_btn)
        
        layout.addLayout(input_layout)

    def add_image_context(self, image_data: bytes):
        """Push a Lens-captured image into the chat context.

        Adds the image to the VLProcessor's ContextManager so that the
        next chat message includes it, and shows a thumbnail placeholder
        in the chat history.
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            self.processor.context_manager.add_frame(image)
            self._append_message("System", "📷 Image captured from Lens (will be included in next message)", "#FF9800")
            logger.info("Image context pushed to chat")
        except Exception as e:
            logger.error("Failed to add image context: %s", e)

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text: return
        
        self.input_field.clear()
        self._append_message("You", text, accent_hex())
        
        # Disable input while processing
        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Inject grounding prompt if user is asking for location
        prompt = text
        if "where" in text.lower() and "click" in text.lower():
            prompt += "\n(Please output the bounding box coordinates of the element I should click in the format [ymin, xmin, ymax, xmax] relative to the image size from 0 to 1000.)"
        
        self.worker = ChatWorker(self.processor, prompt)
        self.worker.result_ready.connect(self._handle_response)
        self.worker.start()
        
    def _handle_response(self, response: str):
        self._append_message("Assistant", response, "#2196F3")
        
        # Parse for grounding coordinates: [ymin, xmin, ymax, xmax]
        match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response)
        if match:
            ymin, xmin, ymax, xmax = map(int, match.groups())
            # Convert 0-1000 range back to screen coordinates
            img = self.processor.context_manager.get_latest_frame()
            if img:
                # Assuming the image represents the full screen
                # For lens crop, we'd need to offset this by the crop rect, but we can assume full screen for now
                screen_rect = QGuiApplication.primaryScreen().geometry()
                sw, sh = screen_rect.width(), screen_rect.height()
                
                real_xmin = int(xmin / 1000.0 * sw)
                real_ymin = int(ymin / 1000.0 * sh)
                real_xmax = int(xmax / 1000.0 * sw)
                real_ymax = int(ymax / 1000.0 * sh)
                
                target_rect = QRect(real_xmin, real_ymin, real_xmax - real_xmin, real_ymax - real_ymin)
                
                # Keep a reference so it doesn't get garbage collected immediately
                self._highlight = GroundingHighlight(target_rect)
        
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()
        
    def _append_message(self, sender: str, text: str, color: str):
        safe_sender = html_escape(sender)
        safe_text = html_escape(text).replace(chr(10), "<br>")
        html = f'<p><b style="color:{color}">{safe_sender}:</b> {safe_text}</p>'
        self.history_display.append(html)
        
    def showEvent(self, event):
        self.input_field.setFocus()
        super().showEvent(event)
