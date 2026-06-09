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

import logging
import asyncio
import re
import io
import os
from PIL import Image
from html import escape as html_escape
from html.parser import HTMLParser
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton, QHBoxLayout, QTextBrowser
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QRect, QSettings, QBuffer, QIODevice, QPoint
from PyQt6.QtGui import QGuiApplication
from mage.ui.grounding import GroundingHighlight
from mage.ui.theme import accent_hex, accent_hover_hex
from mage.ui.overlay_base import MageOverlayWindow
from shared_types import constants
from shared_types.state import t

logger = logging.getLogger(__name__)


class SafeHTMLParser(HTMLParser):
    def __init__(self, allowed_tags, allowed_attrs, media_dir):
        super().__init__()
        self.allowed_tags = allowed_tags
        self.allowed_attrs = allowed_attrs
        self.media_dir = media_dir
        self.result = []
        self.tag_stack = []

    def handle_starttag(self, tag, attrs):
        if tag in self.allowed_tags:
            self.tag_stack.append(tag)
            filtered_attrs = []
            for attr, val in attrs:
                if attr in self.allowed_attrs.get(tag, []):
                    if tag == 'a' and attr == 'href':
                        if val.startswith('play_audio://'):
                            audio_path = val.replace('play_audio://', '')
                            if not self._is_safe_path(audio_path):
                                continue
                        elif val.startswith('file://'):
                            file_path = val.replace('file://', '')
                            if not self._is_safe_path(file_path):
                                continue
                        elif not (val.startswith('http://') or val.startswith('https://')):
                            continue
                    elif tag == 'img' and attr == 'src':
                        if val.startswith('file://'):
                            img_path = val.replace('file://', '')
                            if not self._is_safe_path(img_path):
                                continue
                        else:
                            continue
                    filtered_attrs.append(f'{attr}="{html_escape(val)}"')
            attr_str = f" {' '.join(filtered_attrs)}" if filtered_attrs else ""
            self.result.append(f"<{tag}{attr_str}>")

    def handle_endtag(self, tag):
        if tag in self.allowed_tags and self.tag_stack and self.tag_stack[-1] == tag:
            self.tag_stack.pop()
            self.result.append(f"</{tag}>")

    def handle_data(self, data):
        self.result.append(html_escape(data))

    def _is_safe_path(self, path: str) -> bool:
        real_path = os.path.realpath(path)
        real_media_dir = os.path.realpath(self.media_dir)
        try:
            return os.path.commonpath([real_path, real_media_dir]) == real_media_dir
        except ValueError:
            return False

    def get_safe_html(self):
        return "".join(self.result)


def sanitize_html(html_str: str, media_dir: str) -> str:
    allowed_tags = {'p', 'br', 'code', 'pre', 'em', 'strong', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'b', 'i', 'img'}
    allowed_attrs = {
        'a': {'href'},
        'pre': {'class'},
        'code': {'class'},
        'img': {'src', 'width', 'height'},
    }
    parser = SafeHTMLParser(allowed_tags, allowed_attrs, media_dir)
    parser.feed(html_str)
    return parser.get_safe_html()


class ChatWorker(QThread):
    result_ready = pyqtSignal(str)
    
    def __init__(self, processor, message, source_lang="zh-CN"):
        super().__init__()
        self.processor = processor
        self.message = message
        self.source_lang = source_lang
        
    def run(self):
        async def _init_and_process():
            if not self.processor.client:
                await self.processor.init_engine()
            return await self.processor.process_chat(self.message, source_lang=self.source_lang)

        future = self.processor.engine.submit(_init_and_process())
        try:
            from xian.timeout import CHAT_TIMEOUT_SECONDS
            response = future.result(timeout=CHAT_TIMEOUT_SECONDS)
            self.result_ready.emit(response)
        except Exception as e:
            logger.error("Chat worker error: %s", e)
            self.result_ready.emit(f"Error: {e}")

class SearchWorker(QThread):
    result_ready = pyqtSignal(str)
    
    def __init__(self, processor, query, language):
        super().__init__()
        self.processor = processor
        self.query = query
        self.language = language
        
    def run(self):
        async def _do_search():
            from xian.searcher import WebSearcher
            searcher = WebSearcher()
            try:
                results = await searcher.search(self.query, language=self.language)
            finally:
                await searcher.close()
            return results

        future = self.processor.engine.submit(_do_search())
        try:
            results = future.result(timeout=60)
            if not results:
                response = "No search results found."
            else:
                lines = [f"Found top results for '{self.query}':"]
                for r in results[:3]:
                    title = r.get("title", "")
                    content = r.get("content", "")
                    url = r.get("url", "")
                    lines.append(f"<b>{html_escape(title)}</b><br>{html_escape(content)}<br><a href='{html_escape(url)}'>{html_escape(url)}</a><br>")
                response = "<br>".join(lines)
                
            # Add this search information to the AI context so the assistant knows about it
            context_text = f"User performed a search for '{self.query}'. Results:\n"
            for r in results[:3]:
                context_text += f"- {r.get('title')}: {r.get('content')}\n"
            self.processor.context_manager.add_user_message(f"Search Query: {self.query}", with_image=False)
            self.processor.context_manager.add_assistant_message(context_text)

            self.result_ready.emit(response)
        except Exception as e:
            logger.error("Search worker error: %s", e)
            self.result_ready.emit(f"Search Error: {e}")


class ChatSidebar(MageOverlayWindow):
    def __init__(self, processor, parent=None):
        self.processor = processor
        self._highlight = None  # reference to active GroundingHighlight
        self.worker = None  # reference to active ChatWorker
        
        super().__init__(window_id="chat_sidebar", app=parent, parent=parent)
        
        self.setFixedWidth(350)
        
        # Position on the right side of the screen if not restored
        preset = self.app.settings.value("layout_preset", "Default")
        key = f"layout/{preset}/chat_sidebar"
        if not self.app.settings.contains(key):
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
            QTextEdit, QTextBrowser {{
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
        self.history_display = QTextBrowser()
        self.history_display.setReadOnly(True)
        self.history_display.setOpenLinks(False)
        self.history_display.anchorClicked.connect(self._on_anchor_clicked)
        layout.addWidget(self.history_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText(t("chat.sidebar.placeholder"))
        self.input_field.returnPressed.connect(self._send_message)
        
        self.attach_btn = QPushButton(t("chat.sidebar.button.attach"))
        self.attach_btn.setStyleSheet("background-color: #f57c00;")
        self.attach_btn.clicked.connect(self._attach_screenshot)
        
        self.send_btn = QPushButton(t("chat.sidebar.button.send"))
        self.send_btn.clicked.connect(self._send_message)
        
        self.close_btn = QPushButton(t("chat.sidebar.button.close"))
        self.close_btn.setStyleSheet("background-color: #d32f2f;")
        self.close_btn.clicked.connect(self.hide)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.attach_btn)
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
            
            # Save the captured image to the media folder
            media_dir = os.path.join(self.processor.wiki_dir, "media")
            os.makedirs(media_dir, exist_ok=True)
            import time
            filename = f"capture_{int(time.time())}.png"
            filepath = os.path.join(media_dir, filename)
            image.save(filepath)
            
            self._append_message("System", f"📷 Screenshot attached:<br><img src='file://{filepath}' width='150'>", "#FF9800")
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
        
        if text.startswith("/search"):
            parts = text.split(" ", 2)
            if len(parts) >= 3 and ("-" in parts[1] or len(parts[1]) == 2):
                language = parts[1]
                query = parts[2]
            else:
                language = "zh-CN"
                query = text[len("/search"):].strip()
                
            self.worker = SearchWorker(self.processor, query, language)
            self.worker.result_ready.connect(self._handle_response)
            self.worker.start()
            return
        
        # Inject grounding prompt if user is asking for location
        prompt = text
        if "where" in text.lower() and "click" in text.lower():
            prompt += "\n(Please output the bounding box coordinates of the element I should click in the format [ymin, xmin, ymax, xmax] relative to the image size from 0 to 1000.)"
        
        # Read source language from settings
        settings = QSettings(constants.ORGANIZATION_NAME, constants.APPLICATION_NAME)
        source_lang = settings.value("source_lang", "zh-CN")
        
        self.worker = ChatWorker(self.processor, prompt, source_lang=source_lang)
        self.worker.result_ready.connect(self._handle_response)
        self.worker.start()
        
    def _handle_response(self, response: str):
        is_search = isinstance(self.sender(), SearchWorker)
        self._append_message("Assistant", response, "#2196F3", raw_html=is_search)
        
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
        
    def _append_message(self, sender: str, text: str, color: str, raw_html: bool = False):
        safe_sender = html_escape(sender)
        media_dir = os.path.join(self.processor.wiki_dir, "media")
        
        if sender == "System" and "📷 Screenshot attached" in text:
            safe_text = text
        elif raw_html:
            safe_text = text
        else:
            safe_text = html_escape(text).replace(chr(10), "<br>")
            
            def repl_img(match):
                filepath = match.group(1)
                real_path = os.path.realpath(filepath)
                real_media_dir = os.path.realpath(media_dir)
                try:
                    is_safe = os.path.commonpath([real_path, real_media_dir]) == real_media_dir
                except ValueError:
                    is_safe = False
                if is_safe:
                    return f'<br><img src="file://{filepath}" width="280"><br><a href="file://{filepath}">View Image</a>'
                else:
                    logger.warning("Blocked image rendering from outside media dir: %s", filepath)
                    return "[Blocked image]"
            
            safe_text = re.sub(r'file://([^\s\'"&<>]+?\.(?:png|jpg|jpeg))', repl_img, safe_text, flags=re.IGNORECASE)
            
            def repl_audio(match):
                filepath = match.group(1)
                real_path = os.path.realpath(filepath)
                real_media_dir = os.path.realpath(media_dir)
                try:
                    is_safe = os.path.commonpath([real_path, real_media_dir]) == real_media_dir
                except ValueError:
                    is_safe = False
                if is_safe:
                    return f'<br>🎵 <a href="play_audio://{filepath}">Play Synthesized Audio</a>'
                else:
                    logger.warning("Blocked audio link from outside media dir: %s", filepath)
                    return "[Blocked audio]"
                
            safe_text = re.sub(r'file://([^\s\'"&<>]+?\.(?:wav|mp3))', repl_audio, safe_text, flags=re.IGNORECASE)

        html = f'<p><b style="color:{color}">{safe_sender}:</b><br>{safe_text}</p>'
        sanitized_html = sanitize_html(html, media_dir)
        self.history_display.append(sanitized_html)

    def _attach_screenshot(self):
        screen = QGuiApplication.primaryScreen()
        if screen:
            pixmap = screen.grabWindow(0)
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            pixmap.save(buffer, "PNG")
            self.add_image_context(buffer.data().data())

    def _on_anchor_clicked(self, url):
        url_str = url.toString()
        if url_str.startswith("play_audio://"):
            audio_path = url_str.replace("play_audio://", "")
            # Validate path is within allowed media directory
            media_dir = os.path.join(self.processor.wiki_dir, "media")
            real_path = os.path.realpath(audio_path)
            real_media_dir = os.path.realpath(media_dir)
            try:
                is_safe = os.path.commonpath([real_path, real_media_dir]) == real_media_dir
            except ValueError:
                is_safe = False
            if not is_safe:
                logger.warning("Blocked audio playback from outside media dir: %s", audio_path)
                return
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                from mage.capture.audio import play_audio_async
                play_audio_async(audio_bytes)
            except Exception as e:
                logger.error("Failed to replay audio: %s", e)
        
    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
        self.input_field.setFocus()
