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

import json
import logging
import os
import uuid
from datetime import datetime, timezone

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)
from PyQt6.QtCore import QStandardPaths

from mage.ui.overlay_base import MageOverlayWindow
from mage.ui.theme import accent_hex
from shared_types.state import t

logger = logging.getLogger(__name__)


def _get_notes_path() -> str:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    notes_dir = os.path.join(base, "notes")
    os.makedirs(notes_dir, exist_ok=True)
    return os.path.join(notes_dir, "notes.json")


class NotesSidebar(MageOverlayWindow):
    def __init__(self, app, parent=None):
        super().__init__(window_id="notes_sidebar", app=app, parent=parent)

        self.setFixedWidth(350)
        self._notes: list[dict] = []
        self._editing_id: str | None = None

        # Position on the left side of the screen if not restored
        preset = self.app.settings.value("layout_preset", "Default")
        key = f"layout/{preset}/notes_sidebar"
        if not self.app.settings.contains(key):
            primary = QGuiApplication.primaryScreen()
            if primary is None:
                self.setGeometry(0, 0, 350, 600)
            else:
                screen = primary.geometry()
                self.setGeometry(screen.left(), screen.top(), 350, screen.height())

        # --- Root layout ---
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.bg_frame = QFrame()
        self.bg_frame.setObjectName("NotesBg")
        self.bg_frame.setStyleSheet(f"""
            #NotesBg {{
                background-color: rgba(20, 20, 20, 220);
                border: 1px solid {accent_hex()};
                border-radius: 0px;
            }}
        """)
        root.addWidget(self.bg_frame)

        inner = QVBoxLayout(self.bg_frame)
        inner.setContentsMargins(10, 10, 10, 10)
        inner.setSpacing(8)

        # --- Header ---
        header = QHBoxLayout()
        title_lbl = QLabel(t("notes.sidebar.title"))
        title_lbl.setStyleSheet(f"color: {accent_hex()}; font-size: 14px; font-weight: bold;")
        header.addWidget(title_lbl)
        header.addStretch()

        self.new_btn = QPushButton(t("notes.sidebar.button.new"))
        self.new_btn.setStyleSheet(f"background-color: {accent_hex()}; color: #111; font-weight: bold; padding: 4px 8px;")
        self.new_btn.clicked.connect(lambda: self._show_form(note_id=None))
        header.addWidget(self.new_btn)

        inner.addLayout(header)

        # --- Edit form (hidden by default) ---
        self.form_widget = QWidget()
        form_layout = QVBoxLayout(self.form_widget)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(4)

        self.title_field = QLineEdit()
        self.title_field.setPlaceholderText(t("notes.sidebar.field.title_placeholder"))
        self.title_field.setStyleSheet("background-color: #2A2A2A; color: #DDD; border: 1px solid #444; border-radius: 0px; padding: 4px;")
        form_layout.addWidget(self.title_field)

        self.content_field = QTextEdit()
        self.content_field.setPlaceholderText(t("notes.sidebar.field.content_placeholder"))
        self.content_field.setStyleSheet("background-color: #2A2A2A; color: #DDD; border: 1px solid #444; border-radius: 0px; padding: 4px;")
        self.content_field.setFixedHeight(100)
        form_layout.addWidget(self.content_field)

        form_btns = QHBoxLayout()
        self.save_btn = QPushButton(t("notes.sidebar.button.save"))
        self.save_btn.setStyleSheet(f"background-color: {accent_hex()}; color: #111; font-weight: bold; padding: 4px 8px;")
        self.save_btn.clicked.connect(self._commit_form)
        self.cancel_btn = QPushButton(t("notes.sidebar.button.cancel"))
        self.cancel_btn.setStyleSheet("background-color: #555; color: #DDD; padding: 4px 8px;")
        self.cancel_btn.clicked.connect(self._hide_form)
        form_btns.addWidget(self.save_btn)
        form_btns.addWidget(self.cancel_btn)
        form_layout.addLayout(form_btns)

        self.form_widget.setVisible(False)
        inner.addWidget(self.form_widget)

        # --- Notes scroll area ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("background: transparent;")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.list_container = QWidget()
        self.list_container.setStyleSheet("background: transparent;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(4)
        self.list_layout.addStretch()

        self.scroll.setWidget(self.list_container)
        inner.addWidget(self.scroll, stretch=1)

        # --- Footer close button ---
        close_btn = QPushButton(t("notes.sidebar.button.close"))
        close_btn.setStyleSheet("background-color: #d32f2f; color: #FFF; padding: 4px 8px; font-weight: bold;")
        close_btn.clicked.connect(self.hide)
        inner.addWidget(close_btn)

        self.load_notes()

    # ── Persistence ──────────────────────────────────────────────────────────

    def load_notes(self):
        path = _get_notes_path()
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._notes = json.load(f)
            except Exception as e:
                logger.error("Failed to load notes: %s", e)
                self._notes = []
        else:
            self._notes = []
        self._refresh_list()

    def save_notes(self):
        path = _get_notes_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._notes, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save notes: %s", e)

    # ── List rendering ────────────────────────────────────────────────────────

    def _refresh_list(self):
        # Remove all items except the trailing stretch
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._notes:
            empty = QLabel(t("notes.sidebar.empty_state"))
            empty.setWordWrap(True)
            empty.setStyleSheet("color: #888; font-size: 12px; padding: 8px;")
            self.list_layout.insertWidget(0, empty)
            return

        for note in self._notes:
            self.list_layout.insertWidget(self.list_layout.count() - 1, self._make_card(note))

    def _make_card(self, note: dict) -> QWidget:
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(40, 40, 40, 200);
                border: 1px solid #444;
                border-radius: 0px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)

        title = QLabel(note.get("title", ""))
        title.setStyleSheet(f"color: {accent_hex()}; font-weight: bold; font-size: 13px;")
        title.setWordWrap(True)
        card_layout.addWidget(title)

        content = note.get("content", "")
        preview = content[:100] + ("…" if len(content) > 100 else "")
        if preview:
            preview_lbl = QLabel(preview)
            preview_lbl.setStyleSheet("color: #CCC; font-size: 12px;")
            preview_lbl.setWordWrap(True)
            card_layout.addWidget(preview_lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        edit_btn = QPushButton(t("notes.sidebar.button.edit"))
        edit_btn.setStyleSheet("background-color: #444; color: #DDD; padding: 2px 8px; font-size: 11px;")
        note_id = note["id"]
        edit_btn.clicked.connect(lambda _, nid=note_id: self._show_form(note_id=nid))
        btn_row.addWidget(edit_btn)

        del_btn = QPushButton(t("notes.sidebar.button.delete"))
        del_btn.setStyleSheet("background-color: #8B0000; color: #FFF; padding: 2px 8px; font-size: 11px;")
        del_btn.clicked.connect(lambda _, nid=note_id: self.delete_note(nid))
        btn_row.addWidget(del_btn)

        card_layout.addLayout(btn_row)
        return card

    # ── Form ──────────────────────────────────────────────────────────────────

    def _show_form(self, note_id: str | None):
        self._editing_id = note_id
        if note_id is not None:
            note = next((n for n in self._notes if n["id"] == note_id), None)
            if note:
                self.title_field.setText(note.get("title", ""))
                self.content_field.setPlainText(note.get("content", ""))
        else:
            self.title_field.clear()
            self.content_field.clear()
        self.form_widget.setVisible(True)
        self.title_field.setFocus()

    def _hide_form(self):
        self._editing_id = None
        self.form_widget.setVisible(False)

    def _commit_form(self):
        title = self.title_field.text().strip()
        content = self.content_field.toPlainText().strip()
        if not title and not content:
            self._hide_form()
            return

        if self._editing_id is not None:
            for note in self._notes:
                if note["id"] == self._editing_id:
                    note["title"] = title
                    note["content"] = content
                    break
        else:
            self._notes.insert(0, {
                "id": str(uuid.uuid4()),
                "title": title,
                "content": content,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tags": [],
            })

        self.save_notes()
        self._hide_form()
        self._refresh_list()

    # ── Public API ────────────────────────────────────────────────────────────

    def delete_note(self, note_id: str):
        self._notes = [n for n in self._notes if n["id"] != note_id]
        self.save_notes()
        self._refresh_list()

    def add_note(self, title: str, content: str, tags: list[str] | None = None):
        """Insert a note, persist, and refresh the list."""
        self._notes.insert(0, {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tags": tags or [],
        })
        self.save_notes()
        self._refresh_list()

    def add_note_from_translation(self, original: str, translation: str):
        """Save a translation result directly as a note."""
        title = original[:60] + ("…" if len(original) > 60 else "")
        self.add_note(title, translation, tags=["translation"])

    def set_opacity(self, value: int):
        self.setWindowOpacity(value / 100)

    def set_text_size(self, px: int):
        self._refresh_list()

    # ── Qt overrides ──────────────────────────────────────────────────────────

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.raise_()
