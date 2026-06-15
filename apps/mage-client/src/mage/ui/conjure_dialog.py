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

"""The "Conjure…" modal: describe a creature, generate it with Lemonade, preview
it live, reroll, and apply.

The preview reuses the very same vector renderer the desktop familiar uses
(``_RecipeArt``), animated on a timer, so what you see in the dialog is exactly
what lands on the desktop.
"""

import logging

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QProgressBar, QWidget,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor

from mage.ui.familiar_pet import _RecipeArt, FamiliarState, TransitPhase
from mage.familiar_recipe import coerce_recipe
from mage.familiar_conjure import ConjureWorker
from mage.ui.theme import accent_hex
from shared_types.state import t

logger = logging.getLogger(__name__)

_PREVIEW_PX = 168
# A short loop that shows the creature idle, walking, then casting/reacting so
# its glow and motion are visible in the preview.
_PREVIEW_CYCLE = [
    FamiliarState.IDLE, FamiliarState.WALK, FamiliarState.WALK,
    FamiliarState.CAST, FamiliarState.REACT, FamiliarState.IDLE,
]


class FamiliarPreview(QWidget):
    """Animated preview of a recipe, rendered with the live familiar renderer."""

    def __init__(self, recipe=None, parent=None):
        super().__init__(parent)
        self.setFixedSize(_PREVIEW_PX, _PREVIEW_PX)
        self._art = _RecipeArt(coerce_recipe(recipe)) if recipe else None
        self._frame = 0
        self._tick = 0
        self._state = FamiliarState.IDLE
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(150)

    def set_recipe(self, recipe):
        self._art = _RecipeArt(coerce_recipe(recipe))
        self._frame = 0
        self.update()

    def _on_tick(self):
        self._frame += 1
        self._tick += 1
        if self._tick % 6 == 0:
            self._state = _PREVIEW_CYCLE[(self._tick // 6) % len(_PREVIEW_CYCLE)]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Soft panel so the sprite reads over any theme.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(16, 16, 20))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 14, 14)
        p.setBrush(QColor(accent_hex()))
        if self._art is None:
            p.end()
            return
        scale = self.height() / 96.0
        p.scale(scale, scale)
        self._art.draw(p, self._state, self._frame, 1, TransitPhase.GROUNDED, 96, 96)
        p.end()


class ConjureDialog(QDialog):
    """Describe → conjure (Lemonade) → preview → reroll → use."""

    EXAMPLES = [
        "a tiny angry storm cloud",
        "a sleepy moss golem",
        "a regal fox scholar",
        "a mischievous lantern wisp",
    ]

    def __init__(self, processor, initial_recipe=None, parent=None):
        super().__init__(parent)
        self.processor = processor
        self._recipe = coerce_recipe(initial_recipe) if initial_recipe else None
        self._worker = None

        self.setWindowTitle(t("familiar.conjure.title"))
        self.setModal(True)
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(t("familiar.conjure.prompt")))

        self.input = QLineEdit()
        self.input.setPlaceholderText(t("familiar.conjure.placeholder"))
        self.input.returnPressed.connect(self._conjure)
        lay.addWidget(self.input)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        for ex in self.EXAMPLES:
            b = QPushButton(ex)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setProperty("chip", True)
            b.clicked.connect(lambda _=False, e=ex: self.input.setText(e))
            chips.addWidget(b)
        lay.addLayout(chips)

        prow = QHBoxLayout()
        prow.addStretch()
        self.preview = FamiliarPreview(self._recipe)
        prow.addWidget(self.preview)
        prow.addStretch()
        lay.addLayout(prow)

        self.name_label = QLabel(self._recipe["name"] if self._recipe else "")
        # The name is model-generated; render it as plain text so it can never
        # be auto-detected as rich text / markup.
        self.name_label.setTextFormat(Qt.TextFormat.PlainText)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("font-weight: 600; font-size: 15px;")
        lay.addWidget(self.name_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)        # indeterminate
        self.progress.setTextVisible(False)
        self.progress.hide()
        lay.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.status)

        brow = QHBoxLayout()
        self.conjure_btn = QPushButton(t("familiar.conjure.button"))
        self.conjure_btn.clicked.connect(self._conjure)
        self.cancel_btn = QPushButton(t("settings.button.cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        self.use_btn = QPushButton(t("familiar.conjure.use"))
        self.use_btn.setEnabled(self._recipe is not None)
        self.use_btn.setDefault(True)
        self.use_btn.clicked.connect(self.accept)
        brow.addWidget(self.conjure_btn)
        brow.addStretch()
        brow.addWidget(self.cancel_btn)
        brow.addWidget(self.use_btn)
        lay.addLayout(brow)

        self.setStyleSheet(
            f"QDialog {{ background: #15151a; }}"
            f"QLabel {{ color: #e8e8ec; }}"
            f"QLineEdit {{ color: #f4f4f4; background: #202028; border: 1px solid {accent_hex()};"
            f" border-radius: 6px; padding: 6px 8px; }}"
            f"QPushButton {{ color: #f0f0f0; background: #2a2a33; border: 1px solid #3a3a44;"
            f" border-radius: 6px; padding: 6px 12px; }}"
            f"QPushButton:hover {{ border-color: {accent_hex()}; }}"
            f"QPushButton[chip=\"true\"] {{ padding: 3px 8px; font-size: 11px; color: #c8c8d0; }}"
        )

    # ----- generation -------------------------------------------------------
    def _conjure(self):
        prompt = self.input.text().strip()
        if not prompt or self._worker is not None:
            return
        self.conjure_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        self.progress.show()
        self.status.setText(t("familiar.conjure.working"))
        self._worker = ConjureWorker(self.processor, prompt)
        self._worker.conjured.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_done(self, recipe: dict):
        self._recipe = recipe
        self.preview.set_recipe(recipe)
        self.name_label.setText(recipe.get("name", ""))
        self.status.setText("")
        self.conjure_btn.setText(t("familiar.conjure.reroll"))

    def _on_error(self, msg: str):
        logger.warning("Conjure failed: %s", msg)
        self.status.setText(t("familiar.conjure.failed"))

    def _on_finished(self):
        self.progress.hide()
        self.conjure_btn.setEnabled(True)
        self.use_btn.setEnabled(self._recipe is not None)
        self._worker = None

    def result_recipe(self):
        return self._recipe
