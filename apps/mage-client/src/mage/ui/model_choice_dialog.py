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

"""First-run choices: what to download, and how MAGE should work.

MAGE needs models before it can translate anything, and the first launch used
to pick a collection from installed memory and start fetching it — up to 25.8
GB — without asking. On a slow connection that is a multi-hour surprise, and
the machine's RAM is a poor proxy for what someone actually wants to wait for.

So the choice is put to the user, with the sizes stated plainly: one of the
three collections, or a single vision model that covers screen translation
alone and downloads in minutes. Memory still picks the *default* selection —
it is a reasonable guess — but it no longer decides silently.

The second choice is the architecture. Classic is the original design and the
default: frame a region, translate it, read the result. Real-time keeps a
locked region continuously translated in place — newer, heavier, and still
experimental, so it is offered rather than assumed. It can be switched on and
off later under Experimental in Settings.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from shared_types import constants
from shared_types.state import t
from xian.collections import COLLECTIONS

logger = logging.getLogger(__name__)

__all__ = ["ModelChoiceDialog"]


class ModelChoiceDialog(QDialog):
    """Pick a collection, or the single translation-only model.

    ``chosen_tier`` is the tier value to store, or ``None`` when the single
    model was picked; ``chosen_model`` is the model id to install either way.
    """

    def __init__(self, recommended_tier: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("model_choice.title"))
        self.setModal(True)

        self._group = QButtonGroup(self)
        self._options: list[tuple[QRadioButton, str | None, str]] = []
        self._modes: list[tuple[QRadioButton, str]] = []

        layout = QVBoxLayout(self)

        intro = QLabel(t("model_choice.intro"))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        models_heading = QLabel(t("model_choice.heading.models"))
        models_heading.setStyleSheet("font-weight: bold;")
        layout.addWidget(models_heading)

        for tier, collection in COLLECTIONS.items():
            label = t("model_choice.option.collection").format(
                name=collection.name, size=f"{collection.size_gb:.1f}"
            )
            if str(tier) == str(recommended_tier):
                label += " " + t("model_choice.recommended")
            button = QRadioButton(label)
            button.setToolTip(", ".join(collection.components))
            self._group.addButton(button)
            layout.addWidget(button)

            detail = QLabel(", ".join(collection.components))
            detail.setWordWrap(True)
            detail.setIndent(24)
            detail.setEnabled(False)
            layout.addWidget(detail)

            self._options.append((button, str(tier), collection.name))

        single = QRadioButton(
            t("model_choice.option.single").format(
                name=constants.SINGLE_MODEL_NAME,
                size=f"{constants.SINGLE_MODEL_SIZE_GB:.1f}",
            )
        )
        self._group.addButton(single)
        layout.addWidget(single)

        single_detail = QLabel(t("model_choice.single.detail"))
        single_detail.setWordWrap(True)
        single_detail.setIndent(24)
        single_detail.setEnabled(False)
        layout.addWidget(single_detail)

        self._options.append((single, None, constants.SINGLE_MODEL_NAME))

        note = QLabel(t("model_choice.note"))
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(note)

        mode_heading = QLabel(t("model_choice.heading.mode"))
        mode_heading.setStyleSheet("font-weight: bold;")
        layout.addWidget(mode_heading)

        self._mode_group = QButtonGroup(self)
        for architecture in constants.ARCHITECTURES:
            button = QRadioButton(t(f"model_choice.mode.{architecture}"))
            self._mode_group.addButton(button)
            layout.addWidget(button)

            detail = QLabel(t(f"model_choice.mode.{architecture}.detail"))
            detail.setWordWrap(True)
            detail.setIndent(24)
            detail.setEnabled(False)
            layout.addWidget(detail)

            button.setChecked(architecture == constants.DEFAULT_ARCHITECTURE)
            self._modes.append((button, architecture))

        # Default to the memory-recommended tier, falling back to the first
        # option so there is always exactly one selection to accept.
        for button, tier, _model in self._options:
            if tier is not None and str(tier) == str(recommended_tier):
                button.setChecked(True)
                break
        else:
            self._options[0][0].setChecked(True)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ── result ───────────────────────────────────────────────────────

    @property
    def chosen_tier(self) -> str | None:
        """The tier to store, or None when the single model was chosen."""
        for button, tier, _model in self._options:
            if button.isChecked():
                return tier
        return None

    @property
    def chosen_model(self) -> str:
        """The model id to install."""
        for button, _tier, model in self._options:
            if button.isChecked():
                return model
        return constants.SINGLE_MODEL_NAME

    @property
    def chosen_architecture(self) -> str:
        """"classic" or "realtime"."""
        for button, architecture in self._modes:
            if button.isChecked():
                return architecture
        return constants.DEFAULT_ARCHITECTURE

    @property
    def live_mode_enabled(self) -> bool:
        """Whether continuous translation should be switched on."""
        return self.chosen_architecture == "realtime"
