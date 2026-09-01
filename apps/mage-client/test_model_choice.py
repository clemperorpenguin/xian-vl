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

"""The first-run model picker.

What matters here is that every option is offered, that the default lands on
the memory recommendation, and that whatever is chosen comes back as something
the caller can store and install — because the alternative to asking was
silently downloading up to 25.8 GB.
"""

import os
import sys

import pytest
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.ui.model_choice_dialog import ModelChoiceDialog  # noqa: E402
from shared_types import constants  # noqa: E402
from xian.collections import COLLECTIONS, collection_for_name  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def q_app():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_every_collection_and_the_single_model_are_offered():
    dialog = ModelChoiceDialog("ultra")
    offered = {model for _button, _tier, model in dialog._options}

    assert offered == {c.name for c in COLLECTIONS.values()} | {constants.SINGLE_MODEL_NAME}


def test_the_recommended_tier_is_selected_by_default():
    """Memory still guesses; it just no longer decides on its own."""
    dialog = ModelChoiceDialog("lite")

    assert dialog.chosen_tier == "lite"
    assert collection_for_name(dialog.chosen_model) is not None


@pytest.mark.parametrize("tier", [str(t) for t in COLLECTIONS])
def test_each_tier_can_be_recommended(tier):
    dialog = ModelChoiceDialog(tier)
    assert dialog.chosen_tier == tier


def test_an_unknown_recommendation_still_leaves_one_option_selected():
    """A stale tier in settings must not produce a dialog with no selection."""
    dialog = ModelChoiceDialog("does-not-exist")

    assert dialog.chosen_model
    assert sum(button.isChecked() for button, _t, _m in dialog._options) == 1


def test_choosing_the_single_model_reports_no_tier():
    """It is not a collection, so there is no tier to store for it."""
    dialog = ModelChoiceDialog("ultra")
    single = next(b for b, tier, _m in dialog._options if tier is None)
    single.setChecked(True)

    assert dialog.chosen_tier is None
    assert dialog.chosen_model == constants.SINGLE_MODEL_NAME


def test_the_single_model_is_far_smaller_than_any_collection():
    """The point of offering it: minutes rather than hours on a slow link."""
    assert constants.SINGLE_MODEL_SIZE_GB < min(c.size_gb for c in COLLECTIONS.values())


def test_every_option_label_resolves_a_localized_string():
    """Hardcoded UI text is prohibited; a missing key shows up as the key."""
    dialog = ModelChoiceDialog("ultra")

    for button, _tier, _model in dialog._options:
        assert "GB" in button.text()
        assert not button.text().startswith("model_choice.")


# ── first-run seeding ────────────────────────────────────────────────

def test_seeding_only_writes_settings_not_the_processor():
    """``_seed_collection_tier`` runs before ``self.processor`` exists.

    It is called from ``__init__`` immediately before ``VLProcessor`` is
    constructed from KEY_API_MODEL, so touching the processor there raises
    AttributeError and takes the whole app down on first launch.
    """
    import ast
    import inspect
    import textwrap

    from mage.app import XianApp

    tree = ast.parse(textwrap.dedent(inspect.getsource(XianApp._seed_collection_tier)))
    touches_processor = any(
        isinstance(node, ast.Attribute)
        and node.attr == "processor"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(tree)
    )
    assert not touches_processor, (
        "the picker runs before the processor is built; write settings only"
    )

    init = inspect.getsource(XianApp.__init__)
    assert init.index("_seed_collection_tier") < init.index("self.processor = VLProcessor"), (
        "seeding must precede the processor so it reads the chosen model"
    )
