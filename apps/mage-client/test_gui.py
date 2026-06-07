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

import sys
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings, QRect, Qt
from mage.ui.overlay_base import MageOverlayWindow


class DummyApp:
    def __init__(self):
        self.settings = QSettings("XianProject", "MageTest")


@pytest.fixture(scope="session", autouse=True)
def q_app():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    yield app


def test_mage_overlay_window_geometry_persistence(q_app):
    app = DummyApp()
    # Clear any previous settings
    app.settings.clear()
    app.settings.setValue("layout_preset", "Default")
    
    # Create window and save coordinates
    win = MageOverlayWindow("test_win_persistence", app)
    win.setGeometry(QRect(120, 240, 360, 480))
    win.save_geometry()
    
    # Create new window instance and restore
    win2 = MageOverlayWindow("test_win_persistence", app)
    # Verify geometry restores correctly
    assert win2.geometry().x() == 120
    assert win2.geometry().y() == 240
    assert win2.geometry().width() == 360
    assert win2.geometry().height() == 480
    
    # Clean up settings
    app.settings.clear()


def test_mage_overlay_window_click_through(q_app):
    app = DummyApp()
    app.settings.clear()
    
    win = MageOverlayWindow("test_win_click_through", app)
    
    # Test toggling click through state
    win.set_click_through(True)
    assert win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is True
    
    win.set_click_through(False)
    assert win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is False
    
    # Test click-through is disabled in edit mode (to allow dragging)
    win.set_click_through(True)
    win.set_edit_mode(True)
    assert win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is False
    
    # Exiting edit mode restores the click-through state
    win.set_edit_mode(False)
    assert win.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents) is True
    
    app.settings.clear()
