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

import os
import json
import pytest
from PyQt6.QtCore import QRect, QPoint
from mage.ui.hud import get_hud_presets_dir, get_pinyin_for_text

class MockDictionary:
    def lookup(self, word: str):
        mapping = {
            "普通": [("普通", "pu3 tong1", "common/ordinary")],
            "攻击": [("攻击", "gong1 ji1", "attack/assault")],
            "普": [("普", "pu3", "general")],
            "通": [("通", "tong1", "go through")],
            "攻": [("攻", "gong1", "attack")],
            "击": [("击", "ji1", "strike")]
        }
        return mapping.get(word, [])

def test_presets_dir():
    dir_path = get_hud_presets_dir()
    assert os.path.exists(dir_path)
    assert os.path.isdir(dir_path)

def test_pinyin_generation():
    mock_dict = MockDictionary()
    text = "普通攻击"
    pinyin = get_pinyin_for_text(text, mock_dict)
    # Longest match first:
    # "普通" -> "pu3 tong1"
    # "攻击" -> "gong1 ji1"
    # Should result in "pu3 tong1 gong1 ji1"
    assert pinyin == "pu3 tong1 gong1 ji1"

def test_hover_trigger_geometry():
    # Verify geometry rect bounds logic
    rect = QRect(100, 100, 50, 30)
    
    # Inside point
    pt_inside = QPoint(120, 115)
    assert rect.contains(pt_inside)
    
    # Outside points
    pt_outside = QPoint(90, 115)
    assert not rect.contains(pt_outside)
    pt_outside2 = QPoint(120, 140)
    assert not rect.contains(pt_outside2)

def test_json_save_load(tmp_path):
    preset_file = tmp_path / "test_preset.json"
    preset_data = {
        "name": "test_preset",
        "buttons": [
            {
                "hover_rect": [10, 10, 100, 100],
                "display_rect": [20, 20, 50, 50],
                "original_text": "攻击",
                "translated_text": "Attack",
                "pinyin": "gong1 ji1"
            }
        ]
    }
    
    # Save
    with open(preset_file, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, indent=2)
        
    # Load and assert
    with open(preset_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        
    assert loaded["name"] == "test_preset"
    assert len(loaded["buttons"]) == 1
    assert loaded["buttons"][0]["hover_rect"] == [10, 10, 100, 100]
    assert loaded["buttons"][0]["translated_text"] == "Attack"


class DummyApp:
    def __init__(self):
        from PyQt6.QtCore import QSettings
        self.settings = QSettings("XianProject", "MageTest")
        self.tray = None

    def _apply_transient_parent(self, widget):
        pass


@pytest.fixture(scope="module", autouse=True)
def q_app():
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    yield app


def test_wayland_hover_resolution(tmp_path, monkeypatch, q_app):
    from unittest.mock import MagicMock, patch
    from mage.ui.hud import HudManager
    import threading

    app = DummyApp()
    app.settings.clear()
    
    mock_getter = MagicMock(return_value=(200, 200))
    mock_controller = MagicMock()
    mock_controller.width = 3840
    mock_controller.height = 2160
    mock_controller.x = 1920
    mock_controller.y = 1080
    mock_controller._lock = threading.Lock()
    mock_pick = MagicMock(return_value=("evdev", mock_getter, mock_controller))
    
    # Mock QGuiApplication.platformName to return "wayland"
    monkeypatch.setattr("PyQt6.QtGui.QGuiApplication.platformName", lambda: "wayland")
    
    # Mock ScreenCapture.get_virtual_desktop_geometry
    from PyQt6.QtCore import QRect, QPoint
    monkeypatch.setattr("mage.capture.screen.ScreenCapture.get_virtual_desktop_geometry", lambda: QRect(0, 0, 1920, 1080))
    
    # Mock QCursor.pos to return (960, 540)
    monkeypatch.setattr("PyQt6.QtGui.QCursor.pos", lambda: QPoint(960, 540))

    with patch("wayland_automation.mouse_position.pick_backend_and_start", mock_pick):
        manager = HudManager(app)
        
        # Create a dummy preset file
        preset_file = tmp_path / "test_preset.json"
        preset_data = {
            "name": "test_preset",
            "buttons": [
                {
                    "hover_rect": [10, 10, 100, 100],
                    "display_rect": [20, 20, 50, 50],
                    "original_text": "攻击",
                    "translated_text": "Attack",
                    "pinyin": "gong1 ji1"
                }
            ]
        }
        with open(preset_file, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2)
            
        manager.load_preset(str(preset_file))
        
        assert manager.wayland_mouse_getter == mock_getter
        tracker = manager.wayland_mouse_controller
        assert tracker is not None
        # Verify bounds and scale factors were computed and tracker seeded
        assert tracker.width == 3840
        assert tracker.height == 2160
        assert tracker.x == 1920
        assert tracker.y == 1080
        assert manager.wayland_scale_x == 0.5
        assert manager.wayland_scale_y == 0.5
        assert manager.timer.isActive() is True
        
        # Test _check_hover outside hover_rect
        # Return physical coords that scale outside of hover_rect
        mock_getter.return_value = (500, 500)
        manager._check_hover()
        assert manager.hovered_button is None
        
        # Test _check_hover inside hover_rect
        # Return physical coords that scale inside hover_rect [10, 10, 100, 100]
        # (30, 30) * 0.5 = (15, 15) which is inside
        mock_getter.return_value = (30, 30)
        manager.show_single_tooltip = MagicMock()
        manager._check_hover()
        manager.show_single_tooltip.assert_called_once()
        
        # Test deactivate stops controller
        manager.deactivate()
        mock_controller.stop.assert_called_once()
        assert manager.wayland_mouse_getter is None
        assert manager.wayland_mouse_controller is None
        assert manager.timer.isActive() is False
        
    app.settings.clear()


def test_wayland_no_backend_deactivates(tmp_path, monkeypatch, q_app):
    from unittest.mock import MagicMock, patch
    from mage.ui.hud import HudManager

    app = DummyApp()
    app.settings.clear()
    
    mock_pick = MagicMock(return_value=(None, None, None))
    
    # Mock QGuiApplication.platformName to return "wayland"
    monkeypatch.setattr("PyQt6.QtGui.QGuiApplication.platformName", lambda: "wayland")
    
    # Mock QMessageBox.warning to prevent blocking popup
    mock_warn = MagicMock()
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.warning", mock_warn)
    
    with patch("wayland_automation.mouse_position.pick_backend_and_start", mock_pick):
        manager = HudManager(app)
        
        # Create a dummy preset file
        preset_file = tmp_path / "test_preset_no_backend.json"
        preset_data = {
            "name": "test_preset_no_backend",
            "buttons": [
                {
                    "hover_rect": [10, 10, 100, 100],
                    "display_rect": [20, 20, 50, 50],
                    "original_text": "攻击",
                    "translated_text": "Attack",
                    "pinyin": "gong1 ji1"
                }
            ]
        }
        with open(preset_file, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2)
            
        manager.load_preset(str(preset_file))
        
        assert manager.wayland_mouse_getter is None
        assert manager.wayland_mouse_controller is None
        assert manager.timer.isActive() is False
        assert manager.active_preset is None
        mock_warn.assert_called_once()
        
    app.settings.clear()


def test_multi_device_tracker(monkeypatch):
    from unittest.mock import MagicMock
    import sys
    
    # Mock evdev module
    mock_evdev = MagicMock()
    mock_evdev.ecodes.EV_REL = 2
    mock_evdev.ecodes.REL_X = 0
    mock_evdev.ecodes.REL_Y = 1
    mock_evdev.ecodes.EV_ABS = 3
    mock_evdev.ecodes.ABS_X = 0
    mock_evdev.ecodes.ABS_Y = 1
    mock_evdev.ecodes.EV_KEY = 1
    mock_evdev.ecodes.BTN_TOUCH = 330
    
    # Mock absinfo
    mock_absinfo = MagicMock()
    mock_absinfo.min = 0
    mock_absinfo.max = 1000
    
    # Mock InputDevice
    mock_device = MagicMock()
    mock_device.fd = 10
    mock_device.name = "Test Touchscreen"
    mock_device.capabilities.return_value = {
        3: [(0, mock_absinfo), (1, mock_absinfo)]
    }
    mock_device.absinfo.return_value = mock_absinfo
    
    mock_evdev.list_devices.return_value = ["/dev/input/event0"]
    mock_evdev.InputDevice.return_value = mock_device
    
    # We test the patched EvdevFallback class
    import wayland_automation.mouse_position
    
    # Set up evdev mock attributes on the tracker instance
    tracker = wayland_automation.mouse_position.EvdevFallback(seed=(100, 200))
    tracker.evdev = True
    tracker.list_devices = mock_evdev.list_devices
    tracker.InputDevice = mock_evdev.InputDevice
    tracker.ecodes = mock_evdev.ecodes
    
    # Mock selectors
    mock_selector = MagicMock()
    monkeypatch.setattr("selectors.DefaultSelector", lambda: mock_selector)
    
    # Mock ScreenCapture virtual geometry
    from PyQt6.QtCore import QRect
    mock_geo = QRect(0, 0, 1920, 1080)
    monkeypatch.setattr("mage.capture.screen.ScreenCapture.get_virtual_desktop_geometry", MagicMock(return_value=mock_geo))
    
    assert tracker.start() is True
    assert len(tracker.devices) == 1
    
    # Verify device classification
    dev_info = tracker.devices[10]
    assert dev_info["type"] == "absolute"
    assert dev_info["abs_x"] == mock_absinfo
    
    # Test absolute event processing
    mock_event = MagicMock()
    mock_event.type = 3  # EV_ABS
    mock_event.code = 0  # ABS_X
    mock_event.value = 500
    
    tracker._process_event(dev_info, mock_event)
    assert tracker.x == 960  # 500 / 1000 * 1920 = 960
    
    # Clean up
    tracker.stop()



