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
