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

import os
import sys
from unittest.mock import patch

def test_clean_subprocess_env_not_frozen():
    from mage.utils.env import clean_subprocess_env
    
    mock_env = {
        "PATH": "/usr/bin",
        "SOME_VAR": "value"
    }
    
    with patch("sys.frozen", False, create=True), patch("os.environ", mock_env):
        cleaned = clean_subprocess_env()
        assert cleaned["PATH"] == "/usr/bin"
        assert cleaned["SOME_VAR"] == "value"
        assert "LD_LIBRARY_PATH" not in cleaned
        assert "DYLD_LIBRARY_PATH" not in cleaned

def test_clean_subprocess_env_frozen_with_orig():
    from mage.utils.env import clean_subprocess_env
    
    mock_env = {
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "/tmp/_MEI12345/lib",
        "LD_LIBRARY_PATH_ORIG": "/usr/lib/custom:/usr/local/lib",
        "DYLD_LIBRARY_PATH": "/tmp/_MEI12345/dylib",
        "DYLD_LIBRARY_PATH_ORIG": "/usr/local/lib"
    }
    
    with patch("sys.frozen", True, create=True), patch("os.environ", mock_env):
        cleaned = clean_subprocess_env()
        assert cleaned["PATH"] == "/usr/bin"
        assert cleaned["LD_LIBRARY_PATH"] == "/usr/lib/custom:/usr/local/lib"
        assert cleaned["DYLD_LIBRARY_PATH"] == "/usr/local/lib"
        assert "LD_LIBRARY_PATH_ORIG" not in cleaned
        assert "DYLD_LIBRARY_PATH_ORIG" not in cleaned

def test_clean_subprocess_env_frozen_without_orig():
    from mage.utils.env import clean_subprocess_env
    
    mock_env = {
        "PATH": "/usr/bin",
        "LD_LIBRARY_PATH": "/tmp/_MEI12345/lib",
        "DYLD_LIBRARY_PATH": "/tmp/_MEI12345/dylib"
    }
    
    with patch("sys.frozen", True, create=True), patch("os.environ", mock_env):
        cleaned = clean_subprocess_env()
        assert cleaned["PATH"] == "/usr/bin"
        assert "LD_LIBRARY_PATH" not in cleaned
        assert "DYLD_LIBRARY_PATH" not in cleaned
        assert "LD_LIBRARY_PATH_ORIG" not in cleaned
        assert "DYLD_LIBRARY_PATH_ORIG" not in cleaned
