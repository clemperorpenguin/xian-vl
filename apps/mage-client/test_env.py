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

def test_clean_subprocess_env_meipass_stripping():
    from mage.utils.env import clean_subprocess_env
    
    mock_env = {
        "PATH": "/tmp/_MEI12345/usr/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/tmp/_MEI12345/lib:/usr/lib64",
        "LD_LIBRARY_PATH_ORIG": "/tmp/_MEI12345/lib:/usr/lib64",
        "DYLD_LIBRARY_PATH": "/tmp/_MEI12345/dylib:/usr/local/lib",
        "DYLD_LIBRARY_PATH_ORIG": "/tmp/_MEI12345/dylib:/usr/local/lib",
        "QT_PLUGIN_PATH": "/tmp/_MEI12345/plugins:/usr/lib/qt/plugins",
        "XDG_DATA_DIRS": "/usr/local/share:/usr/share",
        "PYTHONHOME": "/tmp/_MEI12345/python_env",
        "PYTHONPATH": "/usr/lib/python3.11"
    }
    
    with patch("sys.frozen", True, create=True), patch("os.environ", mock_env):
        cleaned = clean_subprocess_env()
        assert cleaned["PATH"] == "/usr/bin"
        assert cleaned["LD_LIBRARY_PATH"] == "/usr/lib64"
        assert cleaned["DYLD_LIBRARY_PATH"] == "/usr/local/lib"
        assert cleaned["QT_PLUGIN_PATH"] == "/usr/lib/qt/plugins"
        assert cleaned["XDG_DATA_DIRS"] == "/usr/local/share:/usr/share"
        assert cleaned["PYTHONPATH"] == "/usr/lib/python3.11"
        assert "PYTHONHOME" not in cleaned
        assert "DYLD_LIBRARY_PATH_ORIG" not in cleaned


def test_clean_subprocess_env_meipass_stripping_windows():
    from mage.utils.env import clean_subprocess_env
    
    mock_env = {
        "PATH": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\usr\\bin;C:\\Windows\\system32",
        "LD_LIBRARY_PATH": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\lib;C:\\usr\\lib64",
        "LD_LIBRARY_PATH_ORIG": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\lib;C:\\usr\\lib64",
        "DYLD_LIBRARY_PATH": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\dylib;C:\\usr\\local\\lib",
        "DYLD_LIBRARY_PATH_ORIG": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\dylib;C:\\usr\\local\\lib",
        "QT_PLUGIN_PATH": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\plugins;C:\\usr\\lib\\qt\\plugins",
        "XDG_DATA_DIRS": "C:\\usr\\local\\share;C:\\usr\\share",
        "PYTHONHOME": "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345\\python_env",
        "PYTHONPATH": "C:\\usr\\lib\\python3.11"
    }
    
    # Mock sys._MEIPASS as a Windows path
    with patch("sys.frozen", True, create=True), \
         patch("sys._MEIPASS", "C:\\Users\\User\\AppData\\Local\\Temp\\_MEI12345", create=True), \
         patch("os.environ", mock_env), \
         patch("os.pathsep", ";"):
        cleaned = clean_subprocess_env()
        assert cleaned["PATH"] == "C:\\Windows\\system32"
        assert cleaned["LD_LIBRARY_PATH"] == "C:\\usr\\lib64"
        assert cleaned["DYLD_LIBRARY_PATH"] == "C:\\usr\\local\\lib"
        assert cleaned["QT_PLUGIN_PATH"] == "C:\\usr\\lib\\qt\\plugins"
        assert cleaned["XDG_DATA_DIRS"] == "C:\\usr\\local\\share;C:\\usr\\share"
        assert cleaned["PYTHONPATH"] == "C:\\usr\\lib\\python3.11"
        assert "PYTHONHOME" not in cleaned
        assert "LD_LIBRARY_PATH_ORIG" not in cleaned
        assert "DYLD_LIBRARY_PATH_ORIG" not in cleaned

