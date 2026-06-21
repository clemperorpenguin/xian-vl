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

def test_window_binder_fallback_and_api():
    # Verify WindowBinder can be imported and instantiated without side effects
    from mage.utils.window_binder import WindowBinder

    binder = WindowBinder("Test Target Window")
    assert binder.target_title == "Test Target Window"

    # Test public API execution and type safety
    exists = binder.exists()
    assert isinstance(exists, bool)

    geom = binder.get_geometry()
    assert geom is None or (isinstance(geom, tuple) and len(geom) == 4)

    active = binder.is_active()
    assert isinstance(active, bool)

    minimized = binder.is_minimized()
    assert isinstance(minimized, bool)

    native_id = binder.get_native_id()
    assert native_id is None or isinstance(native_id, int)

    binder.close()


def test_window_binder_platform_mocking():
    from mage.utils.window_binder import WindowBinder

    # Test Windows platform resolution
    with patch("sys.platform", "win32"):
        binder = WindowBinder("WinTest")
        assert binder.platform == "windows"

    # Test macOS platform resolution
    with patch("sys.platform", "darwin"):
        binder = WindowBinder("MacTest")
        assert binder.platform == "macos"


def test_window_binder_get_active_titles():
    from mage.utils.window_binder import WindowBinder

    titles = WindowBinder.get_active_window_titles()
    assert isinstance(titles, list)
    for t in titles:
        assert isinstance(t, str)
