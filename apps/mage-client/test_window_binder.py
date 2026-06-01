import sys
from unittest.mock import MagicMock, patch
import pytest

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
