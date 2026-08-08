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

import sys
import threading
import logging
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

__all__ = ["MouseListener", "create_mouse_listener"]

class MouseListener(QObject):
    """Base class for global mouse click listeners."""
    left_click = pyqtSignal()

    def start(self):
        pass

    def stop(self):
        pass

try:
    import evdev
except ImportError:  # not installed, or a non-Linux platform
    evdev = None

try:
    from pynput import mouse as _pynput_mouse
except ImportError:  # headless CI, or no display server bindings
    _pynput_mouse = None


if evdev is not None:

    class EvdevMouseListener(MouseListener):
        """
        Listens for global left clicks across Wayland/X11 using evdev.
        Must be run as a user in the 'input' group.
        """
        def __init__(self):
            super().__init__()
            self.running = False
            self.devices = []
            self._threads = []
            
        def _find_mice(self):
            """Find all mouse devices in /dev/input/"""
            try:
                for path in evdev.list_devices():
                    try:
                        device = evdev.InputDevice(path)
                    except Exception:
                        continue
                    
                    is_mouse = False
                    try:
                        if evdev.ecodes.EV_KEY in device.capabilities():
                            keys = device.capabilities()[evdev.ecodes.EV_KEY]
                            if evdev.ecodes.BTN_LEFT in keys:
                                is_mouse = True
                    except Exception:
                        pass
                        
                    if is_mouse:
                        self.devices.append(device)
                        logger.info("EvdevMouseListener: Found mouse - %s at %s", device.name, device.path)
                    else:
                        try:
                            device.close()
                        except Exception:
                            pass
            except Exception as e:
                logger.error("EvdevMouseListener: Failed to find mice: %s", e)

        def has_devices(self) -> bool:
            """Whether any mouse is readable — the caller falls back if not."""
            if not self.devices:
                self._find_mice()
            return bool(self.devices)

        def start(self):
            if self.running:
                return

            # Clean up and ensure fresh open devices
            for device in self.devices:
                try:
                    device.close()
                except Exception:
                    pass
            self.devices = []

            self._find_mice()
            if not self.devices:
                logger.warning("EvdevMouseListener: No mice found to listen to.")
                return

            self.running = True
            self._threads = []
            for device in self.devices:
                thread = threading.Thread(target=self._listen_device, args=(device,), daemon=True)
                thread.start()
                self._threads.append(thread)
                
            logger.info("EvdevMouseListener: Started listening on %d devices.", len(self.devices))

        def stop(self):
            self.running = False
            for device in self.devices:
                try:
                    device.close()
                except Exception:
                    pass
            self.devices = []
            self._threads = []

        def _listen_device(self, device: evdev.InputDevice):
            try:
                for event in device.read_loop():
                    if not self.running:
                        break
                        
                    if event.type == evdev.ecodes.EV_KEY:
                        if event.code == evdev.ecodes.BTN_LEFT and event.value == 1: # 1 is down
                            self.left_click.emit()
            except Exception as e:
                if self.running:
                    logger.error("EvdevMouseListener: Error reading from %s: %s", device.name, e)

if _pynput_mouse is not None:

    class PynputMouseListener(MouseListener):
        """
        Listens for global mouse clicks using pynput.
        Used on macOS and Windows, and on Linux when evdev is unavailable.
        """
        def __init__(self):
            super().__init__()
            self.listener = None

        def on_click(self, x, y, button, pressed):
            if button == _pynput_mouse.Button.left and pressed:
                self.left_click.emit()

        def start(self):
            if not self.listener:
                self.listener = _pynput_mouse.Listener(on_click=self.on_click)
                self.listener.start()
                logger.info("PynputMouseListener: Started listening.")

        def stop(self):
            if self.listener:
                self.listener.stop()
                self.listener = None


def create_mouse_listener() -> MouseListener:
    """Prefer evdev on Linux (the only Wayland-wide option), else pynput."""
    if sys.platform == "linux" and evdev is not None:
        listener = EvdevMouseListener()
        if listener.has_devices():
            return listener
        logger.warning(
            "No readable mouse devices found (is this user in the 'input' group?); "
            "falling back to pynput for click detection."
        )

    if _pynput_mouse is not None:
        return PynputMouseListener()

    logger.error("No mouse backend available (need evdev or pynput); clicks are ignored.")
    return MouseListener()
