import sys
import threading
import logging
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class HotkeyListener(QObject):
    """Base class for global hotkey listeners."""
    trigger_lens = pyqtSignal()
    trigger_chat = pyqtSignal()
    trigger_settings = pyqtSignal()

    def start(self):
        pass

    def stop(self):
        pass

if sys.platform == "linux":
    import evdev

    class EvdevHotkeyListener(HotkeyListener):
        """
        Listens for global hotkeys across all Wayland windows using evdev.
        Must be run as a user in the 'input' group.
        """
        def __init__(self):
            super().__init__()
            self.running = False
            self.devices = []
            self._threads = []
            
            # Track modifier states per device
            self.modifiers = {}
            
            # Setup devices
            self._find_keyboards()
            
        def _find_keyboards(self):
            """Find all keyboard devices in /dev/input/"""
            try:
                devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
                for device in devices:
                    # Check if it has keys
                    if evdev.ecodes.EV_KEY in device.capabilities():
                        # Check if it has standard keyboard keys (like KEY_A)
                        if evdev.ecodes.KEY_A in device.capabilities()[evdev.ecodes.EV_KEY]:
                            self.devices.append(device)
                            self.modifiers[device.path] = {
                                'super': False,
                                'shift': False,
                                'ctrl': False,
                                'alt': False
                            }
                            logger.info(f"EvdevListener: Found keyboard - {device.name} at {device.path}")
            except Exception as e:
                logger.error(f"EvdevListener: Failed to find keyboards (are you in the 'input' group?): {e}")

        def start(self):
            """Start listening threads for all keyboards."""
            if not self.devices:
                logger.warning("EvdevListener: No keyboards found to listen to.")
                return
                
            self.running = True
            for device in self.devices:
                thread = threading.Thread(target=self._listen_device, args=(device,), daemon=True)
                thread.start()
                self._threads.append(thread)
                
            logger.info(f"EvdevListener: Started listening on {len(self.devices)} devices.")

        def stop(self):
            """Stop listening."""
            self.running = False

        def _listen_device(self, device: evdev.InputDevice):
            """Listen loop for a single device."""
            try:
                for event in device.read_loop():
                    if not self.running:
                        break
                        
                    if event.type == evdev.ecodes.EV_KEY:
                        key_event = evdev.categorize(event)
                        self._handle_key_event(device.path, key_event)
            except Exception as e:
                if self.running:
                    logger.error(f"EvdevListener: Error reading from {device.name}: {e}")

        def _handle_key_event(self, device_path: str, event):
            """Process individual key events and detect hotkeys."""
            keycode = event.scancode
            
            is_pressed = event.keystate in (1, 2)  # down or hold
            
            # Update modifier tracking on any state change
            if keycode in (125, 126):
                self.modifiers[device_path]['super'] = is_pressed
            elif keycode in (42, 54):  # Shift
                self.modifiers[device_path]['shift'] = is_pressed
            elif keycode in (29, 97):  # Ctrl
                self.modifiers[device_path]['ctrl'] = is_pressed
            elif keycode in (56, 100):  # Alt
                self.modifiers[device_path]['alt'] = is_pressed
                
            # Check hotkeys only on key down (1)
            if event.keystate == 1:
                mods = self.modifiers[device_path]
                
                # Super + Shift + C -> Lens Overlay
                # KEY_C is 46
                if keycode == 46 and mods['super'] and mods['shift']:
                    logger.info("EvdevListener: Triggered Lens (Super+Shift+C)")
                    self.trigger_lens.emit()
                    
                # Super + A -> Chat Assistant
                # KEY_A is 30
                elif keycode == 30 and mods['super'] and not mods['shift']:
                    logger.info("EvdevListener: Triggered Chat (Super+A)")
                    self.trigger_chat.emit()

                # Super + Shift + S -> Settings
                # KEY_S is 31
                elif keycode == 31 and mods['super'] and mods['shift']:
                    logger.info("EvdevListener: Triggered Settings (Super+Shift+S)")
                    self.trigger_settings.emit()

else:
    from pynput import keyboard

    class PynputHotkeyListener(HotkeyListener):
        """
        Listens for global hotkeys using pynput.
        Used on macOS and Windows.
        """
        def __init__(self):
            super().__init__()
            self.listener = None
            self.current_keys = set()
            
        def on_press(self, key):
            self.current_keys.add(key)
            
            has_super = keyboard.Key.cmd in self.current_keys or keyboard.Key.cmd_l in self.current_keys or keyboard.Key.cmd_r in self.current_keys
            has_shift = keyboard.Key.shift in self.current_keys or keyboard.Key.shift_l in self.current_keys or keyboard.Key.shift_r in self.current_keys
            
            try:
                # Check combinations
                if hasattr(key, 'char') and key.char:
                    char = key.char.lower()
                    if has_super and has_shift and char == 'c':
                        logger.info("PynputListener: Triggered Lens (Super+Shift+C)")
                        self.trigger_lens.emit()
                    elif has_super and not has_shift and char == 'a':
                        logger.info("PynputListener: Triggered Chat (Super+A)")
                        self.trigger_chat.emit()
                    elif has_super and has_shift and char == 's':
                        logger.info("PynputListener: Triggered Settings (Super+Shift+S)")
                        self.trigger_settings.emit()
            except Exception as e:
                logger.debug(f"PynputListener error: {e}")
                
        def on_release(self, key):
            try:
                self.current_keys.remove(key)
            except KeyError:
                pass

        def start(self):
            if not self.listener:
                self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
                self.listener.start()
                logger.info("PynputListener: Started listening.")

        def stop(self):
            if self.listener:
                self.listener.stop()
                self.listener = None

def create_hotkey_listener() -> HotkeyListener:
    if sys.platform == "linux":
        return EvdevHotkeyListener()
    else:
        return PynputHotkeyListener()
