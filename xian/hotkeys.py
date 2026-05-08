import sys
import threading
import logging
import time
import logging
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class HotkeyListener(QObject):
    """Base class for global hotkey listeners."""
    trigger_lens = pyqtSignal()
    trigger_chat = pyqtSignal()
    trigger_settings = pyqtSignal()
    command_mode_started = pyqtSignal()

    def start(self):
        pass

    def stop(self):
        pass

    def set_leader_key(self, leader_string: str):
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
            
            self.leader_mod = 'shift'
            self.leader_key = 57  # KEY_SPACE
            self.command_mode_active = False
            self.command_mode_end_time = 0.0
            
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

        def set_leader_key(self, leader_string: str):
            parts = leader_string.lower().split('+')
            if len(parts) == 2:
                self.leader_mod = parts[0]
                self.leader_key = 57  # space

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
                
            now = time.time()
            if self.command_mode_active and now > self.command_mode_end_time:
                self.command_mode_active = False
                
            # Check hotkeys only on key down (1)
            if event.keystate == 1:
                mods = self.modifiers[device_path]
                
                if keycode == self.leader_key and mods.get(self.leader_mod):
                    self.command_mode_active = True
                    self.command_mode_end_time = now + 15.0
                    logger.info(f"EvdevListener: Command Mode ACTIVATED via {self.leader_mod}+space")
                    self.command_mode_started.emit()
                    return
                
                if self.command_mode_active:
                    # KEY_C is 46
                    if keycode == 46:
                        logger.info("EvdevListener: Triggered Lens")
                        self.trigger_lens.emit()
                        self.command_mode_active = False
                        
                    # KEY_A is 30
                    elif keycode == 30:
                        logger.info("EvdevListener: Triggered Chat")
                        self.trigger_chat.emit()
                        self.command_mode_active = False

                    # KEY_S is 31
                    elif keycode == 31:
                        logger.info("EvdevListener: Triggered Settings")
                        self.trigger_settings.emit()
                        self.command_mode_active = False

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
            self.leader_mod = 'shift'
            self.command_mode_active = False
            self.command_mode_end_time = 0.0

        def set_leader_key(self, leader_string: str):
            parts = leader_string.lower().split('+')
            if len(parts) == 2:
                self.leader_mod = parts[0]
            
        def on_press(self, key):
            self.current_keys.add(key)
            
            now = time.time()
            if self.command_mode_active and now > self.command_mode_end_time:
                self.command_mode_active = False

            has_mod = False
            if self.leader_mod == 'shift':
                has_mod = keyboard.Key.shift in self.current_keys or keyboard.Key.shift_l in self.current_keys or keyboard.Key.shift_r in self.current_keys
            elif self.leader_mod == 'ctrl':
                has_mod = keyboard.Key.ctrl in self.current_keys or keyboard.Key.ctrl_l in self.current_keys or keyboard.Key.ctrl_r in self.current_keys
            elif self.leader_mod == 'alt':
                has_mod = keyboard.Key.alt in self.current_keys or keyboard.Key.alt_l in self.current_keys or keyboard.Key.alt_r in self.current_keys
            elif self.leader_mod == 'super':
                has_mod = keyboard.Key.cmd in self.current_keys or keyboard.Key.cmd_l in self.current_keys or keyboard.Key.cmd_r in self.current_keys

            if has_mod and key == keyboard.Key.space:
                self.command_mode_active = True
                self.command_mode_end_time = now + 15.0
                logger.info(f"PynputListener: Command Mode ACTIVATED via {self.leader_mod}+space")
                self.command_mode_started.emit()
                return

            if self.command_mode_active:
                try:
                    if hasattr(key, 'char') and key.char:
                        char = key.char.lower()
                        if char == 'c':
                            logger.info("PynputListener: Triggered Lens")
                            self.trigger_lens.emit()
                            self.command_mode_active = False
                        elif char == 'a':
                            logger.info("PynputListener: Triggered Chat")
                            self.trigger_chat.emit()
                            self.command_mode_active = False
                        elif char == 's':
                            logger.info("PynputListener: Triggered Settings")
                            self.trigger_settings.emit()
                            self.command_mode_active = False
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
