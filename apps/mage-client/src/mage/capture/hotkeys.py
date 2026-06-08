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
import time
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

__all__ = ["HotkeyListener", "create_hotkey_listener"]

class HotkeyListener(QObject):
    """Base class for global hotkey listeners."""
    trigger_lens = pyqtSignal()
    trigger_chat = pyqtSignal()
    trigger_settings = pyqtSignal()
    trigger_dialogue_mode = pyqtSignal()
    trigger_cinematic_mode = pyqtSignal()
    trigger_how_to_say = pyqtSignal()
    cinematic_capture = pyqtSignal()
    command_mode_started = pyqtSignal()
    command_mode_cancelled = pyqtSignal()
    trigger_raid_mode = pyqtSignal()
    trigger_hud = pyqtSignal()
    trigger_layout_edit = pyqtSignal()
    mouse_position_updated = pyqtSignal(int, int)

    def start(self):
        pass

    def stop(self):
        pass

    def set_leader_key(self, leader_string: str):
        pass

    def seed_mouse_position(self, x: int, y: int, width: int, height: int):
        pass
        
    def cancel_command_mode(self):
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
            self._lock = threading.RLock()
            self.running = False
            self.devices = []
            self._threads = []
            self._monitor_thread = None
            
            self.leader_mod = 'shift'
            self.command_mode_active = False
            self.command_mode_end_time = 0.0
            
            self.cinematic_mode_active = False
            
            # Track modifier states per device
            self.modifiers = {}
            self.mod_clean = {}
            self.last_leader_press_time = {}
            
            # Setup devices
            self.mouse_x = 0
            self.mouse_y = 0
            self.screen_width = 1920
            self.screen_height = 1080
            self._find_devices()
            

        def seed_mouse_position(self, x: int, y: int, width: int, height: int):
            with self._lock:
                self.mouse_x = max(0, min(width, x))
                self.mouse_y = max(0, min(height, y))
                self.screen_width = width
                self.screen_height = height

        def _find_devices(self):
            """Find all keyboard and pointer devices in /dev/input/ and initialize them."""
            try:
                for path in evdev.list_devices():
                    if any(d.path == path for d in self.devices):
                        continue
                    try:
                        device = evdev.InputDevice(path)
                    except Exception:
                        continue
                    
                    is_interesting = False
                    caps = device.capabilities()
                    
                    if evdev.ecodes.EV_KEY in caps:
                        if evdev.ecodes.KEY_A in caps[evdev.ecodes.EV_KEY]:
                            is_interesting = True
                            
                    if evdev.ecodes.EV_REL in caps:
                        if evdev.ecodes.REL_X in caps[evdev.ecodes.EV_REL] and evdev.ecodes.REL_Y in caps[evdev.ecodes.EV_REL]:
                            is_interesting = True
                            
                    if evdev.ecodes.EV_ABS in caps:
                        if evdev.ecodes.ABS_X in caps[evdev.ecodes.EV_ABS] and evdev.ecodes.ABS_Y in caps[evdev.ecodes.EV_ABS]:
                            is_interesting = True
                            
                    if is_interesting:
                        self.devices.append(device)
                        with self._lock:
                            self.modifiers[device.path] = {
                                'super': False,
                                'shift': False,
                                'ctrl': False,
                                'alt': False
                            }
                            self.mod_clean[device.path] = True
                        logger.info("EvdevListener: Found device - %s at %s", device.name, device.path)
                        if self.running:
                            thread = threading.Thread(target=self._listen_device, args=(device,), daemon=True)
                            thread.start()
                            self._threads.append((device.path, thread))
                    else:
                        try:
                            device.close()
                        except Exception:
                            pass
            except Exception as e:
                logger.error("EvdevListener: Failed to find devices (are you in the 'input' group?): %s", e)

        def set_leader_key(self, leader_string: str):
            with self._lock:
                # Handle both "Double-Tap Shift" and legacy "Shift+Space"
                leader_string = leader_string.lower().replace('+space', '')
                parts = leader_string.split()
                if len(parts) > 0:
                    self.leader_mod = parts[-1]  # "shift", "ctrl", "alt", "super"

        def _is_leader_mod_key(self, keycode):
            if self.leader_mod == 'shift': return keycode in (42, 54)
            if self.leader_mod == 'ctrl': return keycode in (29, 97)
            if self.leader_mod == 'alt': return keycode in (56, 100)
            if self.leader_mod == 'super': return keycode in (125, 126)
            return False

        def _monitor_devices_loop(self):
            """Periodically check for new input devices while running."""
            while self.running:
                for _ in range(50):
                    if not self.running:
                        return
                    time.sleep(0.1)
                self.mouse_x = 0
            self.mouse_y = 0
            self.screen_width = 1920
            self.screen_height = 1080
            self._find_devices()

        def start(self):
            """Start listening threads for all keyboards."""
            self.running = True
            
            # Start listener threads for current devices
            for device in self.devices:
                thread = threading.Thread(target=self._listen_device, args=(device,), daemon=True)
                thread.start()
                self._threads.append((device.path, thread))
                
            self._monitor_thread = threading.Thread(target=self._monitor_devices_loop, daemon=True)
            self._monitor_thread.start()
                
            logger.info("EvdevListener: Started listening on %d devices.", len(self.devices))

        def stop(self):
            """Stop listening and clean up all resources."""
            self.running = False
            for device in list(self.devices):
                try:
                    device.close()
                except Exception:
                    pass
            for path, thread in self._threads:
                try:
                    thread.join(timeout=2.0)
                except Exception:
                    pass
            self._threads.clear()
            self.devices.clear()
            if self._monitor_thread:
                try:
                    self._monitor_thread.join(timeout=2.0)
                except Exception:
                    pass
                self._monitor_thread = None

        def cancel_command_mode(self):
            with self._lock:
                self.command_mode_active = False
                self.command_mode_end_time = 0.0

        def _listen_device(self, device: evdev.InputDevice):
            """Listen loop for a single device."""
            try:
                for event in device.read_loop():
                    if not self.running:
                        break
                        
                    if event.type in (evdev.ecodes.EV_KEY, evdev.ecodes.EV_REL, evdev.ecodes.EV_ABS):
                        with self._lock:
                            self._handle_event(device.path, event)
            except Exception as e:
                if self.running:
                    logger.warning("EvdevListener: Device disconnected or error on %s: %s", device.path, e)
            finally:
                try:
                    device.close()
                except Exception:
                    pass
                with self._lock:
                    if device in self.devices:
                        self.devices.remove(device)
                    if device.path in self.modifiers:
                        del self.modifiers[device.path]
                    if device.path in self.mod_clean:
                        del self.mod_clean[device.path]
                    if device.path in self.last_leader_press_time:
                        del self.last_leader_press_time[device.path]

        def _handle_event(self, device_path: str, event):
            """Process input events."""
            if event.type == evdev.ecodes.EV_REL:
                if event.code == evdev.ecodes.REL_X:
                    self.mouse_x = max(0, min(self.screen_width, self.mouse_x + event.value))
                    self.mouse_position_updated.emit(self.mouse_x, self.mouse_y)
                elif event.code == evdev.ecodes.REL_Y:
                    self.mouse_y = max(0, min(self.screen_height, self.mouse_y + event.value))
                    self.mouse_position_updated.emit(self.mouse_x, self.mouse_y)
                return
            elif event.type == evdev.ecodes.EV_ABS:
                # Provide a basic direct mapping for ABS if needed
                if event.code == evdev.ecodes.ABS_X:
                    self.mouse_x = max(0, min(self.screen_width, event.value))
                    self.mouse_position_updated.emit(self.mouse_x, self.mouse_y)
                elif event.code == evdev.ecodes.ABS_Y:
                    self.mouse_y = max(0, min(self.screen_height, event.value))
                    self.mouse_position_updated.emit(self.mouse_x, self.mouse_y)
                return
                
            if event.type != evdev.ecodes.EV_KEY:
                return
                
            key_event = evdev.categorize(event)
            keycode = key_event.scancode
            is_pressed = key_event.keystate in (1, 2)
            is_modifier = keycode in (125, 126, 42, 54, 29, 97, 56, 100)
            
            # Update modifier tracking on any state change
            if device_path not in self.modifiers:
                self.modifiers[device_path] = {'super': False, 'shift': False, 'ctrl': False, 'alt': False}
                
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
                if not is_modifier:
                    self.mod_clean[device_path] = False

                if self._is_leader_mod_key(keycode):
                    if self.mod_clean.get(device_path, True) and (now - self.last_leader_press_time.get(device_path, 0)) < 0.4:
                        # Double-tap detected!
                        self.last_leader_press_time[device_path] = 0.0 # reset
                        if self.command_mode_active:
                            logger.info("EvdevListener: Command Mode TOGGLED OFF via leader double-tap")
                            self.command_mode_active = False
                            self.command_mode_cancelled.emit()
                        else:
                            self.command_mode_active = True
                            self.command_mode_end_time = now + 15.0
                            logger.info("EvdevListener: Command Mode ACTIVATED via double-tap %s", self.leader_mod)
                            self.command_mode_started.emit()
                        return
                    else:
                        self.last_leader_press_time[device_path] = now
                        self.mod_clean[device_path] = True

                # KEY_ESC is 1
                if self.command_mode_active and keycode == evdev.ecodes.KEY_ESC:
                    logger.info("EvdevListener: Command Mode CANCELLED via ESC")
                    self.command_mode_active = False
                    self.command_mode_cancelled.emit()
                    return

                # Check cinematic trigger globally (if mode is active)
                # KEY_GRAVE is 41 (backtick/tilde key)
                if self.cinematic_mode_active and keycode == evdev.ecodes.KEY_GRAVE:
                    logger.info("EvdevListener: Triggered Cinematic Capture")
                    self.cinematic_capture.emit()
                    return
                
                if self.command_mode_active:
                    # KEY_C is 46
                    if keycode == evdev.ecodes.KEY_C:
                        logger.info("EvdevListener: Triggered Lens")
                        self.trigger_lens.emit()
                        self.command_mode_active = False
                        
                    # KEY_A is 30
                    elif keycode == evdev.ecodes.KEY_A:
                        logger.info("EvdevListener: Triggered Chat")
                        self.trigger_chat.emit()
                        self.command_mode_active = False

                    # KEY_S is 31
                    elif keycode == evdev.ecodes.KEY_S:
                        logger.info("EvdevListener: Triggered Settings")
                        self.trigger_settings.emit()
                        self.command_mode_active = False

                    # KEY_O is 24
                    elif keycode == evdev.ecodes.KEY_O:
                        logger.info("EvdevListener: Triggered Dialogue Mode")
                        self.trigger_dialogue_mode.emit()
                        self.command_mode_active = False

                    # KEY_M is 50
                    elif keycode == evdev.ecodes.KEY_M:
                        logger.info("EvdevListener: Triggered Cinematic Mode")
                        self.trigger_cinematic_mode.emit()
                        self.command_mode_active = False

                    # KEY_T is 20
                    elif keycode == evdev.ecodes.KEY_T:
                        logger.info("EvdevListener: Triggered Translate")
                        self.trigger_how_to_say.emit()
                        self.command_mode_active = False

                    # KEY_R is 19
                    elif keycode == evdev.ecodes.KEY_R:
                        logger.info("EvdevListener: Triggered Raid Mode")
                        self.trigger_raid_mode.emit()
                        self.command_mode_active = False

                    # KEY_H is 35
                    elif keycode == evdev.ecodes.KEY_H:
                        logger.info("EvdevListener: Triggered HUD")
                        self.trigger_hud.emit()
                        self.command_mode_active = False

                    # KEY_L is 38
                    elif keycode == evdev.ecodes.KEY_L:
                        logger.info("EvdevListener: Triggered Layout Edit")
                        self.trigger_layout_edit.emit()
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
            self.lock = threading.RLock()
            self.current_keys = set()
            self.leader_mod = 'shift'
            self.command_mode_active = False
            self.command_mode_end_time = 0.0
            self.cinematic_mode_active = False
            self.mod_clean = True
            self.last_leader_press_time = 0.0

        def seed_mouse_position(self, x: int, y: int, width: int, height: int):
            pass

        def set_leader_key(self, leader_string: str):
            with self.lock:
                leader_string = leader_string.lower().replace('+space', '')
                parts = leader_string.split()
                if len(parts) > 0:
                    self.leader_mod = parts[-1]

        def _is_leader_mod_key(self, key):
            if self.leader_mod == 'shift': return key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r)
            if self.leader_mod == 'ctrl': return key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)
            if self.leader_mod == 'alt': return key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r)
            if self.leader_mod == 'super': return key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
            return False
            
        def on_press(self, key):
            with self.lock:
                is_new_press = key not in self.current_keys
                self.current_keys.add(key)
                
                now = time.time()
                if self.command_mode_active and now > self.command_mode_end_time:
                    self.command_mode_active = False

                if is_new_press:
                    is_modifier = key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
                                          keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                                          keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
                                          keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)
                    if not is_modifier:
                        self.mod_clean = False

                    if self._is_leader_mod_key(key):
                        if self.mod_clean and (now - self.last_leader_press_time) < 0.4:
                            self.last_leader_press_time = 0.0
                            if self.command_mode_active:
                                logger.info("PynputListener: Command Mode TOGGLED OFF via leader double-tap")
                                self.command_mode_active = False
                                self.command_mode_cancelled.emit()
                            else:
                                self.command_mode_active = True
                                self.command_mode_end_time = now + 15.0
                                logger.info("PynputListener: Command Mode ACTIVATED via double-tap %s", self.leader_mod)
                                self.command_mode_started.emit()
                            return
                        else:
                            self.last_leader_press_time = now
                            self.mod_clean = True

                if self.command_mode_active and key == keyboard.Key.esc:
                    logger.info("PynputListener: Command Mode CANCELLED via ESC")
                    self.command_mode_active = False
                    self.command_mode_cancelled.emit()
                    return

                if self.cinematic_mode_active:
                    if hasattr(key, 'char') and key.char == '`':
                        logger.info("PynputListener: Triggered Cinematic Capture")
                        self.cinematic_capture.emit()
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
                            elif char == 'o':
                                logger.info("PynputListener: Triggered Dialogue Mode")
                                self.trigger_dialogue_mode.emit()
                                self.command_mode_active = False
                            elif char == 'm':
                                logger.info("PynputListener: Triggered Cinematic Mode")
                                self.trigger_cinematic_mode.emit()
                                self.command_mode_active = False
                            elif char == 't':
                                logger.info("PynputListener: Triggered Translate")
                                self.trigger_how_to_say.emit()
                                self.command_mode_active = False
                            elif char == 'r':
                                logger.info("PynputListener: Triggered Raid Mode")
                                self.trigger_raid_mode.emit()
                                self.command_mode_active = False
                            elif char == 'h':
                                logger.info("PynputListener: Triggered HUD")
                                self.trigger_hud.emit()
                                self.command_mode_active = False
                            elif char == 'l':
                                logger.info("PynputListener: Triggered Layout Edit")
                                self.trigger_layout_edit.emit()
                                self.command_mode_active = False
                    except Exception as e:
                        logger.debug("PynputListener error: %s", e)
                
        def on_release(self, key):
            with self.lock:
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
            with self.lock:
                self.current_keys.clear()

        def cancel_command_mode(self):
            with self.lock:
                self.command_mode_active = False
                self.command_mode_end_time = 0.0

def create_hotkey_listener() -> HotkeyListener:
    if sys.platform == "linux":
        return EvdevHotkeyListener()
    else:
        return PynputHotkeyListener()
