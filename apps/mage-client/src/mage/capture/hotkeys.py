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

"""Global hotkeys: a double-tap leader that opens a short command mode.

Two backends implement the same gesture vocabulary. **evdev** reads
``/dev/input`` directly, which is the only way to see keys under Wayland, and
needs the user to be in the ``input`` group. **pynput** is the portable
fallback used on macOS and Windows, and on Linux whenever evdev is unusable.

The gesture logic itself — double-tap windows, command-mode lifetime, which
letter maps to which action — lives once in :class:`_CommandModeCore`. Each
backend only translates its own key representation into the predicates the core
asks about, so the two paths cannot drift apart.
"""

import sys
import threading
import logging
import time
from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

__all__ = ["HotkeyListener", "create_hotkey_listener"]


@dataclass
class _TapState:
    """Double-tap bookkeeping for one input source.

    evdev keeps one of these per device (a keyboard and a mouse each report
    their own stream); pynput has a single merged stream and so a single state.
    """

    #: False once a non-modifier key has been pressed, which disqualifies the
    #: next modifier tap from completing a double-tap. Stops "Shift+A, Shift"
    #: from being read as a deliberate double-tap of Shift.
    mod_clean: bool = True
    last_leader: float = 0.0
    last_overlay: float = 0.0


class _CommandModeCore:
    """The leader/command-mode state machine, free of Qt and of key encodings."""

    #: Two taps closer together than this count as a double-tap.
    DOUBLE_TAP_SECONDS = 0.4
    #: How long command mode waits for its letter before lapsing.
    COMMAND_MODE_SECONDS = 15.0

    #: Command-mode letter → the signal the listener should emit.
    COMMANDS = {
        "c": "trigger_lens",
        "a": "trigger_chat",
        "s": "trigger_settings",
        "m": "trigger_cinematic_mode",
        "t": "trigger_how_to_say",
        "r": "trigger_raid_mode",
        "n": "trigger_notes",
    }

    OVERLAY_TOGGLE_KEYS = ("rshift", "lshift", "rctrl", "ralt", "super")

    def __init__(self):
        self.leader_mod = "shift"
        self.overlay_toggle_key = "rshift"
        self.command_mode_active = False
        self.command_mode_end_time = 0.0
        self.cinematic_mode_active = False
        self._scopes: dict[object, _TapState] = {}

    # ── configuration ────────────────────────────────────────────────

    def set_leader_key(self, leader_string: str) -> None:
        leader_string = (leader_string or "").lower().replace("+space", "")
        parts = leader_string.split()
        if parts:
            self.leader_mod = parts[-1]  # "shift", "ctrl", "alt", "super"

    def set_overlay_toggle_key(self, key_token: str) -> None:
        token = (key_token or "").strip().lower()
        if token in self.OVERLAY_TOGGLE_KEYS:
            self.overlay_toggle_key = token

    def cancel_command_mode(self) -> None:
        self.command_mode_active = False
        self.command_mode_end_time = 0.0

    def forget_scope(self, scope) -> None:
        """Drop a disconnected device's state."""
        self._scopes.pop(scope, None)

    # ── the state machine ────────────────────────────────────────────

    def on_key_press(
        self,
        scope,
        *,
        now: float,
        is_modifier: bool,
        is_leader: bool,
        is_overlay_toggle: bool,
        is_escape: bool,
        is_grave: bool,
        char: str | None = None,
    ) -> str | None:
        """Feed one initial key-down; return the signal to emit, or None.

        Auto-repeat must not reach this (evdev keystate 2, pynput's repeat
        presses): a held-down leader would otherwise self-double-tap.
        """
        state = self._scopes.setdefault(scope, _TapState())

        if self.command_mode_active and now > self.command_mode_end_time:
            self.command_mode_active = False

        if not is_modifier:
            state.mod_clean = False

        # Checked before the leader and always returns, so binding the overlay
        # toggle to a key that is also the leader modifier still works.
        if is_overlay_toggle:
            if state.mod_clean and (now - state.last_overlay) < self.DOUBLE_TAP_SECONDS:
                state.last_overlay = 0.0
                logger.info("Toggle Overlays via %s double-tap", self.overlay_toggle_key)
                return "toggle_overlays"
            state.last_overlay = now
            # Clear the other timer so alternating taps of the two shifts
            # (rshift → lshift → rshift) can't be read as one gesture.
            state.last_leader = 0.0
            state.mod_clean = True
            return None

        if is_leader:
            if state.mod_clean and (now - state.last_leader) < self.DOUBLE_TAP_SECONDS:
                state.last_leader = 0.0
                if self.command_mode_active:
                    logger.info("Command Mode toggled off via leader double-tap")
                    self.command_mode_active = False
                    return "command_mode_cancelled"
                self.command_mode_active = True
                self.command_mode_end_time = now + self.COMMAND_MODE_SECONDS
                logger.info("Command Mode activated via double-tap %s", self.leader_mod)
                return "command_mode_started"
            state.last_leader = now
            state.last_overlay = 0.0
            state.mod_clean = True
            # Deliberately falls through: a leader tap is still an ordinary key.

        if self.command_mode_active and is_escape:
            logger.info("Command Mode cancelled via ESC")
            self.command_mode_active = False
            return "command_mode_cancelled"

        if self.cinematic_mode_active and is_grave:
            logger.info("Triggered Cinematic Capture")
            return "cinematic_capture"

        if self.command_mode_active and char:
            signal = self.COMMANDS.get(char.lower())
            if signal:
                logger.info("Command Mode: %s", signal)
                self.command_mode_active = False
                return signal

        return None


class HotkeyListener(QObject):
    """Base class for global hotkey listeners."""

    trigger_lens = pyqtSignal()
    trigger_chat = pyqtSignal()
    trigger_settings = pyqtSignal()

    trigger_cinematic_mode = pyqtSignal()
    trigger_how_to_say = pyqtSignal()
    cinematic_capture = pyqtSignal()
    command_mode_started = pyqtSignal()
    command_mode_cancelled = pyqtSignal()
    trigger_raid_mode = pyqtSignal()
    trigger_notes = pyqtSignal()
    toggle_overlays = pyqtSignal()

    mouse_position_updated = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self._core = _CommandModeCore()
        self._lock = threading.RLock()

    # ``cinematic_mode_active`` is flipped from the app when Cinematic Mode is
    # entered, so it stays a plain attribute on the listener.
    @property
    def cinematic_mode_active(self) -> bool:
        return self._core.cinematic_mode_active

    @cinematic_mode_active.setter
    def cinematic_mode_active(self, value: bool) -> None:
        self._core.cinematic_mode_active = bool(value)

    def _dispatch(self, action: str | None) -> None:
        """Emit the signal the core asked for."""
        if action:
            getattr(self, action).emit()

    def start(self):
        pass

    def stop(self):
        pass

    def set_leader_key(self, leader_string: str):
        with self._lock:
            self._core.set_leader_key(leader_string)

    def set_overlay_toggle_key(self, key_token: str):
        with self._lock:
            self._core.set_overlay_toggle_key(key_token)

    def seed_mouse_position(self, x: int, y: int, width: int, height: int):
        pass

    def cancel_command_mode(self):
        with self._lock:
            self._core.cancel_command_mode()


try:
    import evdev
except ImportError:  # not installed, or a non-Linux platform
    evdev = None


try:
    from pynput import keyboard as _pynput_keyboard
except ImportError:  # headless CI, or no display server bindings
    _pynput_keyboard = None


if evdev is not None:

    class EvdevHotkeyListener(HotkeyListener):
        """Reads raw key events from /dev/input, the only Wayland-wide option.

        Requires membership of the ``input`` group; without it no devices are
        found and :meth:`has_devices` reports False so the caller can fall back.
        """

        # Linux input event codes, which evdev reports as raw scancodes.
        _LEADER_CODES = {
            "shift": (42, 54),
            "ctrl": (29, 97),
            "alt": (56, 100),
            "super": (125, 126),
        }
        _OVERLAY_CODES = {
            "rshift": (54,),
            "lshift": (42,),
            "rctrl": (97,),
            "ralt": (100,),
            "super": (125, 126),
        }
        _MODIFIER_CODES = frozenset({125, 126, 42, 54, 29, 97, 56, 100})

        def __init__(self):
            super().__init__()
            self.running = False
            self.devices = []
            self._threads = []
            self._monitor_thread = None

            self.mouse_x = 0
            self.mouse_y = 0
            self.screen_width = 1920
            self.screen_height = 1080
            self._command_codes = {
                getattr(evdev.ecodes, f"KEY_{letter.upper()}"): letter
                for letter in _CommandModeCore.COMMANDS
            }
            self._find_devices()

        def has_devices(self) -> bool:
            with self._lock:
                return bool(self.devices)

        def seed_mouse_position(self, x: int, y: int, width: int, height: int):
            with self._lock:
                self.mouse_x = max(0, min(width, x))
                self.mouse_y = max(0, min(height, y))
                self.screen_width = width
                self.screen_height = height

        def _find_devices(self):
            """Find keyboard and pointer devices, and start listening if running."""
            try:
                with self._lock:
                    known_paths = {d.path for d in self.devices}
                for path in evdev.list_devices():
                    if path in known_paths:
                        continue
                    try:
                        device = evdev.InputDevice(path)
                    except Exception:
                        continue

                    caps = device.capabilities()
                    is_interesting = (
                        evdev.ecodes.KEY_A in caps.get(evdev.ecodes.EV_KEY, ())
                        or {evdev.ecodes.REL_X, evdev.ecodes.REL_Y}.issubset(
                            caps.get(evdev.ecodes.EV_REL, ())
                        )
                        or {evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y}.issubset(
                            caps.get(evdev.ecodes.EV_ABS, ())
                        )
                    )

                    if not is_interesting:
                        try:
                            device.close()
                        except Exception:
                            pass
                        continue

                    # Register and (if already running) spawn the listener
                    # thread atomically, so a concurrent device teardown can't
                    # race the device list.
                    with self._lock:
                        self.devices.append(device)
                        if self.running:
                            thread = threading.Thread(
                                target=self._listen_device, args=(device,), daemon=True
                            )
                            thread.start()
                            self._threads.append((device.path, thread))
                    logger.info(
                        "EvdevListener: Found device - %s at %s", device.name, device.path
                    )
            except Exception as e:
                logger.error(
                    "EvdevListener: Failed to find devices "
                    "(are you in the 'input' group?): %s", e
                )

        def _monitor_devices_loop(self):
            """Periodically rescan for hot-plugged input devices while running."""
            while self.running:
                for _ in range(50):
                    if not self.running:
                        return
                    time.sleep(0.1)
                # Every ~5s, pick up newly connected keyboards/mice.
                self._find_devices()

        def start(self):
            self.running = True

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
            for _path, thread in self._threads:
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

        def _listen_device(self, device):
            """Listen loop for a single device."""
            try:
                for event in device.read_loop():
                    if not self.running:
                        break
                    if event.type in (
                        evdev.ecodes.EV_KEY, evdev.ecodes.EV_REL, evdev.ecodes.EV_ABS
                    ):
                        self._handle_event(device.path, event)
            except Exception as e:
                if self.running:
                    logger.warning(
                        "EvdevListener: Device disconnected or error on %s: %s", device.path, e
                    )
            finally:
                try:
                    device.close()
                except Exception:
                    pass
                with self._lock:
                    if device in self.devices:
                        self.devices.remove(device)
                    self._core.forget_scope(device.path)

        def _handle_event(self, device_path: str, event):
            if event.type == evdev.ecodes.EV_REL:
                self._handle_pointer(event, relative=True)
                return
            if event.type == evdev.ecodes.EV_ABS:
                self._handle_pointer(event, relative=False)
                return
            if event.type != evdev.ecodes.EV_KEY:
                return

            key_event = evdev.categorize(event)
            if key_event.keystate != 1:  # 1 = down; 2 is auto-repeat, 0 is up
                return

            keycode = key_event.scancode
            with self._lock:
                core = self._core
                action = core.on_key_press(
                    device_path,
                    now=time.time(),
                    is_modifier=keycode in self._MODIFIER_CODES,
                    is_leader=keycode in self._LEADER_CODES.get(core.leader_mod, ()),
                    is_overlay_toggle=keycode in self._OVERLAY_CODES.get(
                        core.overlay_toggle_key, ()
                    ),
                    is_escape=keycode == evdev.ecodes.KEY_ESC,
                    is_grave=keycode == evdev.ecodes.KEY_GRAVE,
                    char=self._command_codes.get(keycode),
                )
            self._dispatch(action)

        def _handle_pointer(self, event, *, relative: bool):
            with self._lock:
                if event.code in (evdev.ecodes.REL_X, evdev.ecodes.ABS_X):
                    base = self.mouse_x if relative else 0
                    self.mouse_x = max(0, min(self.screen_width, base + event.value))
                elif event.code in (evdev.ecodes.REL_Y, evdev.ecodes.ABS_Y):
                    base = self.mouse_y if relative else 0
                    self.mouse_y = max(0, min(self.screen_height, base + event.value))
                else:
                    return
                position = (self.mouse_x, self.mouse_y)
            self.mouse_position_updated.emit(*position)


if _pynput_keyboard is not None:

    class PynputHotkeyListener(HotkeyListener):
        """Portable listener for macOS, Windows, and Linux without evdev."""

        _SCOPE = "pynput"  # one merged event stream, so one tap state

        def __init__(self):
            super().__init__()
            self.listener = None
            self.current_keys = set()
            key = _pynput_keyboard.Key
            self._leader_keys = {
                "shift": (key.shift, key.shift_l, key.shift_r),
                "ctrl": (key.ctrl, key.ctrl_l, key.ctrl_r),
                "alt": (key.alt, key.alt_l, key.alt_r),
                "super": (key.cmd, key.cmd_l, key.cmd_r),
            }
            self._overlay_keys = {
                "rshift": (key.shift_r,),
                "lshift": (key.shift_l,),
                "rctrl": (key.ctrl_r,),
                "ralt": (key.alt_r,),
                "super": (key.cmd, key.cmd_l, key.cmd_r),
            }
            self._modifier_keys = frozenset(
                k for keys in self._leader_keys.values() for k in keys
            )

        def on_press(self, key):
            with self._lock:
                # pynput repeats presses while a key is held; only the first
                # one is a real press.
                if key in self.current_keys:
                    return
                self.current_keys.add(key)

                char = getattr(key, "char", None)
                core = self._core
                action = core.on_key_press(
                    self._SCOPE,
                    now=time.time(),
                    is_modifier=key in self._modifier_keys,
                    is_leader=key in self._leader_keys.get(core.leader_mod, ()),
                    is_overlay_toggle=key in self._overlay_keys.get(
                        core.overlay_toggle_key, ()
                    ),
                    is_escape=key == _pynput_keyboard.Key.esc,
                    is_grave=char == "`",
                    char=char,
                )
            self._dispatch(action)

        def on_release(self, key):
            with self._lock:
                self.current_keys.discard(key)

        def start(self):
            if not self.listener:
                self.listener = _pynput_keyboard.Listener(
                    on_press=self.on_press, on_release=self.on_release
                )
                self.listener.start()
                logger.info("PynputListener: Started listening.")

        def stop(self):
            if self.listener:
                self.listener.stop()
                self.listener = None
            with self._lock:
                self.current_keys.clear()


def create_hotkey_listener() -> HotkeyListener:
    """Pick the best available backend.

    evdev is preferred on Linux because it is the only one that sees keys under
    Wayland, but it needs both the module and read access to ``/dev/input``
    (the ``input`` group). When either is missing it finds no devices and would
    listen silently forever, so pynput takes over — which is at least correct
    under X11.
    """
    if sys.platform == "linux" and evdev is not None:
        listener = EvdevHotkeyListener()
        if listener.has_devices():
            return listener
        logger.warning(
            "No readable input devices found (is this user in the 'input' group?); "
            "falling back to pynput, which cannot see keys under Wayland."
        )

    if _pynput_keyboard is not None:
        return PynputHotkeyListener()

    logger.error("No hotkey backend available (need evdev or pynput); hotkeys are disabled.")
    return HotkeyListener()
