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

"""The leader / command-mode gesture machine shared by both hotkey backends."""

from mage.capture.hotkeys import _CommandModeCore


def _press(core, scope="kbd", *, now, leader=False, overlay=False, escape=False,
           grave=False, char=None, modifier=None):
    """Press one key; modifier defaults to "is this a leader/overlay key"."""
    if modifier is None:
        modifier = leader or overlay
    return core.on_key_press(
        scope,
        now=now,
        is_modifier=modifier,
        is_leader=leader,
        is_overlay_toggle=overlay,
        is_escape=escape,
        is_grave=grave,
        char=char,
    )


def test_double_tap_leader_opens_and_closes_command_mode():
    core = _CommandModeCore()

    assert _press(core, now=1.00, leader=True) is None
    assert _press(core, now=1.20, leader=True) == "command_mode_started"
    assert core.command_mode_active

    assert _press(core, now=2.00, leader=True) is None
    assert _press(core, now=2.20, leader=True) == "command_mode_cancelled"
    assert not core.command_mode_active


def test_taps_further_apart_than_the_window_do_not_trigger():
    core = _CommandModeCore()

    assert _press(core, now=1.0, leader=True) is None
    assert _press(core, now=1.0 + _CommandModeCore.DOUBLE_TAP_SECONDS + 0.01, leader=True) is None
    assert not core.command_mode_active


def test_an_intervening_letter_disqualifies_the_double_tap():
    """Shift, A, Shift is typing — not a deliberate double-tap."""
    core = _CommandModeCore()

    _press(core, now=1.00, leader=True)
    _press(core, now=1.05, char="a", modifier=False)
    assert _press(core, now=1.10, leader=True) is None
    assert not core.command_mode_active


def test_command_letters_map_to_their_actions():
    core = _CommandModeCore()
    for letter, expected in _CommandModeCore.COMMANDS.items():
        _press(core, now=1.0, leader=True)
        assert _press(core, now=1.2, leader=True) == "command_mode_started"
        assert _press(core, now=1.3, char=letter, modifier=False) == expected
        assert not core.command_mode_active, "the letter should close command mode"


def test_command_mode_lapses_after_its_timeout():
    core = _CommandModeCore()
    _press(core, now=1.0, leader=True)
    _press(core, now=1.2, leader=True)

    late = 1.2 + _CommandModeCore.COMMAND_MODE_SECONDS + 0.1
    assert _press(core, now=late, char="c", modifier=False) is None
    assert not core.command_mode_active


def test_escape_cancels_command_mode():
    core = _CommandModeCore()
    _press(core, now=1.0, leader=True)
    _press(core, now=1.2, leader=True)

    assert _press(core, now=1.3, escape=True, modifier=False) == "command_mode_cancelled"
    assert not core.command_mode_active


def test_overlay_toggle_wins_over_the_leader_on_a_shared_key():
    """Right shift is both the default toggle and part of the shift leader."""
    core = _CommandModeCore()

    assert _press(core, now=1.0, leader=True, overlay=True) is None
    assert _press(core, now=1.2, leader=True, overlay=True) == "toggle_overlays"
    assert not core.command_mode_active, "the overlay gesture must not open command mode"


def test_alternating_shifts_are_not_read_as_either_gesture():
    """rshift → lshift → rshift is two half-gestures, not one double-tap."""
    core = _CommandModeCore()

    _press(core, now=1.00, leader=True, overlay=True)   # right shift
    _press(core, now=1.10, leader=True)                 # left shift
    assert _press(core, now=1.20, leader=True, overlay=True) is None
    assert not core.command_mode_active


def test_cinematic_capture_only_fires_while_cinematic_mode_is_on():
    core = _CommandModeCore()
    assert _press(core, now=1.0, grave=True, char="`", modifier=False) is None

    core.cinematic_mode_active = True
    assert _press(core, now=2.0, grave=True, char="`", modifier=False) == "cinematic_capture"


def test_each_device_keeps_its_own_double_tap_state():
    """One tap on each of two keyboards is not a double-tap."""
    core = _CommandModeCore()

    assert _press(core, "kbd-a", now=1.0, leader=True) is None
    assert _press(core, "kbd-b", now=1.1, leader=True) is None
    assert not core.command_mode_active

    assert _press(core, "kbd-a", now=1.2, leader=True) == "command_mode_started"


def test_forget_scope_drops_a_disconnected_device():
    core = _CommandModeCore()
    _press(core, "kbd-a", now=1.0, leader=True)
    core.forget_scope("kbd-a")

    # State is gone, so the next tap starts a fresh gesture rather than
    # completing the one from before the disconnect.
    assert _press(core, "kbd-a", now=1.2, leader=True) is None


def test_configured_leader_and_toggle_are_honoured():
    core = _CommandModeCore()
    core.set_leader_key("Double-Tap Ctrl")
    assert core.leader_mod == "ctrl"

    core.set_overlay_toggle_key("ralt")
    assert core.overlay_toggle_key == "ralt"

    core.set_overlay_toggle_key("not-a-key")
    assert core.overlay_toggle_key == "ralt", "an unknown token must be ignored"
