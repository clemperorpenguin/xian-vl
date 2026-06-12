# Implementation Plan — Perfecting Overlay Window Focus (get-lucky)

## Goal

Make MAGE behave like a true always-on-top overlay:

1. **Auto-promote on new translation** — every time a translation bubble (or any
   overlay) appears or updates, it is forcibly raised above the active game
   window, with no alt-tab required.
2. **Command-key double-tap to show/hide all overlays** — double-tapping the
   command/super key (mirroring the existing double-shift "command mode"
   gesture) toggles the visibility of every overlay window at once.

## Current behavior (why it falls short)

- All overlays derive from `MageOverlayWindow`
  (`apps/mage-client/src/mage/ui/overlay_base.py`) and set
  `WindowStaysOnTopHint | Tool | WindowDoesNotAcceptFocus` plus
  `WA_ShowWithoutActivating`. They never take focus — good — but they rely
  purely on the Qt stays-on-top hint for stacking.
- `ResultBubble.__init__` and `update_text`
  (`apps/mage-client/src/mage/ui/result_bubble.py:214,340`) call `raise_()`.
  On most Linux WMs a single `raise_()` on a non-activating `Tool` window is
  **not** honored above a focused/fullscreen game, so the bubble renders behind
  it until the user alt-tabs to MAGE.
- Reliable fronting only happens today when the user configures a target window
  title, which enables `_track_target_window` (`app.py:1901`) to hide/show and
  re-align overlays. That is the "manually set it to always come to the front"
  workaround the user describes.
- The leader double-tap (`hotkeys.py`, `command_mode_active`) is wired only to
  **command mode** (the OSD). There is no global show/hide-all gesture.

## Part A — Reliable "bring to front" on every new translation

### A1. Add an X11 "stay above" hint (`utils/window_binder.py`)
Add `set_above_state_x11(win_id)` parallel to the existing
`set_bypass_compositor_hint_x11` (`window_binder.py:556`). It sends a
`_NET_WM_STATE` client message to the root window adding
`_NET_WM_STATE_ABOVE` (and `_NET_WM_STATE_STAYS_ON_TOP` where present). This is
honored by EWMH WMs far more reliably than the Qt hint and does **not** steal
focus, so the click-through / no-focus overlay contract is preserved. Guard on
`QGuiApplication.platformName() == "xcb"` exactly like the existing helper; it
is a no-op on Wayland/macOS/Windows.

### A2. Add a `promote()` helper to `MageOverlayWindow` (`overlay_base.py`)
```
def promote(self):
    """Force this overlay above the active window without taking focus."""
    if not self.isVisible():
        return
    # Re-assert the stays-on-top flag in case the WM dropped it
    if not (self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint):
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.show()  # re-show without activating (WA_ShowWithoutActivating is set)
    self.raise_()
    set_above_state_x11(self.winId())
```
Call `set_above_state_x11(self.winId())` from `showEvent` too (next to the
existing `set_bypass_compositor_hint_x11` call at `overlay_base.py:160`).

### A3. Promote whenever a translation appears (`result_bubble.py`)
- Replace the bare `self.raise_()` in `__init__` (`:215`) and `update_text`
  (`:340`) with `self.promote()`.
- Persistent dialogue/cinematic/raid bubbles update through `update_text`, so
  they get promoted on every refresh automatically.

### A4. Periodic re-assert in the tracking loop (`app.py`)
The 100 ms `window_tracking_timer` (`app.py:526-528`) currently returns early
when no target binder is set, so unbound overlays never get re-raised. Add a
lightweight, throttled re-assert that runs **regardless** of the binder:

- Add `self._last_promote_ts = 0.0` in `__init__`.
- At the top of `_track_target_window` (before the `if not self.target_binder:
  return`), every ~750 ms call `promote()` on each currently-visible overlay
  from `_all_overlays()` (extended — see A5) **unless** overlays are
  user-hidden (Part B). This keeps the HUD above a game that periodically
  re-raises itself, without the cost of doing it every tick.

### A5. Make `_all_overlays()` complete (`app.py:1804`)
It currently omits the active/persistent bubbles. Reuse the fuller
`raw_overlays` collection already built in `_track_target_window`
(`app.py:1936-1952`) — extract it into a single `_collect_overlays()` helper and
have both `_all_overlays()` and the promote tick use it, so nothing is missed.

## Part B — Right-shift double-tap to show/hide all overlays

The toggle gesture defaults to **double-tap right shift**, mirroring the
existing double-tap left-shift command-mode gesture. Because the default leader
is also `shift`, the two physical shift keys must be split: **left shift = leader
(command mode), right shift = overlay toggle.**

### B1. New signal + right-shift double-tap detector (`capture/hotkeys.py`)
- Add `toggle_overlays = pyqtSignal()` to the `HotkeyListener` base
  (`hotkeys.py:29-43`).
- Introduce a dedicated toggle key, default `'rshift'` (right shift),
  independent of the leader. Add:
  - `self.overlay_toggle_key = 'rshift'`
  - per-device `self.last_overlay_press_time = {}` (Evdev) /
    `self.last_overlay_press_time = 0.0` (Pynput)
  - `_is_overlay_toggle_key(...)` keyed to the *specific* side:
    - Evdev: `keycode == 54` (right shift) — vs. left shift `42`.
    - Pynput: `key == keyboard.Key.shift_r`.
- In the key-press handling of **both** `EvdevHotkeyListener._handle_event`
  (around `:292`) and `PynputHotkeyListener.on_press` (around `:414`), add the
  overlay-toggle branch **before** the leader branch: if the key is the overlay
  toggle key, run its own clean double-tap check (`mod_clean` true, within
  0.4 s), `emit toggle_overlays` on success, and **`return` in all cases** so
  the key never falls through to the leader logic. Right shift is thereby fully
  owned by the overlay toggle; left shift continues to drive command mode.
- **Why early-return instead of editing `_is_leader_mod_key`:** modifier *state*
  tracking (`modifiers[...]['shift']`, `mod_clean`) happens earlier in the
  handler (`hotkeys.py:275-290`) and should still treat either shift as "shift"
  for chord/clean detection — only the *double-tap trigger* needs to be
  side-specific. Returning early from the toggle branch achieves that without
  changing `_is_leader_mod_key` (which still matches both 42/54 for state, but
  never sees 54 as a trigger because we returned).
- **Conflict guard:** if the resolved overlay toggle key is one the leader would
  also trigger on (e.g. user sets leader to right shift), the overlay toggle
  wins for that specific key via the early return. With defaults (leader = left
  shift, toggle = right shift) they are fully distinct.
- **Pynput caveat (macOS/Windows):** pynput reliably reports `Key.shift_r` for
  the right shift on these platforms, but left shift may arrive as the generic
  `Key.shift`; the leader matcher already accepts `shift`/`shift_l`, so this is
  fine. On Linux we use evdev scancodes (42 vs 54), which are unambiguous.

### B2. Toggle handler (`app.py`)
- Wire it next to the existing command-mode connections (`app.py:463-465`):
  `self.hotkey_listener.toggle_overlays.connect(self.toggle_all_overlays)`.
- Implement:
```
def toggle_all_overlays(self):
    self._overlays_hidden = not getattr(self, "_overlays_hidden", False)
    overlays = [w for w in self._collect_overlays() if self._is_valid_widget(w)]
    if self._overlays_hidden:
        self._user_hidden_overlays = [w for w in overlays if w.isVisible()]
        for w in self._user_hidden_overlays:
            w.setVisible(False)
    else:
        for w in getattr(self, "_user_hidden_overlays", []):
            if self._is_valid_widget(w):
                w.setVisible(True)
                if hasattr(w, "promote"):
                    w.promote()
        self._user_hidden_overlays = []
```

### B3. Respect the user-hide state in the binder (`app.py:1959-1970`)
The visibility loop in `_track_target_window` auto-restores overlays via
`_temp_hidden_by_binder`. Gate both the restore path and the Part A promote tick
on `not getattr(self, "_overlays_hidden", False)`, so a user-initiated hide is
not undone on the next 100 ms tick. (User hide takes precedence over binder
auto-show; toggling back returns control to the binder.)

## Part C — Settings & polish (optional, low risk)

- Add `KEY_OVERLAY_TOGGLE_KEY = "overlay_toggle_key"` to `settings_keys.py` and
  push it to the listener via a `set_overlay_toggle_key()` setter (parallel to
  `set_leader_key`), defaulting to `rshift`. Surface it in the settings UI beside
  the leader-key picker. If skipped for v1, hard-default to `rshift`.
- Tray menu: add a "Show/Hide Overlays" action calling `toggle_all_overlays`
  for discoverability and as a Wayland-friendly fallback.

## Files touched

| File | Change |
|------|--------|
| `apps/mage-client/src/mage/utils/window_binder.py` | `set_above_state_x11()` helper |
| `apps/mage-client/src/mage/ui/overlay_base.py` | `promote()`, call `set_above_state_x11` in `showEvent` |
| `apps/mage-client/src/mage/ui/result_bubble.py` | `raise_()` → `promote()` in `__init__` + `update_text` |
| `apps/mage-client/src/mage/capture/hotkeys.py` | `toggle_overlays` signal + right-shift double-tap (both listeners) |
| `apps/mage-client/src/mage/app.py` | `_collect_overlays()`, promote tick, `toggle_all_overlays()`, binder gate, signal wiring |
| `apps/mage-client/src/mage/settings_keys.py` (+ settings UI) | optional toggle-mod setting |

## Platform notes / risks

- **X11**: full functionality via `_NET_WM_STATE_ABOVE` + `raise_()`.
- **Wayland**: client-side restacking is unavailable; global hotkeys still work
  (evdev), and the toggle/visibility logic works, but forced "above" relies on
  the compositor honoring the Qt hint. Document this; the tray action (Part C)
  is the fallback.
- **Focus contract**: keep `WA_ShowWithoutActivating` + `WindowDoesNotAcceptFocus`.
  `promote()` must never call `activateWindow()` — that would steal focus from
  the game and break click-through expectations.
- **Throttle** the periodic promote (~750 ms) to avoid X11 chatter / flicker at
  the 100 ms tick rate.

## Verification

1. No target-window binding configured: start a translation over a focused
   fullscreen game → bubble appears on top immediately, no alt-tab.
2. Refresh a dialogue/cinematic bubble → stays on top across updates.
3. Double-tap **right** shift → all overlays hide; double-tap again → all
   return on top.
4. Confirm **left** shift double-tap still opens command mode (no regression),
   and that a left-shift double-tap does NOT toggle overlays (no cross-fire).
5. macOS smoke test: right-shift (`Key.shift_r`) double-tap toggles overlays via
   pynput; left shift still opens command mode.
