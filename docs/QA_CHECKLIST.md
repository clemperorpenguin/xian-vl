# MAGE Release QA Checklist (Smoke & Regression)

Run this list **before every release** and whenever you touch the capture, overlay,
worker, or hotkey paths. The goal is to shake out the bugs automated tests can't
catch on a GUI/Wayland/audio app. Copy this into an issue per release and tick boxes.

> Legend: 🟢 = must pass to ship · 🟡 = should pass · ⏱️ = run for a long session (soak)

## 0. Pre-flight (automated — do these first)
- [ ] 🟢 `uv sync --all-packages` succeeds on a clean checkout (no pre-existing `.venv`).
- [ ] 🟢 `uv run pytest` is green (all packages, see CI).
- [ ] 🟢 `ruff check .` is clean (or only known/ignored findings).
- [ ] 🟢 `git status` is clean after a full launch+quit — **no stray `-` file** appears
      in the repo root or `apps/mage-client/` (regression guard for the parecord bug).

## 1. First-run / install (do on a clean machine or VM if possible)
- [ ] 🟢 Fresh `git clone` + `./mage.sh` launches to a usable state, following only the README.
- [ ] 🟢 On a minimal Linux box (no `python3-dev`/build tools): install either succeeds
      via wheels, or fails with a **clear, actionable** message (evdev needs a compiler).
- [ ] 🟡 `./mage.sh --install` registers the launcher; `--uninstall` fully removes it.
- [ ] 🟡 Lemonade server **not running** → app shows a clear "server unreachable" state,
      does not hang or crash, and recovers when the server comes up.

## 2. Hotkeys & OSD
- [ ] 🟢 Double-tap `Shift` opens the Action Menu / OSD (evdev path, user in `input` group).
- [ ] 🟢 Fallback path: with no `/dev/input` access, `pynput` still triggers hotkeys.
- [ ] 🟢 Double-tap `Right Shift` toggles Show/Hide all overlays; tray menu entry does too.
- [ ] 🟡 Re-bind the leader key in Settings; new binding works, old one stops.
- [ ] 🟢 **Hot-plug**: connect a USB keyboard/mouse *after* launch → hotkeys work on it
      within ~5s (the device-rescan loop, previously dead, now actually runs).
- [ ] 🟢 **Mouse tracking is stable**: leave the app idle 30s+, then ask "where do I
      click?" — the highlight tracks the real cursor (no periodic jump to the corner).

## 3. Translation / Lens overlay
- [ ] 🟢 Action Menu → `C`, select a region, Translate → overlay appears over the game.
- [ ] 🟢 Overlay is click-through (mouse events reach the game underneath).
- [ ] 🟡 Hover a bubble + `Alt` → CC-CEDICT breakdown (pinyin/definition) shows.
- [ ] 🟡 X11 vs Wayland: confirm always-on-top behaviour matches README caveats.
- [ ] 🟢 **Multi-monitor**: on a setup whose secondary monitor sits left/above the
      primary (non-zero virtual-desktop origin), draw a selection — the box stays put
      (doesn't drift across repaints) and translation crops the region you selected.

## 4. Dialogue mode
- [ ] 🟢 Lock a region; clicking advances/refreshes the translation inline.
- [ ] 🟡 ⏱️ Leave it running through 50+ advances — no leak, no slowdown, no zombie overlays.

## 5. Chat sidebar & visual grounding
- [ ] 🟢 Action Menu → `A` toggles the chat sidebar.
- [ ] 🟢 Ask "where do I click?" → a highlight lands on plausible screen coordinates.
- [ ] 🟡 Action Menu → `T` translates chat input.

## 6. Audio modes (Cinematic / Raid) — most fragile, test hard
- [ ] 🟢 Cinematic mode produces translated voice output coupled to on-screen vision.
- [ ] 🟢 Raid mode runs the continuous transcription loop and shows live results.
- [ ] 🟢 **System-audio capture actually contains audio** on both PipeWire (`pw-record`)
      **and** PulseAudio (`parecord`) systems. (parecord path has been a known weak spot.)
- [ ] 🟢 Start/stop raid repeatedly (10×) — recorder subprocess is always cleaned up
      (`pgrep -a parecord; pgrep -a pw-record` shows nothing after stop).
- [ ] 🟡 ⏱️ Soak: run raid/cinematic for 30+ min — stable RAM/VRAM, no audio drift, no crash.

## 7. Settings & persistence
- [ ] 🟢 Change settings (API URL, model, hotkeys), restart, settings persist.
- [ ] 🟢 Point API URL at a wrong port → graceful error, no crash; fix it → recovers.
- [ ] 🟡 Performance Report exports valid Markdown; opt-in JSONL writes only when enabled.

## 8. Lifecycle & resources
- [ ] 🟢 Clean quit (tray → Quit) leaves no orphaned threads/subprocesses.
- [ ] 🟢 No Python tracebacks in logs during a normal session.
- [ ] 🟡 ⏱️ Long idle (1 hr) then resume — overlays and hotkeys still responsive.

## 9. Localization
- [ ] 🟡 Switch UI language; no missing-key fallbacks (no raw `settings.dialog.title` text).
- [ ] 🟡 `uv run -m localize.cli` regenerates locales without errors.

## 10. Cross-platform spot checks (if shipping those builds)
- [ ] 🟡 Windows lite + full `.zip` launch and connect to Lemonade.
- [ ] 🟡 macOS `.app` launches; Accessibility prompt appears; hotkeys work after grant.
