# AI Developer Guide (AGENTS.md)

Welcome, AI coding assistant! To maintain codebase integrity and prevent architectural drift, you **must** strictly adhere to the structural, concurrency, and localization guidelines defined in this document when modifying the `xian-vl` repository.

---

## 1. Monorepo Landscape

This repository is structured as a Python-based monorepo managed via the **`uv`** package manager. The configuration is defined in the root [pyproject.toml](file:///home/clem/src/cursor/xian-vl/pyproject.toml).

The project is split into applications and shared library packages:

### Applications (`apps/`)
* **[mage-client](file:///home/clem/src/cursor/xian-vl/apps/mage-client)**: The primary client application. A lightweight PyQt6-based desktop overlay/HUD that manages visual translation (Lens overlay), chat integration (Sidebar), settings configuration, and audio translation.
* **[luduan-client](file:///home/clem/src/cursor/xian-vl/apps/luduan-client)**: Document processing and translation client.
* **[lore-client](file:///home/clem/src/cursor/xian-vl/apps/lore-client)**: Local encyclopedia / knowledge base search client.

### Shared Library Packages (`packages/`)
* **[xian-vl](file:///home/clem/src/cursor/xian-vl/packages/xian-vl)**: The core orchestration engine. Contains [pipeline.py](file:///home/clem/src/cursor/xian-vl/packages/xian-vl/src/xian/pipeline.py) which manages VLM prompt construction, image scaling, and response parsing, as well as [async_engine.py](file:///home/clem/src/cursor/xian-vl/packages/xian-vl/src/xian/async_engine.py) which drives the background event loop.
* **[shared-types](file:///home/clem/src/cursor/xian-vl/packages/shared-types)**: Canonical data models (Pydantic v2 models) shared across the workspace, defined in [models.py](file:///home/clem/src/cursor/xian-vl/packages/shared-types/src/shared_types/models.py), along with the localization runtime state.
* **[localize](file:///home/clem/src/cursor/xian-vl/packages/localize)**: CLI tool that automates the translation of base UI localization strings using Lemonade's completions endpoint.

### Package & Dependency Management (`uv`)
Always use `uv` commands when executing tasks in this monorepo:
* **Dependency installation**: Run `uv sync` from the workspace root to synchronize virtual environments for all members.
* **Running scripts**: Run tools using `uv run` to ensure correct virtual environment resolution.
* **Adding dependencies**: Modify the appropriate `pyproject.toml` under `apps/` or `packages/` and run `uv sync` to update `uv.lock`.

---

## 2. Concurrency Mandate

MAGE is designed to provide a low-latency, stutter-free real-time overlay during gaming. Stalls in the UI loop are unacceptable.

### Strict Rules for Threading & Network I/O
1. **Never Block the PyQt6 Main Thread**: Under no circumstances should synchronous network requests, disk I/O, or heavy CPU operations run on the main thread.
2. **Use `QThread` Workers**: Network operations must be wrapped in PyQt6-compatible background worker threads extending `QThread`. Examples include:
   * [InferenceWorker](file:///home/clem/src/cursor/xian-vl/apps/mage-client/src/mage/workers.py#L36-L144): Handles standard/streaming visual translations.
   * [CinematicWorker](file:///home/clem/src/cursor/xian-vl/apps/mage-client/src/mage/workers.py#L216-L287): Orchestrates fused audio and screen translation.
   * [RaidWorker](file:///home/clem/src/cursor/xian-vl/apps/mage-client/src/mage/workers.py#L365-L519): Runs continuous voice-transcription loops.
   * [StatusWorker](file:///home/clem/src/cursor/xian-vl/apps/mage-client/src/mage/workers.py#L520-L548): Performs periodic HTTP health-checks.
3. **Submit to `AsyncEngine`**: Long-lived background connections and client requests are managed by [AsyncEngine](file:///home/clem/src/cursor/xian-vl/packages/xian-vl/src/xian/async_engine.py). Submit your `asyncio` coroutines using `self.processor.engine.submit(coro)` to run them on the background thread's loop (`xian-async-engine`) and resolve the resulting `Future`.
4. **Communicate via Qt Signals**: Background workers must communicate progress, partial stream updates, errors, and completed payloads back to the main thread via Qt Signals (`pyqtSignal`), ensuring thread safety when interacting with UI widgets.

### Wayland Compositor Compatibility & Input Capture
* **Wayland Key Grabs**: Standard window managers/compositors under Wayland block standard X11 global key grabs.
* **evdev Background Hook**: The background hotkey hook is driven by [EvdevHotkeyListener](file:///home/clem/src/cursor/xian-vl/apps/mage-client/src/mage/capture/hotkeys.py#L55) which listens directly to raw device events in `/dev/input/` asynchronously using the `evdev` package.
* **Fallbacks**: If the user lacks access to `/dev/input/` (e.g. not in the `input` group) or is running on macOS/Windows, the system falls back to `pynput` as a user-space hook.
* **Guidance**: Any changes to input capturing must preserve this Wayland-compatible `evdev` implementation and keep event listener loops off the PyQt6 GUI thread.

---

## 3. Localization Guardrail

Hardcoding user-facing strings directly in UI layouts or components is **strictly prohibited**. 

### JSON-Based Translation Framework
All user-facing interface text must route through the global localization manager.
* **The canonical manager**: Defined in [state.py](file:///home/clem/src/cursor/xian-vl/packages/shared-types/src/shared_types/state.py). It uses `RuntimeState` to dynamically load locale JSON dictionaries from the shared locales directory [packages/shared-types/locales](file:///home/clem/src/cursor/xian-vl/packages/shared-types/locales).
* **Usage**: Import the global translation helper `t` from the shared types state:
  ```python
  from shared_types.state import t
  ```
  And resolve localized strings via key identifiers:
  ```python
  self.setWindowTitle(t("settings.dialog.title"))
  ```

### Maintaining Translation Dictionaries
1. **Adding Strings**: All new UI strings must be added first as key-value pairs inside the English reference dictionary: [en.json](file:///home/clem/src/cursor/xian-vl/packages/shared-types/locales/en.json). Write both a `"value"` (the English text) and a `"context"` description to help the AI translation model.
2. **Regenerating Locales**: To translate and build target locale files (`zh.json`, `ja.json`, etc.) automatically, run the CLI utility defined in [cli.py](file:///home/clem/src/cursor/xian-vl/packages/localize/src/localize/cli.py):
  ```bash
  uv run -m localize.cli
  ```
  This script communicates with the local running Lemonade instance using the active model to translate new entries, outputting flat JSON structures directly into the locales folder.
