<!--
 Masha — Browser extension selection translator.
 Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
 Licensed under the GNU General Public License v3.0 or later.
-->

# Porting MASHA to Falkon (and other non-WebExtension browsers)

Falkon (KDE, built on QtWebEngine) does **not** implement the WebExtensions API
— there is no `chrome.*`/`browser.*` runtime. So MASHA cannot be loaded there as
a WXT extension; Falkon needs a **separate plugin** written against its own
system (C++ or Python via [PyFalkon]). This document is the contract for that
port. It is deliberately small.

## What you reuse for free

Two things mean the port is *not* a rewrite of the translation logic:

1. **`src/core/` is browser-free.** `constants.ts`, `prompt.ts`, and
   `translator.ts` contain zero platform calls. They are ~150 lines of obvious
   logic (build messages → POST → strip `<think>` → clean). Read them as the
   spec.
2. **The prompt + model already live in Python.** The Falkon plugin runs in the
   same monorepo, so it can `import` the canonical definitions from
   `packages/xian-vl` / `packages/shared-types` directly instead of mirroring
   `src/core/`. The TS `core/` is itself a mirror of those Python sources
   (`shared_types/constants.py`, `xian/pipeline.py`), so both platforms trace
   back to one source of truth.

## What you must implement

Reimplement exactly one interface — `PlatformBridge` in
[`src/platform/bridge.ts`](../src/platform/bridge.ts) — in the host language.
Six methods, no more:

| Method | WXT realization | Falkon equivalent |
| --- | --- | --- |
| `getConfig` / `setConfig` | `chrome.storage.local` | Falkon `Settings` / a JSON file |
| `onTranslateCommand` | `chrome.contextMenus` + `tabs.sendMessage` | Falkon context-menu plugin hook |
| `getSelectionContext` | DOM selection + nearest block `innerText` | injected JS via `QWebEnginePage.runJavaScript` |
| `showOverlay` | injected `<div>` overlay | injected JS overlay (same HTML/CSS) |
| `replaceSelection` | input `value` splice | injected JS on the focused input |
| `showError` | injected `<div>` | injected JS |

The selection-capture and overlay JS can be lifted almost verbatim from
[`src/entrypoints/content.ts`](../src/entrypoints/content.ts) and injected as a
string — QtWebEngine runs the same DOM APIs.

## Network note

Route the Lemonade call from the **plugin process**, not page JS — same reason
the extension routes it through the background worker: a page-context `fetch` to
a local `http://` node from an `https://` page is mixed-content-blocked.

[PyFalkon]: https://api.kde.org/falkon/pyfalkon/
