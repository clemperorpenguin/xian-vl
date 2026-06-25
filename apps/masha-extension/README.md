# MASHA - Multilingual Access & Site Handling Assistant 🌐

MASHA is a lightweight browser extension that provides **context-aware** text
translation on web pages, powered by a local [Lemonade Server]. Unlike snippet
translators, MASHA sends the surrounding page context to the model as reference,
so it disambiguates homographs, pronouns, honorifics, and domain terms — the
same prompt approach used by MAGE.

## Features

- **Right-click to translate**: select any text, right-click → **Translate with MASHA**.
- **Context-aware**: uses the page title + surrounding section as reference, translating only your selection. (e.g. Spanish *banco* → "bench" in a park sentence, "bank" in a finance sentence.)
- **Non-destructive overlay**: read-only page text shows the translation in a bubble with **Show original** / **Copy**; the page is never overwritten.
- **Replace-in-place for inputs**: in an editable `<input>`/`<textarea>`, the selection is replaced so you can translate text you are composing.
- **Lemonade integration**: OpenAI-compatible, defaults to the canonical `LMX-Omni-5.5B-Lite` model; reasoning (`<think>`) output is stripped automatically.
- **Configurable**: server URL, source/target language, and optional style terms.

## Browser support

| Browser | Status | Build |
| --- | --- | --- |
| Chrome / Edge / Chromium | ✅ Supported | `npm run build` (chrome-mv3) |
| Firefox | ✅ Supported | `npm run build:firefox` (firefox-mv2) |
| Falkon | ⛔ Not supported as an extension | needs a separate plugin — see [docs/FALKON.md](docs/FALKON.md) |

Falkon (QtWebEngine) does not implement the WebExtensions API. The code is
structured so a future Falkon plugin reimplements a single ~6-method interface
(`src/platform/bridge.ts`) and reuses the browser-free `src/core/` logic.

## Development

```bash
cd apps/masha-extension
npm install
npm run compile        # type-check
npm run build          # chrome-mv3 -> dist/chrome-mv3
npm run build:firefox  # firefox-mv2 -> dist/firefox-mv2
npm run dev            # live-reload dev build
```

### Install (developer mode)

- **Chrome/Edge**: `chrome://extensions` → enable Developer mode → **Load unpacked** → select `dist/chrome-mv3`.
- **Firefox**: `about:debugging` → This Firefox → **Load Temporary Add-on** → pick `dist/firefox-mv2/manifest.json`.

## Configuration

Open the popup and set the Lemonade URL (default `http://localhost:13305/v1`),
the source/target languages, and any style terms. Settings persist via
`chrome.storage`.

## Architecture

```
src/
├── core/        # browser-free: constants, prompt builder, translator (no chrome.*)
├── platform/    # bridge.ts — the one interface a platform implements
├── utils/       # config + Lemonade URL helpers
└── entrypoints/ # WXT realization: background (menu + fetch), content (capture + overlay), popup
```

`core/` mirrors the Python sources in `packages/xian-vl` and
`packages/shared-types`; see [docs/FALKON.md](docs/FALKON.md) for the porting
contract.

[Lemonade Server]: https://lemonade-server.ai/
