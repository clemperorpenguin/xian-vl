# Luduan Client 🦤

<img width="1024" height="1024" alt="luduan" src="https://github.com/user-attachments/assets/2785f66a-ce56-48c7-a37b-240fcc11ed42" />

Luduan is a CLI that translates books into another language and narrates them as
an audiobook, powered by the **[Lemonade Server](https://lemonade-server.ai/)**.
Translation and narration are verified working end to end against a live
Lemonade node (chat model for translation, `kokoro-v1` for TTS).

## Features

- **Translation**: translate a book from one language to another (defaults Chinese → English, tuned for web novels — wuxia/xianxia/xuanhuan).
- **Narration**: generate an audiobook from the translated text via Lemonade TTS.
- **Opus output, no system tools required**: audio is encoded to `.opus` via `soundfile` (bundles libsndfile + libopus), so no `ffmpeg`/`opusenc` install is needed. If those CLIs *are* present they're used as a fallback.
- **KOReader sync manifest**: a `.audio.json` sidecar maps each passage to its audio offset for synchronised playback on e-readers.
- **Batch Processing**: process every book in a directory.

## Input formats

| Format | Status | How |
| --- | --- | --- |
| **EPUB** (`.epub`) | ✅ Working | text extraction via `ebooklib` + BeautifulSoup |
| **PDF — text** (`.pdf`) | 🔜 Planned | direct text extraction (`pypdf`/`pdfminer`) → same translate/narrate path |
| **PDF — scanned** (`.pdf`) | 🔜 Planned | page images routed through the Xian **vision OCR pipeline** (the MAGE OCR+translate engine) |
| **Comics** (`.cbz`, `.cbr`, `.cb7`) | 🔜 Planned | archive of page images → vision OCR per panel/page → translated text → narration |

Text formats (EPUB, text PDF) are deterministic and cheap. Image formats
(scanned PDF, comics) reuse the existing `xian` vision pipeline, so dialogue is
OCR'd and translated with the same prompt logic as MAGE. See the roadmap entry
in [docs/ROADMAP.md](../../docs/ROADMAP.md).

## Usage

```bash
uv run luduan --help
```

### Commands

- `translate`: Translate a book, saving Markdown + structured JSON.
- `robobook`: Full pipeline — translate → narrate → encode to `.opus` (+ manifest).
- `batch`: Batch-process all books in a directory.

Example:

```bash
uv run luduan robobook mybook.epub --source Chinese --target English --voice af_heart -o ./out
```
