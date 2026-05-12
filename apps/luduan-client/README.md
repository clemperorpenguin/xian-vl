# Luduan Client 🦤

<img width="1024" height="1024" alt="luduan" src="https://github.com/user-attachments/assets/2785f66a-ce56-48c7-a37b-240fcc11ed42" />

Luduan is a CLI tool for EPUB-to-Robobook translation and narration. It allows you to translate EPUB files and generate audiobooks using the Lemonade Server.

## Features

- **EPUB Translation**: Translate EPUB files from one language to another.
- **Narration**: Generate audiobooks from translated text using TTS engines.
- **Batch Processing**: Process multiple EPUB files in a directory.

## Usage

To run Luduan from the monorepo root:

```bash
uv run luduan --help
```

### Commands

- `translate`: Translate an EPUB file.
- `robobook`: Full pipeline: translate → narrate → encode to Robobook.
- `batch`: Batch-process all EPUBs in a directory.

## Status

Note: The pipeline is currently a scaffold and some features are not yet fully wired up.
