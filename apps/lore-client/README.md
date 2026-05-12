# LORE: Localized Organizer for Research & Entities 📜

Interactive CLI tool for researching Chinese entities and compiling localized English Markdown wiki. Collects translated information into a unified knowledge base.

## Features

- **Dynamic SearXNG Discovery**: Automatically fetches and rotates through high-performance public instances.
- **Playwright Scraping**: High-fidelity web scraping with Readability-based content cleaning.
- **Interactive Loop**: Search, select sources, and ingest content interactively.
- **Wiki Compilation**: Generates Obsidian-compatible Markdown files with YAML frontmatter and bidirectional linking.

## Installation

Ensure you are in the monorepo root and run:

```bash
uv sync
```

Then install Playwright browsers:

```bash
uv run --package lore-client python -m playwright install chromium
```

## Usage

```bash
uv run lore "Entity Name"
```

The tool will output Obsidian-compatible Markdown files to the `wiki/` directory.
