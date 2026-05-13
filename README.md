# Xian-VL Monorepo 🇨🇳🇯🇵🇰🇷🎮📚🕸️🍋🪄

Xian-VL is a hub-and-spoke monorepo containing applications and packages for vision-language tasks, real-time translation, and gaming assistance, all powered by the **[Lemonade Server](https://lemonade-server.ai/)**.

## Project Structure

This repository uses a `uv` workspace to manage multiple packages and applications:

### Applications

- **[MAGE](apps/mage-client/README.md)**: A PyQt6 gaming HUD for real-time OCR, translation, and visual grounding on Wayland (Works!).
- **[MASHA](apps/masha-extension/README.md)**: A browser extension for translating selected text on web pages (WIP).
- **[NATE](apps/nate/README.md)**: An offline-first Chinese OCR reader, dictionary, and translation tool with optional remote LLM acceleration via Lemonade (Android).
- **[LORE](apps/lore-client/README.md)**: Compiles collected information into a unified knowledge base. (WIP)
- **[Luduan](apps/luduan-client/README.md)**: A CLI tool for EPUB-to-Robobook translation and narration (WIP).

### Packages

- **`packages/shared-types`**: Canonical Pydantic models, enums, and constants used across the monorepo.
- **`packages/xian-vl`**: Core engine handling prompt engineering, context management, and Lemonade API interactions.

## Getting Started

### Requirements

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **[Lemonade Server](https://lemonade-server.ai/)** running locally or accessible remotely.

### Installation

Clone the repository and sync the workspace:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
uv sync --all-packages
```

## Roadmap

- [x] Monorepo migration and workspace setup
- [x] Cinematic mode — translate audio with on-screen context
- [x] Dialogue mode — keep an area selected, advance with a click (VNs / story RPGs)
- [x] Game oracle / chatbot integration / complete LORE
- [x] Complete MASHA extension
- [x] OCR/dictionary Android app
- [ ] UI and documentation localized
- [ ] Make everything pretty
- [ ] Server benchmark to select best settings
- [ ] Flatpak / SteamOS support for MAGE client

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.
