# Xian-MAGE — Multilingual Assistant for Gaming Environments 🧙‍♂️

> [!IMPORTANT]
> **Project Status**: MAGE is the only fully tested, verified, and officially supported client application in the Xian-monorepo. 

> [!WARNING]
> **ASR / Raid Mode Limitation**: Live audio stream transcription (Raid Mode) is currently broken and non-functional because the backend Lemonade Server does not support receiving/processing audio uploads at this time. Standard vision OCR, dialogue, and chat modes are fully functional.

Xian-MAGE is a persistent, stateful Wayland desktop assistant for Linux. It provides real-time OCR, translation, visual grounding, and interactive chat directly on top of your desktop.

Xian-MAGE acts as a lightweight client powered by the **[Lemonade-SDK OmniRouter](https://lemonade-server.ai/)**, bringing powerful Vision-Language models to your screen. Because Lemonade handles the inference, **all major GPUs with sufficient VRAM should work with Vulkan.**

## Features

- **Configurable Action Key**: Seamlessly trigger actions via global Wayland hotkeys (requires `evdev`).
  - `Double-Tap Shift`: Opens the Action Key menu (default).
- **Contextual Memory**: Xian-MAGE maintains a sliding-window memory of your recent screen captures and chat history, allowing you to ask follow-up questions about what you just captured.
- **Visual Grounding**: Ask the assistant "where do I click?" and watch it highlight the UI element directly on your screen.
- **Instant Dictionary Lookups**: Hover over translated text bubbles and press `Alt` for real-time CC-CEDICT dictionary breakdowns.
- **Click-Through Overlay**: Translations overlay the original content but are transparent to mouse inputs, letting you play games or read seamlessly.

## Requirements

- **Linux with Wayland** (Windows and macOS are experimental but should work)
- **User Permissions** (Linux): Your user must be in the `input` group for global hotkeys (`sudo usermod -aG input $USER`)
- **[Lemonade Server](https://lemonade-server.ai/)** running at `http://localhost:13305`

## Usage

To launch the MAGE client from the monorepo root:

```bash
uv run --package mage-client mage
```

### Hotkeys
- **Capture & Translate** — Action Key (`Double-Tap Shift`), then `C` to freeze the screen. Drag to select, then Translate / Explain / Chat.
- **Chat Sidebar** — Action Key, then `A`.
- **Settings** — Action Key, then `S`.
