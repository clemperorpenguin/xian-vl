# Xian-VL: Multilingual Assistant for Gaming Environments 🧙‍♂️

<img width="768" height="768" alt="xian" src="https://github.com/user-attachments/assets/7b9498fd-4786-481f-b2c9-e29632b2ec24" />

Xian-VL is a persistent, stateful Wayland desktop assistant for Linux. It provides real-time OCR, translation, visual grounding, and interactive chat directly on top of your desktop. 

Xian-VL acts as a lightweight client powered by the **[Lemonade-SDK OmniRouter](https://lemonade-server.ai/)**, bringing powerful Vision-Language models to your screen. Because Lemonade handles the inference, **all major GPUs with sufficient VRAM should work with Vulkan.**

## Features

- **Global Interaction**: Seamlessly trigger actions via global Wayland hotkeys (requires `evdev`).
  - `Super+Shift+C`: Open the "Lens" overlay to freeze the screen and capture a specific region.
  - `Super+A`: Toggle the persistent Chat Sidebar.
  - `Super+Shift+S`: Open the settings.
- **Contextual Memory**: Xian-VL maintains a sliding-window memory of your recent screen captures and chat history, allowing you to ask follow-up questions about what you just captured.
- **Visual Grounding**: Ask the assistant "where do I click?" and watch it highlight the UI element directly on your screen.
- **Instant Dictionary Lookups**: Hover over translated text bubbles and press `Alt` for real-time CC-CEDICT dictionary breakdowns.
- **Click-Through Overlay**: Translations overlay the original content but are transparent to mouse inputs, letting you play games or read seamlessly.

## Requirements

- **Linux with Wayland** (Windows and MacOS are experimental but should work)
- **Python 3.10+**
- **Lemonade-SDK**: Assumes a local instance running at `http://localhost:13305`. (For remote nodes, see Configuration below).
- **User Permissions**: Your user must be in the `input` group to read global hotkeys (`sudo usermod -aG input $USER`).
- **uv** package manager

## Installation

Xian-VL uses `uv` for fast dependency management. Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
uv venv
uv pip install -e .
```

*Note: Xian-VL is a lightweight client. You do not need to install PyTorch, CUDA, or ROCm locally. Ensure your Lemonade Server is running with an appropriate model collection loaded. The server will download a Qwen3.5-9B-GGUF model automatically, which requires about 6GB of spare VRAM to run.*

## Usage

1. **Start the Lemonade Server**: Make sure your local Lemonade instance is running (`http://localhost:13305`) and has a model collection loaded.
2. **Start Xian-VL**:
   ```bash
   uv run python main.py
   ```
3. **Capture & Translate**: Press `Super+Shift+C` to freeze the screen. Drag to select a region, then use the action bar to Translate, Explain, or Chat about the selected area.
4. **Chat Sidebar**: Press `Super+A` to open the sidebar and interact with the assistant based on your screen context.

## Roadmap

- Literary style selection
- Game oracle/chatbot
- Flatpak/SteamOS support
- Expanded language support

## License

GNU General Public License v3.0 - See LICENSE file for details.
