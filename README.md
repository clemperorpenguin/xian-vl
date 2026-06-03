# Xian-MAGE — Real-Time Vision-Language Assistant for Gaming Environments 🧙‍♂️🎮🤖

Xian-MAGE is a real-time, persistent desktop gaming HUD and assistant for Linux (Wayland). Powered by the **[Lemonade Server](https://lemonade-server.ai/)** backend, it overlays real-time OCR, translation, visual grounding clicking visualizers, and interactive conversational chat directly on top of active gaming environments.

Because inference is orchestrated via Lemonade, **MAGE supports Vulkan-accelerated execution, running smoothly on AMD Radeon™ GPUs and other accelerators.**

See it in action on YouTube: https://www.youtube.com/watch?v=Izu_8pql7cE

<img width="400" height="340" alt="mage" src="https://github.com/user-attachments/assets/bb51b2c6-378f-4a3e-b25d-05ad284e374b" />

---

## 🌟 Core Features (Fully Operational)

- **Vulkan / AMD GPU Accelerated**: Low-latency Vision-Language Model execution powered by backend GPU hardware acceleration.
- **Click-Through Desktop Overlay**: Transparent PyQt6 overlay windows that display translated text directly over game HUDs and dialogue while remaining completely invisible to mouse inputs.
- **Wayland Global Hotkey & Command OSD**: Trigger translation, OSD configurations, and sidebars seamlessly using customizable system-wide leader hotkeys.
- **Dialogue Mode (Autoplay VNs / Story RPGs)**: Lock onto a screen region, translate automatically, and advance/refresh translations inline with a simple mouse click.
- **Visual Grounding Target Highlighting**: Ask the assistant *"where do I click?"* or *"where is the exit?"* and watch it highlight the exact physical coordinates on your screen.
- **Cinematic Mode (Contextual Voice Translation)**: Seamlessly couples screen capture vision analysis with audio playback translation.
- **Local CC-CEDICT Dictionary**: Instantly hover over any translation bubble and press `Alt` for a local thread-safe parsing breakdown of Chinese characters, pinyin, and definitions.

---

## 🛠️ Getting Started (MAGE Client)

### Requirements

- **Linux with Wayland** (X11 also supported; global key-bindings require `evdev` inputs)
- **User Permissions**: Your user must be in the `input` group for global hotkey capturing (`sudo usermod -aG input $USER` and log out/in)
- **Lemonade Server**: A running Lemonade Server instance (accessible at `http://localhost:13305` by default)

### Quick Setup (Linux)

Clone the repository and run the bootstrap script — it installs [`uv`](https://docs.astral.sh/uv/), syncs all dependencies, and launches MAGE automatically:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
./mage.sh
```

To add MAGE to your desktop application menu:

```bash
./mage.sh --install
```

To add MAGE to your menu **and** automatically install system dependencies, build the embeddable Lemonade server from source, and pull the default vision-language model:

```bash
./mage.sh --install --build
```

To remove the desktop entry and icon:

```bash
./mage.sh --uninstall
```

### Pre-built Releases (Windows & macOS)

If you are running Windows or macOS, or prefer not to build from source, download pre-built packages from the [GitHub Releases](https://github.com/clemperorpenguin/xian-vl/releases) page.

Releases come in two variants:
- **Lite** (`mage-client-lite.zip`): Standalone lightweight version. Requires connecting to an external running Lemonade Server.
- **Full** (`mage-client-full.zip` / `mage-client-full.dmg`): Bundled with the embedded `lemond` server, which starts and stops automatically when the application runs.

#### Windows
1. Download `mage-client-lite.zip` or `mage-client-full.zip`.
2. Extract the archive.
3. Double-click `mage-client.exe` to run.

#### macOS
1. Download `mage-client-lite.dmg` or `mage-client-full.dmg`.
2. Double-click the DMG and drag `mage-client.app` to your **Applications** directory.
3. Open and run the application.

### Manual Setup (All Platforms)

If you prefer to manage the environment yourself:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
uv sync --all-packages
uv run --package mage-client mage
```

### Controls
- **Open Action Menu / OSD** — Double-Tap `Shift` (Default Leader Key)
- **Trigger Screen Capture** — Action Menu `C` (select screen region, then Translate / Explain / Chat)
- **Toggle Chat Sidebar** — Action Menu `A`
- **Translate for Chat (Input)** — Action Menu `T`
- **Settings Panel** — Action Menu `S`

---

## 📁 Architecture & Spoke Projects

The monorepo contains the core production-ready MAGE client as well as experimental companion scaffolds:

```
├── apps/
│   ├── mage-client/      # 🧙‍♂️ The main verified, PyQt6-based gaming HUD application
│   ├── nate/             # 📱 Android companion OCR reader & dictionary (Experimental)
│   ├── masha-extension/  # 🌐 Browser extension selection translator (Experimental)
│   ├── lore-client/      # 📜 RAG knowledge wiki builder CLI (Experimental)
│   └── luduan-client/    # 🦤 EPUB translation & audiobook CLI (Experimental)
└── packages/
    ├── xian-vl/          # ⚙️ Core LLM/ASR orchestration engine & context managers
    └── shared-types/     # 📦 Canonical models, constants, and shared types
```

> [!WARNING]
> **ASR / Audio Limitation**: Sending live audio stream uploads to Lemonade is currently broken due to server-side backend limitations. Consequently, live speech translation features (such as **Raid Mode**) are disabled. All visual OCR, text translation, dictionary, and chat grounding features are fully functional.

---

## 📜 License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
