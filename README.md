# Xian-MAGE — Real-Time Vision-Language Assistant for Gaming Environments 🧙‍♂️🎮🤖

[العربية](docs/README.ar.md) | [Español](docs/README.es.md) | [हिन्दी](docs/README.hi.md) | [Русский](docs/README.ru.md) | [Tiếng Việt](docs/README.vi.md) | [简体中文](docs/README.zh.md)

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

- **Linux**: Wayland or X11. To capture global hotkeys, your user must be in the `input` group (`sudo usermod -aG input $USER` followed by a log out/in).
- **macOS**: Accessibility API permissions (granted when prompted on launch for input capturing).
- **Lemonade Server**: A running Lemonade Server instance (accessible at `http://localhost:13305` by default), unless using a bundled/embedded setup.

### Quick Setup (Linux & macOS)

Clone the repository and run the bootstrap script — it automatically installs [`uv`](https://docs.astral.sh/uv/), syncs all dependencies, and launches MAGE:

```bash
git clone https://github.com/clemperorpenguin/xian-vl.git
cd xian-vl
./mage.sh
```

#### Installing / Registering the Application
You can register MAGE in your system's application menu (Linux) or Applications directory (macOS):

- **Default Installation**:
  ```bash
  ./mage.sh --install
  ```
  - **Linux**: Creates a `.desktop` shortcut launcher.
  - **macOS**: Creates a launcher at `/Applications/MAGE.app`, automatically downloads the pre-built embeddable Lemonade server binary, and pre-pulls the default vision-language model.

- **Build Lemonade from Source**:
  If you want to build the embeddable Lemonade server from scratch (useful for custom optimizations or specific platforms):
  ```bash
  ./mage.sh --install --build
  ```

- **Uninstallation**:
  To clean up and remove the registered shortcut and files:
  ```bash
  ./mage.sh --uninstall
  ```

### Pre-built Releases (Windows)

If you are running Windows, or prefer not to run via the command line, download pre-built packages from the [GitHub Releases](https://github.com/clemperorpenguin/xian-vl/releases) page.

Releases come in two variants:
- **Lite**: Standalone lightweight version. Requires connecting to an external running Lemonade Server.
- **Full**: Bundling the embedded `lemond` server, which starts and stops automatically when MAGE runs.

#### Windows Setup
1. Download `mage-client-Windows-x86-64-lite.zip` or `mage-client-Windows-x86-64-full.zip`.
2. Extract the archive.
3. Double-click `mage-client.exe` to run.

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
- **Show / Hide All Overlays** — Double-Tap `Right Shift` (configurable in Settings; also available from the tray menu)
- **Trigger Screen Capture** — Action Menu `C` (select screen region, then Translate / Dialogue / Chat)
- **Toggle Chat Sidebar** — Action Menu `A`
- **Translate for Chat (Input)** — Action Menu `T`
- **Settings Panel** — Action Menu `S`

> **Always-on-top behaviour & Wayland note.** MAGE keeps its overlays above the
> active game and brings each new translation to the front automatically. On
> **X11** this is enforced via EWMH `_NET_WM_STATE_ABOVE` and works reliably
> over fullscreen games. On **Wayland**, client-side window restacking is not
> permitted by the compositor, so "always on top" depends on the compositor
> honouring the stays-on-top hint — fronting may be less reliable there. The
> global hotkeys and the **Show / Hide All Overlays** toggle still work on
> Wayland; use that toggle (or the tray menu entry) if an overlay is ever
> obscured.

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

---

## 📜 License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
