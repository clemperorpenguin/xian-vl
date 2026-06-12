# Xian-MAGE Development Roadmap

This document outlines the planned milestones, upcoming features, and long-term vision for the Xian-MAGE ecosystem.

---

## 🗺️ Current Roadmap Status

- [x] **Core HUD & Visual Translation**: PyQt6 overlay, Wayland hotkeys, Grim/Spectacle screenshot integration, local dictionary lookup, translation OSD menus, and settings panel.
- [x] **Multimodal Omni Integration**: Local Omni routing (`OmniModelRouter`), component auto-discovery via Lemonade, and basic TTS playback.
- [x] **Unified Audio & Speech (Raid Mode & Live Streaming)**: Client-side Raid Mode Window overlay, custom ToggleSwitch, StatusDot LED, and optimized background TTS pipeline complete. 

---

## ⚡ Short-Term Goals (Next 1-3 Months)

### 1. High-Priority MAGE Desktop Features
* [x] **Raid Mode UI Integration**: Draggable overlay window (`RaidWindow`), custom slide switches, status LEDs, and real-time speech logs integrated into the MAGE HUD.
=======
* [x] **Static Translations** Static translations on mouseover, eg tooltips for toolbar buttons - implemented in dev branch.
* [x] **Windows Support** Lightly tested.
* [x] **Mac Support** Lightly tested.
* [ ] **Embedded Lemonade** Currently produces garbled text, more testing is necessary. Install Lemonade as a separate service.
* [ ] **Bilibili Support**: Support overlay capture and live-translation flows for Bilibili video streams and interface windows, making it easier for users to translate on-screen dialogue and commentary in real-time.
* [ ] **Perfect Window Focus** Perfect window focus means the overlay is either up or down. I need to unify this behavior across platforms, there is some weirdness on wayland.

### 2. Expansion of UI Target Locales
* **Approach**: Adding additional languages is straightforward. New strings can be appended to the reference [en.json](file:///packages/shared-types/locales/en.json) with description contexts, followed by running:
  ```bash
  uv run -m localize.cli
  ```
  This automatically translates new strings into targeted locale files (`zh.json`, `ja.json`, etc.) using the Lemonade model.

---

## 🌀 Medium-Term Goals (3-6 Months)

### 1. Nate (Android Companion)
* **Goal**: Focus the scope of the mobile companion specifically on serving as a **highly efficient, high-performance dictionary tool**.
* **Features**: Mobile OCR scanner, rapid CC-CEDICT parsing, and local/clipboard history sync with the main MAGE client.

### 2. Masha (Browser Extension)
* **Goal**: Transition MASHA into a **high-quality, context-aware full-page or partial-page translator**.
* **Features**:
  * Block selection translation.
  * Full-page DOM translation leveraging the LLM's context window.
  * In-context rendering of translated paragraphs, preserving webpage layout while utilizing local Lemonade inference models.

---

## 🌌 Long-Term Vision (6+ Months)

### 1. Contextual Game State and Memory
* Keep a sliding window of historical screen actions to build contextual game-state understanding.
* Enable the VLM to remember past conversation logs, quest journals, and in-game decisions, providing context-aware advice when the user asks: *"What should I do next?"*.

### 2. Direct On-Device NPU Acceleration
* Fully leverage Ryzen AI NPUs (NPU driver hooks) for all OCR/VLM operations to minimize GPU usage, preserving power and hardware resources for heavy gaming frames.
* Direct memory sharing between screen capture frames and NPU hardware buffers.
