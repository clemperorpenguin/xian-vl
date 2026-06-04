# Xian-MAGE Development Roadmap

This document outlines the planned milestones, upcoming features, and long-term vision for the Xian-MAGE ecosystem.

---

## 🗺️ Current Roadmap Status

- [x]  - Core HUD & Visual Translation**: PyQt6 overlay, Wayland hotkeys, Grim/Spectacle screenshot integration, local dictionary lookup, translation OSD menus, and settings panel.
- [/]  - Multimodal Omni Integration**: Local Omni routing (`OmniModelRouter`), component auto-discovery via Lemonade, and basic TTS playback.
- [ ]  - Unified Audio & Speech (Raid Mode & Live Streaming)**: Fully functional local ASR, voice loop translation, and low-latency voice command activation.

---

## ⚡ Short-Term Goals (Next 1-3 Months)

### 1. Raid Mode & ASR Stream Pipeline Hack
* **Status**: Live speech translation (Raid Mode) is currently disabled due to server-side audio uploading/streaming limitations.
* **Goal**: Restore the `RaidWorker` loop.
* **Approach**: Perform hands-on experiments/hacking with chunked audio binaries and connection interfaces to bypass the current Lemonade server-side constraints. Implement noise-cancellation and local voice activity detection (VAD) to improve input quality before sending audio payloads.

### 2. High-Priority MAGE Desktop Features
* **Bilibili Support**: Support overlay capture and live-translation flows for Bilibili video streams and interface windows, making it easier for users to translate on-screen dialogue and commentary in real-time.
* **Raid Mode UI Integration**: Connect the voice capture backend back into the status bar, displaying real-time speech logs directly in the OSD or a dedicated chat bubble overlay.

### 3. Expansion of UI Target Locales
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
