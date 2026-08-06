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
* [x] **Static Translations** Static translations on mouseover, eg tooltips for toolbar buttons - implemented in dev branch.
* [x] **Windows Support** Lightly tested.
* [x] **Mac Support** Lightly tested.
* [x] **Embedded Lemonade** Garbled text bug resolved. Lemonade installs/runs as a separate service.
* [ ] **Bilibili Support**: Support overlay capture and live-translation flows for Bilibili video streams and interface windows, making it easier for users to translate on-screen dialogue and commentary in real-time.
* [ ] **Perfect Window Focus** Perfect window focus means the overlay is either up or down. I need to unify this behavior across platforms, there is some weirdness on wayland.

### 2. Expansion of UI Target Locales
* **Approach**: Adding additional languages is straightforward. New strings can be appended to the reference [en.json](../packages/shared-types/locales/en.json) with description contexts, followed by running:
  ```bash
  uv run --package xuan xuan
  ```
  This automatically translates new strings into targeted locale files (`zh.json`, `ja.json`, etc.) using the Lemonade model.

### 3. Familiar Mode — Generative Companions
* [x] **Switchable familiars**: Five built-in desktop companions (wizard, witch, cat, owl, lemonfae), each with its own transit animation to the top of the screen (teleport-and-float, broom/owl flight, cat edge-climb) and drop-in PNG art override (`familiar/<species>/<state>_<n>.png`).
* [x] **"Conjure…" — Lemonade-authored familiars**: A 6th familiar slot where the user describes a creature in natural language and a local **Lemonade LLM** returns a structured JSON "art recipe" (palette, body/feature primitives, accessory, transit style) that drives a parametric vector renderer. Fully local, single-model, no diffusion dependency and no VRAM contention with the translation model — and it animates live (walk bob, cast glow) like the built-ins.
* [ ] **Follow-up — High-fidelity diffusion sprites** *(stretch)*: An optional higher-quality path where Lemonade expands the prompt into a consistent character sheet and a local **diffusion backend** (SDXL-Turbo via ONNX/DirectML, or stable-diffusion.cpp/Vulkan on Strix Halo) renders the frames, followed by background removal and frame-to-frame consistency (img2img from a base render / sprite-sheet slicing). Gated behind a flag; reliability, latency, and VRAM trade-offs to be weighed against the vector path before promoting it.

---

## 🌀 Medium-Term Goals (3-6 Months)

### 1. Nate (Android Companion)
* **Goal**: Focus the scope of the mobile companion specifically on serving as a **highly efficient, high-performance dictionary tool**.
* **Features**: Mobile OCR scanner, rapid CC-CEDICT parsing, and local/clipboard history sync with the main MAGE client.

### 2. Luduan (Book Translation & Narration)
* **Goal**: Grow Luduan from an EPUB tool into a general **book/comic translate-and-narrate** pipeline.
* **Status**: EPUB translation + Opus narration verified working (pip-only Opus via `soundfile`; no `ffmpeg` required).
* **Input-format expansion** (all converge on the same `ParsedBook` → translate → narrate path):
  * [x] **EPUB** — text extraction via `ebooklib`.
  * [ ] **PDF (text)**: direct text extraction (`pypdf`/`pdfminer`).
  * [ ] **PDF (scanned)**: page images through the `xian` **vision OCR pipeline** (the MAGE OCR+translate engine).
  * [ ] **Comics (`.cbz`, `.cbr`, `.cb7`)**: unpack the image archive, OCR + translate each page/panel via the vision pipeline, then narrate the translated dialogue.
* **Note**: comics and scanned PDFs are a vision-OCR effort (reuse of the MAGE VLM path), not plain text extraction — staged behind the text formats.

### 3. Masha (Browser Extension)
* **Goal**: Transition MASHA into a **high-quality, context-aware full-page or partial-page translator**.
* **Features**:
  * Block selection translation.
  * Full-page DOM translation leveraging the LLM's context window.
  * In-context rendering of translated paragraphs, preserving webpage layout while utilizing local Lemonade inference models.
  * **Document export / print**: export the translated page as a self-contained document (PDF / printable HTML) via the browser's print pipeline (`window.print()` → "Save as PDF"), so a fully translated article can be saved or printed offline. Stretch: a clean reader-mode export that strips page chrome and keeps only the translated body + title.

---

## 🌌 Long-Term Vision (6+ Months)

### 1. Contextual Game State and Memory — ✅ shipped
Everything MAGE translates is now logged to a local SQLite database (`session/sessions.db`
under the app data directory) and fed back into both the OCR and chat prompts, so the
assistant can answer *"What should I do next?"* from the actual playthrough.

* A rolling digest — scene, active objectives, named entities, decisions, open threads — is
  re-summarized by the text LLM as events accumulate, so an hour-old session still fits in
  a prompt.
* Retrieval is FTS5 keyword search with a recency tilt. CJK is segmented per character on
  the way into the index; without that, `unicode61` swallows a whole clause as one token
  and Chinese search silently returns nothing.
* Remaining: semantic retrieval via embeddings (a natural fit for FastFlowLM's `--embed`
  once the NPU path is validated), and importing historical raid transcripts.

### 2. Direct On-Device NPU Acceleration — ⚠️ partially shipped
The honest state of Ryzen AI on Linux, as of Lemonade v10:

* **What works today.** Lemonade serves NPU models through the FastFlowLM (`flm`) recipe,
  which covers text LLMs, Whisper ASR, and embeddings. Chat, translation, session
  summarization, and raid-mode speech can all be routed there via
  **Settings → Backend → Inference backend**. Run `./mage.sh --doctor-npu` to check the
  prerequisites.
* **What does not exist.** There is no NPU *vision* backend on Linux, so the VLM — and
  therefore OCR — stays on the GPU. Routing text work to the NPU frees the GPU; it does
  not make OCR faster. ONNX Runtime's VitisAI execution provider currently falls back to
  CPU on Linux x86-64, so a bespoke NPU OCR path is a spike, not a plan.
* **Hard requirements.** XDNA 2 only (Ryzen AI 300/400 series — XDNA 1 parts are not
  supported), the `amdxdna` driver, and NPU firmware ≥ 1.1.0.0.
* **On "direct memory sharing".** The client talks to `lemond` over HTTP with base64 data
  URLs, which can never be zero-copy. What is achievable — and what the capture path now
  does — is removing the redundant encode/decode round trips between screen capture and
  inference. True dmabuf→NPU buffer import would require in-process inference and is out
  of scope at the frame rates this runs at.

### 3. Live In-Place Translation — ✅ shipped (VLM path)
The **Live** lens action continuously translates a locked region and paints each
translation over the original text — Google Lens / DeepL style — instead of showing a
bubble beside it.

* A perceptual-hash change gate skips inference on unchanged frames, and the worker
  recognizes its *own* overlay in the next capture, so the display does not flicker at
  steady state the way dialogue mode does.
* Latency is bounded by the vision model: expect roughly 2–5 s per changed frame with a 4B
  VLM.
* Remaining: a local ONNX text-detection pass to supply boxes (sub-second end to end, with
  the LLM only translating the recognized lines), which would make this genuinely
  real-time.
