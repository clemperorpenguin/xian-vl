# Xian - Screen OCR & Translation Tool

<img width="768" height="768" alt="xian" src="https://github.com/user-attachments/assets/7b9498fd-4786-481f-b2c9-e29632b2ec24" />

Xian is a PyQt6-based screen OCR and translation tool for Linux (Wayland/KDE Plasma). It uses Qwen3.5 models for unified OCR and translation to provide real-time text extraction and translation from any part of your screen.

## Use Cases

- **Translate websites** in your browser
- **Localize games** without built-in translation
- **Read manga/comics** with instant translation
- **Extract text** from videos, PDFs, or images
- **Translate applications** and UI elements
- **OCR any screen content** for accessibility or productivity

## Screenshots

![screenshot1](https://github.com/user-attachments/assets/065e1da7-6cef-4d26-999d-96516769614c)
![Screenshot2](https://github.com/user-attachments/assets/3f9c0c0b-e773-4dac-a6aa-6a7e80cb429c)

## Features

- **Real-time OCR & Translation**: Automatically captures and translates text from your screen.
- **Multiple Modes**:
    - **Full Screen**: Analyzes the entire screen and places translated text boxes over the original content.
    - **Region Selection**: Define specific areas of the screen for targeted OCR/translation.
    - **OCR Only**: Extract text without translation (copy to clipboard or save to file).
- **Click-Through Overlay**: The translation overlay is transparent to mouse input, allowing you to work uninterrupted.
- **Unified Vision-Language Pipeline**:
    - Supports both Qwen3.5 and TranslateGemma models with integrated OCR and translation in a single inference pass.
    - Qwen3.5 supports 32 languages with robustness to blur, tilt, and low light conditions.
    - TranslateGemma offers high-quality translation with optimized performance.
    - No separate OCR or translation models needed.
- **Smart Model Selection**:
    - Automatically detects available VRAM and selects appropriate model (Qwen3.5 or TranslateGemma).
    - CPU support for systems without dedicated GPU.
    - Toggleable thinking mode for complex layouts (Qwen3.5 only).
- **Advanced Caching System**:
    - L0 Cache: dHash perceptual caching to avoid re-processing identical frames
    - L1 Cache: Persistent LMDB storage for cross-session translation reuse
    - Significant performance improvement for static or slowly-changing content
- **Context-Aware Visual Reconstruction**:
    - Automatic text style detection (font, size, color, orientation)
    - Background inpainting and reconstruction
    - Style-matched text rendering for seamless integration
    - Preserves original text appearance while translating content
- **Flexible Hardware Support**:
    - GPU: Qwen3.5-9B for high-end GPUs (20GB+ VRAM), Qwen3.5-4B for mid-range (12GB+ VRAM)
    - GPU: TranslateGemma-12B for high-quality translation (20GB+ VRAM), 4B for lower-resource (10GB+ VRAM)
    - CPU: Qwen3.5-4B or smaller models (slower but functional)
- **Output Options**: Overlay display, clipboard copy, or file export.
- **Persistent Settings**: Saves your configuration, including model selection, languages, and custom regions.
- **Customizable Appearance**: Adjustable overlay opacity.

## Requirements

- **Linux with Wayland** (Tested on KDE Plasma).
- **Python 3.10+**
- **Hardware** (choose one):
  - **GPU**: NVIDIA GPU with CUDA support, or AMD GPU with ROCm support
  - **CPU**: Any modern CPU (slower inference, suitable for occasional use)
- **uv** package manager

## Installation

Xian uses `uv` for fast, reliable dependency management. Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### For NVIDIA/AMD GPUs (Recommended):

1. Clone the repository:
   ```bash
   git clone https://github.com/samexner/xian-vl.git
   cd xian-vl
   ```

2. Create virtual environment and install with CUDA support:
   ```bash
   uv venv
   uv pip install -e ".[cuda]"
   ```

   **For AMD GPUs (ROCm)**:
   ```bash
   uv venv
   # Install ROCm PyTorch first
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
   # Then install remaining dependencies
   uv pip install -e ".[rocm]"
   ```

### For CPU-only systems:

1. Clone the repository:
   ```bash
   git clone https://github.com/samexner/xian-vl.git
   cd xian-vl
   ```

2. Create virtual environment and install:
   ```bash
   uv venv --python 3.12  # Use 3.12, not 3.14+ (llvmlite doesn't support 3.14 yet)
   uv pip install -e ".[cpu]"
   ```

   **Important**: 
   - Python 3.14 is not yet supported for CPU mode due to `llvmlite` dependency. Use Python 3.10-3.12.
   - CPU inference uses the `transformers` library directly with 4-bit quantization (not vLLM).
   - Expect 5-15s latency per frame, with ~2-3GB RAM usage for 4B model.
   - Model loading may take 2-5 minutes on first run (download + quantization).
   - Suitable for single-frame capture or region selection, not real-time video.
   - CPU is limited to 2 threads to prevent system hardlock.

### Manual installation (pip):

If you prefer pip, legacy requirements files are still available:
- `requirements.txt` for CUDA
- `requirements-rocm.txt` for ROCm

## Hardware Requirements

| Hardware          | Recommended Model | Expected Latency |
|-------------------|-------------------|------------------|
| GPU ≥20GB (4090/3090) | Qwen3.5-9B        | ~500ms           |
| GPU 12-20GB (3060+)   | Qwen3.5-4B        | ~800ms           |
| GPU 6-12GB            | Qwen3.5-2B        | ~1200ms          |
| GPU 20-24GB           | TranslateGemma-12B| ~700ms           |
| GPU 10-12GB           | TranslateGemma-4B | ~1200ms          |
| CPU (modern)      | Qwen3.5-2B        | ~15-45s          |
| CPU (older)       | Qwen3.5-2B        | ~30-60s          |

**Note**: CPU inference is significantly slower than GPU. It's suitable for occasional use or when GPU is unavailable. For best performance, use a GPU with at least 6GB VRAM.

## Usage

1. Start the application:
   ```bash
   uv run python main.py
   ```
   Or if installed with pip:
   ```bash
   python main.py
   ```

2. **Configure Translator**:
   - Go to the **Settings** tab.
   - Select the desired **Model** (Qwen3.5-9B, Qwen3.5-4B, TranslateGemma, or CPU mode).
   - The first time you start translation, the model will be downloaded automatically.
   - Adjust **Max Tokens** and **Thinking Mode** settings as needed.

3. **Select Mode**:
   - In the **General** tab, choose between:
     - **Full Screen Analysis**: Translate everything on screen
     - **Region Selection**: Define specific areas for targeted translation
     - **OCR Only**: Extract text without translation

4. **Define Regions (Optional)**:
   - Go to the **Regions** tab to add specific areas for translation.
   - Regions can be named, saved, and reused across sessions.

5. **Start Translating**:
   - Click the **Start Translation** button.
   - The overlay will appear, and translations will be updated periodically.

## Advanced Features

### OCR-Only Mode
Extract text without translation. Useful for:
- Copying text from images or videos
- Accessibility (screen reading)
- Documentation and archiving

### Region Presets
Save named regions for quick access:
- "Game UI" - HUD elements
- "Chat Box" - Dialogue areas
- "Menu" - Navigation elements

### Text Export
Export extracted text to:
- Clipboard (instant copy)
- Text file (with timestamps)
- CSV format (for analysis)

## Troubleshooting

- **CUDA Version Issues**: Ensure your PyTorch and vLLM versions are compatible with your CUDA installation.
- **VRAM Detection**: If VRAM detection fails, manually select the appropriate model size in settings.
- **Slow Performance**: Reduce Max Tokens, disable Thinking Mode, or use a smaller model.
- **ROCm Compatibility**: For AMD GPUs, ensure you have the correct ROCm version installed that matches your PyTorch version.
- **CPU Performance**: CPU inference is slow. Consider using a smaller model or reducing update interval.
- **Region Selection**: If regions don't select properly, ensure you're clicking and dragging with sufficient movement (minimum 10px).

## Migration from pip to uv

If you previously installed Xian with pip, you can migrate to uv:

```bash
# Remove old virtual environment (optional)
rm -rf .venv

# Create new environment with uv
uv venv
uv pip install -e ".[cuda]"  # or [rocm] or [cpu]
```

Your settings and cached translations are preserved in `~/.config/Xian/`.

## License

GNU General Public License v3.0 - See LICENSE file for details.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
