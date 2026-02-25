# Xian - Real-time Video Game Translation Overlay

<img width="768" height="768" alt="xian" src="https://github.com/user-attachments/assets/7b9498fd-4786-481f-b2c9-e29632b2ec24" />

Xian is a PyQt6-based translation overlay designed for Linux Wayland (specifically KDE Plasma). It uses Qwen3-VL Thinking models for unified OCR and translation to provide real-time translations of video games or websites directly on your screen.

## Screenshots

![screenshot1](https://github.com/user-attachments/assets/065e1da7-6cef-4d26-999d-96516769614c)
![Screenshot2](https://github.com/user-attachments/assets/3f9c0c0b-e773-4dac-a6aa-6a7e80cb429c)

## Features

- **Real-time Translation**: Automatically captures and translates text from your screen.
- **Two Translation Modes**:
    - **Full Screen**: Analyzes the entire screen and places translated text boxes over the original content.
    - **Region Selection**: Allows you to define specific areas of the screen for targeted translation.
- **Click-Through Overlay**: The translation overlay is transparent to mouse input, allowing you to play games or browse websites uninterrupted.
- **Unified Vision-Language Pipeline**:
    - Supports both Qwen3-VL and TranslateGemma models with integrated OCR and translation in a single inference pass.
    - Qwen3-VL supports 32 languages with robustness to blur, tilt, and low light conditions.
    - TranslateGemma offers high-quality translation with optimized performance.
    - No separate OCR or translation models needed.
- **Smart Model Selection**:
    - Automatically detects available VRAM and selects appropriate model (Qwen3-VL or TranslateGemma).
    - Option to enable "Thinking" mode for complex layouts.
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
    - Qwen3-VL-8B for high-end GPUs (24GB+ VRAM)
    - Qwen3-VL-4B for mid-range GPUs (12-24GB VRAM)
    - TranslateGemma-12B for high-quality translation (20GB+ VRAM)
    - TranslateGemma-4B as failover for lower-resource systems (10GB+ VRAM)
- **Persistent Settings**:
    - Saves your configuration, including model selection, languages, and custom regions.
- **Customizable Appearance**: Adjustable overlay opacity.

## Requirements

- **Linux with Wayland** (Tested on KDE Plasma).
- **Python 3.10+**
- **GPU with 12GB+ VRAM** (minimum for 4B model, 24GB+ for 8B model)
  - NVIDIA GPU with CUDA support, or
  - AMD GPU with ROCm support
- **PyQt6**
- **vLLM** (for Qwen3-VL inference)
- **PyTorch**

## Installation

### For NVIDIA GPUs (CUDA):

1. Clone the repository:
   ```bash
   git clone https://github.com/samexner/xian-vl.git
   cd xian
   ```

2. Install dependencies:
   ```bash
   pip install PyQt6 torch vllm>=0.11 pynvml Pillow
   ```

### For AMD GPUs (ROCm):

1. Clone the repository:
   ```bash
   git clone https://github.com/samexner/xian-vl.git
   cd xian
   ```

2. Install ROCm-compatible PyTorch:
   ```bash
   # Follow PyTorch ROCm installation guide for your system
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
   ```

3. Install remaining dependencies:
   ```bash
   pip install PyQt6 vllm>=0.11 pynvml Pillow
   ```

## VRAM Requirements

| GPU               | Recommended Model | Expected Latency |
|-------------------|-------------------|------------------|
| ≥24GB (4090/3090) | Qwen3-VL-8B       | ~600ms           |
| 20-24GB           | TranslateGemma-12B| ~700ms           |
| 12-20GB (3060+)   | Qwen3-VL-4B       | ~900ms           |
| 10-12GB           | TranslateGemma-4B | ~1200ms          |
| <10GB             | Not supported*    | -                |

*Models require at least 10GB VRAM for the smallest models. CPU inference is possible but extremely slow.

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. **Configure Translator**:
   - Go to the **Settings** tab.
   - Select the desired **Model** (Qwen3-VL-4B or Qwen3-VL-8B with or without Thinking mode).
   - The first time you start translation, the model will be downloaded automatically.
   - Adjust **Max Tokens** and **Thinking Mode** settings as needed.

3. **Select Mode**:
   - In the **General** tab, choose between **Full Screen Analysis** or **Region Selection**.

4. **Define Regions (Optional)**:
   - Go to the **Regions** tab to add specific areas for translation.

5. **Start Translating**:
   - Click the **Start Translation** button.
   - The overlay will appear, and translations will be updated periodically.

## Troubleshooting

- **CUDA Version Issues**: Ensure your PyTorch and vLLM versions are compatible with your CUDA installation.
- **VRAM Detection**: If VRAM detection fails, manually select the appropriate model size in settings.
- **Slow Performance**: Reduce Max Tokens or disable Thinking Mode for faster inference.
- **ROCm Compatibility**: For AMD GPUs, ensure you have the correct ROCm version installed that matches your PyTorch version.

## License

GNU General Public License v3.0 - See LICENSE file for details.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
