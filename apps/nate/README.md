# NATE (Neural Analysis & Translation Engine)

> [!CAUTION]
> **Experimental / Untested**: NATE is currently an experimental prototype companion and is **not fully tested or verified**. It is provided as-is as a work-in-progress scaffold.

NATE is a standalone mobile companion to the `xian-vl` ecosystem. It functions as an offline-first Chinese OCR reader, dictionary, and translation tool for camera feeds and documents, with optional remote LLM acceleration via the Lemonade Server.

## Features

- **Modular OCR Engine**: Switch between local lightweight PaddleOCR (via Chaquopy), Google MLKit, or high-performance remote OCR via Lemonade (Qwen3.5-9B).
- **Offline Dictionary**: Integrated CC-CEDICT support for instant character and compound lookup without an internet connection.
- **Live Camera Translation**: Real-time bounding box overlays and translation using CameraX and Jetpack Compose.
- **Contextual ELI5**: Analysis of slang, regional dialects, and complex grammar (Local or Remote).
- **SRS Flashcards**: Save vocabulary with original OCR image crops and sentence context for spaced repetition study.

## Tech Stack

- **Frontend**: Kotlin / Jetpack Compose
- **Logic Bridge**: [Chaquopy](https://chaquo.com/chaquopy/) (Python on Android)
- **OCR**: PaddleOCR (Mobile), Google MLKit
- **Database**: Room (SQLite)
- **Networking**: Retrofit

## Getting Started

### Prerequisites

1. **Android Studio**: Arctic Fox or newer.
2. **Python**: The project uses Chaquopy to manage Python dependencies. Ensure your environment allows for Python wheel compilation if necessary.

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/clemperorpenguin/xian-vl.git
   cd xian-vl/apps/nate
   ```

2. **Database Setup**:
   Place your pre-generated `cedict.db` SQLite file in:
   `app/src/main/assets/cedict.db`

3. **Build**:
   Open the `apps/nate` project in Android Studio and sync Gradle. The first sync will take some time as it downloads Python packages like `paddleocr` and `paddlepaddle`.

4. **Run**:
   Deploy to an Android device (Physical device recommended for CameraX testing).

## Architecture

- **`src/main/java`**: Kotlin-native UI, CameraX lifecycle management, and Room database configuration.
- **`src/main/python`**: Core AI logic including PaddleOCR inference and translation model management.
- **`api/`**: Retrofit interfaces for the Lemonade Omni-Router.

## License

GNU General Public License v3.0
