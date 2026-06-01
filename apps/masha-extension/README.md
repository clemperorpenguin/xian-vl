# MASHA - Multilingual Access & Site Handling Assistant 🌐

> [!CAUTION]
> **Experimental / Untested**: MASHA is currently an experimental browser extension prototype and is **not fully tested or verified**. It is provided as-is as a work-in-progress scaffold.

MASHA is a lightweight browser extension designed to provide real-time text translation on web pages. It integrates with the Lemonade Server to deliver powerful translation capabilities directly within your browser.

## Features

- **Text Selection Translation**: Select any text on a webpage and request a translation.
- **Lemonade Integration**: Powered by the Lemonade Server for high-quality, model-driven translations.
- **Overlay Display**: Displays translations in a non-intrusive overlay on top of the page.

## Installation

To install the extension in developer mode (Chrome/Edge):

1. Open your browser and navigate to `chrome://extensions`.
2. Enable "Developer mode" in the top right.
3. Click "Load unpacked" and select the `apps/masha-extension` directory.

## Configuration

MASHA communicates with the Lemonade Server. Ensure the server is running and accessible at the configured URL (default is `http://localhost:13305/v1`).
