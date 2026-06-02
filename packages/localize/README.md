# XUAN — Xian's Universal Automated Native-linguist 🌐

**XUAN** is the specialized, AI-powered localization and translation tool for the Xian Translation Ecosystem. It parses English source translation keys (with contextual descriptions) and orchestrates automated translation generation using the local Lemonade endpoint.

---

## 🚀 Key Features

* **Qt Integration**: Directly reads connection settings (`api_url`, `api_model`) from MAGE's desktop configurations (`~/.config/Xian/VideoGameTranslator.conf`).
* **Context-Aware Translation**: Sends the string values along with their context metadata to the Lemonade LLM for high-accuracy translation.
* **Target Locales**: Generates flat JSON dictionaries for all default project locales:
  * 🇨🇳 Chinese (`zh`)
  * 🇯🇵 Japanese (`ja`)
  * 🇰🇷 Korean (`ko`)
  * 🇷🇺 Russian (`ru`)
  * 🇪🇸 Spanish (`es`)
  * 🇸🇦 Arabic (`ar`)
  * 🇮🇳 Hindi (`hi`)
  * 🇻🇳 Vietnamese (`vi`)
* **Flexible Execution**: Translates all languages by default or focuses on a single target locale via CLI argument.

---

## 🛠️ Usage Guide

XUAN is built as a synced workspace package. You can invoke it using `uv` from the workspace root directory.

### 1. Translate All Default Locales
To translate all default target languages:
```bash
uv run --package xuan xuan
```
*(Or simply `uv run xuan` if your workspace environment is synced).*

### 2. Translate a Specific Locale
To translate only a specific language (e.g. Chinese `zh` or Spanish `es`):
```bash
uv run --package xuan xuan zh
```

### 3. Alternative Command syntax
You can also run it using the `run` alias:
```bash
uv run --package xuan run [language_code]
```

---

## ⚙️ Configuration

XUAN queries MAGE's desktop settings dynamically using `QSettings`. It uses the following options:

| Setting Key | Default Value | Description |
|---|---|---|
| `api_url` | `http://localhost:13305/v1` | Base URL for the Lemonade endpoint (will append `/chat/completions` if missing) |
| `api_model` | `LMX-Omni-5.5B-Lite` | Model used to generate translation completions |

### Source Strings File
The script reads from the canonical English source file:
`packages/shared-types/locales/en.json`

It will write output JSONs into the same directory:
`packages/shared-types/locales/{lang_code}.json`
