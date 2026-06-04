#!/usr/bin/env python3
# Xian-VL Scripts — Development and automation scripts.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import json
import logging
import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

from PyQt6.QtCore import QSettings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def resolve_model(api_url: str, model_name: str) -> str:
    """If model_name is a composite omni model/collection, resolve it to its LLM component."""
    base_url = api_url
    if "/v1" in base_url:
        base_url = base_url.split("/v1")[0]
    models_url = f"{base_url.rstrip('/')}/v1/models?show_all=true"
    
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=5.0) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = data.get("data", [])
            
            model_info = next((m for m in models if m.get("id") == model_name), None)
            if not model_info:
                return model_name
                
            if model_info.get("recipe") == "collection.omni" or model_name.startswith("LMX-Omni-"):
                components = model_info.get("components", [])
                if not components:
                    return model_name
                
                # Try explicit chat/tool-calling labels
                for comp_id in components:
                    comp_info = next((m for m in models if m.get("id") == comp_id), None)
                    labels = comp_info.get("labels", []) if comp_info else []
                    if any(l in labels for l in ["chat", "tool-calling", "reasoning"]):
                        logging.info(f"Resolved Omni model '{model_name}' to chat component '{comp_id}'")
                        return comp_id
                
                # Check keywords in ID
                for comp_id in components:
                    lower_id = comp_id.lower()
                    if any(kw in lower_id for kw in ["qwen", "llama", "gemma", "mistral", "deepseek", "glm"]):
                        logging.info(f"Resolved Omni model '{model_name}' to chat component '{comp_id}' via keyword")
                        return comp_id
                        
                logging.info(f"Resolved Omni model '{model_name}' to default component '{components[0]}'")
                return components[0]
    except Exception as e:
        logging.warning(f"Could not auto-resolve Omni model component ({e}). Using raw model name.")
        
    return model_name

def get_lemonade_config():
    settings = QSettings("Xian", "VideoGameTranslator")
    api_url = settings.value("api_url", "http://localhost:13305/v1")
    model = settings.value("api_model", "LMX-Omni-5.5B-Lite")
    
    resolved_model = resolve_model(api_url, model)
    
    # Ensure it points to the chat completions endpoint
    if not api_url.endswith("/chat/completions"):
        if not api_url.endswith("/"):
            api_url += "/"
        api_url += "chat/completions"
        
    return api_url, resolved_model

LEMONADE_URL, MODEL = get_lemonade_config()

TARGET_LANGUAGES = {
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "es": "Spanish",
    "ar": "Arabic",
    "hi": "Hindi",
    "vi": "Vietnamese"
}

# Number of translation keys to send per LLM request. Small models truncate
# output when asked to translate all ~90 keys at once, so we batch them.
BATCH_SIZE = 15
MAX_RETRIES = 3


def _translate_batch(batch: dict, target_lang_name: str) -> dict:
    """Translate a single batch of en.json entries into the target language."""

    system_prompt = (
        f"You are a professional localization translator. Translate software UI strings into {target_lang_name}.\n"
        "Input format: A JSON object where each key maps to a nested object containing a 'value' (the English string to translate) and a 'context' (description of where it is used).\n"
        "Output format: Return ONLY a flat JSON object mapping the exact same keys directly to their translated string values.\n"
        "Example Input:\n"
        "{\n"
        "  \"settings.dialog.title\": {\n"
        "    \"value\": \"Xian — Settings\",\n"
        "    \"context\": \"Window title\"\n"
        "  }\n"
        "}\n"
        "Example Output:\n"
        "{\n"
        "  \"settings.dialog.title\": \"Xian — 设置\"\n"
        "}\n"
        "Strict Rules:\n"
        "1. Do not include any explanations, markdown code blocks (like ```json), or thinking process in the output.\n"
        "2. Output ONLY the raw JSON object."
    )

    user_content = json.dumps(batch, ensure_ascii=False, indent=2)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
        "frequency_penalty": 0.6,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.15,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    req = urllib.request.Request(
        LEMONADE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req, timeout=120.0) as response:
        result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)


def translate_strings(en_data: dict, target_lang_name: str) -> dict:
    """Translate all en.json entries into the target language using batched requests."""

    keys = list(en_data.keys())
    batches = [keys[i:i + BATCH_SIZE] for i in range(0, len(keys), BATCH_SIZE)]
    merged: dict = {}
    total = len(batches)

    for idx, batch_keys in enumerate(batches, 1):
        batch = {k: en_data[k] for k in batch_keys}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = _translate_batch(batch, target_lang_name)
                merged.update(result)
                logging.info(
                    "  Batch %d/%d OK (%d keys)", idx, total, len(result)
                )
                break
            except urllib.error.URLError as e:
                logging.error(
                    "  Batch %d/%d attempt %d — connection error: %s",
                    idx, total, attempt, e,
                )
            except json.JSONDecodeError as e:
                logging.error(
                    "  Batch %d/%d attempt %d — JSON parse error: %s",
                    idx, total, attempt, e,
                )
            except Exception as e:
                logging.error(
                    "  Batch %d/%d attempt %d — unexpected error: %s",
                    idx, total, attempt, e,
                )
        else:
            logging.warning(
                "  Batch %d/%d FAILED after %d attempts — skipping %d keys",
                idx, total, MAX_RETRIES, len(batch_keys),
            )

    return merged


def main():
    current_dir = Path(__file__).resolve().parent
    locales_dir = current_dir.parent.parent.parent / "shared-types" / "locales"
    en_path = locales_dir / "en.json"

    if not en_path.exists():
        logging.error("Source file not found: %s", en_path)
        sys.exit(1)

    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)

    target_langs = TARGET_LANGUAGES
    if len(sys.argv) > 1:
        lang_arg = sys.argv[1].lower()
        if lang_arg in TARGET_LANGUAGES:
            target_langs = {lang_arg: TARGET_LANGUAGES[lang_arg]}
        else:
            logging.error(
                "Unknown language code: %s. Valid codes are: %s",
                lang_arg, ", ".join(TARGET_LANGUAGES.keys()),
            )
            sys.exit(1)

    for lang_code, lang_name in target_langs.items():
        out_path = locales_dir / f"{lang_code}.json"
        logging.info("Generating translations for %s (%s)...", lang_name, lang_code)

        # Load existing translations so we only fill gaps / update changed keys
        existing: dict = {}
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}

        # Only translate keys that are missing or whose English source changed
        keys_to_translate = {}
        for key, entry in en_data.items():
            if key not in existing:
                keys_to_translate[key] = entry

        if not keys_to_translate:
            logging.info(
                "  All %d keys already translated — skipping %s.",
                len(en_data), lang_name,
            )
            continue

        logging.info(
            "  %d new/changed keys to translate (out of %d total).",
            len(keys_to_translate), len(en_data),
        )

        translated = translate_strings(keys_to_translate, lang_name)
        if translated:
            merged = {**existing, **translated}
            # Preserve the key order from en.json
            ordered = {k: merged[k] for k in en_data if k in merged}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(ordered, f, ensure_ascii=False, indent=2)
                f.write("\n")
            logging.info(
                "  Successfully wrote %s (%d/%d keys).",
                out_path, len(ordered), len(en_data),
            )
        else:
            logging.warning("  Skipped %s due to translation failure.", lang_name)

if __name__ == "__main__":
    main()
