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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LEMONADE_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "lite"  # "lite omnirouter model"

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

def translate_strings(en_data: dict, target_lang_name: str) -> dict:
    """Uses Lemonade to translate the UI strings into the target language."""
    
    system_prompt = (
        f"You are a professional localization engineer translating a software interface into {target_lang_name}. "
        "The user will provide a JSON object containing keys, English values, and contextual descriptions. "
        "Respond ONLY with a valid, flat JSON object where the keys are exactly the same, and the values are the translations of the English values into the target language. "
        "Do not include any other text, markdown formatting like ```json, or explanations."
    )
    
    user_content = json.dumps(en_data, ensure_ascii=False, indent=2)
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    req = urllib.request.Request(
        LEMONADE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            translated_dict = json.loads(content)
            return translated_dict
    except urllib.error.URLError as e:
        logging.error(f"Failed to connect to Lemonade at {LEMONADE_URL}: {e}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Lemonade response as JSON: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error calling Lemonade: {e}")
        return {}

def main():
    root_dir = Path(__file__).resolve().parent.parent
    locales_dir = root_dir / "packages" / "shared-types" / "locales"
    en_path = locales_dir / "en.json"
    
    if not en_path.exists():
        logging.error(f"Source file not found: {en_path}")
        sys.exit(1)
        
    with open(en_path, "r", encoding="utf-8") as f:
        en_data = json.load(f)
        
    for lang_code, lang_name in TARGET_LANGUAGES.items():
        out_path = locales_dir / f"{lang_code}.json"
        logging.info(f"Generating translations for {lang_name} ({lang_code})...")
        
        translated = translate_strings(en_data, lang_name)
        if translated:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully wrote {out_path}")
        else:
            logging.warning(f"Skipped {lang_name} due to translation failure.")

if __name__ == "__main__":
    main()
