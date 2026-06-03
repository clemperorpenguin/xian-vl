import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class RuntimeState:
    """Shared runtime state holding the JSON-backed translation manager."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RuntimeState, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        self.ui_language: str = "en"
        self._locale_data: Dict[str, Any] = {}
        # Locate the locales directory — path differs between dev and frozen builds
        if getattr(sys, 'frozen', False):
            # PyInstaller bundles locales into _MEIPASS/locales via --add-data
            base = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
            self.locales_dir = base / "locales"
        else:
            # Dev mode: __file__ is .../packages/shared-types/src/shared_types/state.py
            current_dir = Path(__file__).resolve().parent
            self.locales_dir = current_dir.parent.parent / "locales"
        self.load_locale(self.ui_language)

    def load_locale(self, lang: str):
        """Load JSON strings for a given language."""
        self.ui_language = lang
        locale_path = self.locales_dir / f"{lang}.json"
        
        if not locale_path.exists():
            logger.warning("Locale file not found: %s, falling back to English strings if en.json isn't present.", locale_path)
            self._locale_data = {}
            # If en.json also missing, that's fine, t() will return the key or we could try to load en.json as fallback
            return
            
        try:
            with open(locale_path, "r", encoding="utf-8") as f:
                self._locale_data = json.load(f)
            logger.info("Loaded locale data for %s", lang)
        except Exception as e:
            logger.error("Failed to load locale %s: %s", lang, e)
            self._locale_data = {}

    def t(self, key: str) -> str:
        """Translate a given key based on loaded locale data."""
        if not self._locale_data:
            return key
            
        val = self._locale_data.get(key)
        
        # For English, it might have the structure {"value": "...", "context": "..."}
        if isinstance(val, dict) and "value" in val:
            return val["value"]
            
        # For generated languages (flat key-value), it will be a string
        if isinstance(val, str):
            return val
            
        # Fallback to key itself
        return key

# Global instance for easy access
state = RuntimeState()

def t(key: str) -> str:
    """Global translation helper function."""
    return state.t(key)
