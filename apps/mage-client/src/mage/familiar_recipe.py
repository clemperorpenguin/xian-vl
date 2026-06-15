# MAGE — Gaming HUD for real-time screen translation.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

"""Familiar "art recipe" schema — the contract a Lemonade model fills to conjure
a custom desktop familiar.

This module is deliberately **pure data** (no PyQt / no Qt imports) so it can be
shared between the renderer (``familiar_pet._RecipeArt``) and the future
Lemonade worker (``familiar_conjure``) without circular imports.

Every field is an enum from a fixed vocabulary or a validated hex color, so a
coerced recipe is always renderable: ``coerce_recipe`` fills defaults and snaps
unknown values to the nearest valid one rather than ever raising.
"""

import re

# --- allowed vocabularies ---------------------------------------------------
BODY_SHAPES = {"round", "triangle", "blob", "tall", "quad", "egg", "ghost"}
HEAD_KINDS = {"round", "merged"}
EARS = {"none", "pointed", "tufts", "long", "horns", "round", "floppy"}
EYES = {"dots", "big", "glow", "sleepy", "wink"}
FEATURES = {"wings", "tail", "halo", "antennae", "whiskers", "spikes",
            "fangs", "scarf", "flame", "gem", "glasses"}
HEADWEAR = {"none", "pointed_hat", "crown", "hood", "leaf", "bow", "antlers"}
ACCESSORY = {"none", "staff", "broom", "wand", "orb", "lantern", "book"}
TRANSIT = {"teleport", "fly", "climb"}

PALETTE_KEYS = ("primary", "secondary", "accent", "skin")

DEFAULTS = {
    "name": "Familiar",
    "palette": {
        "primary": "#6e8bdc",
        "secondary": "#3f4f80",
        "accent": "#ffd86b",
        "skin": "#f0d6b0",
    },
    "body": "round",
    "head": "round",
    "ears": "none",
    "eyes": "dots",
    "features": [],
    "headwear": "none",
    "accessory": "none",
    "transit": "teleport",
    "float_down": True,
    "glow_color": "#9a6bff",
}

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _hex(value, default: str) -> str:
    if isinstance(value, str) and _HEX.match(value.strip()):
        return value.strip().lower()
    return default


def _enum(value, allowed: set, default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def coerce_recipe(raw) -> dict:
    """Return a fully-valid recipe dict, filling defaults and snapping any
    unknown/garbage value to the nearest valid one. Never raises."""
    raw = raw if isinstance(raw, dict) else {}
    d = DEFAULTS

    pal_in = raw.get("palette") if isinstance(raw.get("palette"), dict) else {}
    palette = {k: _hex(pal_in.get(k), d["palette"][k]) for k in PALETTE_KEYS}

    features = []
    feats_in = raw.get("features")
    if isinstance(feats_in, (list, tuple)):
        for f in feats_in:
            if isinstance(f, str) and f in FEATURES and f not in features:
                features.append(f)

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        name = d["name"]
    name = re.sub(r"\s+", " ", name).strip()[:24]

    return {
        "name": name,
        "palette": palette,
        "body": _enum(raw.get("body"), BODY_SHAPES, d["body"]),
        "head": _enum(raw.get("head"), HEAD_KINDS, d["head"]),
        "ears": _enum(raw.get("ears"), EARS, d["ears"]),
        "eyes": _enum(raw.get("eyes"), EYES, d["eyes"]),
        "features": features,
        "headwear": _enum(raw.get("headwear"), HEADWEAR, d["headwear"]),
        "accessory": _enum(raw.get("accessory"), ACCESSORY, d["accessory"]),
        "transit": _enum(raw.get("transit"), TRANSIT, d["transit"]),
        "float_down": bool(raw.get("float_down", d["float_down"])),
        "glow_color": _hex(raw.get("glow_color"), d["glow_color"]),
    }


def schema_hint() -> str:
    """Human/LLM-readable allowed-values block for the conjure system prompt
    (used in Phase 2)."""
    return (
        "Allowed values:\n"
        f"  body: {sorted(BODY_SHAPES)}\n"
        f"  head: {sorted(HEAD_KINDS)}\n"
        f"  ears: {sorted(EARS)}\n"
        f"  eyes: {sorted(EYES)}\n"
        f"  features (subset): {sorted(FEATURES)}\n"
        f"  headwear: {sorted(HEADWEAR)}\n"
        f"  accessory: {sorted(ACCESSORY)}\n"
        f"  transit: {sorted(TRANSIT)}\n"
        "  palette/glow_color: #rrggbb hex strings\n"
        "  name: 1-24 characters\n"
    )


# --- bundled test recipes (Phase 1 visual iteration) ------------------------
# Chosen to exercise every body shape and all three transit styles.
TEST_RECIPES = {
    "ember_drake": {
        "name": "Ember Drake",
        "palette": {"primary": "#c0392b", "secondary": "#7b241c",
                    "accent": "#f1c40f", "skin": "#e8c39e"},
        "body": "round", "head": "round", "ears": "horns", "eyes": "glow",
        "features": ["wings", "tail", "spikes"], "headwear": "none",
        "accessory": "none", "transit": "fly", "float_down": False,
        "glow_color": "#f39c12",
    },
    "storm_sprite": {
        "name": "Storm Sprite",
        "palette": {"primary": "#7f8c9b", "secondary": "#4d5b66",
                    "accent": "#e8f4ff", "skin": "#cfd8dc"},
        "body": "blob", "head": "merged", "ears": "none", "eyes": "big",
        "features": [], "headwear": "none", "accessory": "none",
        "transit": "teleport", "float_down": True, "glow_color": "#7fd4ff",
    },
    "ivy_cat": {
        "name": "Ivy Cat",
        "palette": {"primary": "#3b6b4a", "secondary": "#274a33",
                    "accent": "#a3e635", "skin": "#d9f0c0"},
        "body": "quad", "head": "round", "ears": "pointed", "eyes": "glow",
        "features": ["tail", "whiskers"], "headwear": "leaf",
        "accessory": "none", "transit": "climb", "float_down": False,
        "glow_color": "#a3e635",
    },
    "lantern_fae": {
        "name": "Lantern Fae",
        "palette": {"primary": "#e0c84a", "secondary": "#8a7320",
                    "accent": "#fff3b0", "skin": "#fff7d6"},
        "body": "tall", "head": "round", "ears": "none", "eyes": "dots",
        "features": ["wings", "antennae", "halo"], "headwear": "leaf",
        "accessory": "orb", "transit": "teleport", "float_down": True,
        "glow_color": "#ffe066",
    },
    "stone_golem": {
        "name": "Stone Golem",
        "palette": {"primary": "#7d7468", "secondary": "#544c42",
                    "accent": "#b9aa92", "skin": "#9a8f7d"},
        "body": "triangle", "head": "round", "ears": "none", "eyes": "dots",
        "features": ["spikes", "gem"], "headwear": "crown", "accessory": "staff",
        "transit": "teleport", "float_down": True, "glow_color": "#c9b48a",
    },
    "pumpkin_imp": {
        "name": "Pumpkin Imp",
        "palette": {"primary": "#d8761e", "secondary": "#8f4711",
                    "accent": "#ffd24a", "skin": "#f0b96a"},
        "body": "egg", "head": "merged", "ears": "horns", "eyes": "glow",
        "features": ["fangs", "flame"], "headwear": "none", "accessory": "none",
        "transit": "teleport", "float_down": True, "glow_color": "#ff7a18",
    },
    "scholar_fox": {
        "name": "Scholar Fox",
        "palette": {"primary": "#c0622d", "secondary": "#7a3a18",
                    "accent": "#ffe0b0", "skin": "#fff0db"},
        "body": "quad", "head": "round", "ears": "pointed", "eyes": "dots",
        "features": ["tail", "glasses", "whiskers"], "headwear": "none",
        "accessory": "book", "transit": "climb", "float_down": False,
        "glow_color": "#ffcaa0",
    },
    "ghost_lyra": {
        "name": "Ghost Lyra",
        "palette": {"primary": "#cfc7e8", "secondary": "#9a8fc0",
                    "accent": "#ffffff", "skin": "#efeaff"},
        "body": "ghost", "head": "merged", "ears": "none", "eyes": "big",
        "features": ["scarf"], "headwear": "none", "accessory": "lantern",
        "transit": "teleport", "float_down": True, "glow_color": "#c9b8ff",
    },
    "royal_stag": {
        "name": "Royal Stag",
        "palette": {"primary": "#8a5a34", "secondary": "#5a3a20",
                    "accent": "#f4d27a", "skin": "#e8c7a0"},
        "body": "tall", "head": "round", "ears": "none", "eyes": "dots",
        "features": ["gem"], "headwear": "antlers", "accessory": "staff",
        "transit": "teleport", "float_down": True, "glow_color": "#f4d27a",
    },
    "bow_bunny": {
        "name": "Bow Bunny",
        "palette": {"primary": "#e7a9c6", "secondary": "#b86d92",
                    "accent": "#fff1f6", "skin": "#fff5fa"},
        "body": "round", "head": "round", "ears": "long", "eyes": "sleepy",
        "features": [], "headwear": "bow", "accessory": "none",
        "transit": "climb", "float_down": False, "glow_color": "#ffc0dd",
    },
    "frost_owl": {
        "name": "Frost Owl",
        "palette": {"primary": "#7fa8d8", "secondary": "#4d6f9e",
                    "accent": "#eaf4ff", "skin": "#dfeeff"},
        "body": "round", "head": "round", "ears": "tufts", "eyes": "big",
        "features": ["wings"], "headwear": "none", "accessory": "lantern",
        "transit": "fly", "float_down": False, "glow_color": "#bfe3ff",
    },
}

# Order the Phase-1 dev cycle steps through (broad spread of shapes/styles).
TEST_RECIPE_ORDER = [
    "ember_drake", "storm_sprite", "ivy_cat", "lantern_fae", "stone_golem",
    "pumpkin_imp", "scholar_fox", "ghost_lyra", "royal_stag", "bow_bunny",
    "frost_owl",
]
