# "Conjure…" — Lemonade-Authored Familiars

Design spec for the generative familiar slot in MAGE Familiar Mode.

## Overview

A 6th familiar slot, **"Conjure…"**, lets the user describe a creature in plain
language. A local **Lemonade LLM** turns that description into a constrained
JSON **art recipe** drawn from a fixed vocabulary of shape primitives. A
parametric vector renderer (`_RecipeArt`) draws every pose from that recipe, so
the conjured familiar **animates live** (walk-bob, cast-glow, transit) exactly
like the built-ins — no diffusion, no background removal, no static PNGs, and no
VRAM contention with the translation model. The recipe is persisted so it
survives restarts.

This is the **fully-local, single-model** path. A higher-fidelity diffusion
path is tracked as a follow-up in [ROADMAP.md](./ROADMAP.md).

## The art recipe (the contract)

The LLM's only job is to fill this schema. Every field is an **enum from a
fixed vocabulary** or a hex color, so anything it returns is renderable.
Creativity lives in the *combination* + palette + name + transit choice.

```jsonc
{
  "name": "Ember Drake",                    // 1–24 chars, shown in menus
  "palette": {
    "primary":   "#c0392b",
    "secondary": "#7b241c",                 // darker accents/shading
    "accent":    "#f1c40f",                 // details / headwear gem
    "skin":      "#e8c39e"                  // face
  },
  "body":     "round",      // round | triangle | blob | tall | quad (4-legged)
  "head":     "round",      // round | merged (face on the body)
  "ears":     "horns",      // none | pointed | tufts | long | horns
  "eyes":     "glow",       // dots | big | glow
  "features": ["wings","tail"],  // subset of: wings, tail, halo, antennae, whiskers, spikes
  "headwear": "none",       // none | pointed_hat | crown | hood | leaf
  "accessory":"none",       // none | staff | broom | wand | orb
  "transit":  "fly",        // teleport | fly | climb
  "float_down": false,      // teleport: blink up, float gently down (reserved)
  "glow_color": "#f39c12"   // teleport sparkle / cast aura tint
}
```

`transit` maps onto the existing `TransitStyle`: `teleport`→`TELEPORT_FLOAT`,
`fly`→`FLIGHT`, `climb`→`CLIMB`. So a conjured bat flies and a conjured golem
teleports — reusing the transit machinery already built, untouched. Unknown
enum values are **coerced to the nearest default** by the validator, so a
hallucinated `"body":"hexagonal"` renders as `round` instead of crashing.

## The Lemonade call

- **Where:** a `conjure_familiar(prompt)` on a **worker thread** (never block the
  GUI), reusing the OpenAI-compatible client the translator already uses
  (`self.processor.client.chat.completions.create(...)`, base URL
  `LEMONADE_API_URL` → `http://localhost:13305/v1`).
- **System prompt:** instruct the model to output **only** JSON matching the
  schema and to pick values **only** from the allowed lists, choosing the
  `transit` that best fits the creature.
- **Decoding:** request strict JSON (`response_format={"type":"json_object"}`
  when supported, else extract the first `{…}` block). Temperature ~0.6 — safe,
  because output is enum-bounded.
- **Robustness ladder:** `json.loads` → schema-validate/coerce → on hard failure
  one retry feeding the error back → on second failure, friendly error and keep
  the current familiar.
- **Latency:** one short (~200-token) completion; a second or two on Strix Halo.
  Show a spinner.

## The parametric renderer

`class _RecipeArt(_FamiliarArt)` draws in the same 96px design box as the
built-in species, composing in z-order: aura → broom → wings → tail → body+head
→ spikes → ears/horns → halo/antennae → headwear → whiskers → eyes → held
accessory. It honors the same state conventions (walk-bob, cast/react aura,
sad X-eyes) and transit poses (wings-spread / broom / cling), so it slots into
`paintEvent`'s vector branch with **zero** changes to the transit / animation /
bubble systems. Drop-in PNGs (`familiar/custom/<state>_<n>.png`) still override
it.

## Persistence & integration

- `FamiliarSpecies.CUSTOM = "custom"`.
- Recipe stored in settings (`KEY_FAMILIAR_CUSTOM_RECIPE`) and mirrored to
  `familiar/custom/recipe.json`.
- The 5 built-ins stay in the static `_PROFILES` map; `custom` builds its
  `_Profile` on demand from the stored recipe (`_build_custom_profile`).
- Touch points: `familiar_recipe.py` (schema + `coerce_recipe`, pure data),
  `familiar_pet.py` (enum + `_RecipeArt` + custom-profile builder),
  `familiar_conjure.py` (Phase 2 LLM worker), `app.py` (Conjure dialog + menu),
  `settings_keys.py`, `en.json`.

## UX

- "Conjure…" appears in both the settings dropdown and the right-click
  **"Change familiar ▸"** submenu.
- Opens a modal: description box, example chips, **Conjure** button, spinner,
  **live preview**, then **Reroll** / **Use this**.
- Only `name` is free text (length-clamped + escaped). Everything else is
  enum-bounded — no unsafe-image surface.

## Build phases

1. ✅ **Schema + `_RecipeArt`** with bundled test recipes — all 5 mood states +
   3 transit styles render. Vocabulary later widened to 7 body shapes, 7 ear
   types, 5 eye styles, 11 features, 7 headwear, 7 accessories.
2. ✅ **`familiar_conjure.py`**: `ConjureWorker` (QThread → `processor.engine`)
   with first-balanced-`{}` extraction, trailing-comma tolerance, coercion, and
   one retry-with-nudge. Triggered by **right-click → "Conjure new familiar…"**,
   a plain `QInputDialog` (casting pose while it thinks; name shown in a bubble
   on success; sad pose on failure).
3. ✅ **Conjure dialog + dropdown wiring**: `conjure_dialog.py` — a modal with a
   description box, example chips, an **animated live preview** (the real
   `_RecipeArt`, cycling idle→walk→cast→react), **Reroll**, and **Use this**.
   Surfaced from right-click → "Conjure new familiar…", a **"Conjure…" button**
   next to the settings dropdown, and the dropdown now includes the `custom`
   slot so it round-trips. The Phase-1 dev cycle was removed; selecting
   "Conjure ✨" simply switches to the existing custom familiar.
