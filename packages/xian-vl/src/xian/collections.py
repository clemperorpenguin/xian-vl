# Xian-VL — Core Vision-Language orchestration engine.
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

"""Xian's own Omni collections for Lemonade Server.

A *collection* is a Lemonade model with ``recipe: "collection.omni"``: a virtual
model whose components cover several modalities at once.  Xian registers its own
rather than depending on a vendor bundle like ``LMX-Omni-5.5B-Lite``, because:

* those are *registry-backed* — their registry entry is a bare pointer with no
  ``components`` until the Hugging Face manifest has been pulled, and a
  ``POST /v1/load`` against one in that state falls through to the llama.cpp
  backend and fails with ``GGUF file not found for checkpoint``;
* their component choices are not ours.  Xian needs a **vision** planner above
  all else, and pays download weight for an image generator it never calls.

Three tiers ship, each a separate registered collection so the model id in
settings names exactly what is installed:

======================  =========================  ==========
Collection              Planner (vision + tools)   Footprint
======================  =========================  ==========
``Xian-Lite``           ``Qwen3.5-4B-GGUF``        ~4.0 GB
``Xian-Ultra``          ``Qwen3.5-9B-GGUF``        ~8.9 GB
``Xian-Halo``           ``Qwen3.6-35B-A3B-MTP``    ~25.8 GB
======================  =========================  ==========

Registration goes through ``POST /v1/pull``, which registers *and* downloads
every component in one streamed call.  Two constraints of that endpoint shape
the bodies built here, both learned from the Lemonade server source:

* the name must carry the ``user.`` prefix whenever the body sets ``recipe``
  (the server rejects it otherwise); it is surfaced publicly under the bare
  name, so ``user.Xian-Ultra`` appears in ``/v1/models`` as ``Xian-Ultra``;
* the collection's own ``checkpoint`` must stay **empty**.  A non-empty
  checkpoint marks a collection as registry-backed, and the server then discards
  the authored component list in favour of a downloaded manifest.

Components are referenced by bare built-in name with no inline ``models`` array.
The server validates that every one is already registered, so a name that has
gone stale fails loudly at install time instead of silently registering a
smaller collection.  Every name is checked against the pinned Lemonade registry
in ``tests/test_collections.py``.

**No NPU components.**  Every recipe here (llamacpp, whispercpp, kokoro) is
portable.  ``flm`` models exist only on machines with FastFlowLM installed, so
naming one would make ``/v1/pull`` fail everywhere else — and a collection is
hidden from ``/v1/models`` until *all* of its components are downloaded, which
would silently disable routing.  NPU work is steered by
:mod:`xian.omni_router` on top of whichever collection is installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from shared_types.enums import CollectionTier

logger = logging.getLogger(__name__)

#: Prefix the Lemonade server requires for user-registered models.
USER_PREFIX = "user."

#: Recipe marking an Omni collection.
OMNI_RECIPE = "collection.omni"


@dataclass(frozen=True)
class XianCollection:
    """One tier: a registered Omni collection and the models it bundles."""

    #: Public model id, as it appears in ``/v1/models`` and in settings.
    name: str
    tier: CollectionTier
    #: Vision-capable, tool-calling planner. Everything text and vision uses it.
    planner: str
    #: Speech-to-text component.
    asr: str
    #: Text-to-speech component.
    tts: str
    #: Approximate total download, GB. Informational — the server is the truth.
    size_gb: float
    #: Minimum usable memory this tier expects, GB.
    min_memory_gb: float
    #: Extra components, if a tier ever needs one.
    extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def registered_name(self) -> str:
        """The name to register under (``user.``-prefixed)."""
        return f"{USER_PREFIX}{self.name}"

    @property
    def components(self) -> list[str]:
        return [self.planner, self.asr, self.tts, *self.extra]


XIAN_LITE = XianCollection(
    name="Xian-Lite",
    tier=CollectionTier.LITE,
    planner="Qwen3.5-4B-GGUF",
    asr="Whisper-Tiny",
    tts="kokoro-v1",
    size_gb=4.01,
    min_memory_gb=0.0,
)

XIAN_ULTRA = XianCollection(
    name="Xian-Ultra",
    tier=CollectionTier.ULTRA,
    planner="Qwen3.5-9B-GGUF",
    asr="Whisper-Large-v3-Turbo",
    tts="kokoro-v1",
    size_gb=8.85,
    min_memory_gb=12.0,
)

# A sparse MoE rather than a dense 27B of similar footprint: on a unified-memory
# APU only the ~3B active parameters are read per token, so it answers several
# times faster at comparable quality — which is what a HUD needs.
XIAN_HALO = XianCollection(
    name="Xian-Halo",
    tier=CollectionTier.HALO,
    planner="Qwen3.6-35B-A3B-MTP-GGUF",
    asr="Whisper-Large-v3-Turbo",
    tts="kokoro-v1",
    size_gb=25.77,
    min_memory_gb=48.0,
)

COLLECTIONS: dict[CollectionTier, XianCollection] = {
    CollectionTier.LITE: XIAN_LITE,
    CollectionTier.ULTRA: XIAN_ULTRA,
    CollectionTier.HALO: XIAN_HALO,
}

#: Public ids of every Xian collection, for "is this one of ours?" checks.
COLLECTION_NAMES: frozenset[str] = frozenset(c.name for c in COLLECTIONS.values())


def get_collection(tier: CollectionTier | str) -> XianCollection:
    """Return the collection for a tier.

    Raises ``KeyError`` for ``CollectionTier.CUSTOM`` — the caller picks its own
    models in that case.
    """
    return COLLECTIONS[CollectionTier(tier)]


def collection_for_name(name: str | None) -> XianCollection | None:
    """Return the collection a model id names, accepting the ``user.`` form."""
    if not name:
        return None
    bare = name[len(USER_PREFIX):] if name.startswith(USER_PREFIX) else name
    for collection in COLLECTIONS.values():
        if collection.name == bare:
            return collection
    return None


def is_xian_collection(name: str | None) -> bool:
    return collection_for_name(name) is not None


def recommended_tier(memory_gb: float) -> CollectionTier:
    """Pick the largest tier that fits in ``memory_gb`` of usable memory."""
    fitting = [c for c in COLLECTIONS.values() if memory_gb >= c.min_memory_gb]
    if not fitting:
        return CollectionTier.LITE
    return max(fitting, key=lambda c: c.min_memory_gb).tier


def build_pull_body(tier: CollectionTier | str, *, stream: bool = True) -> dict:
    """Build the ``POST /v1/pull`` body that registers and downloads a tier.

    The shape is Lemonade's exported-collection format, so the same document can
    later be published to a Hugging Face repo named after the collection and
    pulled by name.  ``checkpoints.main`` is deliberately empty: see the module
    docstring.
    """
    collection = get_collection(tier)
    body: dict = {
        "model_name": collection.registered_name,
        "recipe": OMNI_RECIPE,
        "checkpoints": {"main": ""},
        "components": collection.components,
        "labels": [],
        "recipe_options": {},
    }
    if stream:
        body["stream"] = True
    return body


async def install_collection(client, tier: CollectionTier | str) -> str:
    """Register, download, and load a tier.  Returns its public model id.

    Parameters
    ----------
    client:
        A :class:`~xian.lemonade_client.LemonadeClient` instance.
    tier:
        Which tier to install.
    """
    collection = get_collection(tier)
    logger.info(
        "Registering collection %s (%s, ~%.1f GB)…",
        collection.name, ", ".join(collection.components), collection.size_gb,
    )
    await client.pull_model_body(build_pull_body(tier, stream=False))
    logger.info("Loading collection %s …", collection.name)
    await client.load_model(collection.name)
    logger.info("Collection '%s' ready.", collection.name)
    return collection.name


async def get_collection_status(client, tier: CollectionTier | str) -> dict[str, bool]:
    """Map each component of a tier to whether it is currently loaded."""
    collection = get_collection(tier)
    try:
        health = await client.health()
        loaded = set(health.get("all_models_loaded") or health.get("loaded_models") or [])
    except Exception:
        loaded = set()
    return {name: name in loaded for name in collection.components}


def _main(argv: list[str] | None = None) -> int:
    """``python -m xian.collections <tier>`` prints a tier's /v1/pull body.

    Lets ``mage.sh`` install a collection with plain ``curl`` while keeping this
    module the single source of truth for what is in one.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Print a Xian collection's /v1/pull body.")
    parser.add_argument("tier", choices=[t.value for t in COLLECTIONS], help="Collection tier.")
    parser.add_argument("--no-stream", action="store_true", help="Omit the SSE stream flag.")
    args = parser.parse_args(argv)

    print(json.dumps(build_pull_body(args.tier, stream=not args.no_stream), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
