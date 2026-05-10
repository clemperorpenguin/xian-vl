"""Xian Collection definitions for Lemonade Server.

A *collection* is a curated bundle of models that, when loaded together,
give Xian all the modalities it needs (VLM, ASR, TTS).  Two tiers ship
out of the box:

* **Xian Lite** — targets ≤ 8 GB VRAM (integrated GPUs, GTX 1660-class).
* **Xian Ultra** — targets ≥ 12 GB VRAM (RTX 3060+).

Users may also pick ``CollectionTier.CUSTOM`` and manually select models.
"""

from __future__ import annotations

import logging

from shared_types.enums import CollectionTier
from shared_types.models import CollectionModel

logger = logging.getLogger(__name__)

# ── Collection Specs ─────────────────────────────────────────────────

XIAN_LITE: list[CollectionModel] = [
    CollectionModel(
        name="Qwen3-4B-Instruct-GGUF",
        labels=["vision", "tool-calling"],
        load_options={"gpu_memory_utilization": 0.5},
    ),
    CollectionModel(
        name="whisper-tiny-GGUF",
        labels=["audio", "transcription"],
        load_options={},
    ),
    CollectionModel(
        name="kokoro-v1",
        labels=["tts", "speech"],
        load_options={},
    ),
]

XIAN_ULTRA: list[CollectionModel] = [
    CollectionModel(
        name="Qwen3.5-9B-Instruct-GGUF",
        labels=["vision", "tool-calling"],
        load_options={"gpu_memory_utilization": 0.75},
    ),
    CollectionModel(
        name="whisper-large-v3-turbo-GGUF",
        labels=["audio", "transcription"],
        load_options={},
    ),
    CollectionModel(
        name="kokoro-v1",
        labels=["tts", "speech"],
        load_options={},
    ),
]

COLLECTIONS: dict[CollectionTier, list[CollectionModel]] = {
    CollectionTier.LITE: XIAN_LITE,
    CollectionTier.ULTRA: XIAN_ULTRA,
}


def get_collection(tier: CollectionTier) -> list[CollectionModel]:
    """Return the model list for the requested tier.

    Raises ``KeyError`` for ``CollectionTier.CUSTOM`` — the caller is
    responsible for providing the model list in that case.
    """
    return COLLECTIONS[tier]


async def install_collection(
    client,  # xian.lemonade_client.LemonadeClient (avoid circular import)
    tier: CollectionTier,
) -> None:
    """Pull and load every model in a collection.

    Parameters
    ----------
    client:
        A :class:`~xian.lemonade_client.LemonadeClient` instance.
    tier:
        Which collection to install.
    """
    models = get_collection(tier)
    for model in models:
        logger.info("Pulling %s …", model.name)
        await client.pull_model(model.name)
        logger.info("Loading %s …", model.name)
        # Pass None if load_options is empty to avoid passing empty dict to API
        opts = model.load_options if model.load_options else None
        await client.load_model(model.name, opts)
    logger.info("Collection '%s' ready (%d models).", tier.value, len(models))


async def get_collection_status(
    client,
    tier: CollectionTier,
) -> dict[str, bool]:
    """Check which models in a collection are currently loaded.

    Returns a dict mapping model name → ``True`` if loaded.
    """
    models = get_collection(tier)
    try:
        health = await client.health()
        loaded = set(health.get("loaded_models", []))
    except Exception:
        loaded = set()
    return {m.name: m.name in loaded for m in models}
