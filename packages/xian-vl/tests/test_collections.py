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

"""Xian's own Omni collections: their definition and their registration body."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared_types.enums import CollectionTier
from xian.collections import (
    COLLECTION_NAMES,
    COLLECTIONS,
    OMNI_RECIPE,
    build_pull_body,
    collection_for_name,
    get_collection,
    is_xian_collection,
    recommended_tier,
)


def _lemonade_registry() -> dict:
    """The model registry of the pinned Lemonade submodule, or skip."""
    root = Path(__file__).resolve().parents[3]
    path = root / "lemonade" / "src" / "cpp" / "resources" / "server_models.json"
    if not path.is_file():
        pytest.skip("lemonade submodule not checked out")
    return json.loads(path.read_text(encoding="utf-8"))


# ── definitions ──────────────────────────────────────────────────────

@pytest.mark.parametrize("tier", list(COLLECTIONS))
def test_every_component_exists_in_the_pinned_lemonade_registry(tier):
    """A stale name fails /v1/pull outright — catch it here, not at install time.

    Components are referenced by bare built-in name with no inline definitions,
    so the server requires each one to be already registered.  Bumping the
    lemonade submodule past a renamed or dropped model must break this test.
    """
    registry = _lemonade_registry()
    for component in get_collection(tier).components:
        assert component in registry, f"{component} is not in the Lemonade registry"


@pytest.mark.parametrize("tier", list(COLLECTIONS))
def test_the_planner_can_see_and_call_tools(tier):
    """MAGE is a screen translator first: a planner without vision is useless."""
    registry = _lemonade_registry()
    labels = registry[get_collection(tier).planner].get("labels", [])
    assert "vision" in labels
    assert "tool-calling" in labels


@pytest.mark.parametrize("tier", list(COLLECTIONS))
def test_no_component_needs_hardware_that_may_be_absent(tier):
    """Portable recipes only.

    ``flm`` models exist only where FastFlowLM is installed and ``ryzenai-llm``
    is filtered out on Linux entirely; naming either would make /v1/pull fail on
    every other machine.  NPU work is a routing decision, not a component.
    """
    registry = _lemonade_registry()
    for component in get_collection(tier).components:
        assert registry[component]["recipe"] in ("llamacpp", "whispercpp", "kokoro")


def test_tiers_are_ordered_by_footprint():
    ordered = [COLLECTIONS[t] for t in (CollectionTier.LITE, CollectionTier.ULTRA, CollectionTier.HALO)]
    assert [c.size_gb for c in ordered] == sorted(c.size_gb for c in ordered)
    assert [c.min_memory_gb for c in ordered] == sorted(c.min_memory_gb for c in ordered)


# ── registration body ────────────────────────────────────────────────

@pytest.mark.parametrize("tier", list(COLLECTIONS))
def test_pull_body_registers_under_the_user_namespace(tier):
    """The server rejects a body that sets `recipe` without the user. prefix."""
    body = build_pull_body(tier)
    assert body["model_name"].startswith("user.")
    assert body["recipe"] == OMNI_RECIPE


@pytest.mark.parametrize("tier", list(COLLECTIONS))
def test_pull_body_leaves_the_checkpoint_empty(tier):
    """A checkpoint marks a collection registry-backed.

    The server then discards the authored component list and rebuilds it from a
    downloaded manifest — which does not exist for ours, leaving a collection
    with no components.  That is exactly the state that makes /v1/load fall
    through to llama.cpp and fail with "GGUF file not found for checkpoint".
    """
    body = build_pull_body(tier)
    assert body["checkpoints"] == {"main": ""}
    assert "checkpoint" not in body


@pytest.mark.parametrize("tier", list(COLLECTIONS))
def test_pull_body_components_are_non_empty_and_never_self_referential(tier):
    body = build_pull_body(tier)
    components = body["components"]
    assert components
    assert all(isinstance(name, str) and name for name in components)
    bare = body["model_name"].removeprefix("user.")
    assert bare not in components
    assert not COLLECTION_NAMES & set(components), "a component may not be another collection"


def test_pull_body_streams_by_default_and_can_opt_out():
    assert build_pull_body(CollectionTier.LITE)["stream"] is True
    assert "stream" not in build_pull_body(CollectionTier.LITE, stream=False)


def test_pull_body_accepts_a_plain_string_tier():
    """mage.sh passes the tier through as a shell word."""
    assert build_pull_body("halo")["model_name"] == "user.Xian-Halo"


# ── lookup helpers ───────────────────────────────────────────────────

def test_collections_are_recognized_by_either_name_form():
    assert is_xian_collection("Xian-Ultra")
    assert is_xian_collection("user.Xian-Ultra")
    assert collection_for_name("user.Xian-Lite").tier is CollectionTier.LITE


def test_unrelated_models_are_not_mistaken_for_collections():
    for name in ("LMX-Omni-5.5B-Lite", "Qwen3.5-9B-GGUF", "", None):
        assert not is_xian_collection(name)


def test_custom_tier_has_no_collection():
    with pytest.raises(KeyError):
        get_collection(CollectionTier.CUSTOM)


@pytest.mark.parametrize(
    "memory_gb,expected",
    [
        (0, CollectionTier.LITE),
        (8, CollectionTier.LITE),
        (11.9, CollectionTier.LITE),
        (12, CollectionTier.ULTRA),
        (32, CollectionTier.ULTRA),
        (48, CollectionTier.HALO),
        (128, CollectionTier.HALO),
    ],
)
def test_recommended_tier_picks_the_largest_that_fits(memory_gb, expected):
    assert recommended_tier(memory_gb) is expected


# ── install ──────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_install_registers_then_loads():
    client = MagicMock()
    client.pull_model_body = AsyncMock(return_value={"status": "success"})
    client.load_model = AsyncMock(return_value={"status": "success"})

    from xian.collections import install_collection

    name = await install_collection(client, CollectionTier.ULTRA)

    assert name == "Xian-Ultra"
    # Registered with the full body, loaded by public name: /v1/load resolves
    # user.Xian-Ultra from the alias itself.
    assert client.pull_model_body.await_args.args[0]["model_name"] == "user.Xian-Ultra"
    client.load_model.assert_awaited_once_with("Xian-Ultra")


# ── routing against an installed collection ──────────────────────────

# What GET /v1/models returns once Xian-Ultra is installed, alongside an image
# model the user downloaded separately.
INSTALLED = [
    {
        "id": "Xian-Ultra",
        "recipe": "collection.omni",
        "components": ["Qwen3.5-9B-GGUF", "Whisper-Large-v3-Turbo", "kokoro-v1"],
        "downloaded": True,
    },
    {"id": "Qwen3.5-9B-GGUF", "recipe": "llamacpp", "labels": ["vision", "tool-calling"], "downloaded": True},
    {"id": "Whisper-Large-v3-Turbo", "recipe": "whispercpp", "labels": ["transcription"], "downloaded": True},
    {"id": "kokoro-v1", "recipe": "kokoro", "labels": ["tts"], "downloaded": True},
    {"id": "SD-Turbo", "recipe": "sd-cpp", "labels": ["image"], "downloaded": True},
]


def _router(models, active=None):
    from xian.omni_router import OmniModelRouter

    router = OmniModelRouter("http://localhost:13305/v1")
    router.update_with_models(models)
    if active:
        router.active_model = active
    return router


def test_an_installed_collection_is_detected_and_drives_every_modality():
    router = _router(INSTALLED, active="Xian-Ultra")

    assert router.omni_detected is True
    assert router.omni_model_id == "Xian-Ultra"
    assert router.vision() == "Qwen3.5-9B-GGUF"
    assert router.llm() == "Qwen3.5-9B-GGUF"
    assert router.asr() == "Whisper-Large-v3-Turbo"
    assert router.tts() == "kokoro-v1"


def test_modalities_the_collection_omits_still_resolve():
    """Xian collections carry no image model on purpose.

    An active collection used to wipe the whole routing table and remap only
    from its components, which left image and edit resolving to nothing — so a
    user with an image model downloaded lost the image tools the moment a
    collection was installed.  Components win where they cover a modality; the
    rest falls back to what is on the machine.
    """
    router = _router(INSTALLED, active="Xian-Ultra")

    assert router.image() == "SD-Turbo"
    assert router.edit() == "SD-Turbo"


def test_a_xian_collection_wins_over_a_vendor_one():
    """Ours is the collection whose components we chose."""
    models = INSTALLED + [
        {
            "id": "LMX-Omni-5.5B-Lite",
            "recipe": "collection.omni",
            "components": ["Qwen3.5-4B-MTP-GGUF"],
            "downloaded": True,
        },
    ]
    assert _router(models).omni_model_id == "Xian-Ultra"


def test_a_collection_missing_its_components_is_not_treated_as_a_chat_model():
    """A registered-but-unpulled collection carries no components.

    Resolution must fall through to something loadable rather than handing the
    collection name itself back as a model to run inference against.
    """
    router = _router(
        [{"id": "Qwen3.5-9B-GGUF", "recipe": "llamacpp", "labels": ["vision", "tool-calling"], "downloaded": True}],
        active="Xian-Ultra",
    )

    assert router.llm() == "Qwen3.5-9B-GGUF"
    assert router.vision() == "Qwen3.5-9B-GGUF"


# ── pre-warming ──────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_prewarm_never_loads_a_collection_that_is_not_downloaded(monkeypatch):
    """The exact failure this collection work exists to remove.

    POST /v1/load against a collection with no components falls through to the
    llama.cpp backend, which looks for a GGUF at the collection's checkpoint and
    fails.  A registered-but-undownloaded collection must be skipped, not
    loaded.
    """
    from xian.pipeline import VLConfig, VLProcessor

    processor = VLProcessor(VLConfig(model_name="Xian-Ultra"))
    processor.router.update_with_models([])

    # The server registers the collection but omits it from the default listing
    # until every component is downloaded — that gap is the whole hazard.
    async def list_models(*, show_all: bool = False):
        return [{"id": "Xian-Ultra", "recipe": "collection.omni", "downloaded": False}] if show_all else []

    client = MagicMock()
    client.list_models = list_models
    client.load_model = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("xian.pipeline.LemonadeClient", lambda **kwargs: client)

    await processor.prewarm_model()

    client.load_model.assert_not_awaited()
