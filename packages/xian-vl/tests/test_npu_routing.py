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

"""Backend-preference routing between the GPU and the Ryzen AI NPU."""

import pytest

from xian.omni_router import BACKEND_AUTO, BACKEND_GPU, BACKEND_NPU, OmniModelRouter

# A host with both a GPU Omni bundle and NPU-served FastFlowLM models,
# mirroring what GET /v1/models?show_all=true returns.
MIXED_MODELS = [
    {
        "id": "LMX-Omni-5.5B-Lite",
        "recipe": "collection.omni",
        "components": ["Qwen3.5-4B-MTP-GGUF", "whisper-tiny-GGUF", "kokoro-v1", "SD-Turbo"],
        "downloaded": True,
    },
    {"id": "Qwen3.5-4B-MTP-GGUF", "recipe": "llamacpp", "labels": ["chat", "tool-calling", "vision"], "downloaded": True},
    {"id": "whisper-tiny-GGUF", "recipe": "whispercpp", "labels": ["transcription"], "downloaded": True},
    {"id": "kokoro-v1", "recipe": "oga", "labels": ["tts"], "downloaded": True},
    {"id": "SD-Turbo", "recipe": "diffusers", "labels": ["image"], "downloaded": True},
    {"id": "Qwen3-8B-flm", "recipe": "flm", "labels": ["chat", "tool-calling"], "downloaded": True},
    {"id": "whisper-large-v3-flm", "recipe": "flm", "labels": ["transcription"], "downloaded": True},
]

GPU_ONLY_MODELS = [m for m in MIXED_MODELS if m["recipe"] != "flm"]


def _router(models, preference):
    router = OmniModelRouter("http://localhost:13305/v1", backend_preference=preference)
    router.update_with_models(models)
    return router


def test_npu_is_detected_only_when_an_flm_model_is_present():
    assert _router(MIXED_MODELS, BACKEND_AUTO).npu_available() is True
    assert _router(GPU_ONLY_MODELS, BACKEND_AUTO).npu_available() is False


def test_npu_preference_moves_chat_and_speech_off_the_gpu():
    router = _router(MIXED_MODELS, BACKEND_NPU)
    assert router.llm() == "Qwen3-8B-flm"
    assert router.asr() == "whisper-large-v3-flm"


def test_gpu_preference_never_routes_to_the_npu():
    router = _router(MIXED_MODELS, BACKEND_GPU)
    npu_ids = set(router.npu_model_ids())
    for modality in (router.llm(), router.asr(), router.vision(), router.tts(), router.image()):
        assert modality not in npu_ids


def test_auto_moves_speech_but_not_chat():
    """Whisper on the NPU is a free win; relocating the chat LLM is a trade-off."""
    router = _router(MIXED_MODELS, BACKEND_AUTO)
    assert router.asr() == "whisper-large-v3-flm"
    assert router.llm() == "Qwen3.5-4B-MTP-GGUF"


@pytest.mark.parametrize("preference", [BACKEND_AUTO, BACKEND_GPU, BACKEND_NPU])
def test_vision_never_resolves_to_an_npu_model_that_cannot_see(preference):
    """Routing an image at a text-only model fails at the server.

    None of the NPU models here claim vision, so vision must stay on the GPU
    under every preference — including "npu", which steers what it can rather
    than forcing what it cannot.
    """
    router = _router(MIXED_MODELS, preference)
    assert router.vision() not in set(router.npu_model_ids())


#: The same host, plus a FastFlowLM model the server labels as vision-capable.
#: FLM ships these now; the router used to exclude every NPU model from vision
#: on principle, which made them unreachable and left live mode on a CPU
#: fallback that was slower than the model it was standing in for.
VISION_NPU_MODELS = MIXED_MODELS + [
    {"id": "qwen3.5-4b-FLM", "recipe": "flm",
     "labels": ["vision", "chat", "tool-calling"], "downloaded": True},
]


def test_npu_preference_uses_a_vision_capable_npu_model():
    router = _router(VISION_NPU_MODELS, BACKEND_NPU)
    assert router.vision() == "qwen3.5-4b-FLM"


def test_auto_leaves_vision_on_the_gpu_even_when_the_npu_could_serve_it():
    """A GPU generally runs a large VLM faster; auto keeps the NPU for the small
    high-frequency calls and leaves the GPU clear for vision. Opt in to move it."""
    router = _router(VISION_NPU_MODELS, BACKEND_AUTO)
    assert router.vision() == "Qwen3.5-4B-MTP-GGUF"


def test_gpu_preference_ignores_a_vision_capable_npu_model():
    router = _router(VISION_NPU_MODELS, BACKEND_GPU)
    assert router.vision() not in set(router.npu_model_ids())


def test_a_translation_only_npu_model_never_takes_vision():
    """TranslateGemma carries a vision label but follows no instructions."""
    models = MIXED_MODELS + [
        {"id": "translategemma-4b-FLM", "recipe": "flm",
         "labels": ["vision", "chat"], "downloaded": True},
    ]
    router = _router(models, BACKEND_NPU)
    assert router.vision() != "translategemma-4b-FLM"


def test_vision_fallback_skips_the_npu_when_no_vision_label_exists():
    """The LLM fallback path must not smuggle an NPU model into vision."""
    models = [
        {"id": "Qwen3.5-4B-MTP-GGUF", "recipe": "llamacpp", "labels": ["chat", "tool-calling"], "downloaded": True},
        {"id": "Qwen3-8B-flm", "recipe": "flm", "labels": ["chat", "tool-calling"], "downloaded": True},
    ]
    router = _router(models, BACKEND_NPU)

    assert router.llm() == "Qwen3-8B-flm"
    assert router.vision() == "Qwen3.5-4B-MTP-GGUF"


def test_npu_preference_degrades_gracefully_without_npu_models():
    router = _router(GPU_ONLY_MODELS, BACKEND_NPU)
    assert router.llm() == "Qwen3.5-4B-MTP-GGUF"
    assert router.asr() == "whisper-tiny-GGUF"


def test_npu_llm_selection_skips_whisper_and_embedding_models():
    models = [
        {"id": "whisper-large-v3-flm", "recipe": "flm", "labels": ["transcription"], "downloaded": True},
        {"id": "nomic-embed-flm", "recipe": "flm", "labels": ["embedding"], "downloaded": True},
        {"id": "Gemma3-4B-flm", "recipe": "flm", "labels": ["chat"], "downloaded": True},
    ]
    assert _router(models, BACKEND_NPU).llm() == "Gemma3-4B-flm"


# ── translation routing ──────────────────────────────────────────────

# The same host, plus a dedicated machine-translation model served by the NPU.
MT_MODELS = MIXED_MODELS + [
    {"id": "translategemma-4b-FLM", "recipe": "flm", "labels": ["chat"], "downloaded": True},
]


def test_translation_prefers_a_dedicated_model_on_the_npu():
    """Bulk line translation is the hottest call in live mode — it gets the NPU."""
    assert _router(MT_MODELS, BACKEND_AUTO).translation() == "translategemma-4b-FLM"


def test_translation_falls_back_to_the_general_llm_without_one():
    """Every machine without an MT model keeps the behaviour it had."""
    router = _router(MIXED_MODELS, BACKEND_AUTO)
    assert router.translation() == router.llm() == "Qwen3.5-4B-MTP-GGUF"


def test_gpu_preference_keeps_translation_off_the_npu():
    router = _router(MT_MODELS, BACKEND_GPU)
    assert router.translation() not in set(router.npu_model_ids())


@pytest.mark.parametrize("preference", [BACKEND_AUTO, BACKEND_GPU, BACKEND_NPU])
def test_a_translation_model_never_takes_another_modality(preference):
    """It carries a "chat" label and the word "gemma", and can do neither.

    Asked to hold a conversation it translates the question; asked to read a
    screen it has no vision tower at all.  Both would be silent quality
    failures, so the exclusion is by name and applies to every path.
    """
    router = _router(MT_MODELS, preference)
    for resolved in (router.llm(), router.vision(), router.asr(), router.tts()):
        assert resolved != "translategemma-4b-FLM"


def test_a_translation_model_is_identified_as_translation_only():
    router = _router(MT_MODELS, BACKEND_AUTO)
    assert router.is_translation_only("translategemma-4b-FLM") is True
    assert router.is_translation_only("Qwen3.5-4B-MTP-GGUF") is False


def test_load_options_carry_the_power_mode_only_for_npu_models():
    from xian.pipeline import VLConfig, VLProcessor

    processor = VLProcessor(VLConfig(backend_preference=BACKEND_NPU, npu_power_mode="turbo"))
    processor.router.update_with_models(MIXED_MODELS)

    assert processor._load_options_for("Qwen3-8B-flm") == {"power_mode": "turbo"}
    assert processor._load_options_for("Qwen3.5-4B-MTP-GGUF") is None


# ── label-first resolution ───────────────────────────────────────────

def test_a_models_own_labels_beat_its_family_name():
    """An embedding model named after a chat family must not claim chat.

    Family keywords ("qwen", "gemma") say nothing about what a given member
    does — the family spans chat, embedding, transcription and image models —
    so they must never override what the server actually published.
    """
    models = [
        {"id": "Qwen3-Embedding-8B-GGUF", "recipe": "llamacpp",
         "labels": ["embeddings"], "downloaded": True},
        {"id": "Phi-4-mini-instruct-GGUF", "recipe": "llamacpp",
         "labels": ["chat"], "downloaded": True},
    ]
    router = _router(models, BACKEND_AUTO)

    assert router.llm() == "Phi-4-mini-instruct-GGUF"


def test_a_family_name_still_places_a_model_with_no_labels():
    """The guess of last resort survives, for a server that publishes nothing."""
    models = [{"id": "Qwen3-4B-Instruct", "recipe": "llamacpp", "downloaded": True}]
    assert _router(models, BACKEND_AUTO).llm() == "Qwen3-4B-Instruct"


def test_a_product_name_fills_in_a_modality_the_labels_omitted():
    """An "sd" model listed only as `image` still edits — that is a real
    capability of the product, not a guess from its family."""
    models = [{"id": "SD-Turbo", "recipe": "diffusers", "labels": ["image"], "downloaded": True}]
    router = _router(models, BACKEND_AUTO)

    assert router.image() == "SD-Turbo"
    assert router.edit() == "SD-Turbo"
