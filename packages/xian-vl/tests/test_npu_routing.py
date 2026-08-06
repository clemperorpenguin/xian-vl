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


def test_auto_moves_only_speech():
    """Whisper on the NPU is a free win; relocating the chat LLM is a trade-off."""
    router = _router(MIXED_MODELS, BACKEND_AUTO)
    assert router.asr() == "whisper-large-v3-flm"
    assert router.llm() == "Qwen3.5-4B-MTP-GGUF"


@pytest.mark.parametrize("preference", [BACKEND_AUTO, BACKEND_GPU, BACKEND_NPU])
def test_vision_never_resolves_to_an_npu_model(preference):
    """FastFlowLM serves no vision models — routing there would fail at the server."""
    router = _router(MIXED_MODELS, preference)
    assert router.vision() not in set(router.npu_model_ids())


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


def test_load_options_carry_the_power_mode_only_for_npu_models():
    from xian.pipeline import VLConfig, VLProcessor

    processor = VLProcessor(VLConfig(backend_preference=BACKEND_NPU, npu_power_mode="turbo"))
    processor.router.update_with_models(MIXED_MODELS)

    assert processor._load_options_for("Qwen3-8B-flm") == {"power_mode": "turbo"}
    assert processor._load_options_for("Qwen3.5-4B-MTP-GGUF") is None
