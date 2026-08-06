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

from __future__ import annotations

import logging
from typing import Any
import httpx

logger = logging.getLogger(__name__)


# Lemonade recipes that execute on the Ryzen AI NPU. FastFlowLM ("flm") is
# the only one available on Linux, and it serves text LLMs, Whisper ASR, and
# embeddings — there is no NPU vision path, so vision must never route here.
NPU_RECIPES = ("flm",)

#: Modalities FastFlowLM can actually serve.
NPU_MODALITIES = ("llm", "asr")

BACKEND_AUTO = "auto"
BACKEND_GPU = "gpu"
BACKEND_NPU = "npu"


class OmniModelRouter:
    """Discovers, caches, and routes requests to the correct model for each modality.

    Modalities supported:
    - llm: Primary text/chat model (or tool-calling model)
    - vision: Vision-capable VLM
    - asr: Transcription (Speech-to-text) model
    - tts: Text-to-speech model
    - image: Image generation model
    - edit: Image editing model

    Routing is by modality label.  ``backend_preference`` adds a second axis:
    when the host has an NPU-backed model registered, text and speech work can
    be steered onto it so the GPU is left to the vision model.
    """

    def __init__(self, api_url: str, backend_preference: str = BACKEND_AUTO):
        self.api_url = api_url.rstrip("/")
        self.api_url_no_v1 = self.api_url.removesuffix("/v1")
        self.backend_preference = backend_preference
        self._models: dict[str, str] = {}
        self._downloaded_models: list[dict[str, Any]] = []
        self._omni_detected = False
        self._omni_model_id: str | None = None
        self._active_model: str | None = None
        # Last logged routing signature, so a rebuild that produces an
        # identical table doesn't re-emit the same lines on every inference.
        self._last_logged_signature: tuple | None = None

    @property
    def active_model(self) -> str | None:
        return self._active_model

    @active_model.setter
    def active_model(self, value: str | None) -> None:
        self._active_model = value
        self._rebuild_routing_table()

    def update_with_models(self, models_data: list[dict[str, Any]]) -> None:
        """Populate the routing table from the raw model response.
        This parses the models list and maps labels/recipes to modalities.
        """
        self._downloaded_models = [m for m in models_data if m.get("downloaded", True)]
        self._rebuild_routing_table()

    def _rebuild_routing_table(self) -> None:
        """Rebuild self._models routing table, applying active Omni model overrides if set."""
        # 1. Reset mappings
        self._models = {}
        self._omni_detected = False
        self._omni_model_id = None

        # Check for Omni collections
        for m in self._downloaded_models:
            m_id = m.get("id", "")
            recipe = m.get("recipe", "")
            if recipe == "collection.omni" or m_id.startswith("LMX-Omni-"):
                self._omni_detected = True
                self._omni_model_id = m_id
                break

        # Map labels to modalities globally first
        for m in self._downloaded_models:
            m_id = m.get("id", "")
            labels = m.get("labels", [])
            
            # Map based on standard labels
            for label in labels:
                if label not in self._models:
                    self._models[label] = m_id
            
            # If we don't have explicit labels, try fallback heuristics based on id
            lower_id = m_id.lower()
            if "whisper" in lower_id:
                self._models.setdefault("transcription", m_id)
                self._models.setdefault("asr", m_id)
            elif "kokoro" in lower_id or "tts" in lower_id:
                self._models.setdefault("tts", m_id)
                self._models.setdefault("text-to-speech", m_id)
            elif "flux" in lower_id or "sd" in lower_id or "stable-diffusion" in lower_id:
                self._models.setdefault("image", m_id)
                self._models.setdefault("edit", m_id)
            elif "vision" in lower_id or "vl" in lower_id or "qwen-vl" in lower_id:
                self._models.setdefault("vision", m_id)
            elif "qwen" in lower_id or "llama" in lower_id or "mistral" in lower_id or "gemma" in lower_id:
                self._models.setdefault("chat", m_id)
                self._models.setdefault("tool-calling", m_id)

        # 2. If an Omni model is active, override self._models mapping with its components
        active = self._active_model or self._omni_model_id
        if active and self.is_omni_model(active):
            components = []
            for m in self._downloaded_models:
                if m.get("id") == active:
                    components = m.get("components", [])
                    break
            
            if components:
                # Clear all standard modality mappings to prevent pollution from non-components
                self._models = {}
                
                # Re-map only from components
                for comp_id in components:
                    comp_info = next((m for m in self._downloaded_models if m.get("id") == comp_id), None)
                    comp_labels = comp_info.get("labels", []) if comp_info else []
                    
                    # Map explicit labels
                    for label in comp_labels:
                        self._models[label] = comp_id
                        
                    # Map heuristics
                    lower_id = comp_id.lower()
                    if "whisper" in lower_id:
                        self._models["transcription"] = comp_id
                        self._models["asr"] = comp_id
                    elif "kokoro" in lower_id or "tts" in lower_id:
                        self._models["tts"] = comp_id
                        self._models["text-to-speech"] = comp_id
                    elif "flux" in lower_id or "sd" in lower_id or "stable-diffusion" in lower_id:
                        self._models["image"] = comp_id
                        self._models["edit"] = comp_id
                    elif "vision" in lower_id or "vl" in lower_id or "qwen-vl" in lower_id:
                        self._models["vision"] = comp_id
                    elif "qwen" in lower_id or "llama" in lower_id or "mistral" in lower_id or "gemma" in lower_id:
                        self._models["chat"] = comp_id
                        self._models["tool-calling"] = comp_id
                        self._models["reasoning"] = comp_id

                # Fallback for vision inside the collection
                if "vision" not in self._models and ("chat" in self._models or "tool-calling" in self._models):
                    self._models["vision"] = self._models.get("tool-calling") or self._models.get("chat")

        # Only log when the resolved routing actually changes — this rebuild
        # runs on every inference, so unconditional logging floods the console.
        signature = (self._omni_model_id, tuple(sorted(self._models.items())))
        if signature != self._last_logged_signature:
            self._last_logged_signature = signature
            if self._omni_detected:
                logger.info("Omni model detected: %s", self._omni_model_id)
            logger.debug("OmniModelRouter mapped labels: %s", self._models)

    def discover_sync(self) -> None:
        """Synchronously query the Lemonade server and update the routing table."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.api_url_no_v1}/v1/models", params={"show_all": "true"})
                resp.raise_for_status()
                data = resp.json().get("data", [])
                self.update_with_models(data)
        except Exception as e:
            logger.warning("OmniModelRouter sync discovery failed: %s", e)

    async def discover_async(self) -> None:
        """Asynchronously query the Lemonade server and update the routing table."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.api_url_no_v1}/v1/models", params={"show_all": "true"})
                resp.raise_for_status()
                data = resp.json().get("data", [])
                self.update_with_models(data)
        except Exception as e:
            logger.warning("OmniModelRouter async discovery failed: %s", e)

    def is_omni_model(self, model_id: str | None) -> bool:
        """Returns True if the given model_id is a Lemonade Omni Model."""
        if not model_id or model_id in ("omni-router", "default"):
            return False
        for m in self._downloaded_models:
            if m.get("id") == model_id:
                return m.get("recipe") == "collection.omni" or model_id.startswith("LMX-Omni-")
        return model_id.startswith("LMX-Omni-")

    def resolve_omni_component(self, omni_model_id: str, modality: str) -> str | None:
        """Resolves the sub-component model ID of an Omni Model for a given modality."""
        components = []
        for m in self._downloaded_models:
            if m.get("id") == omni_model_id:
                components = m.get("components", [])
                break
        
        if not components:
            return None

        if modality == "llm":
            return self._resolve_component(components, ["tool-calling", "chat", "reasoning"], ["qwen", "llama", "gemma", "mistral"])
        elif modality == "vision":
            val = self._resolve_component(components, ["vision"], ["vision", "vl", "qwen-vl"])
            return val or self.resolve_omni_component(omni_model_id, "llm")
        elif modality == "asr":
            return self._resolve_component(components, ["transcription", "realtime-transcription", "asr"], ["whisper", "whispercpp"])
        elif modality == "tts":
            return self._resolve_component(components, ["tts", "text-to-speech"], ["kokoro", "tts"])
        elif modality == "image":
            return self._resolve_component(components, ["image"], ["flux", "sd", "stable-diffusion"])
        elif modality == "edit":
            return self._resolve_component(components, ["edit"], ["flux", "sd", "stable-diffusion"])
        
        return None

    def _resolve_component(self, components: list[str], labels_to_match: list[str], fallback_keywords: list[str]) -> str | None:
        for comp_id in components:
            comp_info = next((m for m in self._downloaded_models if m.get("id") == comp_id), None)
            if comp_info:
                comp_labels = comp_info.get("labels", [])
                if any(l in comp_labels for l in labels_to_match):
                    return comp_id

        for comp_id in components:
            comp_id_lower = comp_id.lower()
            if any(kw in comp_id_lower for kw in fallback_keywords):
                return comp_id

        for comp_id in components:
            comp_info = next((m for m in self._downloaded_models if m.get("id") == comp_id), None)
            if comp_info:
                recipe = comp_info.get("recipe", "").lower()
                if any(kw in recipe for kw in fallback_keywords):
                    return comp_id

        return None

    # ── NPU routing ──────────────────────────────────────────────────

    @staticmethod
    def is_npu_model(model: dict[str, Any]) -> bool:
        return str(model.get("recipe", "")).lower() in NPU_RECIPES

    def npu_available(self) -> bool:
        """True when the server has at least one NPU-backed model registered."""
        return any(self.is_npu_model(m) for m in self._downloaded_models)

    def npu_model_ids(self) -> list[str]:
        return [m["id"] for m in self._downloaded_models if self.is_npu_model(m) and "id" in m]

    def _wants_npu(self, modality: str) -> bool:
        """Whether this modality should be steered to the NPU."""
        if modality not in NPU_MODALITIES:
            return False
        if self.backend_preference == BACKEND_NPU:
            return True
        # In auto mode only speech moves: it is a strict win (Whisper is small,
        # the NPU handles it comfortably, and it frees the GPU during raids).
        # Moving the chat LLM is a quality trade-off, so that stays opt-in.
        return self.backend_preference == BACKEND_AUTO and modality == "asr"

    def _npu_model_for(self, modality: str) -> str | None:
        """Find an NPU-backed model that can serve this modality."""
        if modality == "llm":
            labels = ("tool-calling", "chat", "reasoning", "llm")
            keywords = ("qwen", "llama", "gemma", "mistral", "deepseek")
            excluded = ("whisper", "embed")
        elif modality == "asr":
            labels = ("transcription", "realtime-transcription", "asr")
            keywords = ("whisper",)
            excluded = ()
        else:
            return None

        for m in self._downloaded_models:
            if not self.is_npu_model(m):
                continue
            model_id = m.get("id", "")
            lower_id = model_id.lower()
            if any(x in lower_id for x in excluded):
                continue
            if any(label in m.get("labels", []) for label in labels):
                return model_id
            if any(kw in lower_id for kw in keywords):
                return model_id
        return None

    def _npu_override(self, modality: str) -> str | None:
        """Resolve this modality to an NPU model, or None to route normally."""
        if not self._wants_npu(modality):
            return None
        model_id = self._npu_model_for(modality)
        if model_id:
            logger.debug("Routing %s to NPU model %s", modality, model_id)
        return model_id

    # Modality getters
    def llm(self, active_model: str | None = None) -> str:
        """Returns the best chat/LLM model id."""
        npu = self._npu_override("llm")
        if npu:
            return npu

        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "llm")
            if resolved:
                return resolved

        if self._omni_detected and self._omni_model_id:
            resolved = self.resolve_omni_component(self._omni_model_id, "llm")
            if resolved:
                return resolved

        return (self._models.get("tool-calling") or 
                self._models.get("chat") or 
                (self._downloaded_models[0]["id"] if self._downloaded_models else ""))

    def vision(self, active_model: str | None = None) -> str:
        """Returns the best vision-capable model id."""
        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "vision")
            if resolved:
                return resolved

        if self._omni_detected and self._omni_model_id:
            resolved = self.resolve_omni_component(self._omni_model_id, "vision")
            if resolved:
                return resolved

        if self._models.get("vision"):
            return self._models["vision"]

        # Falling back to the LLM must not pick up an NPU model: FastFlowLM
        # has no vision path, so such a request would fail at the server.
        # Every candidate below is re-checked, because the label map and the
        # downloaded list both contain NPU models on a machine that has them.
        npu_ids = set(self.npu_model_ids())
        fallback = self.llm(active_model)
        if fallback and fallback not in npu_ids:
            return fallback

        for candidate in (self._models.get("tool-calling"), self._models.get("chat")):
            if candidate and candidate not in npu_ids:
                return candidate
        for m in self._downloaded_models:
            model_id = m.get("id")
            if model_id and model_id not in npu_ids:
                return model_id
        return ""

    def asr(self, active_model: str | None = None) -> str:
        """Returns the best speech-to-text (ASR) model id."""
        npu = self._npu_override("asr")
        if npu:
            return npu

        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "asr")
            if resolved:
                return resolved

        model_id = self._models.get("transcription") or self._models.get("asr")
        if model_id:
            return model_id

        if self._omni_detected and self._omni_model_id:
            resolved = self.resolve_omni_component(self._omni_model_id, "asr")
            if resolved:
                return resolved

        return ""

    def tts(self, active_model: str | None = None) -> str:
        """Returns the best text-to-speech (TTS) model id."""
        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "tts")
            if resolved:
                return resolved

        model_id = self._models.get("tts") or self._models.get("text-to-speech")
        if model_id:
            return model_id

        if self._omni_detected and self._omni_model_id:
            resolved = self.resolve_omni_component(self._omni_model_id, "tts")
            if resolved:
                return resolved

        return ""

    def image(self, active_model: str | None = None) -> str:
        """Returns the best image generation model id."""
        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "image")
            if resolved:
                return resolved
        return self._models.get("image") or ""

    def edit(self, active_model: str | None = None) -> str:
        """Returns the best image editing model id."""
        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "edit")
            if resolved:
                return resolved
        return self._models.get("edit") or ""

    @property
    def omni_detected(self) -> bool:
        return self._omni_detected

    @property
    def omni_model_id(self) -> str | None:
        return self._omni_model_id

    @property
    def downloaded_model_ids(self) -> list[str]:
        return [m["id"] for m in self._downloaded_models if "id" in m]
