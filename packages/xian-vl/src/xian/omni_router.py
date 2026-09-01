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

from xian.collections import is_xian_collection

logger = logging.getLogger(__name__)


# Lemonade recipes that execute on the Ryzen AI NPU. FastFlowLM ("flm") is the
# only one available on Linux; "ryzenai-llm" is the Windows-only equivalent.
NPU_RECIPES = ("flm", "ryzenai-llm")

#: Modalities an NPU model may serve.
#:
#: Vision is on this list because FastFlowLM now ships vision models — the
#: server labels them, and asking it is the only reliable way to know. An
#: earlier version hard-coded "the NPU has no vision path", which stopped being
#: true and left the whole live overlay on a CPU OCR fallback. Nothing is
#: forced: a modality only reaches the NPU if the server actually lists a model
#: claiming it, so this is inert on a machine without one.
NPU_MODALITIES = ("llm", "asr", "translation", "vision")

# Machine-translation models: they translate, and do nothing else. They have no
# tool-calling, no usable vision, and do not follow instructions — they
# translate them. TranslateGemma is labelled `vision` and is genuinely an
# image-text-to-text model, yet cannot answer a grounding prompt, which is why
# a name check still guards the modalities its labels would otherwise claim.
TRANSLATION_KEYWORDS = ("translategemma", "madlad", "nllb", "opus-mt", "seamless", "towerinstruct")

#: Labels a purpose-built translation model may carry.
TRANSLATION_LABELS = ("translation", "translate")

#: modality -> (labels that claim it, id/recipe keywords that imply it).
#: ``resolve_omni_component`` and ``_npu_model_for`` both resolve through this,
#: so a modality taught to one is taught to the other.
MODALITY_MATCHERS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "llm": (
        ("tool-calling", "chat", "reasoning", "llm"),
        ("qwen", "llama", "gemma", "mistral", "deepseek"),
    ),
    "vision": (("vision",), ("vision", "vl", "qwen-vl")),
    "translation": (TRANSLATION_LABELS, TRANSLATION_KEYWORDS),
    "asr": (
        ("transcription", "realtime-transcription", "asr"),
        ("whisper", "whispercpp"),
    ),
    "tts": (("tts", "text-to-speech"), ("kokoro", "tts")),
    "image": (("image",), ("flux", "sd", "stable-diffusion")),
    "edit": (("edit",), ("flux", "sd", "stable-diffusion")),
}

#: Modality each id keyword implies, for models the server under-describes.
#: Ordered: a translation-only model is name-matched before everything else,
#: because "TranslateGemma" would otherwise land in chat on the "gemma" keyword
#: and displace a planner that can actually hold a conversation.
#:
#: These name a *product*, so they say something real about capability — a
#: "whisper" model transcribes, an "sd" model both generates and edits images.
#: They may therefore add modalities a model's labels left out.
_PRODUCT_MODALITIES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("whisper",), ("transcription", "asr")),
    (("kokoro", "tts"), ("tts", "text-to-speech")),
    (("flux", "sd", "stable-diffusion"), ("image", "edit")),
    (("vision", "vl", "qwen-vl"), ("vision",)),
)

#: These name a model *family*, which implies nothing about what a given
#: member can do — "qwen" spans chat, embedding, transcription and image
#: models. Applied only to a model the server published with no labels at all,
#: as a rough guess in the absence of anything better. Trusting them over a
#: model's own labels is how an embedding model ends up claiming chat, and how
#: the list rots every time a new family ships.
_FAMILY_MODALITIES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("qwen", "llama", "mistral", "gemma"), ("chat", "tool-calling", "reasoning")),
)

#: Kept for callers that want the whole set in priority order.
_HEURISTIC_MODALITIES = _PRODUCT_MODALITIES + _FAMILY_MODALITIES


def is_translation_only_model(model_id: str | None) -> bool:
    """True for a machine-translation model that cannot serve other modalities."""
    if not model_id:
        return False
    return any(kw in model_id.lower() for kw in TRANSLATION_KEYWORDS)

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

        # Check for Omni collections. A Xian collection wins over any other on
        # the machine: it is the one whose components we chose.
        for m in self._downloaded_models:
            m_id = m.get("id", "")
            recipe = m.get("recipe", "")
            if recipe == "collection.omni" or m_id.startswith("LMX-Omni-") or is_xian_collection(m_id):
                self._omni_detected = True
                if self._omni_model_id is None or is_xian_collection(m_id):
                    self._omni_model_id = m_id
                if is_xian_collection(m_id):
                    break

        # Map labels to modalities globally first. Under an explicit GPU
        # preference the NPU models are left out of the table altogether:
        # otherwise whether one wins a modality comes down to where it happens
        # to sit in the server's model list, which is not a preference at all.
        for m in self._downloaded_models:
            if self._routable(m):
                self._map_model(m.get("id", ""), m.get("labels", []), self._models, overwrite=False)

        # 2. If an Omni model is active, override self._models mapping with its components
        active = self._active_model or self._omni_model_id
        if active and self.is_omni_model(active):
            components = []
            for m in self._downloaded_models:
                if m.get("id") == active:
                    components = m.get("components", [])
                    break

            if components:
                # Components *overlay* the global map rather than replacing it.
                # A Xian collection deliberately bundles only planner/ASR/TTS, so
                # wiping the map would resolve image and edit to nothing and
                # break the image tools for a user who has an image model
                # downloaded alongside it. Components still win wherever they
                # cover a modality, which is the point of an active collection.
                overlay: dict[str, str] = {}
                for comp_id in components:
                    comp_info = next((m for m in self._downloaded_models if m.get("id") == comp_id), None)
                    comp_labels = comp_info.get("labels", []) if comp_info else []
                    self._map_model(comp_id, comp_labels, overlay, overwrite=True)

                # Fallback for vision inside the collection
                if "vision" not in overlay and ("chat" in overlay or "tool-calling" in overlay):
                    overlay["vision"] = overlay.get("tool-calling") or overlay.get("chat")

                self._models.update(overlay)

        # Only log when the resolved routing actually changes — this rebuild
        # runs on every inference, so unconditional logging floods the console.
        signature = (self._omni_model_id, tuple(sorted(self._models.items())))
        if signature != self._last_logged_signature:
            self._last_logged_signature = signature
            if self._omni_detected:
                logger.info("Omni model detected: %s", self._omni_model_id)
            logger.debug("OmniModelRouter mapped labels: %s", self._models)

    @staticmethod
    def _map_model(
        model_id: str,
        labels: list[str],
        target: dict[str, str],
        *,
        overwrite: bool,
    ) -> None:
        """Record which modalities ``model_id`` can serve, into ``target``.

        ``overwrite`` picks the precedence: the global table is first-wins
        (whichever model the server listed first keeps the label), while a
        collection's component overlay is last-wins, because a component was
        chosen deliberately and must beat whatever else is installed.
        """
        if not model_id:
            return
        mt_only = is_translation_only_model(model_id)

        def claim(modality: str) -> None:
            if overwrite or modality not in target:
                target[modality] = model_id

        # A translation-only model claims no modality but its own, whatever
        # generic labels it was published with — it cannot hold a conversation
        # or read an image.
        for label in labels:
            if mt_only and label not in TRANSLATION_LABELS:
                continue
            claim(label)

        if mt_only:
            claim("translation")
            return

        # A product name fills in what the labels left out — an "sd" model
        # listed only as `image` still edits — but never contradicts them.
        lower_id = model_id.lower()
        for keywords, modalities in _PRODUCT_MODALITIES:
            if any(kw in lower_id for kw in keywords):
                for modality in modalities:
                    claim(modality)
                return

        # A family name is a guess of last resort, for a model the server
        # published with no labels at all. Once a model has said what it does,
        # that is the answer.
        if labels:
            return
        for keywords, modalities in _FAMILY_MODALITIES:
            if any(kw in lower_id for kw in keywords):
                for modality in modalities:
                    claim(modality)
                return

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
        if is_xian_collection(model_id):
            return True
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

        matcher = MODALITY_MATCHERS.get(modality)
        if matcher is None:
            return None
        resolved = self._resolve_component(components, list(matcher[0]), list(matcher[1]))

        # A collection without a dedicated vision model still has to answer
        # vision calls: its planner is vision-capable by construction.
        if resolved is None and modality == "vision":
            return self.resolve_omni_component(omni_model_id, "llm")
        return resolved

    def _resolve_component(self, components: list[str], labels_to_match: list[str], fallback_keywords: list[str]) -> str | None:
        # A translation-only model answers to no modality but its own. Without
        # this it would win the llm slot on the "gemma" keyword, and every chat
        # turn would go to a model that translates the question instead of
        # answering it.
        wants_translation = any(label in TRANSLATION_LABELS for label in labels_to_match)
        candidates = [
            comp_id for comp_id in components
            if is_translation_only_model(comp_id) == wants_translation
        ]

        for comp_id in candidates:
            comp_info = next((m for m in self._downloaded_models if m.get("id") == comp_id), None)
            if comp_info:
                comp_labels = comp_info.get("labels", [])
                if any(l in comp_labels for l in labels_to_match):
                    return comp_id

        for comp_id in candidates:
            comp_id_lower = comp_id.lower()
            if any(kw in comp_id_lower for kw in fallback_keywords):
                return comp_id

        for comp_id in candidates:
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

    def _routable(self, model: dict[str, Any]) -> bool:
        """Whether this model may appear in the routing table at all."""
        return not (self.backend_preference == BACKEND_GPU and self.is_npu_model(model))

    def npu_available(self) -> bool:
        """True when the server has at least one NPU-backed model registered."""
        return any(self.is_npu_model(m) for m in self._downloaded_models)

    def npu_model_ids(self) -> list[str]:
        return [m["id"] for m in self._downloaded_models if self.is_npu_model(m) and "id" in m]

    def _npu_ids_without_vision(self) -> set[str]:
        """NPU models that would fail an image request, by their own labels."""
        vision_labels, _keywords = MODALITY_MATCHERS["vision"]
        return {
            m["id"]
            for m in self._downloaded_models
            if self.is_npu_model(m)
            and "id" in m
            and not any(label in (m.get("labels") or []) for label in vision_labels)
        }

    def _wants_npu(self, modality: str) -> bool:
        """Whether this modality should be steered to the NPU."""
        if modality not in NPU_MODALITIES:
            return False
        if self.backend_preference == BACKEND_NPU:
            return True
        # In auto mode speech and translation move. Both are strict wins: the
        # models are small, the NPU handles them comfortably, and they are the
        # two highest-frequency calls in the app — freeing the GPU for the
        # vision model is exactly what live mode needs. Moving the *chat* LLM is
        # a quality trade-off, so that one stays opt-in.
        #
        # Vision stays opt-in for the same reason in reverse: a discrete or
        # integrated GPU generally runs a multi-billion-parameter VLM faster
        # than the NPU does, and auto mode is already using the NPU to keep the
        # GPU clear for exactly that. Users who measure otherwise on their own
        # hardware select the "npu" preference.
        return self.backend_preference == BACKEND_AUTO and modality in ("asr", "translation")

    #: Keywords that disqualify a model from a modality it otherwise matches.
    #: An NPU chat model must not resolve to the Whisper or embedding models
    #: sitting next to it, which share none of the chat keywords but do carry
    #: generic labels.
    _NPU_EXCLUSIONS = {"llm": ("whisper", "embed")}

    def _npu_model_for(self, modality: str) -> str | None:
        """Find an NPU-backed model that can serve this modality."""
        if modality not in NPU_MODALITIES:
            return None
        labels, keywords = MODALITY_MATCHERS[modality]
        excluded = self._NPU_EXCLUSIONS.get(modality, ())

        for m in self._downloaded_models:
            if not self.is_npu_model(m):
                continue
            model_id = m.get("id", "")
            lower_id = model_id.lower()
            if any(x in lower_id for x in excluded):
                continue
            # Only the translation modality may resolve to a translation-only
            # model, whatever its labels claim.
            if is_translation_only_model(model_id) != (modality == "translation"):
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

        if self._models.get("tool-calling"):
            return self._models["tool-calling"]
        if self._models.get("chat"):
            return self._models["chat"]
        # Last resort: any downloaded model that can hold a conversation.
        for m in self._downloaded_models:
            model_id = m.get("id", "")
            if model_id and self._routable(m) and not is_translation_only_model(model_id):
                return model_id
        return ""

    def vision(self, active_model: str | None = None) -> str:
        """Returns the best vision-capable model id."""
        # First, exactly as in llm(): an explicit backend preference is the
        # user's instruction about their own hardware, and outranks a
        # collection's bundled component.
        npu = self._npu_override("vision")
        if npu:
            return npu

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

        # Falling back to a chat model must not pick up an NPU model that
        # cannot see: such a request fails at the server. Only the ones that
        # never claimed vision are excluded — an NPU model the server labels
        # `vision` is a legitimate candidate, and blanket-excluding every NPU id
        # here is what previously made a vision-capable NPU unreachable.
        blocked = self._npu_ids_without_vision()
        fallback = self.llm(active_model)
        if fallback and fallback not in blocked:
            return fallback

        for candidate in (self._models.get("tool-calling"), self._models.get("chat")):
            if candidate and candidate not in blocked:
                return candidate
        for m in self._downloaded_models:
            model_id = m.get("id")
            if model_id and model_id not in blocked and not is_translation_only_model(model_id):
                return model_id
        return ""

    def translation(self, active_model: str | None = None) -> str:
        """Returns the best model id for bulk machine translation.

        Prefers a purpose-built translation model — on the NPU when one is
        installed — and falls back to the general LLM, which is what every
        machine without one uses.  Callers must ask
        :meth:`is_translation_only` what kind of model they got: a dedicated MT
        model needs a plain text-in/text-out prompt, not the structured
        instructions the general LLM path uses.
        """
        npu = self._npu_override("translation")
        if npu:
            return npu

        model = active_model or self.active_model
        if model and self.is_omni_model(model):
            resolved = self.resolve_omni_component(model, "translation")
            if resolved:
                return resolved

        model_id = self._models.get("translation")
        if model_id:
            return model_id

        return self.llm(active_model)

    def is_translation_only(self, model_id: str | None) -> bool:
        """Whether this model can only translate (see :func:`is_translation_only_model`)."""
        return is_translation_only_model(model_id)

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
