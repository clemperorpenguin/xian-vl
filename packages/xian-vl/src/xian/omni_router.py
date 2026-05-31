from __future__ import annotations

import logging
from typing import Any
import httpx

logger = logging.getLogger(__name__)


class OmniModelRouter:
    """Discovers, caches, and routes requests to the correct model for each modality.

    Modalities supported:
    - llm: Primary text/chat model (or tool-calling model)
    - vision: Vision-capable VLM
    - asr: Transcription (Speech-to-text) model
    - tts: Text-to-speech model
    - image: Image generation model
    - edit: Image editing model
    """

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.api_url_no_v1 = self.api_url.removesuffix("/v1")
        self._models: dict[str, str] = {}
        self._downloaded_models: list[dict[str, Any]] = []
        self._omni_detected = False
        self._omni_model_id: str | None = None

    def update_with_models(self, models_data: list[dict[str, Any]]) -> None:
        """Populate the routing table from the raw model response.
        This parses the models list and maps labels/recipes to modalities.
        """
        self._downloaded_models = [m for m in models_data if m.get("downloaded", True)]
        
        # Reset mappings
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
                logger.info("Omni model detected: %s", m_id)
                break

        # Map labels to modalities
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
            elif "kokoro" in lower_id or "tts" in lower_id:
                self._models.setdefault("tts", m_id)
            elif "flux" in lower_id or "sd" in lower_id or "stable-diffusion" in lower_id:
                self._models.setdefault("image", m_id)
                self._models.setdefault("edit", m_id)
            elif "vision" in lower_id or "vl" in lower_id or "qwen-vl" in lower_id:
                self._models.setdefault("vision", m_id)
            elif "qwen" in lower_id or "llama" in lower_id or "mistral" in lower_id or "gemma" in lower_id:
                self._models.setdefault("chat", m_id)
                self._models.setdefault("tool-calling", m_id)

        # Print routing table for debugging
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

    # Modality getters
    def llm(self) -> str:
        """Returns the best chat/LLM model id."""
        if self._omni_detected and self._omni_model_id:
            return self._omni_model_id
        return (self._models.get("tool-calling") or 
                self._models.get("chat") or 
                (self._downloaded_models[0]["id"] if self._downloaded_models else ""))

    def vision(self) -> str:
        """Returns the best vision-capable model id."""
        if self._omni_detected and self._omni_model_id:
            return self._omni_model_id
        return self._models.get("vision") or self.llm()

    def asr(self) -> str:
        """Returns the best speech-to-text (ASR) model id."""
        return self._models.get("transcription") or ""

    def tts(self) -> str:
        """Returns the best text-to-speech (TTS) model id."""
        return self._models.get("tts") or ""

    def image(self) -> str:
        """Returns the best image generation model id."""
        return self._models.get("image") or ""

    def edit(self) -> str:
        """Returns the best image editing model id."""
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
