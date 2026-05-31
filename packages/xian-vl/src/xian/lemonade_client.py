"""Async wrapper for Lemonade-specific (non-OpenAI) API endpoints.

The ``openai`` SDK covers ``/v1/chat/completions`` and friends, but
Lemonade exposes several proprietary endpoints for model lifecycle
management, health checks, and multimodal tools (TTS, ASR) that the
SDK does not reach.  This client fills that gap.

All methods are ``async`` and use ``httpx.AsyncClient`` under the hood.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from shared_types.constants import DEFAULT_API_URL

logger = logging.getLogger(__name__)


class LemonadeClient:
    """Thin async client for Lemonade Server's proprietary API.

    Parameters
    ----------
    base_url:
        Lemonade Server base URL, *without* the ``/v1`` suffix.
        Default: ``http://localhost:13305``.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL.removesuffix("/v1"),
        timeout: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base,
            timeout=timeout,
        )

    async def __aenter__(self) -> LemonadeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Shut down the underlying HTTP transport."""
        await self._client.aclose()

    # ── Model Lifecycle ──────────────────────────────────────────────

    async def pull_model(
        self,
        name: str,
        *,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Download a model.  ``POST /v1/pull``."""
        resp = await self._client.post(
            "/v1/pull",
            json={"model": name, "stream": stream},
        )
        resp.raise_for_status()
        return resp.json()

    async def load_model(
        self,
        name: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load a model into memory.  ``POST /v1/load``."""
        payload: dict[str, Any] = {"model": name}
        if options:
            payload.update(options)
        resp = await self._client.post("/v1/load", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def unload_model(self, name: str) -> dict[str, Any]:
        """Unload a model from memory.  ``POST /v1/unload``."""
        resp = await self._client.post("/v1/unload", json={"model": name})
        resp.raise_for_status()
        return resp.json()

    # ── Health & Diagnostics ─────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Server health.  ``GET /v1/health``."""
        resp = await self._client.get("/v1/health")
        resp.raise_for_status()
        return resp.json()

    async def stats(self) -> dict[str, Any]:
        """Inference statistics.  ``GET /v1/stats``."""
        resp = await self._client.get("/v1/stats")
        resp.raise_for_status()
        return resp.json()

    async def system_info(self) -> dict[str, Any]:
        """Hardware / backend info.  ``GET /v1/system-info``."""
        resp = await self._client.get("/v1/system-info")
        resp.raise_for_status()
        return resp.json()

    async def list_models(self, *, show_all: bool = False) -> list[dict[str, Any]]:
        """List available models.  ``GET /v1/models``."""
        params = {"show_all": "true"} if show_all else {}
        resp = await self._client.get("/v1/models", params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])

    # ── Multimodal Tools (OmniRouter) ────────────────────────────────

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str = "zh",
        model: str = "",
    ) -> str:
        """Speech-to-text.  ``POST /v1/audio/transcriptions``.

        Returns the transcription text.
        """
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data: dict[str, str] = {"language": language}
        if model:
            data["model"] = model
        resp = await self._client.post(
            "/v1/audio/transcriptions",
            files=files,
            data=data,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    async def tts(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        model: str = "",
    ) -> bytes:
        """Text-to-speech.  ``POST /v1/audio/speech``.

        Returns raw audio bytes (WAV).
        """
        payload: dict[str, str] = {"input": text, "voice": voice}
        if model:
            payload["model"] = model
        resp = await self._client.post("/v1/audio/speech", json=payload)
        resp.raise_for_status()
        return resp.content

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str = "",
        size: str = "1024x1024",
        response_format: str = "b64_json",
    ) -> dict[str, Any]:
        """Generate an image from prompt.  ``POST /v1/images/generations``."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
        }
        if model:
            payload["model"] = model
        resp = await self._client.post("/v1/images/generations", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mask_bytes: bytes | None = None,
        model: str = "",
        size: str = "1024x1024",
        response_format: str = "b64_json",
    ) -> dict[str, Any]:
        """Edit an existing image.  ``POST /v1/images/edits``."""
        files = {"image": ("image.png", image_bytes, "image/png")}
        if mask_bytes:
            files["mask"] = ("mask.png", mask_bytes, "image/png")
        
        data = {
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
        }
        if model:
            data["model"] = model

        resp = await self._client.post(
            "/v1/images/edits",
            files=files,
            data=data,
        )
        resp.raise_for_status()
        return resp.json()

