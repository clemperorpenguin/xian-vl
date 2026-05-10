"""Pydantic v2 models for the Xian ecosystem.

These are the canonical data structures exchanged between the xian-vl
engine, Lemonade Server, and every client application.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from shared_types.enums import SourceLanguage, TargetLanguage, TranslationMode


# ── Rendering ────────────────────────────────────────────────────────


class TextStyle(BaseModel):
    """Text styling metadata for overlay rendering."""

    font_family: str = "sans-serif"
    font_size: float = 16.0
    font_weight: str = "normal"
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: Optional[tuple[int, int, int]] = None
    rotation_angle: float = 0.0
    opacity: float = 1.0


# ── Accuracy ─────────────────────────────────────────────────────────


class AccuracyScore(BaseModel):
    """Quality indicator attached to every translation.

    ``score`` ranges from 0.0 (worst) to 1.0 (best).
    ``reason`` explains *why* the score is what it is.
    """

    score: float = Field(ge=0.0, le=1.0)
    reason: str = "full_pass"


# ── Translation ──────────────────────────────────────────────────────


class TranslationRequest(BaseModel):
    """Payload sent by a client to request a translation."""

    image_b64: str = Field(description="Base64-encoded screenshot")
    source_lang: SourceLanguage = SourceLanguage.CHINESE
    target_lang: TargetLanguage = TargetLanguage.ENGLISH
    mode: TranslationMode = TranslationMode.GAME
    styles: list[str] = Field(default_factory=list)
    timeout_seconds: float = 3.0


class TranslationResult(BaseModel):
    """Single translated text region returned by the engine."""

    translated_text: str
    original_text: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    confidence: float = 1.0
    accuracy: AccuracyScore = Field(
        default_factory=lambda: AccuracyScore(score=1.0, reason="full_pass")
    )
    style: Optional[TextStyle] = None
    rotation_angle: float = 0.0


# ── Cinematic Mode ───────────────────────────────────────────────────


class CinematicPayload(BaseModel):
    """Fused image + audio payload for cinematic translation.

    The image is the current subtitle frame; the audio is a short
    system-audio clip captured via PipeWire / PulseAudio.  Both are
    processed in parallel — image via ``/v1/chat/completions`` (vision)
    and audio via ``/v1/audio/transcriptions`` (ASR) — then fused into
    a single translation context.
    """

    image_b64: str
    audio_b64: str
    audio_format: str = "wav"
    subtitles_hint: Optional[str] = None


# ── Chat ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """A single message from the user to the context engine."""

    message: str
    image_b64: Optional[str] = None


class ChatResponse(BaseModel):
    """Reply from the context engine."""

    reply: str
    sources: list[str] = Field(default_factory=list)


# ── Feedback ─────────────────────────────────────────────────────────


class FeedbackPayload(BaseModel):
    """User feedback on a translation result."""

    translation_id: str
    rating: int = Field(ge=1, le=5)
    correction: Optional[str] = None


# ── Visual Grounding ─────────────────────────────────────────────────


class GroundingResult(BaseModel):
    """Bounding-box coordinates returned by the navigation agent.

    Format follows Qwen-VL: ``[ymin, xmin, ymax, xmax]`` normalised
    to 0–1000.
    """

    label: str
    bbox: tuple[int, int, int, int]  # (ymin, xmin, ymax, xmax)


# ── Robobook (Luduan) ────────────────────────────────────────────────


class RobobookChapter(BaseModel):
    """One chapter's worth of translated + narrated content."""

    index: int
    title: str = ""
    original_text: str
    translated_text: str
    audio_offset_ms: int = 0
    audio_duration_ms: int = 0


class RobobookRequest(BaseModel):
    """Request to process a document through the Robobook pipeline."""

    source_path: str
    source_lang: SourceLanguage = SourceLanguage.CHINESE
    target_lang: TargetLanguage = TargetLanguage.ENGLISH
    voice: str = "af_heart"
    generate_audio: bool = True


# ── Lemonade Collections ────────────────────────────────────────────


class CollectionModel(BaseModel):
    """A single model entry within a Xian collection."""

    name: str = Field(description="Lemonade model ID, e.g. Qwen3-4B-Instruct-GGUF")
    labels: list[str] = Field(default_factory=list)
    load_options: dict = Field(default_factory=dict)


# ── Lemonade Health ──────────────────────────────────────────────────


class LemonadeHealth(BaseModel):
    """Mirrors the structure of Lemonade's ``GET /v1/health`` response."""

    status: str = "ok"
    loaded_models: list[str] = Field(default_factory=list)
    version: str = ""
