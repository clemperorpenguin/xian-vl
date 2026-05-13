"""Lightweight workers for on-demand inference and health checks."""

import asyncio
import logging
import os
import urllib.request
import json as _json

from PyQt6.QtCore import QThread, QRect, pyqtSignal

from mage.capture.audio import capture_system_audio
from xian.lemonade_client import LemonadeClient

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Run a single VLM inference off the main thread.

    Accepts image bytes and a target language, calls VLProcessor.process_frame(),
    and emits the list of TranslationResult objects when done.  For chat messages
    it calls VLProcessor.process_chat() and emits the response string.
    """

    # (list_of_TranslationResult, action_string)
    translation_done = pyqtSignal(list, str)
    # (chat_response_text,)
    chat_done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor, *, image_data: bytes = None,
                 source_lang: str = "Chinese", target_lang: str = "English",
                 mode: str = "Game", styles: list[str] = None,
                 action: str = "translate",
                 chat_message: str = "", anchor_rect: QRect = None):
        super().__init__()
        self.processor = processor
        self.image_data = image_data
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.mode = mode
        self.styles = styles or []
        self.action = action
        self.chat_message = chat_message
        self.anchor_rect = anchor_rect or QRect()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Ensure the engine is initialised
            if not self.processor.client:
                loop.run_until_complete(self.processor.init_engine())

            if self.action == "chat":
                response = loop.run_until_complete(
                    self.processor.process_chat(self.chat_message)
                )
                self.chat_done.emit(response)
            else:
                results = loop.run_until_complete(
                    self.processor.process_frame(self.image_data, self.source_lang, self.target_lang, self.mode, self.styles)
                )
                self.translation_done.emit(results, self.action)
        except Exception as e:
            logger.error(f"InferenceWorker error: {e}")
            self.error.emit(str(e))
        finally:
            loop.close()


class CinematicWorker(QThread):
    """Run Cinematic inference handling audio transcription and visual OCR.

    Accepts image bytes, captures audio, transcribes it, and calls VLProcessor.process_cinematic().
    """

    translation_done = pyqtSignal(list, str)
    error = pyqtSignal(str)

    def __init__(self, processor, *, image_data: bytes = None,
                 target_lang: str = "English", styles: list[str] = None,
                 anchor_rect: QRect = None):
        super().__init__()
        self.processor = processor
        self.image_data = image_data
        self.target_lang = target_lang
        self.styles = styles or []
        self.anchor_rect = anchor_rect or QRect()

    async def _run_async(self):
        # 1. Capture 4 seconds of audio (while user is viewing subtitle)
        # Note: Depending on timing, if triggered right as text appears, 4s is usually enough.
        audio_bytes = await capture_system_audio(duration_seconds=4.0)
        
        transcript = ""
        if audio_bytes:
            # 2. Transcribe via Lemonade
            try:
                base_url = os.environ.get("LEMONADE_API_URL", self.processor.config.api_url)
                # LemonadeClient expects base url without /v1 but DEFAULT_API_URL has it.
                base_url_no_v1 = base_url.removesuffix("/v1")
                client = LemonadeClient(base_url=base_url_no_v1)
                transcript = await client.transcribe(audio_bytes)
                await client.close()
            except Exception as e:
                logger.warning(f"Audio transcription failed: {e}")

        # 3. Process vision + transcript
        if not self.processor.client:
            await self.processor.init_engine()

        results = await self.processor.process_cinematic(
            self.image_data, 
            transcript, 
            self.target_lang, 
            self.styles
        )
        return results

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(self._run_async())
            self.translation_done.emit(results, "cinematic")
        except Exception as e:
            logger.error(f"CinematicWorker error: {e}")
            self.error.emit(str(e))
        finally:
            loop.close()


class StatusWorker(QThread):
    """Check Lemonade server availability via HTTP GET /models."""

    # (is_available, list_of_model_ids)
    status_changed = pyqtSignal(bool, list)

    def __init__(self, api_url: str = "http://localhost:13305/v1"):
        super().__init__()
        self.api_url = api_url

    def run(self):
        try:
            base_url = os.environ.get("LEMONADE_API_URL", self.api_url)
            req = urllib.request.Request(f"{base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = _json.loads(resp.read().decode())
                models = [m.get("id", "unknown") for m in body.get("data", [])]
                self.status_changed.emit(True, models if models else ["omni-router"])
        except Exception as e:
            logger.warning(f"Lemonade health check failed: {e}")
            self.status_changed.emit(False, [])


class ModelPullWorker(QThread):
    """Pull (download) a model via the Lemonade POST /v1/pull endpoint."""

    pull_done = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, api_url: str, model_name: str, gpu_memory_utilization: str = "Default"):
        super().__init__()
        self.api_url = api_url
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization

    def run(self):
        try:
            base_url = os.environ.get("LEMONADE_API_URL", self.api_url)
            
            payload_dict = {"model_name": self.model_name}
            if self.gpu_memory_utilization != "Default":
                try:
                    payload_dict["gpu_memory_utilization"] = float(self.gpu_memory_utilization)
                except ValueError:
                    logger.warning(f"Invalid gpu_memory_utilization value: {self.gpu_memory_utilization}")
            
            payload = _json.dumps(payload_dict).encode()
            req = urllib.request.Request(
                f"{base_url}/pull",
                data=payload,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            # Model downloads can be large — use a generous timeout
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = _json.loads(resp.read().decode())
                status = body.get("status", "unknown")
                message = body.get("message", "")
                if status == "success":
                    logger.info(f"Model pull succeeded: {message}")
                    self.pull_done.emit(True, message)
                else:
                    logger.warning(f"Model pull returned status={status}: {message}")
                    self.pull_done.emit(False, message)
        except Exception as e:
            logger.error(f"Model pull failed: {e}")
            self.pull_done.emit(False, str(e))
