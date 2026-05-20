"""Lightweight workers for on-demand inference and health checks."""

import asyncio
import logging
import os

import httpx

from PyQt6.QtCore import QThread, QRect, pyqtSignal

from mage.capture.audio import capture_system_audio
from xian.lemonade_client import LemonadeClient
from xian.timeout import vision_timeout_for_mode, CHAT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Run a single VLM inference off the main thread.

    Accepts image bytes and a target language, calls VLProcessor.process_frame(),
    and emits the list of TranslationResult objects when done.  For chat messages
    it calls VLProcessor.process_chat() and emits the response string.
    """

    # (list_of_TranslationResult, action_string)
    translation_done = pyqtSignal(list, str)
    # (partial_translation_text, action_string)
    translation_partial = pyqtSignal(str, str)
    # emitted as soon as run() starts
    thinking = pyqtSignal()
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
        self.thinking.emit()

        if self.action == "chat":
            future = self.processor.engine.submit(
                self.processor.process_chat(self.chat_message)
            )
            try:
                response = future.result(timeout=vision_timeout_for_mode(self.mode))
                self.chat_done.emit(response)
            except Exception as e:
                logger.error("InferenceWorker chat error: %s", e)
                self.error.emit(str(e))
            return

        engine_loop = self.processor.engine._loop
        if not engine_loop:
            self.error.emit("Engine loop not running")
            return

        import asyncio
        q: asyncio.Queue[str | None] = asyncio.Queue()

        async def _stream_to_queue():
            async for _orig, trans in self.processor.stream_frame(
                self.image_data, self.source_lang, self.target_lang,
                self.mode, self.styles
            ):
                await q.put(trans)
            await q.put(None)

        stream_future = self.processor.engine.submit(_stream_to_queue())

        timeout = vision_timeout_for_mode(self.mode)
        try:
            while True:
                item = asyncio.run_coroutine_threadsafe(
                    q.get(), engine_loop
                ).result(timeout=timeout)
                if item is None:
                    break
                self.translation_partial.emit(item, self.action)

            stream_future.result()

            results = self.processor.last_stream_results
            self.translation_done.emit(results, self.action)

            # Fire-and-forget stats log
            async def _log_stats():
                try:
                     base_url = os.environ.get("LEMONADE_API_URL", self.processor.config.api_url)
                     async with LemonadeClient(base_url=base_url.removesuffix("/v1")) as c:
                         stats = await c.stats()
                         logger.debug("Lemonade stats post-inference: %s", stats)
                except Exception:
                     pass
            self.processor.engine.submit(_log_stats())

        except Exception as e:
            logger.error("InferenceWorker error: %s", e)
            self.error.emit(str(e))


class ContinueWorker(QThread):
    """Continue a truncated VLM generation by replaying partial output."""

    # (accumulated_text,)
    continuation_partial = pyqtSignal(str)
    # (final_text, still_truncated: bool)
    continuation_done = pyqtSignal(str, bool)
    error = pyqtSignal(str)

    def __init__(self, processor, *, messages: list[dict],
                 partial_output: str, mode: str = "Game"):
        super().__init__()
        self.processor = processor
        self.messages = messages
        self.partial_output = partial_output
        self.mode = mode

    def run(self):
        engine_loop = self.processor.engine._loop
        if not engine_loop:
            self.error.emit("Engine loop not running")
            return

        import asyncio
        q: asyncio.Queue[tuple[str, str | None] | None] = asyncio.Queue()

        async def _stream():
            async for text, finish in self.processor.continue_generation(
                self.messages, self.partial_output, self.mode
            ):
                await q.put((text, finish))
            await q.put(None)

        future = self.processor.engine.submit(_stream())

        try:
            last_text = self.partial_output
            last_finish = None
            while True:
                item = asyncio.run_coroutine_threadsafe(
                    q.get(), engine_loop
                ).result(timeout=vision_timeout_for_mode(self.mode))
                if item is None:
                    break
                last_text, last_finish = item
                self.continuation_partial.emit(last_text)

            future.result()
            still_truncated = (last_finish == "length")
            self.continuation_done.emit(last_text, still_truncated)

        except Exception as e:
            msg = str(e)
            if "context" in msg.lower() or "too long" in msg.lower():
                self.error.emit("Context window full — cannot continue further.")
            else:
                logger.error("ContinueWorker error: %s", e)
                self.error.emit(msg)


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
        # 1. Start audio capture task (4 seconds recording)
        audio_task = asyncio.create_task(capture_system_audio(duration_seconds=4.0))

        # 2. Preprocess and encode the image on CPU
        image = self.processor.preprocess_image(self.image_data)
        b64_image = self.processor.encode_image(image)

        # 3. Await audio recording completion
        audio_bytes = await audio_task
        
        transcript = ""
        if audio_bytes:
            # 4. Transcribe via Lemonade
            try:
                base_url = os.environ.get("LEMONADE_API_URL", self.processor.config.api_url)
                base_url_no_v1 = base_url.removesuffix("/v1")
                client = LemonadeClient(base_url=base_url_no_v1)
                transcript = await client.transcribe(audio_bytes)
                await client.close()
            except Exception as e:
                logger.warning("Audio transcription failed: %s", e)

        # 5. Process vision + transcript with precomputed base64/image
        results = await self.processor.process_cinematic(
            self.image_data, 
            transcript, 
            self.target_lang, 
            self.styles,
            b64_image=b64_image,
            image=image
        )
        return results

    def run(self):
        # CinematicWorker can still run its own one-shot loop or submit to the engine loop
        # Submitting to engine loop is safer and reuses client
        future = self.processor.engine.submit(self._run_async())
        try:
            results = future.result(timeout=vision_timeout_for_mode("Document") + 5.0)
            self.translation_done.emit(results, "cinematic")
        except Exception as e:
            logger.error("CinematicWorker error: %s", e)
            self.error.emit(str(e))


class ChatTranslationWorker(QThread):
    """Translate text for in-game chat."""

    translation_done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor, text: str, target_lang: str, source_lang: str):
        super().__init__()
        self.processor = processor
        self.text = text
        self.target_lang = target_lang
        self.source_lang = source_lang

    async def _run_async(self):
        system_prompt = (
            "You are a translation API. You MUST respond with valid JSON ONLY. "
            "Do NOT include markdown formatting, backticks, or any other text outside the JSON object. "
            "The JSON object must have exactly one key: 'translation'."
        )
        user_prompt = f"Translate from {self.target_lang} to {self.source_lang}:\n\n{self.text}"
        
        try:
            response = await asyncio.wait_for(
                self.processor.client.chat.completions.create(
                    model=self.processor.config.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.processor.config.max_tokens,
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                ),
                timeout=CHAT_TIMEOUT_SECONDS,
            )
            
            choice = response.choices[0] if response.choices else None
            final_output = (choice.message.content or "").strip() if choice else ""
            
            import re
            import json
            
            # Strip <think> tags if they exist outside or inside the JSON
            cleaned_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
            cleaned_output = re.sub(r'<think>.*$', '', cleaned_output, flags=re.DOTALL).strip()
            
            # Find the JSON object
            json_match = re.search(r'\{.*?\}', cleaned_output, flags=re.DOTALL)
            
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    if "translation" in data:
                        return data["translation"].strip()
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails but we have some cleaned output, it might just be the raw string
            if cleaned_output and not cleaned_output.startswith("{"):
                return cleaned_output
                
            raise ValueError("Failed to parse translation from model output.")
            
        except Exception as e:
            logger.error("Error during chat translation inference: %s", e)
            raise e

    def run(self):
        future = self.processor.engine.submit(self._run_async())
        try:
            result = future.result(timeout=CHAT_TIMEOUT_SECONDS)
            self.translation_done.emit(result)
        except Exception as e:
            logger.error("ChatTranslationWorker error: %s", e)
            self.error.emit(str(e))


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
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base_url}/models")
                resp.raise_for_status()
                body = resp.json()
                models = [m.get("id", "unknown") for m in body.get("data", [])]
                self.status_changed.emit(True, models if models else ["omni-router"])
        except Exception as e:
            logger.warning("Lemonade health check failed: %s", e)
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
            payload: dict = {"model": self.model_name, "stream": True}
            if self.gpu_memory_utilization != "Default":
                try:
                    payload["gpu_memory_utilization"] = float(self.gpu_memory_utilization)
                except ValueError:
                    logger.warning("Invalid gpu_memory_utilization value: %s", self.gpu_memory_utilization)

            with httpx.Client(timeout=600.0) as client:
                resp = client.post(f"{base_url}/pull", json=payload)
                resp.raise_for_status()
                body = resp.json()
                status = body.get("status", "unknown")
                message = body.get("message", "")
                if status == "success":
                    logger.info("Model pull succeeded: %s", message)
                    self.pull_done.emit(True, message)
                else:
                    logger.warning("Model pull returned status=%s: %s", status, message)
                    self.pull_done.emit(False, message)
        except Exception as e:
            logger.error("Model pull failed: %s", e)
            self.pull_done.emit(False, str(e))


class PrewarmWorker(QThread):
    """Pre-load the model into Lemonade Server VRAM off the main thread."""

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def run(self):
        if hasattr(self.processor, "prewarm_model"):
            future = self.processor.engine.submit(self.processor.prewarm_model())
            try:
                future.result(timeout=60)
            except Exception as e:
                logger.warning("PrewarmWorker error: %s", e)
