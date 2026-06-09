# MAGE — Gaming HUD for real-time screen translation.
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

"""Lightweight workers for on-demand inference and health checks."""

import asyncio
import logging
import os
import re
import json

import httpx

from PyQt6.QtCore import QThread, QRect, pyqtSignal

from mage.capture.audio import capture_system_audio
from xian.lemonade_client import LemonadeClient
from xian.timeout import vision_timeout_for_mode, CHAT_TIMEOUT_SECONDS, CHAT_AUX_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def extract_translation_json(raw_output: str) -> str:
    # Strip <think> tags if they exist outside or inside the JSON
    cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    cleaned_output = re.sub(r'<think>.*$', '', cleaned_output, flags=re.DOTALL).strip()

    # Try to find the outer-most JSON object or array
    start_idx = cleaned_output.find('{')
    end_idx = cleaned_output.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = cleaned_output[start_idx:end_idx+1]
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict) and "translation" in data:
                return data["translation"].strip()
        except json.JSONDecodeError:
            pass

    # Fallback to searching all { ... } blocks using raw_decode
    for match in re.finditer(r'\{', cleaned_output):
        idx = match.start()
        try:
            data, _ = json.JSONDecoder().raw_decode(cleaned_output, idx)
            if isinstance(data, dict) and "translation" in data:
                return data["translation"].strip()
        except json.JSONDecodeError:
            continue

    # Fallback: if it's not a JSON object but just raw text, return cleaned_output
    if cleaned_output and not cleaned_output.startswith("{"):
        return cleaned_output

    return ""



def sanitize_markdown(text: str) -> str:
    """Escape backticks and HTML tags to prevent markdown injection."""
    return text.replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")

class InferenceWorker(QThread):
    """Run a single VLM inference off the main thread.

    Accepts image bytes and a target language, calls VLProcessor.process_frame(),
    and emits the list of TranslationResult objects when done.  For chat messages
    it calls VLProcessor.process_chat() and emits the response string.
    """

    translation_done = pyqtSignal(list, str)
    translation_partial = pyqtSignal(str, str)
    thinking = pyqtSignal()
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
                # Poll the future and check for interruption
                while not future.done():
                    if self.isInterruptionRequested():
                        future.cancel()
                        logger.info("InferenceWorker chat cancelled by user")
                        return
                    self.msleep(100)
                response = future.result()
                self.chat_done.emit(response)
            except Exception as e:
                logger.error("InferenceWorker chat error: %s", e)
                self.error.emit(str(e))
            return

        engine_loop = self.processor.engine.loop
        if not engine_loop:
            self.error.emit("Engine loop not running")
            return

        q: asyncio.Queue = asyncio.Queue()

        async def _stream_to_queue():
            final_data = None
            async for _orig, trans, results_data in self.processor.stream_frame(
                self.image_data, self.source_lang, self.target_lang,
                self.mode, self.styles
            ):
                if results_data is not None:
                    final_data = results_data
                else:
                    await q.put(trans)
            await q.put((None, final_data))

        stream_future = self.processor.engine.submit(_stream_to_queue())

        timeout = vision_timeout_for_mode(self.mode)
        try:
            final_data = None
            while True:
                if self.isInterruptionRequested():
                    stream_future.cancel()
                    logger.info("InferenceWorker stream cancelled by user")
                    break
                try:
                    item = asyncio.run_coroutine_threadsafe(
                        q.get(), engine_loop
                    ).result(timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if isinstance(item, tuple) and item[0] is None:
                    final_data = item[1]
                    break
                self.translation_partial.emit(item, self.action)

            if not self.isInterruptionRequested():
                stream_future.result()

            if final_data is not None:
                results, messages, accumulated = final_data
                self.continuation_messages = messages
                self.continuation_context_partial = accumulated
            else:
                results = []

            if not self.isInterruptionRequested():
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

    continuation_partial = pyqtSignal(str)
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
        engine_loop = self.processor.engine.loop
        if not engine_loop:
            self.error.emit("Engine loop not running")
            return

        q: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        async def _stream():
            final_data = None
            async for text, finish, continuation_data in self.processor.continue_generation(
                self.messages, self.partial_output, self.mode
            ):
                if continuation_data is not None:
                    final_data = continuation_data
                else:
                    await q.put((text, finish))
            await q.put((_SENTINEL, final_data))

        future = self.processor.engine.submit(_stream())

        try:
            last_text = self.partial_output
            last_finish = None
            final_data = None
            while True:
                item = asyncio.run_coroutine_threadsafe(
                    q.get(), engine_loop
                ).result(timeout=vision_timeout_for_mode(self.mode))
                if isinstance(item, tuple) and item[0] is _SENTINEL:
                    final_data = item[1]
                    break
                last_text, last_finish = item
                self.continuation_partial.emit(last_text)

            future.result()

            if final_data is not None:
                self.continuation_messages = final_data[0]
                self.continuation_context_partial = final_data[1]

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
                 anchor_rect: QRect = None, source_lang: str = "Chinese"):
        super().__init__()
        self.processor = processor
        self.image_data = image_data
        self.target_lang = target_lang
        self.styles = styles or []
        self.anchor_rect = anchor_rect or QRect()
        self.source_lang = source_lang

    async def _run_async(self):
        # 1. Start audio capture task (4 seconds recording)
        audio_task = asyncio.create_task(capture_system_audio(duration_seconds=4.0))

        # 2. Preprocess and encode the image on CPU
        image = self.processor.preprocess_image(self.image_data)
        b64_image = self.processor.encode_image(image)

        # 3. Await audio recording completion
        audio_bytes = await audio_task
        self.audio_bytes = audio_bytes
        
        transcript = ""
        if audio_bytes:
            # 4. Transcribe via Lemonade
            try:
                base_url = os.environ.get("LEMONADE_API_URL", self.processor.config.api_url)
                base_url_no_v1 = base_url.removesuffix("/v1")
                active_model = self.processor.config.model_name
                if not self.processor.router.asr(active_model):
                    await self.processor.router.discover_async()
                asr_model = self.processor.router.asr(active_model)
                async with LemonadeClient(base_url=base_url_no_v1) as client:
                    transcript = await client.transcribe(audio_bytes, model=asr_model)
            except Exception as e:
                logger.warning("Audio transcription failed: %s", e)

        # 5. Process vision + transcript with precomputed base64/image
        results = await self.processor.process_cinematic(
            self.image_data, 
            transcript, 
            self.target_lang, 
            self.styles,
            b64_image=b64_image,
            image=image,
            source_lang=self.source_lang,
            audio_bytes=audio_bytes
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
                    model=self.processor.get_model_name(),
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
            
            translation = extract_translation_json(final_output)
            if translation:
                return translation
                
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


class RaidWorker(QThread):
    """Run Raid Mode handling audio capture, transcription, translation, and TTS voice cloning.

    No screen/image processing is involved.
    """

    chunk_translated = pyqtSignal(str, str, bytes)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, processor, *, target_lang: str = "English", source_lang: str = "Chinese", save_lore: bool = False, audio_enabled: bool = False):
        super().__init__()
        self.processor = processor
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.save_lore = save_lore
        self._audio_enabled = audio_enabled
        self._running = True

    def stop(self):
        self._running = False

    def set_audio_enabled(self, enabled: bool):
        self._audio_enabled = enabled
        logger.info("RaidWorker: Audio output set to %s", enabled)

    async def _run_async(self):
        from mage.capture.audio import ContinuousAudioStreamer
        
        self.progress.emit("Initializing live audio capture...")
        streamer = ContinuousAudioStreamer()
        await streamer.start()

        base_url = os.environ.get("LEMONADE_API_URL", self.processor.config.api_url)
        base_url_no_v1 = base_url.removesuffix("/v1")
        
        if self.save_lore:
            try:
                import datetime
                import pathlib
                lore_dir = pathlib.Path.home() / ".local" / "share" / "xian-vl" / "lore"
                lore_dir.mkdir(parents=True, exist_ok=True)
                self.lore_filepath = lore_dir / f"Raid_Log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                
                def _init_lore():
                    with open(self.lore_filepath, "w", encoding="utf-8") as f:
                        f.write(f"# Raid Log ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n\n")
                
                await asyncio.to_thread(_init_lore)
            except Exception as e:
                logger.error("Failed to initialize Raid Log file: %s", e)
                self.save_lore = False

        try:
            active_model = self.processor.config.model_name
            if not self.processor.router.asr(active_model):
                await self.processor.router.discover_async()
            asr_model = self.processor.router.asr(active_model)
            if not asr_model:
                raise ValueError("No transcription model available on the server.")

            self.progress.emit("Listening for speech...")

            async with LemonadeClient(base_url=base_url_no_v1) as client:
                async for wav_chunk in streamer.read_chunks():
                    if not self._running:
                        break

                    try:
                        self.progress.emit("Transcribing audio...")
                        transcript = await asyncio.wait_for(
                            client.transcribe(
                                wav_chunk,
                                language="zh" if "chin" in self.source_lang.lower() else "en",
                                model=asr_model
                            ),
                            timeout=15.0
                        )
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("Raid Mode transcription failed or timed out: %s", e)
                        err_msg = str(e).lower()
                        if "500" in err_msg or "internal server error" in err_msg or "model_load_error" in err_msg:
                            self.chunk_translated.emit(
                                "[ASR Server Error]",
                                "⚠️ ASR Server Error (500): Lemonade failed to load/start the speech-to-text model on the server. Live speech translation is disabled due to server-side backend limitations. (Ref: https://github.com/lemonade-sdk/lemonade/issues/2083)",
                                b""
                            )
                            self._running = False
                            self.progress.emit("ASR server error (500). Raid mode stopped.")
                            break
                        else:
                            self.progress.emit(f"Transcription failed ({e}), listening...")
                            continue

                    if not transcript or not transcript.strip():
                        self.progress.emit("Listening for speech...")
                        continue

                    self.progress.emit("Translating text...")
                    system_prompt = (
                        "You are a translation API. You MUST respond with valid JSON ONLY. "
                        "Do NOT include markdown formatting, backticks, or any other text outside the JSON object. "
                        "The JSON object must have exactly one key: 'translation'."
                    )
                    user_prompt = f"Translate from {self.source_lang} to {self.target_lang}:\n\n{transcript}"
                    
                    try:
                        response = await asyncio.wait_for(
                            self.processor.client.chat.completions.create(
                                model=self.processor.get_model_name(),
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                max_tokens=self.processor.config.max_tokens,
                                temperature=0.1,
                                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                            ),
                            timeout=CHAT_AUX_TIMEOUT_SECONDS,
                        )
                        choice = response.choices[0] if response.choices else None
                        final_output = (choice.message.content or "").strip() if choice else ""
                    except (asyncio.TimeoutError, Exception) as e:
                        logger.warning("Raid Mode translation failed or timed out: %s", e)
                        self.progress.emit(f"Translation failed ({e}), listening...")
                        continue
                    
                    translation = extract_translation_json(final_output)
                    if not translation:
                        logger.warning("Failed to parse translation from model output.")
                        translation = "[Translation Failed]"

                    # Sequential background TTS synthesis if audio is enabled
                    audio_bytes = b""
                    if getattr(self, "_audio_enabled", False):
                        try:
                            self.progress.emit("Synthesizing audio...")
                            tts_model = self.processor.router.tts(active_model)
                            if tts_model:
                                voice_param = "af_heart"
                                if self.target_lang == "Chinese":
                                    voice_param = "zf_xiaoxiao"
                                elif self.target_lang == "Japanese":
                                    voice_param = "jf_alpha"
                                
                                audio_bytes = await asyncio.wait_for(
                                    client.tts(translation, voice=voice_param, model=tts_model),
                                    timeout=15.0
                                )
                        except Exception as tts_err:
                            logger.error("Raid Mode sequential TTS synthesis failed: %s", tts_err)
                            
                    self.chunk_translated.emit(transcript, translation, audio_bytes)
                    if self.save_lore and hasattr(self, "lore_filepath"):
                        try:
                            def _append_lore():
                                with open(self.lore_filepath, "a", encoding="utf-8") as f:
                                    f.write(f"**Source**: {sanitize_markdown(transcript)}\n\n**Translation**: {sanitize_markdown(translation)}\n\n---\n")
                            await asyncio.to_thread(_append_lore)
                        except Exception as e:
                            logger.error("Failed to append to Raid Log file: %s", e)

                    self.progress.emit("Listening for speech...")

        finally:
            await streamer.stop()

    def run(self):
        future = self.processor.engine.submit(self._run_async())
        try:
            future.result()
        except Exception as e:
            logger.error("RaidWorker error: %s", e)
            self.error.emit(str(e))


class StatusWorker(QThread):
    """Check Lemonade server availability via HTTP GET /models."""

    status_changed = pyqtSignal(bool, list, list)

    def __init__(self, api_url: str = "http://localhost:13305/v1"):
        super().__init__()
        self.api_url = api_url

    def run(self):
        try:
            base_url = os.environ.get("LEMONADE_API_URL", self.api_url)
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base_url}/models", params={"show_all": "true"})
                resp.raise_for_status()
                body = resp.json()
                data = body.get("data", [])
                models = [
                    m.get("id", "unknown")
                    for m in data
                    if m.get("downloaded", True)
                    and (m.get("recipe") == "collection.omni" or m.get("id", "").startswith("LMX-Omni-"))
                ]
                self.status_changed.emit(True, models if models else ["omni-router"], data)
        except Exception as e:
            logger.warning("Lemonade health check failed: %s", e)
            self.status_changed.emit(False, [], [])


class ModelPullWorker(QThread):
    """Pull (download) a model via the Lemonade POST /v1/pull endpoint."""

    pull_done = pyqtSignal(bool, str)

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
    
    status_changed = pyqtSignal(str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def run(self):
        if hasattr(self.processor, "prewarm_model"):
            target_model = getattr(self.processor.config, "model_name", "default")
            self.status_changed.emit(f"Pre-warming/downloading model '{target_model}' on Lemonade server...")
            
            future = self.processor.engine.submit(self.processor.prewarm_model())
            try:
                future.result(timeout=600)
                self.status_changed.emit(f"Model '{target_model}' is active and ready.")
            except Exception as e:
                logger.warning("PrewarmWorker error: %s", e)
                self.status_changed.emit(f"Failed to pre-warm model: {e}")


class CaptureWorker(QThread):
    """Asynchronously captures, crops, composites, and encodes screenshots."""
    capture_done = pyqtSignal(bytes, QRect)
    error = pyqtSignal(str)

    def __init__(self, mode: str, rects: list[QRect], total_geo: QRect):
        super().__init__()
        self.mode = mode  # "dialogue" or "cinematic"
        self.rects = rects
        self.total_geo = total_geo

    def run(self):
        from mage.capture.screen import ScreenCapture
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtCore import QBuffer, QIODevice

        try:
            data = ScreenCapture.capture_screen()
            if not data:
                self.error.emit("Failed to capture screen")
                return

            img = QImage.fromData(data)
            if img.isNull():
                self.error.emit("Failed to load screen capture image")
                return

            if self.mode == "dialogue":
                rect = self.rects[0]
                safe_rect = rect.translated(-self.total_geo.left(), -self.total_geo.top())
                safe_rect = safe_rect.intersected(img.rect())
                cropped = img.copy(safe_rect)
                
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                cropped.save(buf, "JPG", 85)
                cropped_data = bytes(buf.buffer())
                
                self.capture_done.emit(cropped_data, rect)

            elif self.mode == "cinematic":
                total_height = sum(r.height() for r in self.rects)
                max_width = max(r.width() for r in self.rects)

                composite = QImage(max_width, total_height, QImage.Format.Format_RGB32)
                composite.fill(0)  # fills with black

                painter = QPainter(composite)
                y_offset = 0
                for r in self.rects:
                    safe_rect = r.translated(-self.total_geo.left(), -self.total_geo.top())
                    safe_rect = safe_rect.intersected(img.rect())
                    cropped = img.copy(safe_rect)
                    painter.drawImage(0, y_offset, cropped)
                    y_offset += r.height()
                painter.end()

                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                composite.save(buf, "JPG", 85)
                composite_data = bytes(buf.buffer())

                anchor_rect = self.rects[-1] if self.rects else QRect()
                self.capture_done.emit(composite_data, anchor_rect)
        except Exception as e:
            logger.error("CaptureWorker error: %s", e)
            self.error.emit(str(e))

