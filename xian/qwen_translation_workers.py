"""Qwen3-VL Translation Workers for handling translation tasks."""

import time
import logging
from collections import OrderedDict
from typing import List
import asyncio
import imagehash
from PIL import Image
import io

from PyQt6.QtCore import QThread, pyqtSignal, QRect, QElapsedTimer
from PyQt6.QtGui import QImage, QPainter, QColor, QGuiApplication
from PyQt6.QtWidgets import QThreadPool, QRunnable

from .models import TranslationMode, TranslationRegion, TranslationResult
from .screen_capture import ScreenCapture
from .qwen_pipeline import QwenVLProcessor
from .translation_db import TranslationDB

logger = logging.getLogger(__name__)


class QwenTranslationWorker(QThread):
    """Worker thread for handling translations using Qwen3-VL"""

    translation_ready = pyqtSignal(list, object)  # list of results, optional QRect of the updated area
    status_update = pyqtSignal(str)  # Status message for the UI
    request_hide_overlay = pyqtSignal()
    request_show_overlay = pyqtSignal()

    def __init__(self, qwen_processor: QwenVLProcessor):
        super().__init__()
        self.qwen_processor = qwen_processor
        self.running = False
        self.mode = TranslationMode.FULL_SCREEN
        self.regions = []
        self.source_lang = "auto"
        self.target_lang = "English"
        self.interval = 2000  # ms
        self.redaction_margin = 15  # Default margin for redaction
        self.last_hashes = {}  # Map of region key or "full" to last hash
        self.active_geometries = []  # Current bubble geometries for redaction
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # Sane number of threads
        # Caches to reduce refresh churn
        self.image_cache = OrderedDict()  # image_hash -> {"translations": list}
        self.image_cache_max = 8
        self.translation_cache = OrderedDict()  # fingerprint -> translations
        self.translation_cache_max = 16
        self._last_translation_signature = None
        self._empty_signature = ("__empty__",)
        
        # Initialize perceptual cache
        self.perceptual_cache = {}  # dhash -> translation result
        self.translation_db = TranslationDB("./translation_cache.lmdb")

    def set_active_geometries(self, geometries: List[QRect]):
        """Set geometries to be redacted from the capture"""
        self.active_geometries = geometries

    def set_config(self, mode: TranslationMode, regions: List[TranslationRegion],
                   source_lang: str, target_lang: str, interval: int, redaction_margin: int = 15):
        self.mode = mode
        self.regions = regions
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.interval = interval
        self.redaction_margin = redaction_margin

    def start_translation(self):
        self.running = True
        self.start()

    def stop_translation(self):
        """Stop translation process"""
        self.running = False
        self.thread_pool.clear()  # Cancel pending tasks
        self.quit()
        # Non-blocking wait if called from main thread to prevent UI lag
        if QThread.currentThread() == self.thread():
            self.wait(1000)  # Short timeout
        else:
            self.wait()
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'translation_db'):
            self.translation_db.close()

    def clear_hashes(self):
        """Clear the image hashes and translation cache to force re-translation"""
        self.last_hashes = {}
        self.image_cache.clear()
        self.translation_cache.clear()
        self._last_translation_signature = None

    def run(self):
        """Main worker loop"""
        from PyQt6.QtCore import QElapsedTimer
        timer = QElapsedTimer()

        while self.running:
            timer.start()
            # Request latest geometries for redaction
            self.request_hide_overlay.emit()

            try:
                # Use Qwen3-VL for all modes
                self._translate_with_qwen()

                # Calculate remaining sleep time
                elapsed = timer.elapsed()
                # Use 1 second interval as requested
                target_interval = 1000
                remaining = target_interval - elapsed

                if remaining > 0:
                    self.msleep(int(remaining))
                else:
                    self.msleep(1)  # Minimal sleep to prevent CPU hogging

            except Exception as e:
                logger.error(f"Translation worker error: {e}")
                self.msleep(1000)

        logger.info("Qwen translation worker thread stopped")

    def _translate_with_qwen(self):
        """Capture screen, perform OCR and translation with Qwen3-VL"""
        workflow_start = time.time()

        self.status_update.emit("Capturing screen...")
        capture_start = time.time()
        image_data = ScreenCapture.capture_screen()
        if not image_data or not self.running:
            return
        capture_time = time.time() - capture_start

        # Redact existing translations to avoid translating them again
        redact_time = 0
        if self.active_geometries:
            redact_start = time.time()
            image = QImage.fromData(image_data)
            if not image.isNull():
                # Geometries are in global screen coordinates.
                # Capture is assumed to be the full virtual desktop.
                capture_geo = ScreenCapture.get_virtual_desktop_geometry()
                image = self._redact_image(image, self.active_geometries, capture_geo.topLeft())

                from PyQt6.QtCore import QBuffer, QIODevice
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                image.save(buffer, "PNG")
                image_data = bytes(buffer.buffer())
            redact_time = time.time() - redact_start

        # Preprocess image for better results
        preprocess_start = time.time()
        image_data = ScreenCapture.preprocess_image(image_data)
        preprocess_time = time.time() - preprocess_start

        # Calculate dHash for perceptual caching
        hash_start = time.time()
        pil_image = Image.open(io.BytesIO(image_data))
        dhash = str(imagehash.dhash(pil_image))
        hash_time = time.time() - hash_start

        # Check perceptual cache first (L0 cache)
        if dhash in self.perceptual_cache:
            logger.debug("Perceptual cache hit; reusing cached translation")
            cached_result = self.perceptual_cache[dhash]
            self.translation_ready.emit(cached_result, None)
            self.status_update.emit("Using cached translation (dHash)")
            return

        # Check database cache (L1 cache)
        db_cached = self.translation_db.get_translation(dhash)
        if db_cached:
            logger.debug("Database cache hit; reusing cached translation")
            # Convert stored data back to TranslationResult objects
            cached_results = [TranslationResult(**item) for item in db_cached]
            self.perceptual_cache[dhash] = cached_results  # Also add to in-memory cache
            self.translation_ready.emit(cached_results, None)
            self.status_update.emit("Using cached translation (DB)")
            return

        # Use image hash to avoid redundant processing if nothing changed
        image_hash = ScreenCapture.calculate_hash(image_data)
        cached_entry = self.image_cache.get(image_hash)

        if cached_entry and cached_entry.get("translations"):
            logger.debug("Image cache hit with translations; reusing previous results")
            self.last_hashes["full"] = image_hash
            signature = self._fingerprint_translations(cached_entry["translations"])
            if signature == self._last_translation_signature:
                logger.debug("Translation signature unchanged; suppressing overlay refresh")
                return
            self._last_translation_signature = signature
            self.translation_ready.emit(cached_entry["translations"], None)
            self.status_update.emit("Using cached translations")
            return

        if image_hash == self.last_hashes.get("full"):
            logger.debug("Image hash unchanged, skipping processing")
            return
        self.last_hashes["full"] = image_hash

        self.status_update.emit("Processing with vision-language model...")
        vl_start = time.time()
        try:
            # Process the frame using vision-language model
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                translated_results = loop.run_until_complete(
                    self.qwen_processor.process_frame(image_data, self.target_lang)
                )
            finally:
                loop.close()
            
            vl_time = time.time() - vl_start

            if not translated_results:
                logger.info(f"Vision-language model finished in {vl_time:.2f}s: No text detected")
                self.status_update.emit("No text detected")

                # Keep existing translations on screen; just record the empty state
                # to suppress redundant refreshes until something changes.
                self._store_image_cache(image_hash, translations=[])
                self._last_translation_signature = self._empty_signature
                return

            logger.info(f"Vision-language model processed image in {vl_time:.2f}s, got {len(translated_results)} results")

            workflow_total = time.time() - workflow_start
            logger.info(f"Workflow stats: Capture: {capture_time:.2f}s, Redact: {redact_time:.2f}s, "
                        f"Preprocess: {preprocess_time:.2f}s, Hash: {hash_time:.2f}s, "
                        f"VL-Model: {vl_time:.2f}s, Total: {workflow_total:.2f}s")

            if translated_results:
                # Store in both caches
                self._store_image_cache(image_hash, translations=translated_results)
                
                # Store in perceptual cache
                self.perceptual_cache[dhash] = translated_results
                
                # Store in database cache
                db_data = [result.__dict__ for result in translated_results]
                self.translation_db.put_translation(dhash, db_data)

                signature = self._fingerprint_translations(translated_results)
                if signature == self._last_translation_signature:
                    logger.debug("Translation signature unchanged after translate; suppressing overlay refresh")
                    return
                self._last_translation_signature = signature

                self.translation_ready.emit(translated_results, None)
                self.status_update.emit("Translation complete")
            else:
                self.status_update.emit("Translation failed")

        except Exception as e:
            logger.error(f"Vision-language model processing error: {e}")
            self.status_update.emit(f"VL Model Error: {e}")
            return

    def _redact_image(self, image: QImage, geometries: List[QRect], offset: 'QPoint' = None) -> QImage:
        """Draw black boxes over existing translation areas"""
        if not geometries:
            return image

        # Ensure image is in a format we can paint on
        if image.format() == QImage.Format.Format_Invalid:
            return image

        from PyQt6.QtCore import QPoint
        if offset is None:
            offset = QPoint(0, 0)

        redacted = image.copy()
        painter = QPainter(redacted)
        painter.setBrush(QColor(0, 0, 0))
        painter.setPen(QColor(0, 0, 0))  # Solid black pen

        for rect in geometries:
            # Adjust rect by offset (for regions, rect is in screen coords)
            adj_rect = rect.translated(-offset)
            # Draw box to ensure text is fully covered
            margin = self.redaction_margin
            adj_rect.adjust(-margin, -margin, margin, margin)
            painter.drawRect(adj_rect)

        painter.end()
        return redacted

    def _store_image_cache(self, image_hash: str, translations=None):
        entry = self.image_cache.get(image_hash, {})
        if translations is not None:
            entry["translations"] = translations
        self.image_cache[image_hash] = entry
        try:
            self.image_cache.move_to_end(image_hash)
        except Exception:
            pass
        evicted = []
        while len(self.image_cache) > self.image_cache_max:
            try:
                k, _ = self.image_cache.popitem(last=False)
                evicted.append(k)
            except Exception:
                break

        if evicted:
            logger.debug("Image cache evicted %d item(s); max=%d", len(evicted), self.image_cache_max)
        else:
            logger.debug(
                "Image cache stored hash=%s (translations=%s, size=%d/%d)",
                image_hash,
                "yes" if translations is not None else "no",
                len(self.image_cache),
                self.image_cache_max,
            )

    def _fingerprint_translations(self, translations: List[TranslationResult]):
        try:
            return tuple(sorted(
                (
                    int(round(t.x)),
                    int(round(t.y)),
                    int(round(t.width)),
                    int(round(t.height)),
                    t.translated_text.strip()
                )
                for t in translations
            ))
        except Exception:
            return tuple()


class QwenTranslatorStatusWorker(QThread):
    """Worker thread for checking Qwen processor status"""
    status_changed = pyqtSignal(bool, list)

    def __init__(self, qwen_processor: QwenVLProcessor):
        super().__init__()
        self.qwen_processor = qwen_processor

    def run(self):
        # For Qwen3-VL, we can check if the engine is initialized
        # For now, just return True to indicate the processor is available
        is_available = True  # Placeholder - actual implementation would check if model is loaded
        models = ["Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen3-VL-4B-Thinking", "Qwen3-VL-8B-Thinking"]
        self.status_changed.emit(is_available, models)


class QwenModelWarmupWorker(QThread):
    """Worker thread to initialize the Qwen3-VL model before translation starts."""

    warmup_finished = pyqtSignal(bool, str)

    def __init__(self, qwen_processor: QwenVLProcessor):
        super().__init__()
        self.qwen_processor = qwen_processor

    def run(self):
        try:
            # Initialize the engine
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.qwen_processor.init_engine())
                ok = True
                err = ""
            except Exception as e:
                ok = False
                err = str(e)
            finally:
                loop.close()
                
            self.warmup_finished.emit(ok, err)
        except Exception as e:
            logger.error(f"Qwen3-VL model warmup error: {e}")
            self.warmup_finished.emit(False, str(e))