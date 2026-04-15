"""Vision-Language Translation Workers for handling translation tasks."""

import time
import logging
from collections import OrderedDict
from typing import List, Optional, Tuple
import asyncio
import imagehash
from PIL import Image
import io

from PyQt6.QtCore import QThread, pyqtSignal, QRect, QElapsedTimer, QThreadPool, QRunnable
from PyQt6.QtGui import QImage, QPainter, QColor, QGuiApplication

from .models import TranslationMode, TranslationRegion, TranslationResult
from .screen_capture import ScreenCapture
from .qwen_pipeline import QwenVLProcessor
from .translation_db import TranslationDB
from . import constants

logger = logging.getLogger(__name__)


class QwenTranslationWorker(QThread):
    """Worker thread for handling translations using vision-language models"""

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
        self.image_cache_max = constants.IMAGE_CACHE_MAX_SIZE
        self.translation_cache = OrderedDict()  # fingerprint -> translations
        self.translation_cache_max = constants.TRANSLATION_CACHE_MAX_SIZE
        self._last_translation_signature = None
        self._empty_signature = ("__empty__",)
        
        # Initialize perceptual cache
        self.perceptual_cache = {}  # dhash -> translation result
        self.translation_db = TranslationDB("./translation_cache.lmdb")

    def set_active_geometries(self, geometries: List[QRect]):
        """Set geometries to be redacted from the capture"""
        self.active_geometries = geometries

    def set_config(self, mode: TranslationMode, regions: List[TranslationRegion],
                   source_lang: str, target_lang: str, interval: int, redaction_margin: int = constants.REDACTION_DEFAULT_MARGIN):
        """Update the worker configuration."""
        self.mode = mode
        # Filter out disabled regions
        self.regions = [r for r in regions if r.enabled]
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
        logger.info(f"Worker run started, interval={self.interval}ms")

        while self.running:
            timer.start()
            # Request latest geometries for redaction
            self.request_hide_overlay.emit()

            try:
                # Use vision-language model for all modes
                self._translate_with_qwen()

                # Calculate remaining sleep time
                elapsed = timer.elapsed()
                target_interval = self.interval  # Use configured interval
                remaining = target_interval - elapsed

                logger.debug(f"Worker iteration: {elapsed}ms elapsed, {remaining}ms remaining")

                if remaining > 0:
                    self.msleep(int(remaining))
                else:
                    self.msleep(constants.WORKFLOW_MIN_SLEEP_MS)  # Minimal sleep to prevent CPU hogging

            except Exception as e:
                logger.error(f"Translation worker error: {e}")
                self.msleep(constants.WORKFLOW_ERROR_SLEEP_MS)

        logger.info("Qwen translation worker thread stopped")

    def _translate_with_qwen(self):
        """Capture screen, perform OCR and translation with vision-language model."""
        workflow_start = time.time()
        logger.info(f"_translate_with_qwen called, mode={self.mode}, regions={len(self.regions)}")

        # Step 1: Capture
        self.status_update.emit("Capturing screen...")
        image_data = self._capture_screen()
        if not image_data:
            logger.info("Capture returned None")
            return
        logger.info(f"Screen captured, size={len(image_data)} bytes")

        # Step 2: Redact existing translations
        image_data, _redact_time = self._redact_active_geometries(image_data)
        self._redact_time = _redact_time  # Store for logging in _process_and_emit

        # Step 3: Region mode — process each region individually for speed
        if self.mode == TranslationMode.REGION_SELECT and self.regions:
            self._translate_regions_individually(image_data, workflow_start)
            return
        elif self.mode == TranslationMode.REGION_SELECT and not self.regions:
            self.status_update.emit("No regions defined")
            logger.info("No regions defined, skipping")
            return

        # Full-screen mode: process the entire capture
        # Step 4: Preprocess and hash
        image_data = ScreenCapture.preprocess_image(image_data)
        pil_image = Image.open(io.BytesIO(image_data))
        dhash = str(imagehash.dhash(pil_image))
        logger.info(f"Preprocessed image, dhash={dhash}")

        # Step 5: Check caches
        if self._check_and_emit_cached(dhash):
            logger.info("Cache hit, returning early")
            return

        # Step 6: Check image hash for no-change
        image_hash = ScreenCapture.calculate_hash(image_data)
        if self._should_skip_unchanged_image(image_hash):
            logger.info("Image unchanged, skipping")
            return

        # Step 7: Process with VLM
        logger.info("Calling _process_and_emit")
        self._process_and_emit(image_data, image_hash, dhash, workflow_start)

    def _translate_regions_individually(self, full_image_data: bytes, workflow_start: float):
        """Process each region as a separate small image for CPU speed.
        
        Instead of compositing all regions into one wide image (slow: large composite
        = more vision encoder work), we crop each region individually and run inference
        per-region. For typical 400x300 regions on ARM64, this means:
        - Each crop is small -> fast vision encoding
        - Correct coordinates per-region without hacky offset fixup
        - JPEG encoding for intermediate crops (faster than PNG)
        """
        full_image = Image.open(io.BytesIO(full_image_data))
        all_results = []
        is_cpu = getattr(self.qwen_processor, 'use_cpu_mode', False)

        for i, region in enumerate(self.regions):
            if not self.running:
                logger.info("Worker stopped during region processing")
                return

            region_start = time.time()
            self.status_update.emit(f"Processing region {i+1}/{len(self.regions)}: {region.name}...")
            logger.info(f"Processing region {i+1}/{len(self.regions)}: {region.name} ({region.x},{region.y} {region.width}x{region.height})")

            # Crop the region from the full capture
            box = (region.x, region.y, region.x + region.width, region.y + region.height)
            region_image = full_image.crop(box)

            # Encode as JPEG for speed (PNG is slow to encode/decode)
            buffer = io.BytesIO()
            # Convert to RGB (screen captures are often RGBA; JPEG doesn't support alpha)
            if region_image.mode != "RGB":
                region_image = region_image.convert("RGB")
            region_image.save(buffer, format="JPEG", quality=85)
            region_bytes = buffer.getvalue()
            logger.info(f"Region crop: {region_image.size}, JPEG size={len(region_bytes)} bytes")

            # Check per-region cache using dhash
            region_pil = Image.open(io.BytesIO(region_bytes))
            region_dhash = str(imagehash.dhash(region_pil))

            if region_dhash in self.perceptual_cache:
                logger.info(f"Region {i+1} cache hit (dhash)")
                cached = self.perceptual_cache[region_dhash]
                all_results.extend(cached)
                continue

            db_cached = self.translation_db.get_translation(region_dhash)
            if db_cached:
                logger.info(f"Region {i+1} DB cache hit")
                cached_results = [TranslationResult(**item) for item in db_cached]
                self.perceptual_cache[region_dhash] = cached_results
                all_results.extend(cached_results)
                continue

            # Run VLM inference on the small region crop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    region_results = loop.run_until_complete(
                        self.qwen_processor.process_frame(region_bytes, self.target_lang)
                    )
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"VLM error on region {i+1}: {e}")
                continue

            region_time = time.time() - region_start
            logger.info(f"Region {i+1} inference took {region_time:.1f}s, got {len(region_results)} results")

            # Map coordinates back to screen space
            for res in region_results:
                res.x = float(region.x)
                res.y = float(region.y)
                res.width = float(region.width)
                res.height = float(region.height)

            # Cache per-region results
            self.perceptual_cache[region_dhash] = region_results
            db_data = [r.__dict__ for r in region_results]
            self.translation_db.put_translation(region_dhash, db_data)

            all_results.extend(region_results)

        # Emit all region results together
        workflow_total = time.time() - workflow_start
        logger.info(f"All regions done: {len(all_results)} results in {workflow_total:.1f}s")

        if not all_results:
            self.status_update.emit("No text detected in regions")
            self._last_translation_signature = self._empty_signature
            return

        signature = self._fingerprint_translations(all_results)
        if signature == self._last_translation_signature:
            logger.debug("Translation signature unchanged; suppressing overlay refresh")
            return
        self._last_translation_signature = signature

        self.translation_ready.emit(all_results, None)
        self.status_update.emit(f"Translation complete ({workflow_total:.0f}s)")

    def _capture_screen(self) -> Optional[bytes]:
        """Capture the screen and return image bytes or None."""
        capture_start = time.time()
        image_data = ScreenCapture.capture_screen()
        capture_time = time.time() - capture_start
        if not image_data or not self.running:
            return None
        return image_data

    def _redact_active_geometries(self, image_data: bytes) -> Tuple[bytes, float]:
        """Redact active geometries from the image if any are set."""
        redact_time = 0.0
        if self.active_geometries:
            redact_start = time.time()
            image = QImage.fromData(image_data)
            if not image.isNull():
                capture_geo = ScreenCapture.get_virtual_desktop_geometry()
                image = self._redact_image(image, self.active_geometries, capture_geo.topLeft())

                from PyQt6.QtCore import QBuffer, QIODevice
                buffer = QBuffer()
                buffer.open(QIODevice.OpenModeFlag.WriteOnly)
                image.save(buffer, "PNG")
                image_data = bytes(buffer.buffer())
            redact_time = time.time() - redact_start
        return image_data, redact_time

    def _check_and_emit_cached(self, dhash: str) -> bool:
        """Check perceptual and DB caches. Returns True if cached result was emitted."""
        # Perceptual cache (L0)
        if dhash in self.perceptual_cache:
            logger.debug("Perceptual cache hit; reusing cached translation")
            cached_result = self.perceptual_cache[dhash]
            self.translation_ready.emit(cached_result, None)
            self.status_update.emit("Using cached translation (dHash)")
            return True

        # Database cache (L1)
        db_cached = self.translation_db.get_translation(dhash)
        if db_cached:
            logger.debug("Database cache hit; reusing cached translation")
            cached_results = [TranslationResult(**item) for item in db_cached]
            self.perceptual_cache[dhash] = cached_results
            self.translation_ready.emit(cached_results, None)
            self.status_update.emit("Using cached translation (DB)")
            return True

        return False

    def _should_skip_unchanged_image(self, image_hash: str) -> bool:
        """Skip processing if the image hash hasn't changed."""
        cached_entry = self.image_cache.get(image_hash)

        if cached_entry and cached_entry.get("translations"):
            logger.debug("Image cache hit with translations; reusing previous results")
            self.last_hashes["full"] = image_hash
            signature = self._fingerprint_translations(cached_entry["translations"])
            if signature == self._last_translation_signature:
                logger.debug("Translation signature unchanged; suppressing overlay refresh")
                return True
            self._last_translation_signature = signature
            self.translation_ready.emit(cached_entry["translations"], None)
            self.status_update.emit("Using cached translations")
            return True

        if image_hash == self.last_hashes.get("full"):
            logger.debug("Image hash unchanged, skipping processing")
            return True

        self.last_hashes["full"] = image_hash
        return False

    def _process_and_emit(self, image_data: bytes, image_hash: str, dhash: str,
                          workflow_start: float) -> None:
        """Process the frame with the VLM and emit results."""
        logger.info("Starting VLM inference...")
        self.status_update.emit("Processing with vision-language model...")
        vl_start = time.time()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                translated_results = loop.run_until_complete(
                    self.qwen_processor.process_frame(image_data, self.target_lang)
                )
            finally:
                loop.close()

            vl_time = time.time() - vl_start
            workflow_total = time.time() - workflow_start
            logger.info(f"VLM inference took {vl_time:.2f}s, total workflow {workflow_total:.2f}s")

            if not translated_results:
                logger.info(f"Vision-language model finished in {vl_time:.2f}s: No text detected")
                self.status_update.emit("No text detected")
                self._store_image_cache(image_hash, translations=[])
                self._last_translation_signature = self._empty_signature
                return

            logger.info(
                f"Vision-language model processed image in {vl_time:.2f}s, "
                f"got {len(translated_results)} results"
            )
            logger.info(
                f"Workflow stats: Redact: {self._redact_time:.2f}s, "
                f"VL-Model: {vl_time:.2f}s, Total: {workflow_total:.2f}s"
            )

            # Store in caches
            self._store_image_cache(image_hash, translations=translated_results)
            self.perceptual_cache[dhash] = translated_results
            db_data = [result.__dict__ for result in translated_results]
            self.translation_db.put_translation(dhash, db_data)

            signature = self._fingerprint_translations(translated_results)
            if signature == self._last_translation_signature:
                logger.debug("Translation signature unchanged after translate; suppressing overlay refresh")
                return
            self._last_translation_signature = signature

            self.translation_ready.emit(translated_results, None)
            self.status_update.emit("Translation complete")

        except Exception as e:
            logger.error(f"Vision-language model processing error: {e}")
            self.status_update.emit(f"VL Model Error: {e}")

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
        # For vision-language models, we can check if the engine is initialized
        # For now, just return True to indicate the processor is available
        is_available = True  # Placeholder - actual implementation would check if model is loaded
        models = ["Qwen3.5-4B", "Qwen3.5-9B", "TranslateGemma-4B", "TranslateGemma-12B"]
        self.status_changed.emit(is_available, models)


class QwenModelWarmupWorker(QThread):
    """Worker thread to initialize the vision-language model before translation starts."""

    warmup_finished = pyqtSignal(bool, str)

    def __init__(self, qwen_processor: QwenVLProcessor):
        super().__init__()
        self.qwen_processor = qwen_processor

    def run(self):
        try:
            logger.info("Starting model warmup...")
            # Initialize the engine
            import asyncio
            import traceback
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.qwen_processor.init_engine())
                logger.info("Engine initialized successfully")
                ok = True
                err = ""
            except Exception as e:
                logger.error(f"Engine initialization failed: {e}")
                logger.error(traceback.format_exc())
                ok = False
                err = str(e)
            finally:
                loop.close()

            self.warmup_finished.emit(ok, err)
        except Exception as e:
            logger.error(f"Vision-language model warmup error: {e}")
            self.warmup_finished.emit(False, str(e))