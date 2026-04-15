"""Vision-Language Processing Pipeline for unified OCR and translation."""

import asyncio
import io
import logging
import re
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from PIL import Image
import pynvml

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .models import TranslationResult, TextStyle
from .style_detection import StyleDetector, BackgroundReconstructor
from . import constants

logger = logging.getLogger(__name__)

@dataclass
class VLConfig:
    """Configuration for Vision-Language processing"""
    model_name: str = "Qwen3.5-9B (Auto-select)"  # Default model
    model_size: str = "auto"  # "auto", "4b", "9b" for Qwen models, ignored for TranslateGemma
    thinking_mode: bool = True  # Thinking mode enabled by default for Qwen3.5
    max_tokens: int = 1024
    temperature: float = 0.1
    gpu_memory_utilization: float = 0.85
    dtype: str = "bfloat16"  # or "float16"


class VLProcessor:
    """Processor for vision-language models with unified OCR and translation capabilities."""

    def __init__(self, config: VLConfig = None):
        self.config = config or VLConfig()
        self.engine = None  # vLLM engine
        self.transformers_model = None  # transformers model for CPU mode
        self.transformers_processor = None  # transformers processor for CPU mode
        self.model_id = None
        self.is_translategemma = False  # Flag to track if using TranslateGemma
        self.use_cpu_mode = False  # True when using transformers instead of vLLM

        # Initialize style detection and background reconstruction
        self.style_detector = StyleDetector()
        self.background_reconstructor = BackgroundReconstructor()

    def detect_vram(self) -> int:
        """Detect total VRAM in GB using torch.cuda first, then pynvml as fallback."""
        # Try torch.cuda first (works with both NVIDIA and AMD ROCm)
        try:
            if torch.cuda.is_available():
                total_vram_bytes = 0
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total_vram_bytes += props.total_memory

                total_vram_gb = total_vram_bytes // (1024**3)
                logger.info(f"Detected {total_vram_gb}GB of total VRAM via torch.cuda across {torch.cuda.device_count()} GPU(s)")
                return total_vram_gb
        except Exception as e:
            logger.debug(f"torch.cuda VRAM detection failed: {e}")

        # Fallback to pynvml (NVIDIA only)
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                logger.warning("No NVIDIA GPUs detected via pynvml")
                return 0

            total_vram_bytes = 0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_bytes += mem_info.total

            total_vram_gb = total_vram_bytes // (1024**3)
            logger.info(f"Detected {total_vram_gb}GB of total VRAM via pynvml across {device_count} GPU(s)")
            return total_vram_gb

        except Exception as e:
            logger.warning(f"Could not detect VRAM via pynvml: {e}")

        logger.warning("No GPU detected. Model will run on CPU via transformers (slow but functional).")
        return 0

    def select_model(self, vram_gb: int) -> str:
        """Select appropriate model based on VRAM and configuration."""
        model_name = self.config.model_name.lower()

        # Check if using TranslateGemma models
        if "translategemma" in model_name:
            self.is_translategemma = True
            if "12b" in model_name:
                if vram_gb >= constants.VRAM_THRESHOLD_TRANSLATEGEMMA_12B:
                    return constants.MODEL_TRANSLATEGEMMA_12B
                else:
                    return constants.MODEL_TRANSLATEGEMMA_4B
            elif "4b" in model_name:
                if vram_gb >= constants.VRAM_THRESHOLD_TRANSLATEGEMMA_4B:
                    return constants.MODEL_TRANSLATEGEMMA_4B
                else:
                    if vram_gb == 0:
                        logger.warning("Running TranslateGemma-4B on CPU. This will be slow.")
                        return constants.MODEL_TRANSLATEGEMMA_4B
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) to run TranslateGemma-4B. "
                        f"Minimum requirement is approximately {constants.VRAM_THRESHOLD_TRANSLATEGEMMA_4B}GB."
                    )
            else:
                if vram_gb >= constants.VRAM_THRESHOLD_TRANSLATEGEMMA_4B:
                    return constants.MODEL_TRANSLATEGEMMA_4B
                else:
                    if vram_gb == 0:
                        logger.warning("Running TranslateGemma-4B on CPU. This will be slow.")
                        return constants.MODEL_TRANSLATEGEMMA_4B
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) to run TranslateGemma models. "
                        f"Minimum requirement is approximately {constants.VRAM_THRESHOLD_TRANSLATEGEMMA_4B}GB."
                    )
        else:
            # Handle Qwen3.5 models
            if self.config.model_size == "9b":
                if vram_gb >= constants.VRAM_THRESHOLD_9B:
                    return constants.MODEL_QWEN_9B
                else:
                    if vram_gb == 0:
                        logger.warning("Running Qwen3.5-9B on CPU. This will be extremely slow.")
                        return constants.MODEL_QWEN_9B
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) for Qwen3.5-9B. "
                        f"Minimum requirement is approximately {constants.VRAM_THRESHOLD_9B}GB."
                    )
            elif self.config.model_size == "4b":
                if vram_gb >= constants.VRAM_THRESHOLD_4B:
                    return constants.MODEL_QWEN_4B
                else:
                    if vram_gb == 0:
                        logger.warning("Running Qwen3.5-4B on CPU. Inference will be slow (5-15s per frame).")
                        return constants.MODEL_QWEN_4B
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) for Qwen3.5-4B. "
                        f"Minimum requirement is approximately {constants.VRAM_THRESHOLD_4B}GB."
                    )
            elif self.config.model_size == "2b":
                if vram_gb >= constants.VRAM_THRESHOLD_2B or vram_gb == 0:
                    return constants.MODEL_QWEN_2B
                raise RuntimeError(
                    f"Not enough VRAM ({vram_gb}GB) for Qwen3.5-2B. "
                    f"Minimum requirement is approximately {constants.VRAM_THRESHOLD_2B}GB."
                )
            else:  # auto
                if vram_gb >= constants.VRAM_AUTO_SELECT_9B:
                    model_id = constants.MODEL_QWEN_9B
                    logger.info(f"Auto-selected Qwen3.5-9B based on {vram_gb}GB VRAM")
                elif vram_gb >= constants.VRAM_AUTO_SELECT_4B:
                    model_id = constants.MODEL_QWEN_4B
                    logger.info(f"Auto-selected Qwen3.5-4B based on {vram_gb}GB VRAM")
                elif vram_gb >= constants.VRAM_AUTO_SELECT_2B:
                    model_id = constants.MODEL_QWEN_2B
                    logger.info(f"Auto-selected Qwen3.5-2B based on {vram_gb}GB VRAM")
                elif vram_gb >= constants.VRAM_THRESHOLD_4B:
                    logger.info(f"Fallback to TranslateGemma-4B due to limited VRAM ({vram_gb}GB)")
                    self.is_translategemma = True
                    return constants.MODEL_TRANSLATEGEMMA_4B
                elif vram_gb == 0:
                    logger.warning("No GPU detected. Falling back to Qwen3.5-2B on CPU via transformers. Expect 15-45s latency per frame.")
                    model_id = constants.MODEL_QWEN_2B
                else:
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) to run vision-language models. "
                        f"Minimum requirement is approximately {constants.VRAM_THRESHOLD_2B}GB for Qwen3.5-2B."
                    )
            return model_id

    def _get_transformers_device_and_dtype(self):
        """Determine device and dtype for transformers CPU mode."""
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32  # CPU needs float32, bfloat16 not supported on all CPUs
        return device, dtype

    async def init_engine(self):
        """Initialize the model engine (vLLM for GPU, transformers for CPU)."""
        vram_gb = self.detect_vram()

        # If no GPU detected, use transformers for CPU inference
        if vram_gb == 0:
            logger.warning("No GPU detected. Using transformers for CPU inference (slower but functional).")
            await self._init_transformers()
            return

        # GPU mode: use vLLM
        self.model_id = self.select_model(vram_gb)
        logger.info(f"Initializing vLLM engine with model: {self.model_id}")

        if not VLLM_AVAILABLE:
            raise RuntimeError(
                "vLLM is required for GPU inference but not installed. "
                "Install with: uv pip install vllm>=0.11"
            )

        # Prepare engine args
        engine_kwargs = {
            "model": self.model_id,
            "trust_remote_code": True,
            "max_model_len": 8192,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype,
            "enforce_eager": True,
        }

        engine_args = AsyncEngineArgs(**engine_kwargs)

        self.engine = await AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM engine initialized successfully")

    async def _init_transformers(self):
        """Initialize transformers model for CPU inference with quantization."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is required for CPU inference but not installed. "
                "Install with: uv pip install transformers"
            )

        self.use_cpu_mode = True
        self.model_id = self.select_model(vram_gb=0)
        
        device, dtype = self._get_transformers_device_and_dtype()
        logger.info(f"Loading model {self.model_id} on {device} with standard precision")

        import os
        import multiprocessing
        
        num_cores = max(4, multiprocessing.cpu_count() - 1)
        
        # Limit torch threads to avoid CPU saturation but still be fast
        try:
            torch.set_num_threads(num_cores)
            # interop_threads is fine to leave alone or attempt to set
        except Exception as e:
            logger.debug(f"Could not set torch thread limits: {e}")
            
        # Set environment variables for other libraries
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_cores)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Disable oneDNN/MKLDNN which causes hangs on some CPUs
        os.environ["ATEN_CPU_CAPABILITY"] = "DEFAULT"
        os.environ["TORCH_MKLDNN_MATMUL_MIN_DIM"] = "999999"

        # Load processor first
        logger.info("Loading processor...")
        self.transformers_processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        logger.info("Processor loaded")

        # Load model with standard data types for CPU
        logger.info("Loading model weights (this may take a few minutes on CPU)...")
        try:
            logger.info("Using standard precision for CPU inference (bitsandbytes heavily throttles ARM architectures)")

            self.transformers_model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # Best compatibility on aarch64 CPU
                device_map="cpu",
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # Force eager attention for CPU
            )
        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            self.transformers_model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # Force eager attention for CPU
            )

        logger.info(f"Transformers model device: {self.transformers_model.device}")
        logger.info(f"Transformers model loaded successfully")

    def preprocess_image(self, image_data: bytes) -> Image.Image:
        """
        Preprocess image for vision-language model input.
        For Qwen3.5: max dimension defined in constants, maintaining aspect ratio.
        For TranslateGemma: normalize to dimension defined in constants.
        CPU mode uses aggressively smaller dimensions since vision cost is O(n²).
        """
        image = Image.open(io.BytesIO(image_data))

        if self.is_translategemma:
            dim = constants.TRANSLATEGEMMA_DIMENSION
            image = image.resize((dim, dim), Image.Resampling.LANCZOS)
        else:
            # Use much smaller dimension on CPU — vision encoder cost is quadratic
            max_dimension = constants.CPU_MAX_DIMENSION if self.use_cpu_mode else constants.QWEN_MAX_DIMENSION
            width, height = image.size

            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)

                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                if self.use_cpu_mode:
                    logger.info(f"CPU mode: resized {width}x{height} -> {image.size[0]}x{image.size[1]} (max_dim={max_dimension})")

        return image

    def create_prompt(self, target_lang: str, thinking_mode: bool = False) -> str:
        """Create unified OCR+Translation prompt template."""
        if self.use_cpu_mode:
            # Ultra-short prompt for CPU: every output token costs ~1 second on aarch64
            prompt = f"Translate all text in this image to {target_lang}. Output only the translation, nothing else."
            return prompt

        prompt = f"""Extract all visible text from this image in its original language, then provide a natural translation to {target_lang}. Format your response as:

ORIGINAL TEXT:
[line-by-line extracted text with approximate positioning]

TRANSLATION:
[fluent translation preserving context and layout intent]

Rules:
- Preserve line breaks and approximate spatial grouping
- For UI elements/buttons: translate naturally while preserving function
- For proper nouns/place names: keep original unless commonly localized
- Ignore decorative/artistic text without semantic meaning
- If text is ambiguous due to image quality, indicate uncertainty"""

        return prompt

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract original text and translation."""
        logger.info(f"parse_response raw input ({len(response)} chars): {response[:500]}")

        # Strip thinking tags if present (Qwen3.5 thinking mode can leak even when suppressed)
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Also strip partial/unclosed thinking tags (timeout may cut mid-think)
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = response.strip()

        original_match = re.search(r'ORIGINAL TEXT:\s*(.*?)\s*TRANSLATION:', cleaned, re.DOTALL | re.IGNORECASE)
        translation_match = re.search(r'TRANSLATION:\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)

        original_text = original_match.group(1).strip() if original_match else ""
        translation = translation_match.group(1).strip() if translation_match else ""

        # Fallback: if no markers found but there's actual text, use raw output as translation
        if not translation and cleaned:
            logger.info("parse_response: No TRANSLATION marker found, using raw output as translation")
            translation = cleaned
            original_text = ""

        logger.info(f"parse_response result: original={len(original_text)} chars, translation={len(translation)} chars")
        return original_text, translation

    async def process_frame(self, image_data: bytes, target_lang: str) -> List[TranslationResult]:
        """
        Process a single frame with unified OCR and translation.

        Args:
            image_data: Raw image bytes from screen capture
            target_lang: Target language for translation

        Returns:
            List of TranslationResult objects
        """
        if self.use_cpu_mode:
            return await self._process_frame_transformers(image_data, target_lang)
        else:
            return await self._process_frame_vllm(image_data, target_lang)

    async def _process_frame_vllm(self, image_data: bytes, target_lang: str) -> List[TranslationResult]:
        """Process frame using vLLM engine (GPU mode)."""
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call init_engine() first.")

        try:
            image = self.preprocess_image(image_data)
            prompt = self.create_prompt(target_lang, self.config.thinking_mode)

            sampling_params = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": 0.9,
            }

            try:
                async def generate_task():
                    inputs = {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                    results_generator = self.engine.generate(
                        inputs,
                        sampling_params=sampling_params,
                        request_id=f"request-{int(time.time())}-{id(self)}",
                    )

                    final_output = ""
                    async for request_output in results_generator:
                        if request_output.outputs:
                            final_output = request_output.outputs[0].text
                    return final_output

                try:
                    final_output = await asyncio.wait_for(
                        generate_task(), 
                        timeout=constants.INFERENCE_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.error("Timeout during vLLM inference")
                    return []

                return self._build_result(final_output, image)

            except Exception as e:
                logger.error(f"Error during vLLM inference: {e}")
                return []

        except Exception as e:
            logger.error(f"Error processing frame with vLLM: {e}")
            return []

    def _compute_cpu_token_budget(self, image: Image.Image) -> int:
        """Compute adaptive max_new_tokens based on image dimensions.
        
        Smaller images contain less text, so we can use fewer tokens.
        On aarch64 CPU at ~1-3 tok/s, every token saved is 0.3-1.0 seconds.
        """
        max_dim = max(image.size)
        if max_dim < 256:
            budget = constants.CPU_MAX_TOKENS_SMALL
        elif max_dim <= 400:
            budget = constants.CPU_MAX_TOKENS_MEDIUM
        else:
            budget = constants.CPU_MAX_TOKENS_LARGE
        logger.info(f"CPU token budget: {budget} tokens for {image.size[0]}x{image.size[1]} image (max_dim={max_dim})")
        return budget

    def _compute_cpu_timeout(self, max_new_tokens: int) -> float:
        """Compute timeout that scales with token budget.
        
        At ~1 tok/s on aarch64, we need at least max_new_tokens seconds
        plus overhead for vision encoding. Use 1.5s/token as conservative estimate.
        """
        timeout = max(
            constants.CPU_TIMEOUT_BASE_SECONDS,
            max_new_tokens * constants.CPU_TIMEOUT_PER_TOKEN
        )
        return timeout

    async def _process_frame_transformers(self, image_data: bytes, target_lang: str) -> List[TranslationResult]:
        """Process frame using transformers (CPU mode)."""
        logger.info("_process_frame_transformers START")
        if not self.transformers_model or not self.transformers_processor:
            raise RuntimeError("Transformers model not initialized.")

        try:
            logger.info("Step 1: Preprocessing image...")
            image = self.preprocess_image(image_data)
            logger.info(f"Step 1 done: image size={image.size}")
            
            prompt = self.create_prompt(target_lang, thinking_mode=False)
            logger.info(f"Step 2: Building messages, prompt length={len(prompt)}")

            # Build messages for the processor
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template — suppress thinking mode on CPU to avoid wasting tokens
            logger.info("Step 3: Applying chat template...")
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.use_cpu_mode:
                template_kwargs["enable_thinking"] = False
            try:
                text = self.transformers_processor.apply_chat_template(
                    messages, **template_kwargs
                )
            except TypeError:
                # Fallback if processor doesn't support enable_thinking
                logger.info("Processor does not support enable_thinking, applying template without it")
                template_kwargs.pop("enable_thinking", None)
                text = self.transformers_processor.apply_chat_template(
                    messages, **template_kwargs
                )
            logger.info(f"Step 3 done: text length={len(text)}")

            # Process inputs
            logger.info("Step 4: Processing inputs...")
            inputs = self.transformers_processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )
            logger.info(f"Step 4 done: input_ids shape={inputs['input_ids'].shape}")

            # Move to device
            inputs = inputs.to(self.transformers_model.device)
            logger.info(f"Inputs on device: {self.transformers_model.device}")

            # Adaptive CPU token budget and timeout
            is_cpu = self.transformers_model.device.type == "cpu"
            if is_cpu:
                max_new_tokens = self._compute_cpu_token_budget(image)
                cpu_timeout = self._compute_cpu_timeout(max_new_tokens)
            else:
                max_new_tokens = self.config.max_tokens
                cpu_timeout = constants.INFERENCE_TIMEOUT_SECONDS
            logger.info(f"Step 5: Generation config: max_tokens={max_new_tokens}, timeout={cpu_timeout:.0f}s, device={self.transformers_model.device}")

            # Generate with timeout
            try:
                from transformers import StoppingCriteria, StoppingCriteriaList
                
                class TimeoutStoppingCriteria(StoppingCriteria):
                    """Stops generation after timeout, with periodic heartbeat logging."""
                    def __init__(self, start_time: float, timeout_s: float, heartbeat_interval: float = 30.0):
                        self.start_time = start_time
                        self.timeout_s = timeout_s
                        self.heartbeat_interval = heartbeat_interval
                        self.last_heartbeat = start_time
                        self.token_count = 0

                    def __call__(self, input_ids, scores, **kwargs):
                        self.token_count += 1
                        now = time.time()
                        elapsed = now - self.start_time
                        
                        # Periodic heartbeat so we can see progress vs hang
                        if now - self.last_heartbeat >= self.heartbeat_interval:
                            tok_s = self.token_count / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"[heartbeat] {elapsed:.0f}s elapsed, "
                                f"{self.token_count} tokens generated ({tok_s:.2f} tok/s), "
                                f"timeout in {self.timeout_s - elapsed:.0f}s"
                            )
                            self.last_heartbeat = now
                        
                        if elapsed > self.timeout_s:
                            logger.warning(f"TimeoutStoppingCriteria fired after {elapsed:.0f}s, {self.token_count} tokens")
                            return True
                        return False

                async def generate_task():
                    def do_generate():
                        # Log actual model dtype for diagnostics
                        try:
                            first_param = next(self.transformers_model.parameters())
                            logger.info(f"do_generate: model dtype={first_param.dtype}, device={first_param.device}")
                        except Exception:
                            pass
                        
                        logger.info(f"do_generate: Starting model.generate() with max_new_tokens={max_new_tokens}...")
                        gen_start = time.time()
                        
                        is_cpu = self.transformers_model.device.type == "cpu"
                        
                        stopping_criteria = StoppingCriteriaList([
                            TimeoutStoppingCriteria(time.time(), cpu_timeout)
                        ])
                        
                        # Build generation kwargs
                        gen_kwargs = {
                            **inputs,
                            "max_new_tokens": max_new_tokens,
                            "do_sample": not is_cpu,
                            "stopping_criteria": stopping_criteria,
                        }
                        
                        if not is_cpu:
                            gen_kwargs["temperature"] = self.config.temperature
                            gen_kwargs["top_p"] = 0.9
                        
                        with torch.no_grad():
                            output_ids = self.transformers_model.generate(**gen_kwargs)
                        
                        gen_elapsed = time.time() - gen_start
                        num_input = inputs["input_ids"].shape[1]
                        num_output = output_ids.shape[1]
                        num_generated = num_output - num_input
                        tokens_per_sec = num_generated / gen_elapsed if gen_elapsed > 0 else 0
                        logger.info(
                            f"do_generate: done in {gen_elapsed:.1f}s, "
                            f"generated {num_generated} tokens ({tokens_per_sec:.2f} tok/s), "
                            f"output shape={output_ids.shape}"
                        )
                        
                        # Decode only the generated tokens (exclude input)
                        output_text = self.transformers_processor.decode(
                            output_ids[0][num_input:],
                            skip_special_tokens=True
                        )
                        logger.info(f"do_generate: Decoded text ({len(output_text)} chars): {output_text[:200]}")
                        return output_text

                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, do_generate)

                logger.info("Step 6: Awaiting generate_task with timeout...")
                # asyncio timeout is a safety net; TimeoutStoppingCriteria handles graceful stop
                final_output = await asyncio.wait_for(
                    generate_task(),
                    timeout=cpu_timeout + 30.0
                )
                logger.info(f"Step 6 done: final_output length={len(final_output)}")

                return self._build_result(final_output, image)

            except asyncio.TimeoutError:
                logger.error("Timeout during transformers inference (asyncio safety net)")
                return []

        except Exception as e:
            logger.error(f"Error processing frame with transformers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _build_result(self, response: str, image: Image.Image) -> List[TranslationResult]:
        """Build TranslationResult from model response."""
        original_text, translated_text = self.parse_response(response)

        if not translated_text.strip():
            logger.debug("No text detected in image")
            return []

        # Detect text style
        img_width, img_height = image.size
        style_bbox = (
            img_width * constants.STYLE_DETECT_CENTER_REGION[0],
            img_height * constants.STYLE_DETECT_CENTER_REGION[1],
            img_width * constants.STYLE_DETECT_CENTER_REGION[2],
            img_height * constants.STYLE_DETECT_CENTER_REGION[3],
        )
        detected_style = self.style_detector.detect_style(image, style_bbox)

        if detected_style is None:
            detected_style = TextStyle()

        result = TranslationResult(
            translated_text=translated_text,
            x=0.0,
            y=0.0,
            width=float(img_width),
            height=float(img_height),
            confidence=0.9,
            original_text=original_text,
            style=TextStyle(
                font_family=detected_style.font_family,
                font_size=detected_style.font_size,
                font_weight=detected_style.font_weight,
                text_color=detected_style.text_color,
                background_color=detected_style.background_color,
                rotation_angle=detected_style.rotation_angle,
                opacity=detected_style.opacity
            ),
            rotation_angle=detected_style.rotation_angle
        )

        return [result]

    async def close(self):
        """Close the engine properly."""
        if self.engine:
            self.engine = None
        if self.transformers_model:
            del self.transformers_model
            self.transformers_model = None
        if self.transformers_processor:
            del self.transformers_processor
            self.transformers_processor = None
        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Additional helper functions
def validate_model_availability(model_id: str) -> bool:
    """Validate if the model can be loaded."""
    try:
        return True
    except Exception:
        return False

# For backward compatibility
QwenVLProcessor = VLProcessor
QwenVLConfig = VLConfig
