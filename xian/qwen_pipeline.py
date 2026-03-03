"""Vision-Language Processing Pipeline for unified OCR and translation."""

import asyncio
import io
import logging
import os
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

from .models import TranslationResult, TextStyle
from .style_detection import StyleDetector, BackgroundReconstructor

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
        self.engine = None
        self.model_id = None
        self.is_translategemma = False  # Flag to track if using TranslateGemma
        
        # Initialize style detection and background reconstruction
        self.style_detector = StyleDetector()
        self.background_reconstructor = BackgroundReconstructor()
        
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is required for vision-language processing. "
                "Install with: pip install vllm>=0.11"
            )
    
    def detect_vram(self) -> int:
        """Detect total VRAM in GB using pynvml."""
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                logger.warning("No NVIDIA GPUs detected, falling back to CPU")
                return 0
            
            total_vram_bytes = 0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_vram_bytes += mem_info.total
            
            # Convert to GB
            total_vram_gb = total_vram_bytes // (1024**3)
            logger.info(f"Detected {total_vram_gb}GB of total VRAM across {device_count} GPU(s)")
            return total_vram_gb
            
        except Exception as e:
            logger.warning(f"Could not detect VRAM via pynvml: {e}. Falling back to CPU.")
            return 0
    
    def select_model(self, vram_gb: int) -> str:
        """Select appropriate model based on VRAM and configuration."""
        model_name = self.config.model_name.lower()
        
        # Check if using TranslateGemma models
        if "translategemma" in model_name:
            self.is_translategemma = True
            if "12b" in model_name:
                if vram_gb >= 20:
                    return "google/translategemma-12b-it"
                elif vram_gb >= 12:
                    # Fallback to 4B if VRAM insufficient
                    return "google/translategemma-4b-it"
                else:
                    # Even if VRAM is low, try to use the 4B model as a last resort
                    return "google/translategemma-4b-it"
            elif "4b" in model_name:
                if vram_gb >= 10:
                    return "google/translategemma-4b-it"
                else:
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) to run TranslateGemma-4B. "
                        f"Minimum requirement is approximately 10GB."
                    )
            else:
                # Default to 4B if specific size not mentioned
                if vram_gb >= 10:
                    return "google/translategemma-4b-it"
                else:
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) to run TranslateGemma models. "
                        f"Minimum requirement is approximately 10GB."
                    )
        else:
            # Handle Qwen3.5 models
            if self.config.model_size == "9b":
                if vram_gb >= 18:
                    return "Qwen/Qwen3.5-9B"
                else:
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) for Qwen3.5-9B. "
                        f"Minimum requirement is approximately 18GB."
                    )
            elif self.config.model_size == "4b":
                if vram_gb >= 10:
                    return "Qwen/Qwen3.5-4B"
                else:
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) for Qwen3.5-4B. "
                        f"Minimum requirement is approximately 10GB."
                    )
            else:  # auto
                if vram_gb >= 20:
                    model_id = "Qwen/Qwen3.5-9B"
                    logger.info(f"Auto-selected Qwen3.5-9B based on {vram_gb}GB VRAM")
                elif vram_gb >= 12:
                    model_id = "Qwen/Qwen3.5-4B"
                    logger.info(f"Auto-selected Qwen3.5-4B based on {vram_gb}GB VRAM")
                elif vram_gb >= 10:
                    # Fallback to TranslateGemma-4B if Qwen models don't fit
                    logger.info(f"Fallback to TranslateGemma-4B due to limited VRAM ({vram_gb}GB)")
                    self.is_translategemma = True
                    return "google/translategemma-4b-it"
                else:
                    raise RuntimeError(
                        f"Not enough VRAM ({vram_gb}GB) to run vision-language models. "
                        f"Minimum requirement is approximately 10GB for fallback models."
                    )
            return model_id
    
    async def init_engine(self):
        """Initialize the vLLM engine asynchronously."""
        vram_gb = self.detect_vram()
        
        # If no GPU detected, warn user but allow proceeding
        if vram_gb == 0:
            logger.warning("No GPU detected. Vision-language model will run on CPU, which will be slow.")
            # For CPU, we'll use a different approach or warn
            # For now, let's assume we have a GPU or the user knows the implications
        
        self.model_id = self.select_model(vram_gb)
        
        logger.info(f"Initializing vision-language engine with model: {self.model_id}")
        
        # Prepare engine args with thinking mode toggle for Qwen3.5
        engine_kwargs = {
            "model": self.model_id,
            "trust_remote_code": True,
            "max_model_len": 8192,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "dtype": self.config.dtype,
            "enforce_eager": True,  # Avoids CUDA graph capture overhead for single images
        }
        
        # Add thinking mode toggle for Qwen3.5 models
        if not self.is_translategemma:
            engine_kwargs["chat_template_kwargs"] = {
                "enable_thinking": self.config.thinking_mode
            }
        
        engine_args = AsyncEngineArgs(**engine_kwargs)
        
        self.engine = await AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Vision-language engine initialized successfully")
    
    def preprocess_image(self, image_data: bytes) -> Image.Image:
        """
        Preprocess image for vision-language model input.
        For Qwen3.5: max dimension 1024px maintaining aspect ratio.
        For TranslateGemma: normalize to 896x896 as specified.
        """
        image = Image.open(io.BytesIO(image_data))
        
        if self.is_translategemma:
            # For TranslateGemma, normalize to 896x896 as specified
            image = image.resize((896, 896), Image.Resampling.LANCZOS)
        else:
            # For Qwen3.5, maintain aspect ratio, max dimension 1024px
            max_dimension = 1024
            width, height = image.size
            
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                    
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def create_prompt(self, target_lang: str, thinking_mode: bool = False) -> str:
        """Create unified OCR+Translation prompt template."""
        if self.is_translategemma:
            # TranslateGemma-specific prompt
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
        else:
            # Qwen3.5-specific prompt (thinking handled by chat_template_kwargs)
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
        # Look for ORIGINAL TEXT and TRANSLATION sections
        original_match = re.search(r'ORIGINAL TEXT:\s*(.*?)\s*TRANSLATION:', response, re.DOTALL | re.IGNORECASE)
        translation_match = re.search(r'TRANSLATION:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        
        original_text = original_match.group(1).strip() if original_match else ""
        translation = translation_match.group(1).strip() if translation_match else ""
        
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
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call init_engine() first.")
        
        try:
            # Preprocess image
            image = self.preprocess_image(image_data)
            
            # Create prompt
            prompt = self.create_prompt(target_lang, self.config.thinking_mode)
            
            # Prepare inputs for vLLM
            sampling_params = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": 0.9,
            }
            
            # Generate response with timeout
            import asyncio
            try:
                # Create a task to run the generation with timeout
                async def generate_task():
                    # For TranslateGemma, we may need to pass the image differently
                    if self.is_translategemma:
                        inputs = {
                            "prompt": prompt,
                            "multi_modal_data": {"image": image},
                        }
                        results_generator = self.engine.generate(
                            inputs["prompt"], 
                            sampling_params=sampling_params, 
                            request_id=f"request-{int(time.time())}-{id(self)}",
                            multi_modal_data=inputs.get("multi_modal_data")
                        )
                    else:
                        # For Qwen3.5
                        inputs = {
                            "prompt": prompt,
                            "multi_modal_data": {"image": image},
                        }
                        results_generator = self.engine.generate(
                            inputs["prompt"], 
                            sampling_params=sampling_params, 
                            request_id=f"request-{int(time.time())}-{id(self)}"
                        )
                    
                    final_output = ""
                    async for request_output in results_generator:
                        if request_output.outputs:
                            final_output = request_output.outputs[0].text
                    return final_output
                
                # Run with timeout
                try:
                    final_output = await asyncio.wait_for(generate_task(), timeout=15.0)  # 15 second timeout for larger models
                except asyncio.TimeoutError:
                    logger.error("Timeout during vision-language model inference")
                    return []
                
                # Parse the response to extract original text and translation
                original_text, translated_text = self.parse_response(final_output)
                
                # If no text was detected, return empty list
                if not translated_text.strip():
                    logger.debug("No text detected in image")
                    return []

                # Detect text style from the image for context-aware rendering
                # Use the center region of the image as a representative sample
                img_width, img_height = image.size
                style_bbox = (
                    img_width * 0.25,
                    img_height * 0.25,
                    img_width * 0.5,
                    img_height * 0.5
                )
                detected_style = self.style_detector.detect_style(image, style_bbox)
                
                # Create default style if detection failed
                if detected_style is None:
                    detected_style = TextStyle()

                # For now, return a single TranslationResult with the full translation
                # In a more sophisticated implementation, we would extract bounding boxes
                # from the model response to create individual TranslationResult objects
                result = TranslationResult(
                    translated_text=translated_text,
                    x=0.0,  # Placeholder values - would come from OCR in a full implementation
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
                
            except Exception as e:
                logger.error(f"Error during vision-language model inference: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    async def close(self):
        """Close the engine properly."""
        if self.engine:
            # vLLM doesn't have a direct close method, but we can set it to None
            self.engine = None


# Additional helper functions
def validate_model_availability(model_id: str) -> bool:
    """Validate if the model can be loaded."""
    try:
        # This would be a more complex check in practice
        # For now, just return True assuming the model exists
        return True
    except Exception:
        return False

# For backward compatibility
QwenVLProcessor = VLProcessor
QwenVLConfig = VLConfig