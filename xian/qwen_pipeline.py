"""Vision-Language Processing Pipeline for unified OCR and translation using Lemonade-SDK OmniRouter (OpenAI Compatible)."""

import io
import logging
import re
import os
import base64
import threading
from typing import List, Tuple
from dataclasses import dataclass

from PIL import Image
from openai import AsyncOpenAI

from .models import TranslationResult, TextStyle
from .context_manager import ContextManager
from . import constants

logger = logging.getLogger(__name__)

@dataclass
class VLConfig:
    """Configuration for Vision-Language processing"""
    model_name: str = "omni-router"
    api_url: str = "http://localhost:13305/v1"
    max_tokens: int = 1024
    temperature: float = 0.1

class VLProcessor:
    """Processor for vision-language models with unified OCR and translation capabilities via OpenAI API."""

    def __init__(self, config: VLConfig = None):
        self.config = config or VLConfig()
        self.client = None
        self._client_lock = threading.Lock()

        # Initialize context manager for stateful interactions
        self.context_manager = ContextManager(max_frames=3)

    async def init_engine(self):
        """Initialize the OpenAI API client (thread-safe)."""
        with self._client_lock:
            if self.client:
                return  # already initialized by another thread
            # Use configured API URL, falling back to environment variable if set
            base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
            api_key = os.environ.get("LEMONADE_API_KEY", "lemonade")
            logger.info(f"Initializing OpenAI client with base_url: {base_url}")
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )

    def preprocess_image(self, image_data: bytes) -> Image.Image:
        """
        Preprocess image for vision-language model input.
        """
        image = Image.open(io.BytesIO(image_data))
        max_dimension = constants.QWEN_MAX_DIMENSION
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

    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for OpenAI API."""
        # JPEG does not support alpha channels; screen captures are often RGBA
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def create_prompt(self, source_lang: str, target_lang: str, mode: str, styles: list[str]) -> str:
        """Create a terse OCR+Translation prompt tailored by user settings."""
        mode_context = "a web interface" if mode == "Web" else "a video game interface"
        style_context = f" Translate using a {', '.join(styles)} style/tone." if styles else ""
        
        return (
            f"OCR the {source_lang} text from this image of {mode_context}, "
            f"then translate it to {target_lang}.{style_context}\n"
            f"Reply ONLY with two sections, no commentary:\n"
            f"ORIGINAL:\n<extracted {source_lang} text>\n\n"
            f"TRANSLATED:\n<{target_lang} translation>"
        )

    def parse_response(self, response: str) -> Tuple[str, str]:
        """Parse the model response to extract original text and translation.

        Tries structured markers first, then heuristic splitting, then raw fallback.
        """
        logger.info(f"parse_response raw input ({len(response)} chars): {response[:300]}")

        # Strip thinking tags if present
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = response.strip()

        # --- Strategy 1: structured markers (ORIGINAL/TRANSLATED or ORIGINAL TEXT/TRANSLATION) ---
        orig_match = re.search(
            r'(?:ORIGINAL(?:\s*TEXT)?)\s*:\s*(.*?)\s*(?:TRANSLAT(?:ED|ION))\s*:',
            cleaned, re.DOTALL | re.IGNORECASE
        )
        trans_match = re.search(
            r'(?:TRANSLAT(?:ED|ION))\s*:\s*(.*)',
            cleaned, re.DOTALL | re.IGNORECASE
        )

        if orig_match and trans_match:
            original_text = orig_match.group(1).strip()
            translation = trans_match.group(1).strip()
            logger.info(f"parse_response (markers): original={len(original_text)} chars, translation={len(translation)} chars")
            return original_text, translation

        # --- Strategy 2: split on double newline ---
        parts = re.split(r'\n\s*\n', cleaned, maxsplit=1)
        if len(parts) == 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 5:
            original_text = parts[0].strip()
            translation = parts[1].strip()
            logger.info(f"parse_response (split): original={len(original_text)} chars, translation={len(translation)} chars")
            return original_text, translation

        # --- Strategy 3: raw fallback ---
        logger.info("parse_response: using raw output as translation")
        return "", cleaned

    async def process_frame(self, image_data: bytes, source_lang: str, target_lang: str, mode: str, styles: list[str]) -> List[TranslationResult]:
        """
        Process a single frame with unified OCR and translation via OmniRouter.
        """
        if not self.client:
            raise RuntimeError("Engine not initialized. Call init_engine() first.")

        # Preprocess and store in context manager
        image = self.preprocess_image(image_data)
        self.context_manager.add_frame(image)

        prompt = self.create_prompt(source_lang, target_lang, mode, styles)
        b64_image = self.encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]

        try:
            logger.info("Sending translation request to OmniRouter via OpenAI...")
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Debug: log raw response structure
            choice = response.choices[0] if response.choices else None
            if choice:
                logger.info(f"API response: finish_reason={choice.finish_reason}, "
                            f"content_length={len(choice.message.content or '')}, "
                            f"role={choice.message.role}")
            else:
                logger.warning(f"No choices in response. Full response: {response}")

            final_output = (choice.message.content or "") if choice else ""

            # Fallback: if the model spent all tokens on reasoning and produced
            # no content, try to extract a usable translation from reasoning_content.
            if not final_output and choice:
                reasoning = getattr(choice.message, 'reasoning_content', None) or ""
                if reasoning:
                    logger.info(f"Content empty but reasoning_content has {len(reasoning)} chars; extracting from it")
                    final_output = reasoning

            results = self._build_result(final_output, image)
            
            # Update context manager with extracted data
            extracted = [r.original_text for r in results]
            self.context_manager.update_last_frame_data("\n".join(extracted), results)
            
            return results

        except Exception as e:
            logger.error(f"Error during OpenAI API inference: {e}", exc_info=True)
            raise

    async def process_chat(self, message: str) -> str:
        """Process a contextual chat message using the sliding window context via OmniRouter."""
        if not self.client:
            raise RuntimeError("Engine not initialized. Call init_engine() first.")
            
        logger.info(f"Processing chat message: {message}")
        
        # Add user message to context manager
        self.context_manager.add_user_message(message, with_image=True)
        
        # Build OpenAI-compatible chat history
        openai_messages = []
        for msg in self.context_manager.get_chat_history():
            role = msg["role"]
            content = msg["content"]
            
            openai_content = []
            
            for item in content:
                if item["type"] == "text":
                    openai_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image":
                    frame = self.context_manager.get_latest_frame()
                    if frame:
                        b64_img = self.encode_image(frame)
                        openai_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
            
            # OpenAI requires content to be a string if it's just text, or list if multimodal
            if len(openai_content) == 1 and openai_content[0]["type"] == "text":
                final_content = openai_content[0]["text"]
            else:
                final_content = openai_content

            openai_messages.append({"role": role, "content": final_content})

        try:
            logger.info("Sending chat request to OmniRouter via OpenAI...")
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=openai_messages,
                max_tokens=self.config.max_tokens,
                temperature=0.7,  # Higher temp for chat
            )
            
            final_output = response.choices[0].message.content or ""
            cleaned_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
            
            # Add assistant message to context
            self.context_manager.add_assistant_message(cleaned_output)
            
            return cleaned_output
            
        except Exception as e:
            logger.error(f"Error during OpenAI chat inference: {e}")
            return f"Error communicating with the model: {str(e)}"

    def _build_result(self, response: str, image: Image.Image) -> List[TranslationResult]:
        """Build TranslationResult from model response."""
        original_text, translated_text = self.parse_response(response)

        if not translated_text.strip():
            logger.debug("No text detected in image")
            return []

        img_width, img_height = image.size

        result = TranslationResult(
            translated_text=translated_text,
            x=0.0,
            y=0.0,
            width=float(img_width),
            height=float(img_height),
            confidence=0.9,
            original_text=original_text,
            style=TextStyle(),
        )

        return [result]

    async def close(self):
        """Close the client properly."""
        if self.client:
            await self.client.close()
            self.client = None
