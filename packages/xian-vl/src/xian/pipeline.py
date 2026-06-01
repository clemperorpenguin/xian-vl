"""Vision-Language Processing Pipeline for unified OCR and translation using Lemonade-SDK OmniRouter (OpenAI Compatible)."""

import asyncio
import base64
import io
import logging
import os
import re
import json
import threading
import time
from dataclasses import dataclass
import yaml

from PIL import Image, ImageFilter
from openai import AsyncOpenAI
import imagehash

from shared_types.constants import DEFAULT_API_URL, DEFAULT_MAX_TOKENS, QWEN_MAX_DIMENSION, IMAGE_HASH_SIZE, MODE_MAX_TOKENS
from shared_types.models import TranslationResult, TextStyle, AccuracyScore
from xian.compiler import WikiCompiler
from xian.context_manager import ContextManager
from xian.searcher import LocalWikiSearcher, WebSearcher
from xian.url_safety import markdown_http_https_url_or_none
from xian.timeout import (
    CHAT_AUX_TIMEOUT_SECONDS,
    CHAT_TIMEOUT_SECONDS,
    vision_timeout_for_mode,
)
from xian.async_engine import AsyncEngine
from xian.lemonade_client import LemonadeClient
from xian.omni_router import OmniModelRouter
from xian.tools import OMNI_TOOLS

_LANG_CODE_TO_NAME = {
    "zh-CN": "Chinese",
    "zh-TW": "Chinese",
    "en-US": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
}

logger = logging.getLogger(__name__)


def play_audio_simple(audio_bytes: bytes):
    """Play WAV audio bytes using a subprocess runner in a background thread."""
    import shutil
    import subprocess
    import tempfile
    import threading
    from pathlib import Path

    def run():
        player = None
        for cmd in ("pw-play", "paplay", "aplay"):
            if shutil.which(cmd):
                player = cmd
                break
        if not player:
            logger.warning("No audio player found for simple playback.")
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            subprocess.run([player, tmp_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error("Simple playback failed: %s", e)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()


@dataclass
class VLConfig:
    """Configuration for Vision-Language processing."""

    model_name: str = "omni-router"
    api_url: str = DEFAULT_API_URL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.1


class VLProcessor:
    """Processor for vision-language models with unified OCR and translation capabilities via OpenAI API."""

    def __init__(self, config: VLConfig | None = None):
        self.config = config or VLConfig()
        self.engine = AsyncEngine(
            base_url=os.environ.get("LEMONADE_API_URL", self.config.api_url),
            api_key=os.environ.get("LEMONADE_API_KEY", "not-needed")
        )
        self.engine.start()
        self.router = OmniModelRouter(os.environ.get("LEMONADE_API_URL", self.config.api_url))


        # Thread lock for caching properties
        self._lock = threading.Lock()

        # Perceptual hash caching properties
        self._last_phash: str | None = None
        self._last_b64: str | None = None
        self._last_results: list[TranslationResult] | None = None
        self.last_stream_results: list[TranslationResult] = []

        # Continuation context for truncated responses
        self._last_messages: list[dict] | None = None
        self._last_raw_output: str = ""

        # Initialize context manager for stateful interactions
        self.context_manager = ContextManager(max_frames=1)
        self.wiki_dir = self.find_wiki_dir()
        logger.info("Using wiki directory: %s", self.wiki_dir)

    def _validate_file_path(self, path: str, allow_dirs: list[str] | None = None) -> str:
        """Validate that a file path is within allowed directories."""
        if not os.path.isabs(path):
            path = os.path.join(self.wiki_dir, "media", path)
        real = os.path.realpath(path)
        allowed = allow_dirs or [os.path.join(self.wiki_dir, "media")]
        for d in allowed:
            real_d = os.path.realpath(d)
            if real.startswith(real_d + os.sep) or real == real_d:
                return real
        raise PermissionError(f"Access denied: path '{path}' is outside allowed directories")


    @property
    def client(self) -> AsyncOpenAI:
        return self.engine.client

    def get_model_name(self) -> str:
        """Resolve the model name to use for standard completions/vision tasks."""
        self.router.active_model = self.config.model_name
        if self.config.model_name in ("omni-router", "default"):
            return self.router.llm()
        if self.router.is_omni_model(self.config.model_name):
            return self.router.llm(self.config.model_name)
        return self.config.model_name

    def get_vision_model_name(self) -> str:
        """Resolve the model name to use for vision completions tasks."""
        self.router.active_model = self.config.model_name
        if self.config.model_name in ("omni-router", "default"):
            return self.router.vision()
        if self.router.is_omni_model(self.config.model_name):
            return self.router.vision(self.config.model_name)
        return self.config.model_name


    async def init_engine(self):
        """No-op stub for backward compatibility."""
        pass

    def find_wiki_dir(self) -> str:
        # 1. Env var
        env_dir = os.environ.get("XIAN_WIKI_DIR")
        if env_dir and os.path.exists(env_dir):
            return env_dir
        
        # 2. Walk up to find 'wiki' folder
        current = os.path.abspath(os.getcwd())
        for _ in range(4):
            candidate = os.path.join(current, "wiki")
            if os.path.exists(candidate) and os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        
        # 3. Default fallback
        return os.path.abspath(os.path.join(os.getcwd(), "wiki"))

    def load_glossary_from_wiki(self) -> dict[str, str]:
        glossary = {}
        if not os.path.exists(self.wiki_dir):
            return glossary
        
        try:
            for filename in os.listdir(self.wiki_dir):
                if not filename.endswith(".md"):
                    continue
                filepath = os.path.join(self.wiki_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            frontmatter = yaml.safe_load(parts[1])
                            if isinstance(frontmatter, dict):
                                title = frontmatter.get("title")
                                original_names = frontmatter.get("original_names")
                                if title and original_names:
                                    if isinstance(original_names, list):
                                        for name in original_names:
                                            glossary[str(name).strip()] = str(title).strip()
                                    elif isinstance(original_names, str):
                                        glossary[original_names.strip()] = str(title).strip()
                except Exception as e:
                    logger.warning("Failed to parse frontmatter from %s: %s", filepath, e)
        except Exception as e:
            logger.warning("Failed to read glossary from wiki directory: %s", e)
        return glossary

    def get_recent_text_for_search(self) -> str:
        if not hasattr(self, "context_manager") or not self.context_manager:
            return ""
        text = self.context_manager.get_recent_extracted_text()
        if text:
            orig_text, _, _ = self.parse_response(text)
            if orig_text:
                return orig_text
        return ""



    def _get_or_encode_image(self, image: Image.Image) -> tuple[str, bool]:
        """Return (b64_string, was_cached). Skips re-encoding if image is unchanged."""
        phash = str(imagehash.phash(image, hash_size=IMAGE_HASH_SIZE))
        with self._lock:
            if phash == self._last_phash and self._last_b64:
                logger.debug("Image unchanged (phash=%s), reusing cached b64", phash)
                return self._last_b64, True
        b64 = self.encode_image(image)
        with self._lock:
            if phash == self._last_phash and self._last_b64:
                return self._last_b64, True
            self._last_phash = phash
            self._last_b64 = b64
            return b64, False

    def preprocess_image(self, image_data: bytes) -> Image.Image:
        """
        Preprocess image for vision-language model input.
        """
        image = Image.open(io.BytesIO(image_data))
        max_dimension = QWEN_MAX_DIMENSION
        width, height = image.size

        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Pad to a square to prevent the Vision Transformer from stretching the image
        width, height = image.size
        if width != height:
            target_dim = max(width, height)
            new_image = Image.new("RGB", (target_dim, target_dim), (0, 0, 0))
            
            if image.mode == "RGBA":
                image = image.convert("RGB")
                
            paste_x = (target_dim - width) // 2
            paste_y = (target_dim - height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image

        # Sharpen to make text edges crisper for OCR
        image = image.filter(ImageFilter.SHARPEN)

        return image

    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to lossless PNG for OCR accuracy."""
        # Ensure RGBA is converted for compatibility
        if image.mode == "RGBA":
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def create_prompt(self, source_lang: str, target_lang: str, mode: str, styles: list[str]) -> tuple[str, str]:
        """Create a terse OCR+Translation prompt tailored by user settings.
        Returns a tuple of (system_prompt, user_prompt).
        """
        mode_context = "a web interface" if mode == "Web" else "a video game interface"
        style_context = f" (Optionally use {', '.join(styles)} terms if it does not compromise accuracy)" if styles else ""

        system_prompt = (
            f"You are a highly precise OCR and translation engine.\n"
            f"You must strictly output ONLY this exact 3-part layout with NO markdown wrappers:\n"
            f"ORIGINAL:\n[Extracted {source_lang} text]\n\n"
            f"TRANSLATED:\n[Direct translation into {target_lang}]\n\n"
            f"CONFIDENCE:\n[Confidence float score between 0.0 and 1.0]\n\n"
            f"OCR RULES:\n"
            f"- SKIP game UI icons, decorative symbols, item icons, and non-text graphical elements entirely.\n"
            f"- If a character is unclear, output your best guess rather than repeating or stalling.\n"
            f"- Do NOT repeat the same character or word more than it actually appears in the image.\n"
            f"- If no readable text is found, output ORIGINAL: (none) and TRANSLATED: (none).\n\n"
            f"REASONING CONSTRAINTS:\n"
            f"- Keep your thinking process (if any) extremely brief, concise, and focused on resolving ambiguous characters.\n"
            f"- Do NOT write out coordinate grids, row-by-row lists, or long tables in your reasoning."
        )

        # 1. Load Glossary from Wiki
        glossary = self.load_glossary_from_wiki()
        if glossary:
            glossary_lines = ["\nUse the following game terminology glossary for translation mapping:"]
            for ch, en in glossary.items():
                glossary_lines.append(f"- {ch}: {en}")
            system_prompt += "\n" + "\n".join(glossary_lines)

        # 2. Perform RAG Search (using recent frame OCR history)
        query = self.get_recent_text_for_search()
        if query:
            try:
                local_searcher = LocalWikiSearcher(wiki_dir=self.wiki_dir)
                results = local_searcher.search(query, num_results=2)
                if results:
                    context_parts = ["\nLORE REFERENCE ARTICLES:\nUse the following background lore articles for context and translation accuracy:"]
                    for i, res in enumerate(results, 1):
                        title = res.get("title", "Untitled").replace("[LOCAL WIKI] ", "")
                        content = res.get("content", "")
                        context_parts.append(f"--- Article {i}: {title} ---\n{content}\n")
                    system_prompt += "\n" + "\n".join(context_parts)
            except Exception as e:
                logger.warning("RAG search in create_prompt failed: %s", e)

        user_prompt = (
            f"OCR the {source_lang} text from this image of {mode_context}, "
            f"then translate it to {target_lang}.{style_context}"
        )
        
        return system_prompt, user_prompt

    @staticmethod
    def _detect_repetition_loop(text: str) -> bool:
        """Detect if the tail of `text` is stuck in a repetition loop."""
        cleaned = text
        if "<think>" in text:
            if "</think>" in text:
                cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            else:
                # If the model is currently in the thinking block (has <think> but no closing tag),
                # we skip repetition detection entirely to allow the reasoning process to complete.
                return False

        if len(cleaned) < 20:
            return False
            
        # 1. Simple single character repeat
        if re.search(r'(.)\1{14,}$', cleaned):
            return True
            
        # 2. Multi-character regex for short patterns
        match = re.search(r'(.{2,50}?)\1{2,}$', cleaned)
        if match and len(match.group(0)) >= 20:
            return True
            
        # 3. Detect large structural loops (e.g., 15 to 1000+ chars) via string slicing
        tail = cleaned[-4000:] if len(cleaned) > 4000 else cleaned
        max_size = len(tail) // 3
        if max_size >= 15:
            size = 15
            while size <= max_size:
                chunk1 = tail[-size:]
                chunk2 = tail[-size*2:-size]
                chunk3 = tail[-size*3:-size*2]
                if chunk1 == chunk2 and chunk2 == chunk3:
                    return True
                size = max(size + 1, int(size * 1.5))
                    
        return False

    def parse_response(self, response: str) -> tuple[str, str, float]:
        """Parse the model response to extract original text, translation, and confidence.

        Returns (original_text, translated_text, confidence_score).
        """
        # Strip thinking tags if present
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()

        # Parse confidence score using keyword search
        confidence = 0.85
        conf_pattern = r'(?:(?:^|\n)[ \t]*(?:\d+\.\s*)?(?:\*|-)?\s*(?:\*\*)?(?:CONFIDENCE)\b|[\(\[]\s*CONFIDENCE\s*[\)\]])\D*?([0-9.]+)'
        conf_matches = list(re.finditer(conf_pattern, cleaned, re.IGNORECASE))
        if conf_matches:
            try:
                confidence = float(conf_matches[-1].group(1).strip())
                confidence = max(0.0, min(1.0, confidence))  # clamp between 0.0 and 1.0
            except ValueError:
                pass

        # Split on double-newline for fallback parsing (confidence only)
        parts = re.split(r'\n\s*\n', cleaned, maxsplit=2)
        if not conf_matches and len(parts) == 3:
            try:
                val = float(parts[2].strip())
                confidence = max(0.0, min(1.0, val))
            except ValueError:
                pass

        # Extract ORIGINAL and TRANSLATED sections using a highly resilient regex.
        # Markers must be formatted either at the start of a line (with optional list/bold prefixes)
        # or enclosed in brackets/parentheses to avoid matching conversational step titles like "Refine Translation:".
        orig_marker_pat = r'(?:(?:^|\n)[ \t]*(?:\d+\.\s*)?(?:\*|-)?\s*(?:\*\*)?ORIGINAL\b|[\(\[]\s*ORIGINAL\s*[\)\]])[^\w\n]*[:\n]?'
        trans_marker_pat = r'(?:(?:^|\n)[ \t]*(?:\d+\.\s*)?(?:\*|-)?\s*(?:\*\*)?TRANSLAT[a-zA-Z]*\b|[\(\[]\s*TRANSLAT[a-zA-Z]*\s*[\)\]])[^\w\n]*[:\n]?'

        orig_match = None
        orig_matches = list(re.finditer(
            r'(?:(?:^|\n)[ \t]*(?:\d+\.\s*)?(?:\*|-)?\s*(?:\*\*)?ORIGINAL\b|[\(\[]\s*ORIGINAL\s*[\)\]])[^\w\n]*[:\n][ \t]*(.*?)(?=\n\s*(?:\d+\.\s*)?\**\s*(?:[\(\[]?\s*(?:TRANSLAT|CONFIDENCE)|TRANSLAT[a-zA-Z]*\b|CONFIDENCE\b)|\Z)',
            cleaned, re.DOTALL | re.IGNORECASE
        ))
        if orig_matches:
            orig_match = orig_matches[-1]

        trans_match = None
        trans_matches = list(re.finditer(
            r'(?:(?:^|\n)[ \t]*(?:\d+\.\s*)?(?:\*|-)?\s*(?:\*\*)?TRANSLAT[a-zA-Z]*\b|[\(\[]\s*TRANSLAT[a-zA-Z]*\s*[\)\]])[^\w\n]*[:\n][ \t]*(.*?)(?=\n\s*(?:\d+\.\s*)?\**\s*(?:[\(\[]?\s*(?:ORIGINAL|CONFIDENCE)|ORIGINAL\b|CONFIDENCE\b)|\Z)',
            cleaned, re.DOTALL | re.IGNORECASE
        ))
        if trans_matches:
            trans_match = trans_matches[-1]

        has_orig_marker = bool(re.search(orig_marker_pat, cleaned, re.IGNORECASE))
        has_trans_marker = bool(re.search(trans_marker_pat, cleaned, re.IGNORECASE))

        # If it has at least one marker, we trust our regex matches (even if empty because it's streaming)
        if has_orig_marker or has_trans_marker:
            original_text = ""
            translation = ""
            
            if orig_match:
                original_text = orig_match.group(1).strip()
                original_text = original_text.strip(' \t\n\r"\'`*•-')
                original_text = re.sub(
                    r'^(?:The image contains.*?text:|The text in the image is:|The extracted text is:|The.*?text is:)\s*',
                    '', original_text, flags=re.IGNORECASE
                )
                original_text = original_text.strip(' \t\n\r"\'`*•-')
                
            if trans_match:
                translation = trans_match.group(1).strip()
                translation = translation.strip(' \t\n\r"\'`*•-')

            # Filter out template placeholders
            placeholder_pat = re.compile(
                r'^\[\s*.*?(?:text|translation|score|original|translated|english|chinese|placeholder|insert|label|here|extract|direct).*?\]$',
                re.IGNORECASE
            )
            if placeholder_pat.match(original_text):
                original_text = ""
            if placeholder_pat.match(translation):
                translation = ""
                
            return original_text, translation, confidence

        # If no markers are present AT ALL, it is either still streaming unstructured reasoning,
        # or the model completely failed the layout. We return empty strings to prevent leaking
        # unstructured reasoning to the UI, unless the fallback double-newline split succeeded.
        if len(parts) == 3:
            try:
                val = float(parts[2].strip())
                confidence = max(0.0, min(1.0, val))
                return parts[0].strip(), parts[1].strip(), confidence
            except ValueError:
                pass

        return "", "", confidence

    async def process_frame(self, image_data: bytes, source_lang: str, target_lang: str, mode: str, styles: list[str]) -> list[TranslationResult]:
        """
        Process a single frame with unified OCR and translation via OmniRouter.
        """
        results_data = None
        async for _, _, extra in self.stream_frame(image_data, source_lang, target_lang, mode, styles):
            if extra is not None:
                results_data = extra[0]
        if results_data is None:
            raise RuntimeError("Stream frame failed to produce results.")
        return results_data

    async def stream_frame(self, image_data: bytes, source_lang: str, target_lang: str, mode: str, styles: list[str]):
        """Streaming version of process_frame. Yields (original, translated, results_data) partials."""
        if not self.engine:
            raise RuntimeError("Engine not initialized.")

        image = self.preprocess_image(image_data)
        self.context_manager.add_frame(image)
        system_prompt, user_prompt = self.create_prompt(source_lang, target_lang, mode, styles)

        # Check cache
        b64_image, was_cached = self._get_or_encode_image(image)
        with self._lock:
            cached_results = self._last_results
        if was_cached and cached_results is not None:
            logger.info("Returning cached translation result (identical frame) via stream")
            with self._lock:
                self.last_stream_results = cached_results
            original_combined = "\n".join(r.original_text for r in cached_results if r.original_text)
            translation_combined = "\n".join(r.translated_text for r in cached_results if r.translated_text)
            yield original_combined, translation_combined, (cached_results, None, None)
            return

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ]

        max_tok = max(MODE_MAX_TOKENS.get(mode, self.config.max_tokens), 2048)

        max_attempts = 2
        for attempt in range(max_attempts):
            accumulated = ""
            loop_detected = False

            try:
                # Escalate parameters on retry to break out of loop
                temp = self.config.temperature if attempt == 0 else min(0.9, self.config.temperature + 0.5)
                rep_penalty = 1.15 if attempt == 0 else 1.5

                logger.info(
                    "Sending streaming translation request to OmniRouter via OpenAI (attempt %d/%d, temp=%.1f, rep=%.2f)...",
                    attempt + 1, max_attempts, temp, rep_penalty,
                )
                stream = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.get_vision_model_name(),
                        messages=messages,
                        max_tokens=max_tok,
                        temperature=temp,
                        frequency_penalty=0.6,
                        presence_penalty=0.2,
                        stream=True,
                        extra_body={
                            "repetition_penalty": rep_penalty,
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    ),
                    timeout=vision_timeout_for_mode(mode),
                )
                last_log_time = time.time()
                last_log_len = 0
                last_finish = None
                in_thinking = False
                
                async for chunk in stream:
                    delta = ""
                    if chunk.choices:
                        reasoning = getattr(chunk.choices[0].delta, 'reasoning_content', None)
                        content = chunk.choices[0].delta.content or ""
                        
                        if isinstance(reasoning, str) and reasoning:
                            if not in_thinking:
                                delta += "<think>"
                                in_thinking = True
                            delta += reasoning
                        
                        if content:
                            if in_thinking:
                                delta += "</think>"
                                in_thinking = False
                            delta += content
                            
                    accumulated += delta
                    if chunk.choices and chunk.choices[0].finish_reason:
                        last_finish = chunk.choices[0].finish_reason

                    # Throttle stream logging to once per second
                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        new_text = accumulated[last_log_len:]
                        clean_new = new_text.replace('\n', '\\n')
                        logger.debug("Stream progress [%d chars]: ...%s", len(accumulated), clean_new)
                        last_log_time = current_time
                        last_log_len = len(accumulated)

                    # Check for repetition loop every 60+ chars
                    if self._detect_repetition_loop(accumulated):
                        logger.warning(
                            "Repetition loop detected at %d chars (attempt %d/%d), aborting stream. Tail: %r",
                            len(accumulated), attempt + 1, max_attempts, accumulated[-80:],
                        )
                        loop_detected = True
                        # Cancel the stream
                        await stream.response.aclose()
                        break

                    orig, trans, _ = self.parse_response(accumulated)
                    yield orig, trans, None

                if in_thinking:
                    accumulated += "</think>"
                    in_thinking = False

                if loop_detected:
                    if attempt < max_attempts - 1:
                        logger.info("Retrying with escalated parameters...")
                        continue
                    else:
                        # Final attempt also looped — truncate the repeated tail and use what we have
                        logger.warning("Loop persisted after %d attempts, truncating repeated content", max_attempts)
                        # Strip the repeated tail
                        accumulated = re.sub(r'(.{1,12}?)\1{4,}$', r'\1', accumulated)

                results = self._build_result(accumulated, image)

                # Propagate truncation flag
                if last_finish == "length":
                    if not results:
                        results = [TranslationResult(
                            original_text="",
                            translated_text="⚠️ Translation request truncated (VLM token limit reached).",
                            confidence=0.0
                        )]
                    for r in results:
                        r.truncated = True
                        r.raw_output = accumulated

                with self._lock:
                    self._last_results = results
                    self.last_stream_results = results
                    self._last_messages = messages
                    self._last_raw_output = accumulated

                extracted = [r.original_text for r in results]
                self.context_manager.update_last_frame_data("\n".join(extracted), results)

                yield orig, trans, (results, messages, accumulated)
                return  # Success, exit the retry loop

            except asyncio.TimeoutError:
                logger.error("OpenAI streaming translation timed out (mode=%s)", mode)
                raise RuntimeError("Translation request timed out.") from None

            except Exception as e:
                logger.error("Error during OpenAI API streaming inference: %s", e, exc_info=True)
                raise

    async def continue_generation(
        self,
        messages: list[dict],
        partial_output: str,
        mode: str = "Game",
    ):
        """Continue a truncated generation by replaying context.

        Appends the partial output as an assistant message, then asks
        the model to finish where it left off.  Yields
        ``(full_text_so_far, finish_reason)`` tuples for streaming.
        """
        continuation_messages = list(messages)  # shallow copy

        # Add the truncated assistant reply
        continuation_messages.append({
            "role": "assistant",
            "content": partial_output,
        })

        # Add a continuation prompt
        continuation_messages.append({
            "role": "user",
            "content": (
                "Your previous response was cut off. "
                "Continue EXACTLY where you left off. "
                "Do NOT repeat any text you already wrote. "
                "Do NOT restart the ORIGINAL/TRANSLATED/CONFIDENCE format from scratch — "
                "just output the remaining text."
            ),
        })

        max_tok = max(MODE_MAX_TOKENS.get(mode, self.config.max_tokens), 2048)
        accumulated = ""

        try:
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.get_vision_model_name(),
                    messages=continuation_messages,
                    max_tokens=max_tok,
                    temperature=0.2,
                    frequency_penalty=0.6,
                    presence_penalty=0.2,
                    stream=True,
                    extra_body={
                        "repetition_penalty": 1.15,
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                ),
                timeout=vision_timeout_for_mode(mode),
            )

            last_finish = None
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                accumulated += delta
                if chunk.choices and chunk.choices[0].finish_reason:
                    last_finish = chunk.choices[0].finish_reason
                yield partial_output + accumulated, last_finish, None

            # Update stored context for potential further continuations
            with self._lock:
                self._last_raw_output = partial_output + accumulated
                self._last_messages = continuation_messages

            yield partial_output + accumulated, last_finish, (continuation_messages, partial_output + accumulated)

        except asyncio.TimeoutError:
            logger.error("Continuation timed out (mode=%s)", mode)
            raise RuntimeError("Continuation request timed out.") from None

        except Exception as e:
            logger.error("Error during continuation: %s", e, exc_info=True)
            raise

    def create_cinematic_prompt(self, transcript: str, target_lang: str, styles: list[str], source_lang: str = "Chinese") -> str:
        """Create a prompt combining audio transcript and visual OCR for cinematic mode."""
        style_context = f" (Optionally use {', '.join(styles)} terms if it does not compromise accuracy)" if styles else ""
        system_prompt = (
            f"Translate the following dialogue to {target_lang}. Use the provided system audio transcription for context, "
            f"and cross-reference it with the OCR from the provided image of the game interface to ensure accurate character names and tone.{style_context}\n\n"
            f"Audio Transcript: {transcript}\n\n"
            f"CRITICAL SYSTEM DIRECTIVES:\n"
            f"- Output a single confidence estimation float (0.0 to 1.0) based on your certainty of the OCR and translation.\n"
            f"- Output ONLY the 3-part layout below.\n\n"
            f"Reply strictly in this 3-part layout with NO conversational introduction or Markdown formatting:\n"
            f"ORIGINAL:\n[Extracted {source_lang} dialogue text]\n\n"
            f"TRANSLATED:\n[Direct translation into {target_lang}]\n\n"
            f"CONFIDENCE:\n[Confidence float score]\n\n"
            f"REASONING CONSTRAINTS:\n"
            f"- Keep your thinking process (if any) extremely brief, concise, and focused entirely on cross-referencing and translation alignment.\n"
            f"- Do NOT write long explanations or list translations repeatedly in your reasoning."
        )

        # 1. Load Glossary from Wiki
        glossary = self.load_glossary_from_wiki()
        if glossary:
            glossary_lines = ["\nUse the following game terminology glossary for translation mapping:"]
            for ch, en in glossary.items():
                glossary_lines.append(f"- {ch}: {en}")
            system_prompt += "\n" + "\n".join(glossary_lines)

        # 2. Perform RAG Search (using transcript as query)
        if transcript:
            try:
                local_searcher = LocalWikiSearcher(wiki_dir=self.wiki_dir)
                results = local_searcher.search(transcript, num_results=2)
                if results:
                    context_parts = ["\nLORE REFERENCE ARTICLES:\nUse the following background lore articles for context and translation accuracy:"]
                    for i, res in enumerate(results, 1):
                        title = res.get("title", "Untitled").replace("[LOCAL WIKI] ", "")
                        content = res.get("content", "")
                        context_parts.append(f"--- Article {i}: {title} ---\n{content}\n")
                    system_prompt += "\n" + "\n".join(context_parts)
            except Exception as e:
                logger.warning("RAG search in create_cinematic_prompt failed: %s", e)

        return system_prompt

    async def process_cinematic(self, image_data: bytes, transcript: str, target_lang: str, styles: list[str], b64_image: str | None = None, image: Image.Image | None = None, source_lang: str = "Chinese", audio_bytes: bytes | None = None) -> list[TranslationResult]:
        """
        Process a single frame along with an audio transcript using OmniRouter.
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized.")

        if image is None:
            image = self.preprocess_image(image_data)
        self.context_manager.add_frame(image)

        prompt = self.create_cinematic_prompt(transcript, target_lang, styles, source_lang=source_lang)
        if b64_image is None:
            b64_image = self.encode_image(image)

        content_list = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
        ]

        if audio_bytes:
            import base64
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            content_list.append({
                "type": "input_audio",
                "input_audio": {
                    "data": b64_audio,
                    "format": "wav"
                }
            })

        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]


        try:
            max_attempts = 2
            results = []
            
            for attempt in range(max_attempts):
                temp = 0.2 if attempt == 0 else 0.7
                rep_penalty = 1.15 if attempt == 0 else 1.5
                
                logger.info(
                    "Sending cinematic translation request to OmniRouter via OpenAI (attempt %d/%d, temp=%.1f, rep=%.2f)...",
                    attempt + 1, max_attempts, temp, rep_penalty
                )
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.get_vision_model_name(),
                        messages=messages,
                        max_tokens=self.config.max_tokens,
                        temperature=temp,
                        frequency_penalty=0.6,
                        presence_penalty=0.2,
                        extra_body={
                            "repetition_penalty": rep_penalty,
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    ),
                    timeout=vision_timeout_for_mode("Document"),
                )

                choice = response.choices[0] if response.choices else None
                final_output = (choice.message.content or "") if choice else ""

                # Fallback for reasoning models that return content in reasoning_content
                if not final_output and choice:
                    reasoning = getattr(choice.message, 'reasoning_content', None)
                    if isinstance(reasoning, str):
                        final_output = reasoning

                # If the response was truncated while reasoning (content is empty)
                if not final_output and choice and choice.finish_reason == "length":
                    logger.warning("VLM execution was truncated (finish_reason=length) while thinking. Try increasing max_tokens.")
                    results = [TranslationResult(
                        original_text="",
                        translated_text="⚠️ Translation request truncated (VLM token limit reached).",
                        confidence=0.0
                    )]
                    break

                # Loop detection for non-streaming response
                if self._detect_repetition_loop(final_output):
                    if attempt < max_attempts - 1:
                        logger.warning(
                            "Repetition loop detected in cinematic output (attempt %d/%d), retrying.",
                            attempt + 1, max_attempts
                        )
                        continue
                    else:
                        logger.warning("Loop persisted after %d attempts, truncating repeated content", max_attempts)
                        final_output = re.sub(r'(.{1,12}?)\1{4,}$', r'\1', final_output)
                        
                results = self._build_result(final_output, image)
                break

            extracted = [r.original_text for r in results]
            self.context_manager.update_last_frame_data("\n".join(extracted), results)

            return results

        except asyncio.TimeoutError:
            logger.error("OpenAI cinematic translation timed out")
            raise RuntimeError("Cinematic translation request timed out.") from None

        except Exception as e:
            logger.error("Error during OpenAI API inference (cinematic): %s", e, exc_info=True)
            raise

    async def translate_query(self, query: str, target_lang: str) -> str:
        """Translate a search query into another language using the LLM."""
        if not self.client:
            return query
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.get_model_name(),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"Translate the following search query to {target_lang}. "
                                "Output ONLY the translated query, nothing else."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    max_tokens=200,
                    temperature=0.1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
                timeout=CHAT_AUX_TIMEOUT_SECONDS,
            )
            ch0 = response.choices[0] if response.choices else None
            translated = ch0.message.content or "" if ch0 else ""
            # Strip thinking tags if present
            translated = re.sub(r'<think>.*?</think>', '', translated, flags=re.DOTALL).strip()
            translated = re.sub(r'<think>.*$', '', translated, flags=re.DOTALL).strip()
            if translated:
                logger.info("Translated query '%s' → '%s'", query, translated)
                return translated
        except Exception as e:
            logger.warning("Query translation failed: %s. Using original.", e)
        return query

    async def process_chat(self, message: str, source_lang: str = "zh-CN") -> str:
        """Process a contextual chat message using the sliding window context via OmniRouter."""
        if not self.client:
            raise RuntimeError("Engine not initialized. Call init_engine() first.")

        # Strip all XML-like tool-call patterns from user input
        sanitized = re.sub(r'</?(?:tool_call|function|parameter)\b[^>]*>', '', message)
        # Also strip the <function=name> variant used in text-based tool calls
        sanitized = re.sub(r'<function=[^>]*>', '', sanitized)
        # Strip any remaining angle-bracket patterns that look like tool syntax
        sanitized = re.sub(r'<parameter=[^>]*>', '', sanitized)
        logger.info("Processing chat message (%d chars)", len(sanitized))

        # Add user message to context manager
        self.context_manager.add_user_message(sanitized, with_image=True)

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
                        openai_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}})

            # OpenAI requires content to be a string if it's just text, or list if multimodal
            if len(openai_content) == 1 and openai_content[0]["type"] == "text":
                final_content = openai_content[0]["text"]
            else:
                final_content = openai_content

            openai_messages.append({"role": role, "content": final_content})

        system_msg = {
            "role": "system",
            "content": (
                "You are an AI assistant with access to various tools (generating/editing images, text-to-speech, transcription, vision, knowledge search).\n"
                "If the user asks a question about facts, lore, games, or real-world information, "
                "you MUST call the search tool INSTEAD of answering from memory.\n\n"
                "To search, you can use the search_knowledge tool or output EXACTLY this format (and nothing else):\n"
                "<tool_call>\n"
                "<function=search_knowledge>\n"
                "<parameter=query>your search query here</parameter>\n"
                "<parameter=language>zh-CN</parameter>\n"
                "</function>\n"
                "</tool_call>\n\n"
                "The language parameter is optional (defaults to zh-CN). "
                "Use en-US for English queries. "
                "Do NOT answer factual questions without searching first.\n\n"
                "IMPORTANT TOOL GUIDELINES:\n"
                "1. Only call tools if they are directly relevant to fulfilling the user's request. Do not call image generation tools for simple questions.\n"
                "2. Do not use analyze_image for images already visible in the conversation history unless the user explicitly requests a detailed analysis of that specific image/screenshot.\n"
                "3. If the user asks to analyze the current screen or screenshot, use 'screenshot' or 'image' as the image_path in analyze_image."
            ),
        }
        openai_messages.insert(0, system_msg)

        model_to_use = self.get_model_name()

        try:
            final_output = ""
            for turn in range(3):
                logger.info("Sending chat request to OmniRouter via OpenAI (turn %d)...", turn + 1)
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model_to_use,
                        messages=openai_messages,
                        max_tokens=self.config.max_tokens,
                        temperature=0.7,  # Higher temp for chat
                        tools=OMNI_TOOLS,
                    ),
                    timeout=CHAT_TIMEOUT_SECONDS,
                )

                choice = response.choices[0] if response.choices else None
                if not choice:
                    break

                msg = choice.message
                final_output = msg.content or ""

                # Handle structured tool calls
                if msg.tool_calls:
                    # Append assistant message with tool calls
                    openai_messages.append(msg)

                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        raw_args = tool_call.function.arguments or "{}"
                        logger.info("Executing tool call: %s with args: %s", func_name, raw_args)
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = {}

                        tool_result = None
                        try:
                            if func_name == "generate_image":
                                prompt = args.get("prompt", "")
                                size = args.get("size", "1024x1024")
                                
                                media_dir = os.path.join(self.wiki_dir, "media")
                                os.makedirs(media_dir, exist_ok=True)
                                
                                base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
                                base_url_no_v1 = base_url.removesuffix("/v1")
                                async with LemonadeClient(base_url=base_url_no_v1) as lemonade:
                                    res = await lemonade.generate_image(prompt, model=self.router.image(), size=size)
                                
                                img_data = res["data"][0]["b64_json"]
                                img_bytes = base64.b64decode(img_data)
                                filename = f"gen_{int(time.time())}.png"
                                filepath = os.path.join(media_dir, filename)
                                with open(filepath, "wb") as f:
                                    f.write(img_bytes)
                                    
                                tool_result = {"status": "success", "image_path": filepath, "url": f"file://{filepath}"}
                                
                            elif func_name == "edit_image":
                                prompt = args.get("prompt", "")
                                img_path = args.get("image_path", "")
                                
                                clean_path = img_path.replace("file://", "").strip()
                                if clean_path.lower() in ("image", "screenshot", "current", "active", "latest") or not clean_path:
                                    frame = self.context_manager.get_latest_frame()
                                    if frame:
                                        media_dir = os.path.join(self.wiki_dir, "media")
                                        os.makedirs(media_dir, exist_ok=True)
                                        screenshot_path = os.path.join(media_dir, "current_screenshot.png")
                                        frame.save(screenshot_path, "PNG")
                                        clean_path = screenshot_path
                                        logger.info("Intercepted symbolic image path. Saved latest frame to %s", clean_path)

                                clean_path = self._validate_file_path(clean_path)
                                with open(clean_path, "rb") as f:
                                    img_bytes = f.read()
                                    
                                media_dir = os.path.join(self.wiki_dir, "media")
                                os.makedirs(media_dir, exist_ok=True)
                                
                                base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
                                base_url_no_v1 = base_url.removesuffix("/v1")
                                async with LemonadeClient(base_url=base_url_no_v1) as lemonade:
                                    res = await lemonade.edit_image(img_bytes, prompt, model=self.router.edit())
                                    
                                out_data = res["data"][0]["b64_json"]
                                out_bytes = base64.b64decode(out_data)
                                filename = f"edit_{int(time.time())}.png"
                                filepath = os.path.join(media_dir, filename)
                                with open(filepath, "wb") as f:
                                    f.write(out_bytes)
                                    
                                tool_result = {"status": "success", "image_path": filepath, "url": f"file://{filepath}"}
                                
                            elif func_name == "text_to_speech":
                                text = args.get("text", "")
                                voice = args.get("voice", "af_heart")
                                
                                media_dir = os.path.join(self.wiki_dir, "media")
                                os.makedirs(media_dir, exist_ok=True)
                                
                                base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
                                base_url_no_v1 = base_url.removesuffix("/v1")
                                async with LemonadeClient(base_url=base_url_no_v1) as lemonade:
                                    audio_bytes = await lemonade.tts(text, voice=voice, model=self.router.tts())
                                    
                                filename = f"speech_{int(time.time())}.wav"
                                filepath = os.path.join(media_dir, filename)
                                with open(filepath, "wb") as f:
                                    f.write(audio_bytes)
                                    
                                play_audio_simple(audio_bytes)
                                
                                tool_result = {"status": "success", "audio_path": filepath, "url": f"file://{filepath}"}
                                
                            elif func_name == "transcribe_audio":
                                audio_path = args.get("audio_path", "").replace("file://", "").strip()
                                
                                audio_path = self._validate_file_path(audio_path)
                                with open(audio_path, "rb") as f:
                                    audio_bytes = f.read()
                                    
                                base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
                                base_url_no_v1 = base_url.removesuffix("/v1")
                                async with LemonadeClient(base_url=base_url_no_v1) as lemonade:
                                    transcript = await lemonade.transcribe(audio_bytes, model=self.router.asr())
                                    
                                tool_result = {"status": "success", "transcript": transcript}
                                
                            elif func_name == "analyze_image":
                                img_path = args.get("image_path", "").replace("file://", "").strip()
                                question = args.get("question", "")
                                
                                clean_path = img_path
                                if clean_path.lower() in ("image", "screenshot", "current", "active", "latest") or not clean_path:
                                    frame = self.context_manager.get_latest_frame()
                                    if frame:
                                        media_dir = os.path.join(self.wiki_dir, "media")
                                        os.makedirs(media_dir, exist_ok=True)
                                        screenshot_path = os.path.join(media_dir, "current_screenshot.png")
                                        frame.save(screenshot_path, "PNG")
                                        clean_path = screenshot_path
                                        logger.info("Intercepted symbolic image path. Saved latest frame to %s", clean_path)

                                clean_path = self._validate_file_path(clean_path)
                                with open(clean_path, "rb") as f:
                                    img_bytes = f.read()
                                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                                
                                response_vision = await self.client.chat.completions.create(
                                    model=self.router.vision(),
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {"type": "text", "text": question},
                                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                                            ]
                                        }
                                    ],
                                    max_tokens=self.config.max_tokens,
                                )
                                analysis = response_vision.choices[0].message.content or ""
                                tool_result = {"status": "success", "analysis": analysis}
                                
                            elif func_name in ("perform_web_search", "search_knowledge"):
                                query = args.get("query", "")
                                language = args.get("language", "zh-CN")
                                
                                local_searcher = LocalWikiSearcher(wiki_dir=self.wiki_dir)
                                local_results = local_searcher.search(query, num_results=3)

                                web_results = []
                                searcher = None
                                try:
                                    searcher = WebSearcher()
                                    lang_name = _LANG_CODE_TO_NAME.get(source_lang, source_lang)
                                    translated_query = await self.translate_query(query, lang_name)
                                    web_results = await searcher.dual_search(
                                        query=query,
                                        target_lang=language,
                                        translated_query=translated_query,
                                        source_lang=source_lang,
                                    )
                                except Exception as e:
                                    logger.warning("Web search failed: %s. Relying on local wiki.", e)
                                finally:
                                    if searcher is not None:
                                        await searcher.close()

                                results = local_results + web_results

                                if web_results:
                                    compiler = WikiCompiler(wiki_dir=self.wiki_dir)
                                    content_lines = []
                                    for r in web_results:
                                        title = r.get("title", "No Title")
                                        href = markdown_http_https_url_or_none(r.get("url", ""))
                                        if href:
                                            content_lines.append(f"### [{title}]({href})")
                                        else:
                                            content_lines.append(
                                                f"### {title}\n*(source URL omitted — unsupported or disallowed scheme)*"
                                            )
                                        content_lines.append(r.get("content", ""))
                                    compiler.compile(query, "\n".join(content_lines), metadata={"sources": web_results})

                                summary_parts = []
                                for i, r in enumerate(results[:6], 1):
                                    title = r.get('title', 'Untitled')
                                    content = r.get('content', '')
                                    url = r.get('url', '')
                                    summary_parts.append(
                                        f"--- Source {i}: {title} ---\n"
                                        f"URL: {url}\n"
                                        f"{content}\n"
                                    )
                                summary = "\n".join(summary_parts)
                                tool_result = {"status": "success", "search_results": summary}
                            else:
                                tool_result = {"error": f"Unknown function: {func_name}"}
                        except FileNotFoundError as ex:
                            logger.warning("Error executing tool %s (file not found): %s", func_name, ex)
                            tool_result = {"error": f"File not found: {os.path.basename(ex.filename) if ex.filename else 'specified file'}"}
                        except PermissionError as ex:
                            logger.warning("Error executing tool %s (permission error): %s", func_name, ex)
                            tool_result = {"error": "Access denied: path is outside allowed directories"}
                        except Exception as ex:
                            logger.error("Error executing tool %s: %s", func_name, ex, exc_info=True)
                            tool_result = {"error": f"Tool '{func_name}' failed. Please try a different approach."}

                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": json.dumps(tool_result)
                        })
                    continue

                # Detect text-based search/knowledge tool calls
                tool_call_match = re.search(
                    r'<tool_call>.*?<function=(\w+)>(.*?)</function>.*?</tool_call>',
                    final_output, re.DOTALL,
                )
                if tool_call_match:
                    func_body = tool_call_match.group(2)
                    query_match = re.search(
                        r'<parameter=query>\s*(.*?)\s*</parameter>',
                        func_body, re.DOTALL,
                    )
                    lang_match = re.search(
                        r'<parameter=language>\s*(.*?)\s*</parameter>',
                        func_body, re.DOTALL,
                    )
                    query = query_match.group(1).strip() if query_match else ""
                    language = lang_match.group(1).strip() if lang_match else "zh-CN"

                    if query:
                        logger.info("Agent requested search for: %s (lang: %s, source: %s)", query, language, source_lang)
                        local_searcher = LocalWikiSearcher(wiki_dir=self.wiki_dir)
                        local_results = local_searcher.search(query, num_results=3)

                        web_results = []
                        searcher = None
                        try:
                            searcher = WebSearcher()
                            lang_name = _LANG_CODE_TO_NAME.get(source_lang, source_lang)
                            translated_query = await self.translate_query(query, lang_name)
                            web_results = await searcher.dual_search(
                                query=query,
                                target_lang=language,
                                translated_query=translated_query,
                                source_lang=source_lang,
                            )
                        except Exception as e:
                            logger.warning("Web search failed: %s. Relying on local wiki.", e)
                        finally:
                            if searcher is not None:
                                await searcher.close()

                        results = local_results + web_results

                        if web_results:
                            compiler = WikiCompiler(wiki_dir=self.wiki_dir)
                            content_lines = []
                            for r in web_results:
                                title = r.get("title", "No Title")
                                href = markdown_http_https_url_or_none(r.get("url", ""))
                                if href:
                                    content_lines.append(f"### [{title}]({href})")
                                else:
                                    content_lines.append(
                                        f"### {title}\n*(source URL omitted — unsupported or disallowed scheme)*"
                                    )
                                content_lines.append(r.get("content", ""))
                            compiler.compile(query, "\n".join(content_lines), metadata={"sources": web_results})

                        summary_parts = []
                        for i, r in enumerate(results[:6], 1):
                            title = r.get('title', 'Untitled')
                            content = r.get('content', '')
                            url = r.get('url', '')
                            summary_parts.append(
                                f"--- Source {i}: {title} ---\n"
                                f"URL: {url}\n"
                                f"{content}\n"
                            )
                        summary = "\n".join(summary_parts)

                        openai_messages.append({
                            "role": "assistant",
                            "content": f"I searched for '{query}' and found several relevant sources.",
                        })
                        openai_messages.append({
                            "role": "user",
                            "content": (
                                f"Here is the information I found:\n\n{summary}\n\n"
                                "Based on these sources, please answer my original question "
                                "thoroughly and in detail. Synthesize the information from "
                                "all sources into a clear, comprehensive response."
                            ),
                        })
                        continue

                # No tool calls or match, break out of loop
                break

            cleaned_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
            cleaned_output = re.sub(r'<think>.*$', '', cleaned_output, flags=re.DOTALL).strip()
            cleaned_output = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned_output, flags=re.DOTALL).strip()

            # Add assistant message to context
            self.context_manager.add_assistant_message(cleaned_output)

            return cleaned_output

        except asyncio.TimeoutError:
            logger.error("OpenAI chat request timed out")
            raise RuntimeError("Chat request timed out.") from None

        except Exception as e:
            logger.error("Error during OpenAI chat inference: %s", e, exc_info=True)
            raise

    def _build_result(self, response: str, image: Image.Image) -> list[TranslationResult]:
        """Build TranslationResult from model response, populating dynamic confidence."""
        original_text, translated_text, confidence = self.parse_response(response)

        if not translated_text.strip():
            logger.debug("No text detected in image")
            return []

        img_width, img_height = image.size

        # Determine accuracy quality reason
        if confidence >= 0.85:
            reason = "full_pass"
        elif confidence >= 0.70:
            reason = "acceptable_pass"
        else:
            reason = "low_contrast_or_ambiguous_ocr"

        result = TranslationResult(
            translated_text=translated_text,
            x=0.0,
            y=0.0,
            width=float(img_width),
            height=float(img_height),
            confidence=confidence,
            accuracy=AccuracyScore(score=confidence, reason=reason),
            original_text=original_text,
            style=TextStyle(),
        )

        return [result]

    async def prewarm_model(self) -> None:
        """Pre-load the target model into Lemonade Server VRAM."""
        await self.init_engine()
        
        # Discover models
        await self.router.discover_async()
        
        model_to_load = self.config.model_name
        is_omni = False
        
        if not model_to_load or model_to_load in ("omni-router", "default"):
            if self.router.omni_model_id:
                model_to_load = self.router.omni_model_id
                is_omni = True
            else:
                model_to_load = self.router.llm()
        elif self.router.is_omni_model(model_to_load):
            is_omni = True

        if not model_to_load:
            logger.info("No LLM model resolved for virtual routing. Skipping prewarming.")
            return

        base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
        base_url_no_v1 = base_url.removesuffix("/v1")
        try:
            async with LemonadeClient(base_url=base_url_no_v1) as client:
                logger.debug("Checking models on Lemonade server...")
                # Get pulled and registered models
                pulled_models = await client.list_models()
                pulled_ids = [m.get("id") for m in pulled_models if m.get("id")]
                
                registered_models = await client.list_models(show_all=True)
                registered_ids = [m.get("id") for m in registered_models if m.get("id")]

                if is_omni:
                    if model_to_load not in registered_ids:
                        logger.warning(
                            "Omni model '%s' is not registered on Lemonade server. "
                            "Registered models: %s. Skipping pre-warming.",
                            model_to_load, registered_ids
                        )
                        return
                else:
                    if model_to_load not in pulled_ids:
                        logger.warning(
                            "Model '%s' is not pulled/available on Lemonade server. "
                            "Pulled models: %s. Skipping VRAM pre-warming.",
                            model_to_load, pulled_ids
                        )
                        return

                logger.info("Pre-warming model '%s' (is_omni=%s) in Lemonade Server...", model_to_load, is_omni)
                try:
                    await client.load_model(model_to_load)
                except Exception as load_err:
                    if not is_omni:
                        logger.debug("Explicit /v1/load failed, falling back to dummy inference: %s", load_err)
                        await self.client.chat.completions.create(
                            model=model_to_load,
                            messages=[{"role": "user", "content": "warmup"}],
                            max_tokens=1
                        )
                    else:
                        logger.warning("Failed to load omni model '%s': %s", model_to_load, load_err)
        except Exception as e:
            logger.warning("VRAM Pre-warming failed: %s", e)

    async def close(self):
        """Close the client properly."""
        if self.engine:
            self.engine.shutdown()
