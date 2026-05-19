"""Vision-Language Processing Pipeline for unified OCR and translation using Lemonade-SDK OmniRouter (OpenAI Compatible)."""

import asyncio
import base64
import io
import logging
import os
import re
import json
from dataclasses import dataclass

from PIL import Image
from openai import AsyncOpenAI
import imagehash

from shared_types.constants import DEFAULT_API_URL, DEFAULT_MAX_TOKENS, QWEN_MAX_DIMENSION, IMAGE_HASH_SIZE, MODE_MAX_TOKENS
from shared_types.models import TranslationResult, TextStyle
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

logger = logging.getLogger(__name__)


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

        # Perceptual hash caching properties
        self._last_phash: str | None = None
        self._last_b64: str | None = None
        self._last_results: list[TranslationResult] | None = None
        self.last_stream_results: list[TranslationResult] = []

        # Initialize context manager for stateful interactions
        self.context_manager = ContextManager(max_frames=1)

    @property
    def client(self) -> AsyncOpenAI:
        return self.engine.client

    async def init_engine(self):
        """No-op stub for backward compatibility."""
        pass

    def _get_or_encode_image(self, image: Image.Image) -> tuple[str, bool]:
        """Return (b64_string, was_cached). Skips re-encoding if image is unchanged."""
        phash = str(imagehash.phash(image, hash_size=IMAGE_HASH_SIZE))
        if phash == self._last_phash and self._last_b64:
            logger.debug("Image unchanged (phash=%s), reusing cached b64", phash)
            return self._last_b64, True
        b64 = self.encode_image(image)
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
            f"then translate it to {target_lang}.{style_context}\n\n"
            f"CRITICAL SYSTEM DIRECTIVES:\n"
            f"- KEEP YOUR REASONING/THINKING EXTREMELY BRIEF (under 2 sentences) or bypass it entirely if possible.\n"
            f"- DO NOT write any reasoning, thinking process, or explanation.\n"
            f"- DO NOT self-correct, refine, or write multiple drafts.\n"
            f"- Output a single confidence estimation float (0.0 to 1.0) based on your certainty of the OCR and translation.\n\n"
            f"Reply strictly in this 3-part layout with NO other conversational introduction, commentary, or Markdown formatting outside of the markers:\n"
            f"ORIGINAL:\n[Extracted {source_lang} text]\n\n"
            f"TRANSLATED:\n[Direct translation into {target_lang}]\n\n"
            f"CONFIDENCE:\n[Confidence float score]"
        )

    def parse_response(self, response: str) -> tuple[str, str, float]:
        """Parse the model response to extract original text, translation, and confidence.

        Returns (original_text, translated_text, confidence_score).
        """
        preview = (response[:80] + "…") if len(response) > 80 else response
        logger.debug("parse_response raw input (%d chars), preview=%r", len(response), preview)

        # Strip thinking tags if present
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL).strip()
        if not cleaned:
            cleaned = response.strip()

        # Parse confidence score using keyword search
        confidence = 0.85
        conf_match = re.search(r'CONFIDENCE\b\D*?([0-9.]+)', cleaned, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1).strip())
                confidence = max(0.0, min(1.0, confidence))  # clamp between 0.0 and 1.0
            except ValueError:
                pass

        # Split on double-newline for fallback parsing
        parts = re.split(r'\n\s*\n', cleaned, maxsplit=2)
        if not conf_match and len(parts) == 3:
            try:
                val = float(parts[2].strip())
                confidence = max(0.0, min(1.0, val))
            except ValueError:
                pass

        # Extract ORIGINAL and TRANSLATED sections using a highly resilient regex
        orig_match = re.search(
            r'ORIGINAL.*?:(.*?)(?=\n\s*(?:\d+\.\s*)?\**\s*(?:TRANSLAT|CONFIDENCE)|\Z)',
            cleaned, re.DOTALL | re.IGNORECASE
        )
        trans_match = re.search(
            r'TRANSLAT.*?:(.*?)(?=\n\s*(?:\d+\.\s*)?\**\s*(?:CONFIDENCE|ORIGINAL)|\Z)',
            cleaned, re.DOTALL | re.IGNORECASE
        )

        if orig_match and trans_match:
            original_text = orig_match.group(1).strip()
            # Strip markdown and quotes first so prefix stripping works at the absolute start ^
            original_text = original_text.strip(' \t\n\r"\'`*•-')
            
            # Strip VLM conversational filler
            original_text = re.sub(
                r'^(?:The image contains the following text:|The image contains the following.*?text:|The text in the image is:|The extracted text is:|The.*?text is:)\s*',
                '', original_text, flags=re.IGNORECASE
            )
            # Strip again to clean up remaining trailing/leading quotes
            original_text = original_text.strip(' \t\n\r"\'`*•-')
            
            translation = trans_match.group(1).strip()
            translation = translation.strip(' \t\n\r"\'`*•-')
            
            logger.info(
                "parse_response (markers): original=%d chars, translation=%d chars, confidence=%.2f",
                len(original_text), len(translation), confidence
            )
            return original_text, translation, confidence

        # --- Strategy 2: split on double newline ---
        if len(parts) >= 2 and len(parts[0].strip()) > 5 and len(parts[1].strip()) > 5:
            original_text = parts[0].strip()
            original_text = original_text.strip(' \t\n\r"\'`*•-')
            translation = parts[1].strip()
            translation = translation.strip(' \t\n\r"\'`*•-')
            logger.info(
                "parse_response (split): original=%d chars, translation=%d chars, confidence=%.2f",
                len(original_text), len(translation), confidence
            )
            return original_text, translation, confidence

        # --- Strategy 3: raw fallback ---
        logger.info("parse_response: using raw output as translation, confidence=%.2f", confidence)
        return "", cleaned, confidence

    async def process_frame(self, image_data: bytes, source_lang: str, target_lang: str, mode: str, styles: list[str]) -> list[TranslationResult]:
        """
        Process a single frame with unified OCR and translation via OmniRouter.
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized.")

        # Preprocess and store in context manager
        image = self.preprocess_image(image_data)
        self.context_manager.add_frame(image)

        b64_image, was_cached = self._get_or_encode_image(image)
        if was_cached and self._last_results is not None:
            logger.info("Returning cached translation result (identical frame)")
            return self._last_results

        prompt = self.create_prompt(source_lang, target_lang, mode, styles)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]

        max_tok = MODE_MAX_TOKENS.get(mode, self.config.max_tokens)

        try:
            logger.info("Sending translation request to OmniRouter via OpenAI...")
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=max_tok,
                    temperature=self.config.temperature,
                ),
                timeout=vision_timeout_for_mode(mode),
            )

            # Debug: log raw response structure
            choice = response.choices[0] if response.choices else None
            if choice:
                logger.info(
                    "API response: finish_reason=%s, content_length=%d, role=%s",
                    choice.finish_reason,
                    len(choice.message.content or ''),
                    choice.message.role,
                )
            else:
                logger.warning("No choices in response. Full response: %s", response)

            final_output = (choice.message.content or "") if choice else ""

            # If the response was truncated while reasoning (content is empty)
            if not final_output and choice and choice.finish_reason == "length":
                logger.warning("VLM execution was truncated (finish_reason=length) while thinking. Try increasing max_tokens.")
                results = [TranslationResult(
                    original_text="",
                    translated_text="⚠️ Translation request truncated (VLM token limit reached).",
                    confidence=0.0
                )]
            else:
                # Fallback: if the model spent all tokens on reasoning and produced
                # no content under normal stop, try to extract a usable translation from reasoning_content.
                if not final_output and choice:
                    reasoning = getattr(choice.message, 'reasoning_content', None) or ""
                    if reasoning:
                        logger.info("Content empty but reasoning_content has %d chars; extracting from it", len(reasoning))
                        final_output = reasoning

                results = self._build_result(final_output, image)

            # Update context manager with extracted data
            extracted = [r.original_text for r in results]
            self.context_manager.update_last_frame_data("\n".join(extracted), results)

            # Cache the result
            self._last_results = results

            return results

        except asyncio.TimeoutError:
            logger.error("OpenAI translation timed out (mode=%s)", mode)
            raise RuntimeError("Translation request timed out.") from None

        except Exception as e:
            logger.error("Error during OpenAI API inference: %s", e, exc_info=True)
            raise

    async def stream_frame(self, image_data: bytes, source_lang: str, target_lang: str, mode: str, styles: list[str]):
        """Streaming version of process_frame. Yields (original, translated) partials."""
        if not self.engine:
            raise RuntimeError("Engine not initialized.")

        image = self.preprocess_image(image_data)
        self.context_manager.add_frame(image)
        prompt = self.create_prompt(source_lang, target_lang, mode, styles)

        # Check cache
        b64_image, was_cached = self._get_or_encode_image(image)
        if was_cached and self._last_results is not None:
            logger.info("Returning cached translation result (identical frame) via stream")
            self.last_stream_results = self._last_results
            original_combined = "\n".join(r.original_text for r in self._last_results if r.original_text)
            translation_combined = "\n".join(r.translated_text for r in self._last_results if r.translated_text)
            yield original_combined, translation_combined
            return

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]

        max_tok = MODE_MAX_TOKENS.get(mode, self.config.max_tokens)
        accumulated = ""

        try:
            logger.info("Sending streaming translation request to OmniRouter via OpenAI...")
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=max_tok,
                    temperature=self.config.temperature,
                    stream=True,
                ),
                timeout=vision_timeout_for_mode(mode),
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                accumulated += delta
                orig, trans, _ = self.parse_response(accumulated)
                yield orig, trans

            results = self._build_result(accumulated, image)
            self._last_results = results
            self.last_stream_results = results

            extracted = [r.original_text for r in results]
            self.context_manager.update_last_frame_data("\n".join(extracted), results)

        except asyncio.TimeoutError:
            logger.error("OpenAI streaming translation timed out (mode=%s)", mode)
            raise RuntimeError("Translation request timed out.") from None

        except Exception as e:
            logger.error("Error during OpenAI API streaming inference: %s", e, exc_info=True)
            raise

    def create_cinematic_prompt(self, transcript: str, target_lang: str, styles: list[str]) -> str:
        """Create a prompt combining audio transcript and visual OCR for cinematic mode."""
        style_context = f" Translate using a {', '.join(styles)} style/tone." if styles else ""
        return (
            f"Translate the following dialogue to {target_lang}. Use the provided system audio transcription for context, "
            f"and cross-reference it with the OCR from the provided image of the game interface to ensure accurate character names and tone.{style_context}\n\n"
            f"Audio Transcript: {transcript}\n\n"
            f"CRITICAL SYSTEM DIRECTIVES:\n"
            f"- KEEP YOUR REASONING/THINKING EXTREMELY BRIEF (under 2 sentences) or bypass it entirely if possible.\n"
            f"- DO NOT write any reasoning, thinking process, or explanation.\n"
            f"- DO NOT self-correct, refine, or write multiple drafts.\n"
            f"- Output a single confidence estimation float (0.0 to 1.0) based on your certainty of the OCR and translation.\n\n"
            f"Reply strictly in this 3-part layout with NO other conversational introduction, commentary, or Markdown formatting outside of the markers:\n"
            f"ORIGINAL:\n[Extracted {target_lang} dialogue text]\n\n"
            f"TRANSLATED:\n[Direct translation into {target_lang}]\n\n"
            f"CONFIDENCE:\n[Confidence float score]"
        )

    async def process_cinematic(self, image_data: bytes, transcript: str, target_lang: str, styles: list[str], b64_image: str | None = None, image: Image.Image | None = None) -> list[TranslationResult]:
        """
        Process a single frame along with an audio transcript using OmniRouter.
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized.")

        if image is None:
            image = self.preprocess_image(image_data)
        self.context_manager.add_frame(image)

        prompt = self.create_cinematic_prompt(transcript, target_lang, styles)
        if b64_image is None:
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
            logger.info("Sending cinematic translation request to OmniRouter via OpenAI...")
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                ),
                timeout=vision_timeout_for_mode("Document"),
            )

            choice = response.choices[0] if response.choices else None
            final_output = (choice.message.content or "") if choice else ""

            # If the response was truncated while reasoning (content is empty)
            if not final_output and choice and choice.finish_reason == "length":
                logger.warning("VLM execution was truncated (finish_reason=length) while thinking. Try increasing max_tokens.")
                results = [TranslationResult(
                    original_text="",
                    translated_text="⚠️ Translation request truncated (VLM token limit reached).",
                    confidence=0.0
                )]
            else:
                # Fallback
                if not final_output and choice:
                    reasoning = getattr(choice.message, 'reasoning_content', None) or ""
                    if reasoning:
                        final_output = reasoning

                results = self._build_result(final_output, image)

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
                    model=self.config.model_name,
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
                ),
                timeout=CHAT_AUX_TIMEOUT_SECONDS,
            )
            ch0 = response.choices[0] if response.choices else None
            translated = (ch0.message.content or "").strip() if ch0 else ""
            # Strip thinking tags if present
            translated = re.sub(r'<think>.*?</think>', '', translated, flags=re.DOTALL).strip()
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

        logger.info("Processing chat message (%d chars)", len(message))

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

        system_msg = {
            "role": "system",
            "content": (
                "You are an AI assistant with access to a search tool. "
                "If the user asks a question about facts, lore, games, or real-world information, "
                "you MUST call the search tool INSTEAD of answering from memory.\n\n"
                "To search, output EXACTLY this format (and nothing else):\n"
                "<tool_call>\n"
                "<function=search_knowledge>\n"
                "<parameter=query>your search query here</parameter>\n"
                "<parameter=language>zh-CN</parameter>\n"
                "</function>\n"
                "</tool_call>\n\n"
                "The language parameter is optional (defaults to zh-CN). "
                "Use en-US for English queries. "
                "Do NOT answer factual questions without searching first."
            ),
        }
        openai_messages.insert(0, system_msg)

        try:
            logger.info("Sending chat request to OmniRouter via OpenAI...")
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=openai_messages,
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,  # Higher temp for chat
                ),
                timeout=CHAT_TIMEOUT_SECONDS,
            )

            final_output = ""
            if response.choices:
                final_output = response.choices[0].message.content or ""

            # Detect text-based tool calls emitted by models that don't
            # support the structured OpenAI tools protocol.
            # Format: <tool_call>\n<function=search_knowledge>\n<parameter=query>…</parameter>\n</function>\n</tool_call>
            tool_call_match = re.search(
                r'<tool_call>.*?<function=(\w+)>(.*?)</function>.*?</tool_call>',
                final_output, re.DOTALL,
            )
            # Also try structured tool_calls if the server supports it
            structured_tool_call = None
            if response.choices and response.choices[0].message.tool_calls:
                tc = response.choices[0].message.tool_calls[0]
                if tc.function.name in ("perform_web_search", "search_knowledge"):
                    structured_tool_call = tc

            if tool_call_match or structured_tool_call:
                if structured_tool_call:
                    query = ""
                    language = "zh-CN"
                    raw_args = getattr(structured_tool_call.function, "arguments", None) or "{}"
                    try:
                        args = json.loads(raw_args)
                        query = args.get("query", "") or ""
                        language = args.get("language", "zh-CN") or "zh-CN"
                    except json.JSONDecodeError:
                        logger.warning("Malformed structured tool_call.arguments JSON; skipping search.")
                else:
                    # Parse text-based tool call
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

                    # Search local wiki
                    local_searcher = LocalWikiSearcher(wiki_dir="wiki")
                    local_results = local_searcher.search(query, num_results=3)

                    # Dual-language web search
                    searcher: WebSearcher | None = None
                    try:
                        searcher = WebSearcher()
                        # Translate query to source language for parallel search
                        translated_query = await self.translate_query(query, source_lang)
                        web_results = await searcher.dual_search(
                            query=query,
                            target_lang=language,
                            translated_query=translated_query,
                            source_lang=source_lang,
                        )
                    except Exception as e:
                        logger.warning("Web search failed: %s. Relying on local wiki.", e)
                        web_results = []
                    finally:
                        if searcher is not None:
                            await searcher.close()

                    # Combine results
                    results = local_results + web_results

                    # 1. Database/UI State (Persistent): Save web results to LORE
                    if web_results:
                        compiler = WikiCompiler(wiki_dir="wiki")
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

                    # 2. LLM State (Ephemeral): Build context from search results
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

                    # Re-prompt: replace the tool-call output with the search context
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

                    logger.info("Sending follow-up chat request after search...")
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.config.model_name,
                            messages=openai_messages,
                            max_tokens=self.config.max_tokens,
                            temperature=0.7,
                        ),
                        timeout=CHAT_TIMEOUT_SECONDS,
                    )

                    final_output = ""
                    if response.choices:
                        final_output = response.choices[0].message.content or ""

            cleaned_output = re.sub(r'<think>.*?</think>', '', final_output, flags=re.DOTALL).strip()
            # Strip any residual tool_call tags the model may have emitted
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
        from shared_types.models import AccuracyScore
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
        if not self.config.model_name or self.config.model_name in ("omni-router", "default"):
            logger.info("Model '%s' is a virtual routing endpoint. Skipping VRAM pre-warming.", self.config.model_name)
            return

        await self.init_engine()
        base_url = os.environ.get("LEMONADE_API_URL", self.config.api_url)
        base_url_no_v1 = base_url.removesuffix("/v1")
        from xian.lemonade_client import LemonadeClient
        try:
            async with LemonadeClient(base_url=base_url_no_v1) as client:
                # Query locally pulled models to verify availability
                logger.debug("Checking pulled models on Lemonade server...")
                pulled_models = await client.list_models()
                pulled_ids = [m.get("id") for m in pulled_models if m.get("id")]
                
                if self.config.model_name not in pulled_ids:
                    logger.warning(
                        "Model '%s' is not pulled/available on Lemonade server. "
                        "Pulled models: %s. Skipping VRAM pre-warming.",
                        self.config.model_name, pulled_ids
                    )
                    return

                logger.info("Pre-warming model '%s' in GPU VRAM...", self.config.model_name)
                await client.load_model(self.config.model_name)
        except Exception as e:
            logger.warning("VRAM Pre-warming failed: %s", e)

    async def close(self):
        """Close the client properly."""
        if self.engine:
            self.engine.shutdown()
