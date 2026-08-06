# Xian-VL — Core Vision-Language orchestration engine.
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

"""Located text regions: OCR + translation *with coordinates*.

The standard pipeline returns one blob of text, which is all a speech bubble
needs.  Painting a translation over the original — Google Lens style — needs to
know where each line sits, so this module asks the vision model for boxes and
parses them back into pixel space.

Coordinates come back normalized 0-1000 on both axes (the Qwen-VL grounding
convention, already used for the "where do I click?" feature).  Importantly the
frame is *not* padded to a square here, unlike the bubble path: padding would
offset every box, and the inverse mapping is one more thing to get wrong.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from PIL import Image, ImageFilter

from shared_types.constants import QWEN_MAX_DIMENSION
from xian.timeout import vision_timeout_for_mode

logger = logging.getLogger(__name__)

__all__ = ["TextRegion", "parse_regions", "preprocess_for_grounding", "ground_and_translate"]

#: The grounding coordinate space: boxes are integers 0-1000 on each axis.
GROUNDING_SCALE = 1000.0

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass
class TextRegion:
    """One run of text on screen, in source-image pixel coordinates."""

    box: tuple[int, int, int, int]  # (left, top, right, bottom)
    original: str
    translated: str

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0 and bool(self.translated.strip())


def preprocess_for_grounding(image: Image.Image, max_dimension: int = QWEN_MAX_DIMENSION) -> Image.Image:
    """Scale and sharpen a frame without padding it to a square.

    Deliberately not `VLProcessor.preprocess_pil`: that pads to a square for
    the bubble path, which would shift every box the model reports.
    """
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    width, height = image.size
    if width > max_dimension or height > max_dimension:
        if width >= height:
            new_size = (max_dimension, max(1, int(height * max_dimension / width)))
        else:
            new_size = (max(1, int(width * max_dimension / height)), max_dimension)
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image.filter(ImageFilter.SHARPEN)


def parse_regions(raw: str, width: int, height: int) -> list[TextRegion]:
    """Parse the model's JSON array into pixel-space regions.

    Tolerates the wrappers models actually emit: markdown fences, a leading
    thinking block, and prose either side of the array.
    """
    if not raw:
        return []
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]

    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return []
    try:
        entries = json.loads(match.group(0))
    except (ValueError, TypeError):
        logger.debug("Grounding response was not valid JSON: %.200s", raw)
        return []
    if not isinstance(entries, list):
        return []

    regions: list[TextRegion] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue

        left = _clamp(min(x1, x2) / GROUNDING_SCALE * width, 0, width)
        right = _clamp(max(x1, x2) / GROUNDING_SCALE * width, 0, width)
        top = _clamp(min(y1, y2) / GROUNDING_SCALE * height, 0, height)
        bottom = _clamp(max(y1, y2) / GROUNDING_SCALE * height, 0, height)

        original = str(entry.get("original") or "")
        translated = str(entry.get("translated") or "") or original

        region = TextRegion((left, top, right, bottom), original, translated)
        if region.is_valid():
            regions.append(region)

    return regions


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


def build_grounding_prompt(source_lang: str, target_lang: str, glossary: dict[str, str] | None = None) -> str:
    prompt = (
        "You are an OCR and translation engine for a single screenshot. "
        f"Detect every distinct region of {source_lang} text. "
        "Return ONLY a JSON array — no markdown, no prose, no explanation.\n"
        'Each element must be {"box":[x1,y1,x2,y2],"original":"<source text>",'
        f'"translated":"<{target_lang} translation>"}}.\n'
        "Box coordinates are integers normalized to 0-1000, where x runs along the image "
        "width and y along its height. Give one element per visual line or label, tight to "
        "the text itself.\n"
        "Skip icons, decorative symbols, and graphics that contain no text. "
        "If there is no text at all, return []."
    )
    if glossary:
        lines = "\n".join(f"- {src}: {dst}" for src, dst in glossary.items())
        prompt += f"\n\nUse this terminology when it applies:\n{lines}"
    return prompt


async def ground_and_translate(
    processor,
    image: Image.Image,
    source_lang: str,
    target_lang: str,
    *,
    glossary: dict[str, str] | None = None,
    max_tokens: int = 2048,
) -> list[TextRegion]:
    """Locate and translate every text region in a frame, in one vision call."""
    if not processor.client:
        raise RuntimeError("Engine not initialized.")

    prepared = preprocess_for_grounding(image)
    b64_image = processor.encode_image(prepared)

    response = await asyncio.wait_for(
        processor.client.chat.completions.create(
            model=processor.get_vision_model_name(),
            messages=[
                {"role": "system", "content": build_grounding_prompt(source_lang, target_lang, glossary)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"OCR every text region and translate it to {target_lang}. Respond with the JSON array only."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
        timeout=vision_timeout_for_mode("Game"),
    )

    choice = response.choices[0] if response.choices else None
    content = (choice.message.content or "") if choice else ""

    # Boxes are relative to the *prepared* image, so map them back to the
    # caller's coordinate space before returning.
    regions = parse_regions(content, prepared.width, prepared.height)
    if prepared.size != image.size:
        regions = rescale_regions(regions, prepared.size, image.size)
    return regions


def rescale_regions(
    regions: list[TextRegion], from_size: tuple[int, int], to_size: tuple[int, int]
) -> list[TextRegion]:
    """Map regions between two image sizes."""
    if from_size == to_size or not from_size[0] or not from_size[1]:
        return regions
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    scaled = []
    for r in regions:
        box = (
            _clamp(r.box[0] * sx, 0, to_size[0]),
            _clamp(r.box[1] * sy, 0, to_size[1]),
            _clamp(r.box[2] * sx, 0, to_size[0]),
            _clamp(r.box[3] * sy, 0, to_size[1]),
        )
        region = TextRegion(box, r.original, r.translated)
        if region.is_valid():
            scaled.append(region)
    return scaled
