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

from shared_types.constants import (
    LIVE_IMAGE_FORMAT,
    LIVE_IMAGE_QUALITY,
    LIVE_MAX_DIMENSION,
    LIVE_MIN_SCALE,
    QWEN_MAX_DIMENSION,
)
from xian.timeout import vision_timeout_for_mode

logger = logging.getLogger(__name__)

__all__ = [
    "TextRegion",
    "parse_regions",
    "preprocess_for_grounding",
    "ground_and_translate",
    "scan_json_objects",
    "suppress_overlapping_regions",
    "drop_container_regions",
    "drop_frame_sized_regions",
    "live_max_dimension",
]

#: The grounding coordinate space: boxes are integers 0-1000 on each axis.
GROUNDING_SCALE = 1000.0

#: How much of one box another must cover before they count as the same
#: detection rather than two neighbouring lines.
DEFAULT_MAX_COVER = 0.5

#: Smaller boxes a region must contain before it is treated as a container.
CONTAINER_CHILD_COUNT = 2

#: A child has to be meaningfully smaller than the box containing it; two
#: boxes of nearly equal size are competing detections of one line, not
#: nesting, and belong to the deduplication pass instead.
CONTAINER_CHILD_AREA_RATIO = 0.9

#: Fraction of the frame above which a box is the model boxing the whole
#: picture rather than locating a line of text.
MAX_FRAME_COVER = 0.8

#: Generation cap for one grounding call. Each region costs roughly 40-60
#: tokens once its box, source text and translation are written out, so a busy
#: whole-screen frame runs past 2048 and gets cut off part-way down the screen.
#: Only a ceiling: the model stops when it closes the array, so raising it
#: costs nothing on the frames that never reach it.
GROUNDING_MAX_TOKENS = 4096

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


def live_max_dimension(size: tuple[int, int]) -> int:
    """Pick the longest edge to send for a frame of this size.

    A locked region around a dialogue box gets :data:`LIVE_MAX_DIMENSION`,
    which is what that constant was measured for. A whole-screen capture gets
    more, because the text on it is a far smaller fraction of the frame: a 23px
    glyph on a 2880-wide capture arrives at the model as 8.2px under the 1024
    budget, too small for a Chinese character to hold its strokes, and the
    request still looks like it succeeded.

    Growth is proportional and capped, so this never exceeds what the one-shot
    path already sends.
    """
    longest = max(size)
    if longest <= 0:
        return LIVE_MAX_DIMENSION
    needed = int(longest * LIVE_MIN_SCALE)
    return max(LIVE_MAX_DIMENSION, min(needed, QWEN_MAX_DIMENSION))


def preprocess_for_grounding(
    image: Image.Image,
    max_dimension: int = QWEN_MAX_DIMENSION,
    *,
    sharpen: bool = True,
) -> Image.Image:
    """Scale a frame for the model without padding it to a square.

    Deliberately not `VLProcessor.preprocess_pil`: that pads to a square for
    the bubble path, which would shift every box the model reports.

    ``sharpen`` is a convolution over the whole image — 17-19ms at 1920×1200,
    paid on every frame. It earns that on a one-shot capture of small or soft
    text; on the live path it is switched off, where the frame is downscaled
    and JPEG-encoded straight afterwards and the sharpened edges largely do not
    survive the trip anyway.
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

    return image.filter(ImageFilter.SHARPEN) if sharpen else image


def _parse_entries(raw: str) -> list:
    """Decode the array of grounding objects, whole or partial.

    A whole-screen frame can hold more text regions than ``max_tokens`` leaves
    room for, and the response is then cut off mid-array. The array pattern
    cannot match an unterminated array, so every region the model *did* produce
    was being thrown away along with the truncated one — the screen came back
    empty rather than partly translated.
    """
    match = _JSON_ARRAY_RE.search(raw)
    if match:
        try:
            entries = json.loads(match.group(0))
        except (ValueError, TypeError):
            entries = None
        if isinstance(entries, list):
            return entries

    # Fall back to the streaming scanner, which reads complete objects out of
    # an incomplete array and simply stops at the truncation point.
    objects, _ = scan_json_objects(raw)
    recovered = []
    for chunk in objects:
        try:
            entry = json.loads(chunk)
        except (ValueError, TypeError):
            continue
        recovered.append(entry)
    if not recovered and raw.strip():
        # Neither a whole array nor a single complete object. The small models
        # do this — one answered a whole-screen frame with
        # ``{"box":[[326,11,656,207], "WORLD OF WARCRAFT"}``, unbalanced braces
        # and all — and the overlay's only symptom is that it stays blank.
        logger.warning("Grounding response could not be decoded: %.300s", raw)
    return recovered


def parse_regions(raw: str, width: int, height: int) -> list[TextRegion]:
    """Parse the model's JSON array into pixel-space regions.

    Tolerates the wrappers models actually emit: markdown fences, a leading
    thinking block, and prose either side of the array.
    """
    if not raw:
        return []
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1]

    entries = _parse_entries(raw)

    regions: list[TextRegion] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        coordinates = _entry_box(entry)
        if coordinates is None:
            continue
        x1, y1, x2, y2 = coordinates

        left = _clamp(min(x1, x2) / GROUNDING_SCALE * width, 0, width)
        right = _clamp(max(x1, x2) / GROUNDING_SCALE * width, 0, width)
        top = _clamp(min(y1, y2) / GROUNDING_SCALE * height, 0, height)
        bottom = _clamp(max(y1, y2) / GROUNDING_SCALE * height, 0, height)

        original, translated = _entry_text(entry)

        region = TextRegion((left, top, right, bottom), original, translated)
        if region.is_valid():
            regions.append(region)

    if entries and not regions:
        # The model answered, and none of it fitted the schema. Distinct from
        # "no text on screen", and otherwise indistinguishable from it: the
        # overlay simply stays blank while the loop keeps paying for calls.
        logger.warning(
            "Grounding returned %d object(s) but none held a usable box: %.200s",
            len(entries), raw,
        )

    return regions


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, value)))


def _entry_box(entry: dict) -> tuple[float, float, float, float] | None:
    """Read the four coordinates out of one grounding object.

    The prompt asks for ``box``, but the Qwen-VL grounding convention these
    models are trained on names it ``bbox_2d``, and a model reaching for that
    training rather than the prompt is a schema mismatch we can simply accept.
    The same convention nests the coordinates one level deep when it expects to
    return several boxes, so a single-element wrapper is unwrapped too.
    """
    box = None
    for key in ("box", "bbox_2d", "bbox"):
        value = entry.get(key)
        if isinstance(value, (list, tuple)) and value:
            box = value
            break
    if box is None:
        return None

    if isinstance(box[0], (list, tuple)):
        box = box[0]
    if len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _entry_text(entry: dict) -> tuple[str, str]:
    """Source and target text, under whichever key the model used."""
    original = ""
    for key in ("original", "text", "source"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            original = value
            break

    translated = ""
    for key in ("translated", "translation", "target"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            translated = value
            break

    return original, translated or original


def _covered_fraction(inner: tuple, outer: tuple) -> float:
    """How much of ``inner``'s area ``outer`` covers, 0-1."""
    overlap_x = min(inner[2], outer[2]) - max(inner[0], outer[0])
    overlap_y = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if overlap_x <= 0 or overlap_y <= 0:
        return 0.0
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return overlap_x * overlap_y / inner_area if inner_area > 0 else 0.0


def _area(region: TextRegion) -> int:
    return region.width * region.height


def drop_container_regions(
    regions: list[TextRegion], max_cover: float = DEFAULT_MAX_COVER
) -> list[TextRegion]:
    """Drop boxes that swallow several smaller ones.

    Asked for "one element per visual line", a model will still sometimes hand
    back the panel *and* the lines inside it — or, on a whole-screen frame, one
    box around the entire picture. Keeping the container is the worst of the
    options: the overlay fills it, which paints an opaque slab over everything
    the smaller boxes were going to say.

    Two children is the threshold because one is ambiguous. A box containing a
    single smaller box is far more often two attempts at the same line, where
    the larger one is the better answer, and that case is left to the
    deduplication pass below.
    """
    children: dict[int, int] = {}
    for region in regions:
        count = 0
        for other in regions:
            if other is region:
                continue
            if _area(other) >= _area(region) * CONTAINER_CHILD_AREA_RATIO:
                continue
            if _covered_fraction(other.box, region.box) > max_cover:
                count += 1
        children[id(region)] = count

    kept = [r for r in regions if children[id(r)] < CONTAINER_CHILD_COUNT]
    dropped = len(regions) - len(kept)
    if dropped:
        logger.debug("Dropped %d container box(es) covering smaller regions", dropped)
    return kept


def drop_frame_sized_regions(
    regions: list[TextRegion],
    frame_size: tuple[int, int],
    max_cover: float = MAX_FRAME_COVER,
) -> list[TextRegion]:
    """Drop boxes that cover most of the frame, when anything else survives.

    A box this size is the model boxing the picture rather than finding a line,
    and it is the single most destructive thing the overlay can be handed: the
    fill covers the whole capture, so the game disappears behind one flat
    rectangle with one line of text on it.

    It is not dropped unconditionally. On a locked region drawn tightly around
    a single line, a box covering nearly all of the frame is the *correct*
    answer, and there is nothing else to fall back to.

    That reprieve is only available to a frame small enough for the claim to be
    true. The size that separates them is the one :func:`live_max_dimension`
    already draws: at or below the locked-region budget, "this whole frame is
    one line of text" is plausible; above it, the same box asserts that one line
    fills a screen, and it is dropped even if nothing survives it. Painting
    nothing is recoverable — the next tick tries again — while painting it hides
    the game behind a slab.
    """
    frame_area = frame_size[0] * frame_size[1]
    if frame_area <= 0:
        return regions

    kept = [r for r in regions if _area(r) < frame_area * max_cover]
    if len(kept) == len(regions):
        return kept
    if not kept and max(frame_size) <= LIVE_MAX_DIMENSION:
        return regions

    logger.debug(
        "Dropped %d frame-sized box(es), leaving %d region(s)",
        len(regions) - len(kept), len(kept),
    )
    return kept


def suppress_overlapping_regions(
    regions: list[TextRegion],
    max_cover: float = DEFAULT_MAX_COVER,
    *,
    frame_size: tuple[int, int] | None = None,
) -> list[TextRegion]:
    """Reduce the model's boxes to a set the overlay can paint without conflict.

    The overlay fills each region's box and then draws its translation, in list
    order, so two boxes over the same pixels means the later fill paints out
    the earlier translation. Detectors produce these routinely on tightly
    stacked UI — a chat log, a character list — and measured over the
    screenshot corpus, 12 box pairs covered each other by more than half, one
    of them completely.

    Three passes, in this order:

    1. frame-sized boxes, which obscure the entire capture;
    2. containers, which obscure everything inside them;
    3. duplicates, largest first, so the survivor is the box most likely to
       hold the whole line rather than a fragment of it.

    Ordering matters: a container has to be identified against the full set of
    boxes, before deduplication has removed the very children that reveal it.
    """
    if frame_size is not None:
        regions = drop_frame_sized_regions(regions, frame_size, MAX_FRAME_COVER)
    candidates = drop_container_regions(regions, max_cover)

    kept: list[TextRegion] = []
    for region in sorted(candidates, key=_area, reverse=True):
        if any(_covered_fraction(region.box, other.box) > max_cover for other in kept):
            continue
        kept.append(region)

    # Restore the caller's ordering: reading order is what the overlay and the
    # session recorder expect, not descending area.
    order = {id(region): index for index, region in enumerate(regions)}
    return sorted(kept, key=lambda region: order[id(region)])


def scan_json_objects(buffer: str, start: int = 0) -> tuple[list[str], int]:
    """Pull complete ``{...}`` objects out of a partly-received JSON array.

    Streaming the grounding call lets the overlay paint each line as it arrives
    instead of waiting for the model to finish the whole screen, which is the
    difference between text appearing in a few hundred milliseconds and after
    several seconds.  The catch is that a partial response is not valid JSON,
    so this walks the buffer tracking brace depth and string state, and returns
    only the objects that are definitely complete.

    Returns the complete object substrings found from ``start``, and the index
    to resume scanning from next time.
    """
    objects: list[str] = []
    depth = 0
    object_start = -1
    in_string = False
    escaped = False
    consumed = start

    for index in range(start, len(buffer)):
        char = buffer[index]

        if in_string:
            # A backslash escapes the next character, including a quote; a
            # brace inside a string must never move the depth counter.
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and object_start >= 0:
                    objects.append(buffer[object_start:index + 1])
                    consumed = index + 1
                    object_start = -1

    return objects, consumed


def region_from_entry(entry: dict, width: int, height: int) -> TextRegion | None:
    """Turn one parsed grounding object into a pixel-space region."""
    coordinates = _entry_box(entry)
    if coordinates is None:
        return None
    x1, y1, x2, y2 = coordinates

    original, translated = _entry_text(entry)

    region = TextRegion(
        (
            _clamp(min(x1, x2) / GROUNDING_SCALE * width, 0, width),
            _clamp(min(y1, y2) / GROUNDING_SCALE * height, 0, height),
            _clamp(max(x1, x2) / GROUNDING_SCALE * width, 0, width),
            _clamp(max(y1, y2) / GROUNDING_SCALE * height, 0, height),
        ),
        original,
        translated,
    )
    return region if region.is_valid() else None


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
        # Both of these are failures seen on whole-screen captures, and both are
        # also caught in code — a model that ignores the instruction still gets
        # its container and frame-sized boxes dropped before they are painted.
        "Never return one box around a group of lines, a panel, a window, or the whole "
        "image: box each line on its own. A box must not cover most of the image.\n"
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
    max_tokens: int = GROUNDING_MAX_TOKENS,
    max_dimension: int | None = None,
    sharpen: bool = False,
    on_region=None,
) -> list[TextRegion]:
    """Locate and translate every text region in a frame, in one vision call.

    Defaults are the live overlay's: this runs once per changed frame, so the
    frame is downscaled and JPEG-encoded to keep the round trip short. How far
    it is downscaled depends on how big the capture is — see
    :func:`live_max_dimension` — because a budget that suits a locked region
    makes whole-screen text unreadable. A caller that only grounds once can
    pass ``QWEN_MAX_DIMENSION`` and ``sharpen=True`` to trade the latency back
    for fidelity.

    ``on_region`` opts into streaming: the response is consumed as it arrives
    and the callback is invoked with the regions decoded so far, so the first
    lines can be painted long before the model has finished describing the
    screen. The full list is still returned either way. A stream that fails
    part-way falls back to whatever it managed to decode.
    """
    if not processor.client:
        raise RuntimeError("Engine not initialized.")

    if max_dimension is None:
        max_dimension = live_max_dimension(image.size)
    prepared = preprocess_for_grounding(image, max_dimension, sharpen=sharpen)
    b64_image = processor.encode_image(
        prepared, fmt=LIVE_IMAGE_FORMAT, quality=LIVE_IMAGE_QUALITY
    )
    mime = "jpeg" if LIVE_IMAGE_FORMAT.upper() == "JPEG" else LIVE_IMAGE_FORMAT.lower()

    request = {
        "model": processor.get_vision_model_name(),
        "messages": [
            {"role": "system", "content": build_grounding_prompt(source_lang, target_lang, glossary)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"OCR every text region and translate it to {target_lang}. Respond with the JSON array only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_image}"}},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    timeout = vision_timeout_for_mode("Game")

    def finish(regions: list[TextRegion]) -> list[TextRegion]:
        """Deduplicate and map back to the caller's coordinate space."""
        regions = suppress_overlapping_regions(regions, frame_size=prepared.size)
        if prepared.size != image.size:
            regions = rescale_regions(regions, prepared.size, image.size)
        return regions

    if on_region is not None:
        regions = await asyncio.wait_for(
            _stream_regions(processor, request, prepared, finish, on_region),
            timeout=timeout,
        )
        return finish(regions)

    response = await asyncio.wait_for(
        processor.client.chat.completions.create(**request), timeout=timeout
    )

    choice = response.choices[0] if response.choices else None
    content = (choice.message.content or "") if choice else ""

    # Boxes are relative to the *prepared* image, so map them back to the
    # caller's coordinate space before returning.
    return finish(parse_regions(content, prepared.width, prepared.height))


async def _stream_regions(processor, request: dict, prepared, finish, on_region) -> list[TextRegion]:
    """Consume a streamed grounding response, publishing regions as they land.

    Anything that goes wrong mid-stream keeps the regions already decoded: a
    partial screen of translations is worth more than an exception, and the
    next tick will try again anyway.
    """
    regions: list[TextRegion] = []
    buffer = ""
    consumed = 0

    try:
        stream = await processor.client.chat.completions.create(**request, stream=True)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            piece = getattr(choices[0].delta, "content", None)
            if not piece:
                continue

            buffer += piece
            found, consumed = scan_json_objects(buffer, consumed)
            new_regions = []
            for raw in found:
                try:
                    entry = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(entry, dict):
                    continue
                region = region_from_entry(entry, prepared.width, prepared.height)
                if region is not None:
                    new_regions.append(region)

            if new_regions:
                regions.extend(new_regions)
                on_region(finish(list(regions)))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Grounding stream ended early (%s); keeping %d regions", exc, len(regions))

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
