#!/usr/bin/env python3
# Xian-VL Scripts — Development and automation scripts.
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

"""Print what the live pipeline actually does on a corpus of captured frames.

The benchmark tests answer "did anything break".  This answers "what are the
numbers" — per-frame OCR cost against the tick budget, and how far apart the
change gate holds a re-capture and a genuinely new screen.  Run it before
tuning the interval, the hash size, or the change threshold.

    uv run --package xian-vl python scripts/benchmark_live_pipeline.py

Reads the same corpus as the tests: ``XIAN_SCREENSHOT_CORPUS``, or
``../game_screenshots`` beside the checkout.
"""

from __future__ import annotations

import io
import itertools
import statistics
import sys
from pathlib import Path

import imagehash
from PIL import Image, ImageDraw, ImageEnhance

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "mage-client" / "src"))

from benchmark_corpus import corpus_dir, corpus_paths  # noqa: E402

#: Hash sizes to compare: 8 was the shipped default and could not separate the
#: two classes; 16 is what replaced it.
HASH_SIZES = (8, 16)


def _recaptures(image: Image.Image):
    """The same screen captured a second time — see the benchmark tests."""
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    buffer.seek(0)
    yield Image.open(buffer).convert("RGB")

    cursor = image.copy()
    x, y = image.width // 2, image.height // 2
    ImageDraw.Draw(cursor).ellipse([x, y, x + 14, y + 14], fill=(255, 255, 255))
    yield cursor

    yield image.transform(image.size, Image.AFFINE, (1, 0, 1, 0, 1, 1))
    yield ImageEnhance.Brightness(image).enhance(1.03)


def _summarize(label: str, values: list[float], unit: str = "ms") -> None:
    if not values:
        print(f"  {label:<28} (none)")
        return
    ordered = sorted(values)
    print(
        f"  {label:<28} n={len(ordered):<4} median={statistics.median(ordered):>7.1f}{unit}"
        f"  p95={ordered[int(len(ordered) * 0.95) - 1]:>7.1f}{unit}"
        f"  max={ordered[-1]:>7.1f}{unit}"
    )


def report_change_gate(paths: list[Path]) -> None:
    from mage.live_lens import DEFAULT_CHANGE_THRESHOLD, LIVE_HASH_SIZE

    print("\n── change gate: re-capture vs. a genuinely new screen ───────")
    print(f"  shipped: hash_size={LIVE_HASH_SIZE} ({LIVE_HASH_SIZE ** 2} bits), "
          f"threshold={DEFAULT_CHANGE_THRESHOLD}\n")

    images = {}
    for path in paths:
        with Image.open(path) as opened:
            images[path.name] = opened.convert("RGB").copy()

    for hash_size in HASH_SIZES:
        hashes, unchanged = {}, []
        for name, image in images.items():
            reference = imagehash.phash(image, hash_size=hash_size)
            hashes[name] = reference
            unchanged += [
                abs(reference - imagehash.phash(variant, hash_size=hash_size))
                for variant in _recaptures(image)
            ]
        changed = [abs(hashes[a] - hashes[b]) for a, b in itertools.combinations(hashes, 2)]

        ceiling, floor = max(unchanged), min(changed)
        bits = hash_size ** 2
        verdict = (
            f"threshold anywhere in [{ceiling + 1}, {floor}] separates them"
            if floor > ceiling
            else "OVERLAP — no threshold can separate them"
        )
        print(f"  hash_size={hash_size} ({bits} bits)")
        print(f"    re-capture of one screen  max={ceiling:>4}   ({ceiling / bits:.1%} of bits)")
        print(f"    two different screens     min={floor:>4}   ({floor / bits:.1%} of bits)")
        print(f"    -> {verdict}")


def main() -> int:
    print(f"corpus: {corpus_dir()}")
    paths = corpus_paths()
    print(f"frames: {len(paths)}")

    report_change_gate(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
