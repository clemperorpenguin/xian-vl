# Luduan — EPUB to RoboBook narration CLI.
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

"""EPUB text extraction.

Parses an EPUB file, extracts chapter text, and returns structured
paragraph data suitable for the translation pipeline.

Migrated from the original Luduan repository.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Paragraph:
    """A single extractable text block from an EPUB."""

    chapter_index: int
    text: str
    is_heading: bool = False


@dataclass
class ParsedBook:
    """The result of parsing an EPUB file."""

    title: str = ""
    author: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)


def parse_epub(path: Path) -> ParsedBook:
    """Extract text content from an EPUB file.

    Requires ``ebooklib`` and ``BeautifulSoup`` (``bs4``).

    Parameters
    ----------
    path:
        Path to the ``.epub`` file.

    Returns
    -------
    ParsedBook
        Structured paragraph data with chapter indices.
    """
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "EPUB parsing requires 'ebooklib' and 'beautifulsoup4'. "
            "Install them with:  pip install ebooklib beautifulsoup4"
        ) from exc

    book = epub.read_epub(str(path), options={"ignore_ncx": True})

    parsed = ParsedBook(
        title=book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else path.stem,
        author=book.get_metadata("DC", "creator")[0][0] if book.get_metadata("DC", "creator") else "",
    )

    chapter_idx = 0
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")

        for tag in soup.find_all(["p", "h1", "h2", "h3", "h4"]):
            text = tag.get_text(strip=True)
            if not text or len(text) < 2:
                continue

            is_heading = tag.name.startswith("h")
            if is_heading:
                chapter_idx += 1

            parsed.paragraphs.append(
                Paragraph(
                    chapter_index=chapter_idx,
                    text=text,
                    is_heading=is_heading,
                )
            )

    logger.info(
        "Parsed '%s': %d paragraphs across %d chapters.",
        parsed.title,
        len(parsed.paragraphs),
        chapter_idx,
    )
    return parsed
