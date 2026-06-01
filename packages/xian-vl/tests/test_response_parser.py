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

"""Tests for response parser to verify robustness against VLM formatting variations."""

from __future__ import annotations

import pytest
from xian.pipeline import VLProcessor, VLConfig


@pytest.fixture
def processor() -> VLProcessor:
    return VLProcessor(VLConfig())


def test_parse_response_clean(processor: VLProcessor) -> None:
    raw_response = (
        "ORIGINAL:\n"
        "为维护未成年人健康上网环境\n\n"
        "TRANSLATED:\n"
        "To maintain a healthy online environment for minors\n\n"
        "CONFIDENCE:\n"
        "0.95"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == "为维护未成年人健康上网环境"
    assert trans == "To maintain a healthy online environment for minors"
    assert conf == 0.95


def test_parse_response_numbered_spillover(processor: VLProcessor) -> None:
    raw_response = (
        "1.  **Extract Text (ORIGINAL):**\n"
        "    The image contains the following Chinese text:\n"
        '    "为维护未成年人健康上网环境，本游戏暂不支持实名认证18岁以下的用户登录体验"\n\n'
        "2.  **Translate (TRANSLATED):**\n"
        "    In order to maintain a healthy online environment for minors, "
        "this game does not currently support real-name verification for users under 18.\n\n"
        "3.  **Confidence (CONFIDENCE):**\n"
        "    0.92"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == "为维护未成年人健康上网环境，本游戏暂不支持实名认证18岁以下的用户登录体验"
    assert trans == "In order to maintain a healthy online environment for minors, this game does not currently support real-name verification for users under 18."
    assert conf == 0.92


def test_parse_response_markdown_bold_and_quotes(processor: VLProcessor) -> None:
    raw_response = (
        "**ORIGINAL**:\n"
        '*为维护未成年人*\n\n'
        "**TRANSLATED**:\n"
        '"To protect minors"\n\n'
        "**CONFIDENCE**:\n"
        "0.8"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == "为维护未成年人"
    assert trans == "To protect minors"
    assert conf == 0.8


def test_parse_response_fallback_split(processor: VLProcessor) -> None:
    raw_response = (
        "为维护未成年人\n\n"
        "To protect minors\n\n"
        "0.7"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == "为维护未成年人"
    assert trans == "To protect minors"
    assert conf == 0.7


def test_parse_response_multiple_blocks(processor: VLProcessor) -> None:
    raw_response = (
        "4.  **Format Output:**\n"
        "    *   ORIGINAL: [Text]\n"
        "    *   TRANSLATED: [Translation]\n"
        "    *   CONFIDENCE: [Score]\n\n"
        "Let's assemble the final text.\n\n"
        "ORIGINAL:\n"
        "剑侠世界4\n\n"
        "TRANSLATED:\n"
        "The Realm of the Sword Hero IV\n\n"
        "CONFIDENCE:\n"
        "0.95"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == "剑侠世界4"
    assert trans == "The Realm of the Sword Hero IV"
    assert conf == 0.95


def test_parse_response_placeholders(processor: VLProcessor) -> None:
    # Test that template placeholders are cleaned/ignored
    raw_response = (
        "ORIGINAL:\n"
        "[Extracted Chinese text]\n\n"
        "TRANSLATED:\n"
        "[Direct translation into English]\n\n"
        "CONFIDENCE:\n"
        "0.85"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == ""
    assert trans == ""
    assert conf == 0.85


def test_parse_response_unclosed_think(processor: VLProcessor) -> None:
    # Test that unclosed think blocks containing draft ORIGINAL/TRANSLATED markers are stripped
    raw_response = (
        "<think>\n"
        "Thinking Process:\n"
        "1. Identify text.\n"
        "2. Draft ORIGINAL:\n"
        "   剑侠世界4\n"
        "   TRANSLATED:\n"
        "   Sword Hero's Realm 4\n"
    )
    orig, trans, conf = processor.parse_response(raw_response)
    assert orig == ""
    assert trans == ""
    assert conf == 0.85


def test_parse_response_streaming_phases(processor: VLProcessor) -> None:
    # Phase 1: Unclosed think block containing drafts
    p1 = "<think>Thinking... ORIGINAL: draft TRANSLATED: draft"
    orig, trans, _ = processor.parse_response(p1)
    assert orig == ""
    assert trans == ""

    # Phase 2: Closed think block, but final layout not started yet
    p2 = "<think>Thinking... </think>\n"
    orig, trans, _ = processor.parse_response(p2)
    assert orig == ""
    assert trans == ""

    # Phase 3: ORIGINAL starts streaming
    p3 = "<think>Thinking... </think>\nORIGINAL:\nReal Original Text"
    orig, trans, _ = processor.parse_response(p3)
    assert orig == "Real Original Text"
    assert trans == ""

    # Phase 4: TRANSLATED starts streaming
    p4 = "<think>Thinking... </think>\nORIGINAL:\nReal Original Text\n\nTRANSLATED:\nReal Translation"
    orig, trans, _ = processor.parse_response(p4)
    assert orig == "Real Original Text"
    assert trans == "Real Translation"

