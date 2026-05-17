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
