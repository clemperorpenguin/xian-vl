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

"""The chat knowledge-search path, both ways the model can reach it."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from xian.pipeline import VLProcessor, VLConfig
from xian.tools import OMNI_TOOLS


def _make_processor(tmp_path, monkeypatch):
    """A processor whose wiki holds one article and whose web search is stubbed."""
    (tmp_path / "Qingxu Temple.md").write_text(
        "---\ntitle: Qingxu Temple\n---\n\nThe temple sits east of the river.\n",
        encoding="utf-8",
    )

    processor = VLProcessor(VLConfig())
    processor.wiki_dir = str(tmp_path)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    processor.engine = MagicMock()
    processor.engine.client = mock_client

    # The web half is offline in tests; the local wiki alone must still answer.
    class _NoWeb:
        async def dual_search(self, **_kwargs):
            return []

        async def close(self):
            return None

    monkeypatch.setattr("xian.pipeline.WebSearcher", _NoWeb)
    # Query translation is its own completion call; stub it so the side_effect
    # list below maps one-to-one onto the chat turns under test.
    processor.translate_query = AsyncMock(side_effect=lambda q, _lang: q)
    return processor, mock_client.chat.completions.create


def _reply(content=None, tool_calls=()):
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = list(tool_calls)
    response = MagicMock()
    response.choices = [choice]
    return response


def _tool_call(name, arguments):
    call = MagicMock()
    call.id = "call_1"
    call.function.name = name
    call.function.arguments = arguments
    return call


def test_search_knowledge_is_declared_as_a_tool():
    """The system prompt tells the model to call it, so it has to be callable."""
    names = {t["function"]["name"] for t in OMNI_TOOLS}
    assert "search_knowledge" in names

    schema = next(t for t in OMNI_TOOLS if t["function"]["name"] == "search_knowledge")
    params = schema["function"]["parameters"]
    assert params["required"] == ["query"]
    assert "language" in params["properties"]


@pytest.mark.anyio
async def test_structured_tool_call_returns_wiki_hits(tmp_path, monkeypatch):
    """A search_knowledge tool call must yield results, not a swallowed error.

    Regression: LocalWikiSearcher.search is async and was called without await,
    so the coroutine + list concatenation raised TypeError inside the generic
    tool handler and every search came back as "Tool 'search_knowledge' failed".
    """
    processor, create = _make_processor(tmp_path, monkeypatch)
    create.side_effect = [
        _reply(tool_calls=[_tool_call("search_knowledge", '{"query": "Qingxu Temple"}')]),
        _reply(content="It is east of the river."),
    ]

    result = await processor.process_chat("where is Qingxu Temple?")
    assert result == "It is east of the river."

    messages = create.call_args_list[1][1]["messages"]
    tool_message = next(m for m in messages if m.get("role") == "tool")
    assert "east of the river" in tool_message["content"]
    assert "failed" not in tool_message["content"]

    await processor.close()


@pytest.mark.anyio
async def test_text_tool_call_returns_wiki_hits(tmp_path, monkeypatch):
    """The XML-text fallback path must not raise out of process_chat.

    Regression: the same missing await here was not caught by any local handler,
    so the TypeError propagated and killed the whole chat turn.
    """
    processor, create = _make_processor(tmp_path, monkeypatch)
    create.side_effect = [
        _reply(content=(
            "<tool_call>\n<function=search_knowledge>\n"
            "<parameter=query>Qingxu Temple</parameter>\n"
            "</function>\n</tool_call>"
        )),
        _reply(content="It is east of the river."),
    ]

    result = await processor.process_chat("where is Qingxu Temple?")
    assert result == "It is east of the river."

    messages = create.call_args_list[1][1]["messages"]
    assert any("east of the river" in str(m.get("content", "")) for m in messages)

    await processor.close()


@pytest.mark.anyio
async def test_web_failure_still_returns_local_results(tmp_path, monkeypatch):
    """A raising web search must degrade to the wiki, not abort the turn."""
    processor, _create = _make_processor(tmp_path, monkeypatch)

    class _Broken:
        async def dual_search(self, **_kwargs):
            raise RuntimeError("network down")

        async def close(self):
            return None

    monkeypatch.setattr("xian.pipeline.WebSearcher", _Broken)
    processor.translate_query = AsyncMock(return_value="Qingxu Temple")

    summary = await processor._run_knowledge_search("Qingxu Temple", "en-US", "zh-CN")
    assert "east of the river" in summary

    await processor.close()
