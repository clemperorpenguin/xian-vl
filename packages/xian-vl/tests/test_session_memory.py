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

"""Durable session memory: the store, the context assembler, the summarizer."""

import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from xian.game_state import GameStateAssembler, RollingSummarizer, estimate_tokens
from xian.pipeline import VLConfig, VLProcessor
from xian.session_store import SessionEvent, SessionStore


@pytest.fixture
def store(tmp_path):
    s = SessionStore(str(tmp_path / "sessions.db"))
    s.begin_session(window_title="Jianghu Online", game_profile="jx3")
    yield s
    s.close()


def _append(store, kind, original="", translated="", **extra):
    store.append_event(SessionEvent(kind=kind, original=original, translated=translated, extra=extra))


# ── store ────────────────────────────────────────────────────────────

def test_events_round_trip_through_the_writer_thread(store):
    _append(store, "ocr", "前往城門", "Head to the city gate")
    _append(store, "dialogue", "我是李掌櫃", "I am Shopkeeper Li")
    assert store.flush(), "writer thread should commit within the timeout"

    events = store.recent_events()
    assert [e.kind for e in events] == ["ocr", "dialogue"]
    assert events[0].translated == "Head to the city gate"
    assert events[0].ts > 0
    assert events[0].id > 0


def test_appends_are_safe_from_many_threads(store):
    def worker(n):
        for i in range(20):
            _append(store, "ocr", f"原文{n}-{i}", f"line {n}-{i}")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.flush()
    assert len(store.recent_events(limit=200)) == 80


def test_search_finds_events_by_either_language(store):
    _append(store, "ocr", "青龍寺在東邊", "The Azure Dragon Temple is to the east")
    _append(store, "ocr", "藥草在河邊", "Herbs grow by the river")
    store.flush()

    hits = store.search("where is the temple?")
    assert any("Azure Dragon Temple" in e.translated for e in hits)

    hits_cjk = store.search("青龍寺")
    assert any("青龍寺" in e.original for e in hits_cjk)


def test_search_survives_punctuation_that_fts_treats_as_operators(store):
    _append(store, "ocr", "", "Meet the envoy at dusk")
    store.flush()

    # Bare FTS5 would choke on these; the query builder must sanitize them.
    for query in ['"unbalanced', "envoy AND (", "what now?? -- envoy", "*"]:
        store.search(query)

    assert any("envoy" in e.translated for e in store.search("envoy"))


def test_recent_events_can_filter_by_kind(store):
    _append(store, "ocr", "", "screen text")
    _append(store, "chat_user", "", "what do I do next?")
    store.flush()

    assert [e.kind for e in store.recent_events(kinds=("chat_user",))] == ["chat_user"]


def test_purge_drops_history_past_the_retention_window(store):
    _append(store, "ocr", "", "ancient history")
    store.flush()
    with store._connect() as conn:
        conn.execute("UPDATE events SET ts = ?", (time.time() - 90 * 86400,))

    assert store.purge_older_than(30) == 1
    assert store.recent_events() == []


def test_clear_all_empties_the_index_too(store):
    _append(store, "ocr", "", "forget me")
    store.flush()
    store.clear_all()

    store.begin_session()
    assert store.recent_events() == []
    assert store.search("forget") == []


def test_digest_persists_per_session(store):
    assert store.get_digest() == ("", 0)
    store.set_digest("Scene: outside the gate", 42)
    assert store.get_digest() == ("Scene: outside the gate", 42)

    store.begin_session(game_profile="other-game")
    assert store.get_digest() == ("", 0), "a new session starts with a clean digest"


def test_events_since_returns_only_unsummarized(store):
    _append(store, "ocr", "", "first")
    _append(store, "ocr", "", "second")
    store.flush()
    first_id = store.recent_events()[0].id

    pending = store.events_since(first_id)
    assert [e.translated for e in pending] == ["second"]


# ── assembler ────────────────────────────────────────────────────────

def test_context_block_combines_digest_recent_and_retrieved(store):
    store.set_digest("Scene: the player is outside Azure Dragon Temple", 0)
    _append(store, "ocr", "", "A locked gate blocks the path")
    for i in range(30):
        _append(store, "ocr", "", f"filler line {i}")
    _append(store, "dialogue", "", "The abbot mentions a hidden key")
    store.flush()

    block = GameStateAssembler(store).build_context_block("where is the key?", token_budget=1200)

    assert "Azure Dragon Temple" in block
    assert "hidden key" in block, "the newest events must always survive budgeting"
    assert "Recently on screen:" in block


def test_context_block_respects_its_token_budget(store):
    store.set_digest("x " * 4000, 0)
    for i in range(300):
        _append(store, "ocr", "", f"a fairly wordy line of dialogue number {i}")
    store.flush()

    budget = 400
    block = GameStateAssembler(store).build_context_block("dialogue", token_budget=budget)

    assert estimate_tokens(block) <= budget * 1.1, "block should stay near the requested budget"


def test_context_block_is_empty_without_history(store):
    assert GameStateAssembler(store).build_context_block("anything") == ""


def test_context_block_does_not_repeat_retrieved_events(store):
    _append(store, "ocr", "", "the envoy waits at dusk")
    store.flush()

    block = GameStateAssembler(store).build_context_block("envoy", token_budget=1200)

    assert block.count("the envoy waits at dusk") == 1


# ── summarizer ───────────────────────────────────────────────────────

def _processor_with_store(store, reply="Scene: at the gate\nObjectives: find the key"):
    processor = VLProcessor(VLConfig())
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = reply
    response = MagicMock()
    response.choices = [choice]
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    processor.engine = MagicMock()
    processor.engine.client = mock_client
    processor.attach_session_store(store)
    return processor, mock_client


def test_summarizer_waits_until_enough_has_happened(store):
    processor, _ = _processor_with_store(store)
    summarizer = processor.summarizer

    _append(store, "ocr", "", "one line")
    store.flush()
    assert summarizer.should_summarize() is False

    for i in range(RollingSummarizer.EVENT_THRESHOLD):
        _append(store, "ocr", "", f"line {i}")
    store.flush()
    assert summarizer.should_summarize() is True


@pytest.mark.anyio
async def test_summary_folds_events_into_the_digest(store):
    processor, mock_client = _processor_with_store(store)
    for i in range(RollingSummarizer.EVENT_THRESHOLD + 1):
        _append(store, "ocr", "", f"event {i}")
    store.flush()

    await processor.summarizer._summarize()

    digest, last_id = store.get_digest()
    assert "find the key" in digest
    assert last_id == store.recent_events(limit=100)[-1].id

    prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "PREVIOUS SUMMARY" in prompt and "event 0" in prompt


@pytest.mark.anyio
async def test_summary_failure_leaves_the_previous_digest_intact(store):
    processor, mock_client = _processor_with_store(store)
    store.set_digest("Scene: the old summary", 0)
    _append(store, "ocr", "", "something new")
    store.flush()
    mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("server down"))

    await processor.summarizer._summarize()  # must not raise

    assert store.get_digest()[0] == "Scene: the old summary"


def test_summarizer_does_not_stack_concurrent_runs(store):
    processor, _ = _processor_with_store(store)
    for i in range(RollingSummarizer.EVENT_THRESHOLD + 1):
        _append(store, "ocr", "", f"event {i}")
    store.flush()

    # Stand in for the real engine, which would consume the coroutine.
    processor.engine.submit = MagicMock(side_effect=lambda coro: coro.close())

    processor.summarizer.maybe_summarize()
    processor.summarizer.maybe_summarize()

    assert processor.engine.submit.call_count == 1


# ── processor wiring ─────────────────────────────────────────────────

def test_processor_without_a_store_records_nothing():
    processor = VLProcessor(VLConfig())
    processor.record_event("ocr", "原文", "text")  # must not raise
    assert processor.build_game_state_block("anything") == ""


def test_recorded_results_skip_the_no_text_sentinel(store):
    from shared_types.models import TranslationResult

    processor, _ = _processor_with_store(store)
    processor._record_results("ocr", [
        TranslationResult(original_text="(none)", translated_text="(none)"),
        TranslationResult(original_text="城門", translated_text="City gate"),
    ])
    store.flush()

    assert [e.translated for e in store.recent_events()] == ["City gate"]


@pytest.mark.anyio
async def test_chat_prompt_carries_the_session_context(store, tmp_path):
    processor, mock_client = _processor_with_store(store, reply="Head east.")
    processor.wiki_dir = str(tmp_path)
    reply = MagicMock()
    reply.message.tool_calls = []
    reply.message.content = "Head east."
    response = MagicMock()
    response.choices = [reply]
    mock_client.chat.completions.create = AsyncMock(return_value=response)

    store.set_digest("Objectives: deliver the letter to Shopkeeper Li", 0)
    _append(store, "ocr", "", "Shopkeeper Li is at the east market")
    store.flush()
    processor.context_manager.add_frame(Image.new("RGB", (8, 8)))

    await processor.process_chat("what do I do next?")

    system_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "GAME SESSION CONTEXT" in system_prompt
    assert "deliver the letter" in system_prompt
    assert "east market" in system_prompt

    store.flush()
    kinds = [e.kind for e in store.recent_events()]
    assert "chat_user" in kinds and "chat_assistant" in kinds


# ── review follow-ups ────────────────────────────────────────────────

def test_search_recalls_earlier_sessions_of_the_same_game(store):
    """Remembering an NPC met three evenings ago is the point of the feature."""
    _append(store, "ocr", "", "Elder Xu guards the north pass")
    store.flush()
    store.end_session()
    store.begin_session(window_title="Jianghu Online", game_profile="jx3")

    assert any("Elder Xu" in e.translated for e in store.search("Elder Xu"))


def test_search_does_not_leak_across_different_games(store):
    """Retrieved events are shown to the model as fact about *this* game."""
    _append(store, "ocr", "", "Elder Xu guards the north pass")
    store.flush()
    store.end_session()
    store.begin_session(window_title="Some Other RPG", game_profile="other")

    assert store.search("Elder Xu") == []
    assert any("Elder Xu" in e.translated for e in store.search("Elder Xu", same_game_only=False))


def test_context_block_does_not_inject_another_games_quest_text(store):
    _append(store, "ocr", "", "Deliver the jade seal to the magistrate")
    store.flush()
    store.end_session()
    store.begin_session(window_title="Some Other RPG", game_profile="other")

    block = GameStateAssembler(store).build_context_block("jade seal", token_budget=1200)

    assert "magistrate" not in block


def test_flush_waits_for_events_queued_during_the_flush_window(store):
    """clear_all() flushes first; a missed event would be resurrected after."""
    for i in range(50):
        _append(store, "ocr", "", f"line {i}")
    assert store.flush()

    store.clear_all()
    store.begin_session()
    assert store.recent_events(limit=100, session_only=False) == []


def test_connections_are_closed_after_every_operation(store, monkeypatch):
    """Reads and writes must not leave SQLite handles behind.

    Regression: `with sqlite3.connect(...) as conn` commits but never closes,
    so every append and every query leaked a connection and a WAL reader.
    """
    import sqlite3 as sqlite3_module

    opened = []
    real_connect = sqlite3_module.connect

    def _tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr("xian.session_store.sqlite3.connect", _tracking_connect)

    _append(store, "ocr", "原文", "translated")
    store.flush()
    store.recent_events(limit=5)
    store.search("translated")
    store.get_digest()
    store.purge_older_than(days=365)

    assert opened, "expected the store to have opened at least one connection"
    for conn in opened:
        with pytest.raises(sqlite3_module.ProgrammingError):
            conn.execute("SELECT 1")
