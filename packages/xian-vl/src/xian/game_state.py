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

"""Turn a session log into the paragraph the model actually needs.

A play session produces far more text than fits in a prompt, so context is
assembled from three parts under a token budget:

* a **digest** — a rolling summary of the session so far, always included;
* a **recent tail** — the last handful of events verbatim, always included;
* **retrieved** events — full-text matches for whatever the player just asked.

The digest is what makes "what do I do next?" answerable an hour in: raw event
history would have long since fallen out of the window.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from xian.session_store import SessionEvent, SessionStore
from xian.timeout import CHAT_AUX_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

__all__ = ["GameStateAssembler", "RollingSummarizer", "estimate_tokens"]

_DIGEST_SYSTEM_PROMPT = """You maintain a running state summary for a player's game session.
Merge the PREVIOUS SUMMARY with the NEW EVENTS into one updated summary.
Preserve facts from the previous summary unless the new events contradict them.
Be specific: keep names, places, item names, and numbers exactly as written.

Reply with exactly these sections, omitting any that have no content:
Scene: where the player is and what is happening
Objectives: active quests and stated goals, most recent first
Entities: named characters, factions, items, and locations that matter
Decisions: choices the player has made
Open threads: questions raised but not resolved

Write terse bullet points. No preamble, no commentary."""


def estimate_tokens(text: str) -> int:
    """Rough token count that does not lie about CJK.

    Tokenizers split Chinese/Japanese far closer to one token per character
    than the usual ~4-chars-per-token English rule, so this uses ~3 to stay
    conservative for the mixed text MAGE handles.
    """
    return max(1, len(text) // 3)


class GameStateAssembler:
    """Builds the game-state block injected into prompts."""

    #: Share of the budget each part may claim.
    DIGEST_SHARE = 0.35
    RECENT_SHARE = 0.40

    #: Events pulled before budgeting trims them back.
    RECENT_LIMIT = 24
    RETRIEVED_LIMIT = 8

    def __init__(self, store: SessionStore):
        self.store = store

    def build_context_block(self, question: str | None = None, token_budget: int = 1200) -> str:
        """Assemble session context, or "" when there is nothing worth sending."""
        if token_budget <= 0:
            return ""

        sections: list[str] = []
        digest, _ = self.store.get_digest()
        if digest:
            sections.append(
                "Session so far:\n" + _clip(digest, int(token_budget * self.DIGEST_SHARE))
            )

        recent = self.store.recent_events(limit=self.RECENT_LIMIT)
        recent_block = _render_events(recent, int(token_budget * self.RECENT_SHARE))
        if recent_block:
            sections.append("Recently on screen:\n" + recent_block)

        if question:
            seen = {e.id for e in recent}
            hits = [e for e in self.store.search(question, limit=self.RETRIEVED_LIMIT) if e.id not in seen]
            used = sum(estimate_tokens(s) for s in sections)
            retrieved = _render_events(hits, max(0, token_budget - used))
            if retrieved:
                sections.append("Earlier, possibly relevant:\n" + retrieved)

        return "\n\n".join(sections)


class RollingSummarizer:
    """Folds accumulated events into the session digest.

    Runs on the shared async engine and never more than once at a time: a
    second trigger while a summary is in flight is dropped, not queued.
    """

    #: Events (or estimated tokens) that accumulate before re-summarizing.
    EVENT_THRESHOLD = 25
    TOKEN_THRESHOLD = 1500

    #: Cap on the digest itself, so it cannot grow without bound.
    MAX_DIGEST_TOKENS = 500

    def __init__(self, store: SessionStore, processor):
        self.store = store
        self.processor = processor
        self._running = threading.Event()
        # maybe_summarize() is called from several worker threads, so the
        # "is it already running?" test and the claim have to be atomic —
        # an Event alone lets two threads both pass the check.
        self._claim_lock = threading.Lock()

    def should_summarize(self) -> bool:
        _, last_id = self.store.get_digest()
        pending = self.store.events_since(last_id)
        if not pending:
            return False
        if len(pending) >= self.EVENT_THRESHOLD:
            return True
        return sum(estimate_tokens(e.summary_line()) for e in pending) >= self.TOKEN_THRESHOLD

    def maybe_summarize(self) -> None:
        """Kick off a summary if one is due. Returns immediately."""
        if self._running.is_set() or not self.processor.engine:
            return
        if not self.should_summarize():
            return

        with self._claim_lock:
            if self._running.is_set():
                return
            self._running.set()

        coro = self._summarize()
        try:
            self.processor.engine.submit(coro)
        except Exception as exc:
            # The coroutine was never handed to a loop, so close it explicitly
            # instead of leaving an un-awaited coroutine for the GC to warn on.
            coro.close()
            self._running.clear()
            logger.debug("Could not schedule session summary: %s", exc)

    async def _summarize(self) -> None:
        try:
            previous, last_id = self.store.get_digest()
            pending = self.store.events_since(last_id)
            if not pending:
                return

            new_events = _render_events(pending, self.TOKEN_THRESHOLD * 2)
            user_content = (
                f"PREVIOUS SUMMARY:\n{previous or '(none — this is the start of the session)'}\n\n"
                f"NEW EVENTS:\n{new_events}"
            )

            response = await asyncio.wait_for(
                self.processor.client.chat.completions.create(
                    model=self.processor.get_model_name(),
                    messages=[
                        {"role": "system", "content": _DIGEST_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=self.MAX_DIGEST_TOKENS * 2,
                    temperature=0.2,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                ),
                timeout=CHAT_AUX_TIMEOUT_SECONDS,
            )
            choice = response.choices[0] if response.choices else None
            digest = (choice.message.content or "").strip() if choice else ""
            digest = _strip_thinking(digest)
            if not digest:
                logger.debug("Summarizer returned nothing; keeping previous digest")
                return

            self.store.set_digest(_clip(digest, self.MAX_DIGEST_TOKENS), pending[-1].id)
            logger.info("Session digest updated from %d events", len(pending))
        except asyncio.TimeoutError:
            logger.warning("Session summary timed out after %.0fs", CHAT_AUX_TIMEOUT_SECONDS)
        except Exception as exc:
            # Memory is an enhancement; a failed summary must not surface to the player.
            logger.warning("Session summary failed: %s", exc)
        finally:
            self._running.clear()


def _render_events(events: list[SessionEvent], token_budget: int) -> str:
    """Render events newest-last, dropping the oldest that do not fit."""
    if not events or token_budget <= 0:
        return ""
    lines: list[str] = []
    used = 0
    for event in reversed(events):
        line = event.summary_line()
        cost = estimate_tokens(line)
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    return "\n".join(reversed(lines))


def _clip(text: str, token_budget: int) -> str:
    """Trim text to a token budget on a line boundary."""
    if estimate_tokens(text) <= token_budget:
        return text
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        cost = estimate_tokens(line)
        if used + cost > token_budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept) if kept else text[: token_budget * 3]


def _strip_thinking(text: str) -> str:
    """Drop a leading <think> block some models emit even when disabled."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
