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

"""Durable log of what the player has seen, said, and decided.

Everything MAGE translates is thrown away the moment its bubble closes, so the
assistant cannot answer "what do I do next?" — it has never seen the quest text
it translated ten minutes ago.  This module keeps that history on disk.

Writes go through a single background thread: callers append from the Qt main
thread and from worker threads, and neither may block on disk I/O (AGENTS.md
§2).  Reads run on the calling thread against their own short-lived connection.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SessionEvent", "SessionStore", "EVENT_KINDS"]

# Where an event came from. Kept open-ended on purpose: new capture modes
# should be able to log without a schema migration.
EVENT_KINDS = (
    "ocr",           # a translated screen region
    "dialogue",      # a line advanced in dialogue mode
    "chat_user",     # something the player asked
    "chat_assistant",  # something the assistant answered
    "raid",          # a transcribed/translated voice line
    "note",          # a note the player saved
    "inpaint",       # a line rendered by the live overlay
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   REAL NOT NULL,
    ended_at     REAL,
    window_title TEXT,
    game_profile TEXT,
    digest       TEXT,
    digest_event_id INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    original    TEXT NOT NULL DEFAULT '',
    translated  TEXT NOT NULL DEFAULT '',
    confidence  REAL,
    region_json TEXT,
    extra_json  TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

-- Rows are written here explicitly rather than by trigger, because the text
-- is CJK-segmented on the way in (see _segment_cjk) and SQL triggers cannot
-- do that. rowid always mirrors events.id.
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    original, translated, tokenize='unicode61 remove_diacritics 2'
);
"""

# Scripts whose text is written without spaces, so FTS5's unicode61 tokenizer
# would swallow a whole clause as one token and make substring search useless.
_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _segment_cjk(text: str) -> str:
    """Space out CJK characters so each becomes its own FTS token.

    "青龍寺在東邊" indexes as six tokens, which lets a query for 青龍寺 match it
    as a three-token phrase. Without this the whole clause is a single token
    and only an exact-clause query would ever hit.
    """
    if not text:
        return ""
    out: list[str] = []
    for char in text:
        if _is_cjk(char):
            out.append(" ")
            out.append(char)
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


@dataclass
class SessionEvent:
    """One thing that happened during play."""

    kind: str
    original: str = ""
    translated: str = ""
    confidence: float | None = None
    region: tuple[int, int, int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0
    id: int = 0
    session_id: int = 0

    def summary_line(self) -> str:
        """Render the event as a single line for prompt context."""
        stamp = time.strftime("%H:%M", time.localtime(self.ts)) if self.ts else "--:--"
        if self.original and self.translated and self.original != self.translated:
            body = f"{self.translated}  (orig: {self.original})"
        else:
            body = self.translated or self.original
        return f"[{stamp}] {self.kind}: {body}".strip()


class SessionStore:
    """SQLite-backed history of a play session.

    The Qt layer owns the path (``QStandardPaths.AppDataLocation``); this class
    stays free of Qt so the engine package remains importable by the CLI apps.
    """

    #: How long the writer waits to batch appends before committing.
    FLUSH_INTERVAL_SECONDS = 0.5

    def __init__(self, db_path: str, *, retention_days: int = 30):
        self.db_path = db_path
        self.retention_days = retention_days
        self._session_id: int | None = None
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._flushed = threading.Event()
        self._flushed.set()
        # Counted rather than inferred from queue.empty(): the writer can
        # observe an empty queue in the window between a producer clearing
        # ``_flushed`` and its put(), which would let flush() return before
        # that event was committed — and clear_all() would then delete rows
        # the writer is about to re-insert.
        self._pending_writes = 0

        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        with self._connect() as conn:
            conn.executescript(_SCHEMA)

        self._writer = threading.Thread(
            target=self._writer_loop, name="xian-session-writer", daemon=True
        )
        self._writer.start()
        logger.info("SessionStore ready at %s", db_path)

    # ── connections ──────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """Yield a short-lived connection, committing and closing on exit.

        sqlite3's own connection context manager commits but does *not* close,
        which would leak a handle (and a WAL reader) per read and per write.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ── session lifecycle ────────────────────────────────────────────

    def begin_session(self, window_title: str | None = None, game_profile: str | None = None) -> int:
        """Open a new session and make it the target for later appends."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (started_at, window_title, game_profile) VALUES (?, ?, ?)",
                (time.time(), window_title, game_profile),
            )
            session_id = int(cursor.lastrowid)
        with self._lock:
            self._session_id = session_id
        logger.info("Session %d started (%s)", session_id, game_profile or window_title or "unnamed")
        return session_id

    def end_session(self) -> None:
        """Close the current session, if one is open."""
        session_id = self.session_id
        if session_id is None:
            return
        self.flush()
        with self._connect() as conn:
            conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id))
        with self._lock:
            self._session_id = None

    @property
    def session_id(self) -> int | None:
        with self._lock:
            return self._session_id

    # ── writes ───────────────────────────────────────────────────────

    def append_event(self, event: SessionEvent) -> None:
        """Queue an event for durable storage. Never blocks on disk."""
        if self._closed.is_set():
            return
        session_id = self.session_id
        if session_id is None:
            session_id = self.begin_session()
        if not event.ts:
            event.ts = time.time()
        event.session_id = session_id
        with self._lock:
            self._pending_writes += 1
            self._flushed.clear()
        self._queue.put(event)

    def _mark_written(self, count: int) -> None:
        """Account for ``count`` committed events and release flush() waiters."""
        with self._lock:
            self._pending_writes = max(0, self._pending_writes - count)
            if self._pending_writes == 0:
                self._flushed.set()

    def _writer_loop(self) -> None:
        pending: list[SessionEvent] = []
        while True:
            try:
                item = self._queue.get(timeout=self.FLUSH_INTERVAL_SECONDS)
                if item is None:  # shutdown sentinel
                    self._flush_pending(pending)
                    self._flushed.set()
                    return
                pending.append(item)
                # Opportunistically drain so bursts commit together.
                while len(pending) < 200:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        self._flush_pending(pending)
                        self._flushed.set()
                        return
                    pending.append(item)
            except queue.Empty:
                pass

            self._flush_pending(pending)
            pending = []

    def _flush_pending(self, pending: list[SessionEvent]) -> None:
        """Write a batch and account for it, whether or not the write worked."""
        if not pending:
            return
        count = len(pending)
        try:
            self._write_batch(pending)
        finally:
            self._mark_written(count)

    def _write_batch(self, pending: list[SessionEvent]) -> None:
        if not pending:
            return
        try:
            with self._connect() as conn:
                for e in pending:
                    cursor = conn.execute(
                        "INSERT INTO events (session_id, ts, kind, original, translated,"
                        " confidence, region_json, extra_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            e.session_id, e.ts, e.kind, e.original or "", e.translated or "",
                            e.confidence,
                            json.dumps(list(e.region)) if e.region else None,
                            json.dumps(e.extra) if e.extra else None,
                        ),
                    )
                    conn.execute(
                        "INSERT INTO events_fts (rowid, original, translated) VALUES (?, ?, ?)",
                        (
                            cursor.lastrowid,
                            _segment_cjk(e.original or ""),
                            _segment_cjk(e.translated or ""),
                        ),
                    )
        except Exception as exc:  # a lost log line must never break translation
            logger.warning("SessionStore write failed (%d events dropped): %s", len(pending), exc)
        finally:
            pending.clear()

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until queued events have been committed. For tests/shutdown."""
        return self._flushed.wait(timeout=timeout)

    # ── reads ────────────────────────────────────────────────────────

    def recent_events(
        self, limit: int = 20, kinds: tuple[str, ...] | None = None, session_only: bool = True
    ) -> list[SessionEvent]:
        """Return the newest events, oldest first."""
        sql = "SELECT * FROM events"
        clauses, params = [], []
        if session_only and self.session_id is not None:
            clauses.append("session_id = ?")
            params.append(self.session_id)
        if kinds:
            clauses.append(f"kind IN ({','.join('?' * len(kinds))})")
            params.extend(kinds)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in reversed(rows)]

    def search(self, query: str, limit: int = 8, same_game_only: bool = True) -> list[SessionEvent]:
        """Full-text search over the play history, newest-relevant first.

        Ranked by BM25 with a mild recency tilt: in a long session the same
        NPC name recurs, and the latest mention is usually the useful one.

        Results span earlier sessions of the *same* game by default, which is
        the point — recalling an NPC met three evenings ago is exactly what
        this is for. They must not span different games, though: the retrieved
        block is presented to the model as established fact about the current
        playthrough, so a quest line from last week's other game would be
        asserted as true of this one.
        """
        terms = _fts_terms(query)
        if not terms:
            return []

        sql = (
            "SELECT e.*, bm25(events_fts) AS rank FROM events_fts"
            " JOIN events e ON e.id = events_fts.rowid"
        )
        params: list = [terms]
        scope = self._current_game_scope() if same_game_only else None
        if scope is not None:
            sql += (
                " JOIN sessions s ON s.id = e.session_id"
                " WHERE events_fts MATCH ?"
                " AND IFNULL(s.game_profile,'') = ? AND IFNULL(s.window_title,'') = ?"
            )
            params.extend(scope)
        else:
            sql += " WHERE events_fts MATCH ?"
        sql += " ORDER BY rank ASC, e.id DESC LIMIT ?"
        params.append(limit)

        try:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("FTS query %r rejected: %s", terms, exc)
            return []
        return [_row_to_event(r) for r in rows]

    def _current_game_scope(self) -> tuple[str, str] | None:
        """The (game_profile, window_title) identifying the current game."""
        session_id = self.session_id
        if session_id is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT game_profile, window_title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if not row:
            return None
        return (row["game_profile"] or "", row["window_title"] or "")

    # ── rolling digest ───────────────────────────────────────────────

    def get_digest(self) -> tuple[str, int]:
        """Return ``(digest_text, last_summarized_event_id)`` for this session."""
        session_id = self.session_id
        if session_id is None:
            return "", 0
        with self._connect() as conn:
            row = conn.execute(
                "SELECT digest, digest_event_id FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if not row:
            return "", 0
        return (row["digest"] or ""), int(row["digest_event_id"] or 0)

    def set_digest(self, digest: str, last_event_id: int) -> None:
        session_id = self.session_id
        if session_id is None:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET digest = ?, digest_event_id = ? WHERE id = ?",
                (digest, last_event_id, session_id),
            )

    def events_since(self, event_id: int, limit: int = 200) -> list[SessionEvent]:
        """Events logged after ``event_id`` in the current session, oldest first."""
        session_id = self.session_id
        if session_id is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
                (session_id, event_id, limit),
            ).fetchall()
        return [_row_to_event(r) for r in rows]

    # ── maintenance ──────────────────────────────────────────────────

    def purge_older_than(self, days: int | None = None) -> int:
        """Drop sessions older than the retention window. Returns rows removed."""
        days = self.retention_days if days is None else days
        if days <= 0:
            return 0
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM events_fts WHERE rowid IN"
                " (SELECT id FROM events WHERE ts < ?)",
                (cutoff,),
            )
            removed = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,)).rowcount
            conn.execute(
                "DELETE FROM sessions WHERE started_at < ? AND id != ?",
                (cutoff, self.session_id or -1),
            )
        if removed:
            logger.info("Purged %d events older than %d days", removed, days)
        return removed

    def clear_all(self) -> None:
        """Erase the entire history (the settings dialog's "clear memory")."""
        self.flush()
        with self._connect() as conn:
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM events_fts")
        with self._lock:
            self._session_id = None

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._writer.join(timeout=5.0)


def _row_to_event(row: sqlite3.Row) -> SessionEvent:
    return SessionEvent(
        id=int(row["id"]),
        session_id=int(row["session_id"]),
        ts=float(row["ts"]),
        kind=row["kind"],
        original=row["original"] or "",
        translated=row["translated"] or "",
        confidence=row["confidence"],
        region=tuple(json.loads(row["region_json"])) if row["region_json"] else None,
        extra=json.loads(row["extra_json"]) if row["extra_json"] else {},
    )


def _fts_terms(query: str) -> str:
    """Turn free text into a safe FTS5 OR-query.

    User questions carry quotes, hyphens, and punctuation that FTS5 would read
    as operators, so every term is emitted as an explicitly quoted phrase.  A
    run of CJK becomes a multi-token phrase, matching the segmentation used at
    index time, which keeps 青龍寺 matching inside 青龍寺在東邊.
    """
    phrases: list[str] = []
    run: list[str] = []
    run_is_cjk = False

    def flush() -> None:
        if not run:
            return
        term = "".join(run)
        if run_is_cjk:
            phrases.append('"' + " ".join(term) + '"')
        elif len(term) > 1:
            phrases.append(f'"{term}"')
        run.clear()

    for char in query:
        if _is_cjk(char):
            if not run_is_cjk:
                flush()
                run_is_cjk = True
            run.append(char)
        elif char.isalnum():
            if run_is_cjk:
                flush()
                run_is_cjk = False
            run.append(char)
        else:
            flush()
            run_is_cjk = False
    flush()

    return " OR ".join(phrases[:12])
