"""Persistent async engine thread.

Owns a single asyncio event loop and the AsyncOpenAI client.
All callers submit coroutines via submit() and await the Future.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, TypeVar

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AsyncEngine(threading.Thread):
    """Background thread with a persistent event loop and OpenAI client."""

    def __init__(self, base_url: str, api_key: str = "not-needed") -> None:
        super().__init__(daemon=True, name="xian-async-engine")
        self._base_url = base_url
        self._api_key = api_key
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: AsyncOpenAI | None = None
        self._ready = threading.Event()

    # ── Public API ────────────────────────────────────────────────────

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        self._ready.wait()      # block until loop is running
        assert self._loop is not None
        return self._loop

    @property
    def client(self) -> AsyncOpenAI:
        self._ready.wait()      # block until loop is running
        assert self._client is not None
        return self._client

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule *coro* on the engine loop; return a concurrent.futures.Future."""
        self._ready.wait()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    def reconfigure(self, base_url: str, api_key: str = "not-needed") -> None:
        """Replace the client with a new base_url (e.g. after settings change)."""
        self._ready.wait()
        # Verify self._loop exists and is running
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._do_reconfigure, base_url, api_key)
        else:
            logger.warning("AsyncEngine loop not running, updating variables directly")
            self._base_url = base_url
            self._api_key = api_key

    def shutdown(self) -> None:
        if self._loop and self._loop.is_running():
            async def _graceful_stop():
                current = asyncio.current_task(self._loop)
                tasks = [t for t in asyncio.all_tasks(self._loop) if t is not current and not t.done()]
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._loop.stop()
            self._loop.call_soon_threadsafe(
                lambda: self._loop.create_task(_graceful_stop())
            )

    # ── Thread internals ──────────────────────────────────────────────

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        self._ready.set()
        logger.info("AsyncEngine started (base_url=%s)", self._base_url)
        try:
            self._loop.run_forever()
        finally:
            # Check if event loop is closed to avoid run_until_complete errors
            if not self._loop.is_closed():
                self._loop.run_until_complete(self._client.close())
                self._loop.close()
            logger.info("AsyncEngine stopped")

    def _do_reconfigure(self, base_url: str, api_key: str) -> None:
        """Must be called from within the engine loop."""
        async def _swap():
            old = self._client
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            logger.info("AsyncEngine reconfigured (base_url=%s)", base_url)
            if old:
                await old.close()
        self._loop.create_task(_swap())
