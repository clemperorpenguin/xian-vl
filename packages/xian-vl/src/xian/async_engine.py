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


from urllib.parse import urlparse

def sanitize_url(url: str) -> str:
    try:
        p = urlparse(url)
        if p.password or p.username:
            netloc = p.netloc.split("@")[-1]
            return f"{p.scheme}://{netloc}{p.path}"
    except Exception:
        pass
    return url


class AsyncEngine(threading.Thread):
    """Background thread with a persistent event loop and OpenAI client."""

    def __init__(self, base_url: str, api_key: str = "not-needed") -> None:
        super().__init__(daemon=True, name="xian-async-engine")
        self._base_url = base_url
        self._api_key = api_key
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: AsyncOpenAI | None = None
        self._ready = threading.Event()

    def _handle_exception(self, loop, context):
        msg = context.get("exception", context.get("message"))
        logger.error("Unhandled exception in AsyncEngine loop: %s", msg)

    # ── Public API ────────────────────────────────────────────────────

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("AsyncEngine failed to start (timeout)")
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("AsyncEngine loop is not running or has stopped")
        return self._loop

    @property
    def client(self) -> AsyncOpenAI:
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("AsyncEngine failed to start (timeout)")
        if not self._client:
            raise RuntimeError("AsyncEngine client is not initialized")
        return self._client

    def submit(self, coro: Coroutine[Any, Any, T]) -> Future[T]:
        """Schedule *coro* on the engine loop; return a concurrent.futures.Future."""
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("AsyncEngine failed to start (timeout)")
        if not self._loop or not self._loop.is_running():
            raise RuntimeError("AsyncEngine loop is not running or has stopped")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    def reconfigure(self, base_url: str, api_key: str = "not-needed") -> None:
        """Replace the client with a new base_url (e.g. after settings change)."""
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("AsyncEngine failed to start (timeout)")
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
                if self._client:
                    try:
                        await self._client.close()
                    except Exception as e:
                        logger.error("Error closing OpenAI client in shutdown: %s", e)
                self._loop.stop()
            self._loop.call_soon_threadsafe(
                lambda: self._loop.create_task(_graceful_stop())
            )

    # ── Thread internals ──────────────────────────────────────────────

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.set_exception_handler(self._handle_exception)
        asyncio.set_event_loop(self._loop)
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        self._loop.call_soon(self._ready.set)
        logger.info("AsyncEngine started (base_url=%s)", sanitize_url(self._base_url))
        try:
            self._loop.run_forever()
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.close()
            logger.info("AsyncEngine stopped")

    def _do_reconfigure(self, base_url: str, api_key: str) -> None:
        """Must be called from within the engine loop."""
        async def _swap():
            old = self._client
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            logger.info("AsyncEngine reconfigured (base_url=%s)", sanitize_url(base_url))
            if old:
                await old.close()
        self._loop.create_task(_swap())
