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

"""Tests for URL safety helpers (SSRF guardrails)."""

from __future__ import annotations

import asyncio
import ipaddress

import httpx
import pytest

from xian import url_safety as us
from xian.lemonade_url import normalize_lemonade_api_base_url, should_warn_http_to_non_loopback
from xian.url_safety import httpx_get_with_safe_redirects, markdown_http_https_url_or_none


@pytest.fixture
def public_dns(monkeypatch: pytest.MonkeyPatch):
    """Avoid real DNS; pretend every hostname resolves to a public address."""

    def _fake(host: str, *, timeout_seconds: float = 2.0) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return [ipaddress.ip_address("8.8.8.8")]

    monkeypatch.setattr(us, "_resolve_host_ips", _fake)


def test_literal_loopback_blocked() -> None:
    assert not us.is_safe_http_url_for_untrusted_fetch("http://127.0.0.1/")
    assert not us.is_safe_http_url_for_untrusted_fetch("http://[::1]/x")


def test_literal_public_allowed(public_dns: None) -> None:
    assert us.is_safe_http_url_for_untrusted_fetch("http://example.com/path")


def test_forbidden_hostname(public_dns: None) -> None:
    assert not us.is_safe_http_url_for_untrusted_fetch("http://metadata.google.internal/")


def test_markdown_only_http_https() -> None:
    assert markdown_http_https_url_or_none("javascript:alert(1)") is None
    assert markdown_http_https_url_or_none("https://example.com/x") == "https://example.com/x"


def test_normalize_lemonade() -> None:
    assert normalize_lemonade_api_base_url("  http://localhost:13305 ") == "http://localhost:13305/v1"
    assert normalize_lemonade_api_base_url("http://host:1/v1") == "http://host:1/v1"


def test_should_warn_http_remote() -> None:
    assert should_warn_http_to_non_loopback("http://192.168.1.5/v1")
    assert not should_warn_http_to_non_loopback("http://localhost:13305/v1")
    assert not should_warn_http_to_non_loopback("https://remote.example/v1")


def test_httpx_get_aborts_redirect_to_loopback(public_dns: None) -> None:
    async def _run() -> httpx.Response | None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "example.com":
                return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
            return httpx.Response(200, text="ok")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await httpx_get_with_safe_redirects(client, "http://example.com/start")

    assert asyncio.run(_run()) is None


def test_httpx_get_follows_safe_redirect(public_dns: None) -> None:
    async def _run() -> httpx.Response | None:
        def handler(request: httpx.Request) -> httpx.Response:
            u = str(request.url)
            if u.endswith("/a"):
                return httpx.Response(302, headers={"location": "/b"})
            if u.endswith("/b"):
                return httpx.Response(200, text="<html><body>ok</body></html>")
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await httpx_get_with_safe_redirects(client, "http://example.com/a")

    out = asyncio.run(_run())
    assert out is not None
    assert out.status_code == 200
