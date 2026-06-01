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

"""Block obvious SSRF targets for HTTP(S) fetches driven by untrusted URLs (search, etc.)."""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Final

import httpx

logger = logging.getLogger(__name__)

MAX_ENRICHMENT_REDIRECTS: Final[int] = 15
_RESOLVE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="url_safety_resolve")

_REDIRECT_STATUS: Final[frozenset[int]] = frozenset({301, 302, 303, 307, 308})

_FORBIDDEN_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "metadata.google.internal",
        "metadata.google.internal.",
    }
)


def _blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    if ip.is_loopback or ip.is_link_local or ip.is_private:
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        ula = ipaddress.IPv6Network("fc00::/7")
        if ip in ula:
            return True
        if ip.ipv4_mapped is not None and _blocked_ip(ip.ipv4_mapped):
            return True
    return False


def _literal_ip_from_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _resolve_host_ips(host: str, *, timeout_seconds: float = 2.0) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve *host* to IP addresses; raises on failure or timeout."""

    def _ga() -> list[tuple[int, int, int, str, tuple]]:
        return socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)

    fut = _RESOLVE_EXECUTOR.submit(_ga)
    try:
        infos = fut.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise socket.gaierror(f"resolve timeout for {host!r}") from exc

    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _fam, _type, _proto, _canon, sockaddr in infos:
        if not sockaddr:
            continue
        addr = sockaddr[0]
        try:
            ips.append(ipaddress.ip_address(addr))
        except ValueError:
            continue
    return ips


def is_safe_http_url_for_untrusted_fetch(url: str, *, resolve_timeout_seconds: float = 2.0) -> bool:
    """Return False if *url* must not be fetched (private IP, localhost, non-http(s), etc.)."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    if host.lower() in _FORBIDDEN_HOSTNAMES:
        return False

    literal = _literal_ip_from_host(host)
    if literal is not None:
        return not _blocked_ip(literal)

    try:
        ips = _resolve_host_ips(host, timeout_seconds=resolve_timeout_seconds)
    except OSError as exc:
        logger.debug("DNS resolution failed for %s: %s", host, exc)
        return False

    if not ips:
        return False

    for ip in ips:
        if _blocked_ip(ip):
            return False

    return True


def markdown_http_https_url_or_none(url: str | None) -> str | None:
    """If *url* is ``http`` or ``https``, return it stripped; otherwise None (no ``javascript:``, etc.)."""
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return None
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        return None
    return url.strip()


async def httpx_get_with_safe_redirects(
    client: httpx.AsyncClient,
    initial_url: str,
    *,
    max_hops: int = MAX_ENRICHMENT_REDIRECTS,
    resolve_timeout_seconds: float = 2.0,
) -> httpx.Response | None:
    """
    GET *initial_url* without automatic redirects; validate each hop.

    Returns the final non-redirect response, or None if a hop is unsafe or too many redirects.
    """
    current = initial_url.strip()
    for _hop in range(max_hops + 1):
        if not is_safe_http_url_for_untrusted_fetch(current, resolve_timeout_seconds=resolve_timeout_seconds):
            logger.debug("Blocked unsafe enrichment URL: %s", current)
            return None

        try:
            resp = await client.get(current, follow_redirects=False)
        except httpx.HTTPError as exc:
            logger.debug("HTTP error for %s: %s", current, exc)
            return None

        if resp.status_code in _REDIRECT_STATUS:
            loc = resp.headers.get("location")
            if not loc:
                return None
            current = urllib.parse.urljoin(str(resp.request.url), loc)
            continue

        return resp

    logger.debug("Too many redirects for enrichment starting at %s", initial_url)
    return None
