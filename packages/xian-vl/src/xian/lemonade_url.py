"""Normalize Lemonade OpenAI-compatible base URLs (MAGE, CLI, etc.)."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def normalize_lemonade_api_base_url(raw: str) -> str:
    """Return a base URL with exactly one ``/v1`` suffix, mirroring the MASHA extension."""
    trimmed = raw.strip().rstrip("/")
    if not trimmed:
        return "http://localhost:13305/v1"
    return trimmed if trimmed.endswith("/v1") else f"{trimmed}/v1"


def is_loopback_http_api_host(host: str | None) -> bool:
    """True if *host* is a typical local Lemonade binding (loopback only)."""
    if host is None:
        return False
    h = host.lower().strip("[]")
    if not h:
        return False
    if h == "localhost":
        return True
    if h in ("127.0.0.1", "::1", "0:0:0:0:0:0:0:1"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return bool(ip.is_loopback)


def should_warn_http_to_non_loopback(url: str) -> bool:
    """Suggest TLS in front of Lemonade when using HTTP to a non-loopback host."""
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "http":
        return False
    return not is_loopback_http_api_host(parsed.hostname)
