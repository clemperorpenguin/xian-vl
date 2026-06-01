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


import logging
import os

logger = logging.getLogger(__name__)


def should_warn_http_to_non_loopback(url: str) -> bool:
    """Suggest TLS in front of Lemonade when using HTTP to a non-loopback host."""
    if os.environ.get("XIAN_ALLOW_INSECURE_HTTP") == "1":
        return False
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "http":
        return False
    warn = not is_loopback_http_api_host(parsed.hostname)
    if warn:
        logger.warning(
            "SECURITY WARNING: Communicating with remote Lemonade host '%s' over cleartext HTTP. "
            "API keys and data can be intercepted or modified on the network path. "
            "Consider using HTTPS or setting XIAN_ALLOW_INSECURE_HTTP=1 to suppress this warning.",
            parsed.hostname
        )
    return warn
