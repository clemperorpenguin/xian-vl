from __future__ import annotations

import asyncio
import glob
import logging
import os
import re
import urllib.parse

import httpx

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class DuckDuckGoSearcher:
    """Primary web searcher using DuckDuckGo Lite (HTML).

    DuckDuckGo Lite is a lightweight HTML interface that does not
    require an API key and is far more tolerant of automated queries
    than SearXNG public instances, which aggressively rate-limit or
    block ``format=json`` requests.
    """

    DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": _BROWSER_UA},
            follow_redirects=True,
        )

    async def search(self, query: str, num_results: int = 10, language: str = "zh-CN") -> list[dict]:
        """Perform a search via DuckDuckGo Lite and parse HTML results."""
        # Map language codes to DDG region codes
        region_map = {
            "zh": "cn-zh", "zh-CN": "cn-zh", "zh-TW": "tw-tzh", "Chinese": "cn-zh",
            "ja": "jp-jp", "ko": "kr-kr", "Japanese": "jp-jp", "Korean": "kr-kr",
            "en": "us-en", "en-US": "us-en", "English": "us-en",
            "ru": "ru-ru", "Russian": "ru-ru",
            "hi": "in-hi", "Hindi": "in-hi",
            "bn": "in-bn", "Bengali": "in-bn",
            "tr": "tr-tr", "Turkish": "tr-tr",
            "ar": "xa-ar", "Arabic": "xa-ar",
            "es": "es-es", "Spanish": "es-es",
            "fr": "fr-fr", "French": "fr-fr",
            "pt": "pt-pt", "Portuguese": "pt-pt",
            "vi": "vn-vi", "Vietnamese": "vn-vi",
        }
        kl = region_map.get(language, "")

        data = {"q": query}
        if kl:
            data["kl"] = kl

        try:
            response = await self.client.post(self.DDG_LITE_URL, data=data)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("DuckDuckGo Lite request failed: %s", exc)
            return []

        return self._parse_results(response.text, num_results)

    # ------------------------------------------------------------------
    # HTML parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(html: str, limit: int) -> list[dict]:
        """Extract titles, URLs, and snippets from DDG Lite HTML."""
        # Link format:  <a … href="URL" class='result-link'>TITLE</a>
        link_pattern = re.compile(
            r"""href=["']([^"']+)["']\s+class=["']result-link["'][^>]*>(.*?)</a>""",
            re.DOTALL,
        )
        # Snippet format: <td class='result-snippet'>…</td>
        snippet_pattern = re.compile(
            r"""class=["']result-snippet["'][^>]*>(.*?)</td>""",
            re.DOTALL,
        )

        raw_links = link_pattern.findall(html)
        raw_snippets = snippet_pattern.findall(html)

        results: list[dict] = []
        for i, (raw_href, raw_title) in enumerate(raw_links):
            if i >= limit:
                break

            title = re.sub(r"<[^>]+>", "", raw_title).strip()
            snippet = re.sub(r"<[^>]+>", "", raw_snippets[i]).strip() if i < len(raw_snippets) else ""

            # Resolve DDG redirect URLs (//duckduckgo.com/l/?uddg=…)
            parsed = urllib.parse.urlparse(raw_href)
            qs = urllib.parse.parse_qs(parsed.query)
            url = qs.get("uddg", [raw_href])[0]

            results.append({"title": title, "content": snippet, "url": url})

        logger.info("DuckDuckGo Lite returned %d results.", len(results))
        return results

    async def close(self):
        await self.client.aclose()


class SearXNGSearcher:
    """Fallback searcher using public SearXNG instances.

    Most public instances aggressively rate-limit ``format=json`` requests,
    so this class is kept as a secondary option behind :class:`DuckDuckGoSearcher`.
    """

    def __init__(self):
        self.instances: list[str] = []
        self.current_index = 0
        self.client = httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": _BROWSER_UA},
        )

    async def discover_instances(self):
        """Fetches public instances from searx.space and filters for high performance."""
        logger.info("Fetching SearXNG instances from searx.space...")
        try:
            response = await self.client.get("https://searx.space/data/instances.json")
            response.raise_for_status()
            data = response.json()

            instances_data = data.get("instances", {})
            valid_instances = []

            for url, info in instances_data.items():
                # Filter for instances that are public, have no error, and are online
                if info.get("network_type") not in ("normal", "public"):
                    continue
                if info.get("error") is not None:
                    continue

                # Check timing if available (prefer faster instances)
                timing = info.get("timing", {}).get("initial", {}).get("all", {}).get("value", 999)
                if timing > 2.0:  # Skip slow instances (> 2 seconds)
                    continue

                valid_instances.append((url, timing))

            # Sort by timing (fastest first)
            valid_instances.sort(key=lambda x: x[1])
            self.instances = [url for url, _ in valid_instances]

            # If list is empty, fallback to some known defaults
            if not self.instances:
                logger.warning("No high-performance instances found on searx.space. Using defaults.")
                self.instances = [
                    "https://searx.be/",
                    "https://searxng.site/",
                    "https://paulgo.io/",
                ]

            # Shuffle the top entries to distribute load
            top_n = min(10, len(self.instances))
            top_instances = self.instances[:top_n]
            import random
            random.shuffle(top_instances)
            self.instances[:top_n] = top_instances

            logger.info("Discovered %d valid SearXNG instances.", len(self.instances))

        except Exception as e:
            logger.error("Failed to discover instances: %s. Using fallback defaults.", e)
            self.instances = [
                "https://searx.be/",
                "https://searxng.site/",
                "https://paulgo.io/",
            ]

    def _get_next_instance(self) -> str:
        if not self.instances:
            raise RuntimeError("No SearXNG instances available.")

        url = self.instances[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.instances)
        return url

    async def search(self, query: str, num_results: int = 10, language: str = "zh-CN") -> list[dict]:
        """Performs a search using rotating instances."""
        if not self.instances:
            await self.discover_instances()

        attempts = 0
        max_attempts = min(20, len(self.instances))

        while attempts < max_attempts:
            instance_url = self._get_next_instance()
            logger.info("Attempting search on %s (Attempt %d/%d)", instance_url, attempts + 1, max_attempts)

            try:
                search_url = instance_url.rstrip("/") + "/search"
                params = {
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "language": language,
                }

                response = await self.client.get(search_url, params=params, timeout=5.0)

                if response.status_code == 429:
                    logger.warning("Rate limited by %s, switching...", instance_url)
                    attempts += 1
                    continue

                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                logger.info("Successfully retrieved %d results from %s", len(results), instance_url)
                return results[:num_results]

            except (httpx.HTTPError, ValueError) as e:
                logger.warning("Failed to search on %s: %s. Trying next instance.", instance_url, e)
                attempts += 1
                await asyncio.sleep(0.5)  # Short pause before retry

        raise RuntimeError("All attempted SearXNG instances failed.")

    async def close(self):
        await self.client.aclose()


class WebSearcher:
    """Unified web search with automatic fallback and content enrichment.

    Tries :class:`DuckDuckGoSearcher` first (reliable, no rate-limits).
    Falls back to :class:`SearXNGSearcher` if DDG returns zero results.
    After obtaining results, fetches actual page content from the top URLs
    using ``readability-lxml`` to replace short snippets with real content.
    """

    # Skip URLs that won't yield useful text content
    _SKIP_DOMAINS = {"youtube.com", "youtu.be", "twitter.com", "x.com", "reddit.com"}

    def __init__(self):
        self._ddg = DuckDuckGoSearcher()
        self._searxng = SearXNGSearcher()
        self._client = httpx.AsyncClient(
            timeout=8.0,
            headers={"User-Agent": _BROWSER_UA},
            follow_redirects=True,
        )

    async def search(self, query: str, num_results: int = 10, language: str = "zh-CN",
                     enrich: bool = True, enrich_top_n: int = 2) -> list[dict]:
        """Search the web, trying DDG first then SearXNG.

        If *enrich* is True, the top *enrich_top_n* results will have their
        snippet replaced with actual page content fetched via HTTP.
        """
        results = await self._ddg.search(query, num_results=num_results, language=language)
        if not results:
            logger.info("DuckDuckGo returned no results; falling back to SearXNG.")
            try:
                results = await self._searxng.search(query, num_results=num_results, language=language)
            except RuntimeError:
                logger.warning("SearXNG fallback also failed.")
                return []

        if enrich and results:
            # Prioritize results likely to have detailed content for enrichment
            self._prioritize_for_enrichment(results)
            await self._enrich_results(results, enrich_top_n)

        return results

    async def dual_search(
        self,
        query: str,
        target_lang: str,
        translated_query: str,
        source_lang: str,
        num_results: int = 10,
        enrich_top_n: int = 3,
    ) -> list[dict]:
        """Run two searches concurrently and merge the results.

        Fires one search with *query* in *target_lang* and another with
        *translated_query* in *source_lang*.  Results are deduplicated by
        URL, prioritized for enrichment, and the top *enrich_top_n* are
        fetched for full-page content.
        """
        target_results, source_results = await asyncio.gather(
            self.search(query, num_results=num_results, language=target_lang, enrich=False),
            self.search(translated_query, num_results=num_results, language=source_lang, enrich=False),
        )

        logger.info(
            "Dual search: %d target-lang results, %d source-lang results",
            len(target_results), len(source_results),
        )

        # Deduplicate by URL
        seen_urls: set[str] = set()
        merged: list[dict] = []
        for r in target_results + source_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(r)

        # Prioritize and enrich the combined list
        self._prioritize_for_enrichment(merged)
        await self._enrich_results(merged, enrich_top_n)

        logger.info("Dual search: %d merged results after dedup.", len(merged))
        return merged

    @staticmethod
    def _prioritize_for_enrichment(results: list[dict]) -> None:
        """Re-order results so wiki/info pages come first for enrichment.

        This ensures the enrichment budget is spent on pages that contain
        detailed articles (Wikipedia, Fandom, Baidu Baike, etc.) rather
        than shallow landing pages or news feeds.
        """
        priority_domains = (
            "wikipedia.org", "fandom.com", "baike.baidu.com",
            "mmotop.com", "mmorpg.com", "wtfast.com",
        )

        def _sort_key(r: dict) -> int:
            url = r.get("url", "").lower()
            for i, domain in enumerate(priority_domains):
                if domain in url:
                    return i
            return len(priority_domains)

        results.sort(key=_sort_key)

    async def _enrich_results(self, results: list[dict], top_n: int) -> None:
        """Fetch actual page content for the top N results.

        Replaces the short DDG snippet with a truncated extract of the real
        page body (up to ~1500 chars) using readability-lxml.
        """
        from readability import Document as ReadabilityDocument
        from bs4 import BeautifulSoup

        enriched = 0
        for result in results:
            if enriched >= top_n:
                break

            url = result.get("url", "")
            if not url:
                continue
            parsed_url = urllib.parse.urlparse(url)
            if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                logger.debug("Skipping enrich for non-http(s) URL: %s", url)
                continue

            # Skip domains that won't have useful text
            domain = parsed_url.netloc.lower()
            if any(skip in domain for skip in self._SKIP_DOMAINS):
                continue

            try:
                resp = await self._client.get(url)
                if resp.status_code != 200:
                    continue

                doc = ReadabilityDocument(resp.text)
                soup = BeautifulSoup(doc.summary(), "lxml")
                text = soup.get_text(separator="\n", strip=True)

                # Clean up reference markers like [1], [2] etc.
                text = re.sub(r"\[\s*\d+\s*\]", "", text)

                if len(text) > 100:  # Only replace if we got meaningful content
                    # Truncate to ~1500 chars at a sentence boundary
                    if len(text) > 1500:
                        cut = text[:1500].rfind(".")
                        if cut > 500:
                            text = text[:cut + 1]
                        else:
                            text = text[:1500] + "…"

                    result["content"] = text
                    enriched += 1
                    logger.info("Enriched result '%s' with %d chars from %s",
                                result.get("title", "")[:40], len(text), url)

            except Exception as exc:
                logger.debug("Failed to enrich %s: %s", url, exc)

    async def close(self):
        await self._ddg.close()
        await self._searxng.close()
        await self._client.aclose()


class LocalWikiSearcher:
    """Searches the local LORE wiki for persistent knowledge."""

    def __init__(self, wiki_dir: str = "wiki"):
        self.wiki_dir = wiki_dir

    def search(self, query: str, num_results: int = 5) -> list[dict]:
        """Performs a simple keyword search over local wiki markdown files."""
        if not os.path.exists(self.wiki_dir):
            return []

        results = []
        query_terms = query.lower().split()
        if not query_terms:
            return []

        # Simple scoring based on term frequency
        for filepath in glob.glob(os.path.join(self.wiki_dir, "*.md")):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                content_lower = content.lower()
                score = sum(content_lower.count(term) for term in query_terms)

                if score > 0:
                    filename = os.path.basename(filepath)
                    title = filename.replace(".md", "")

                    # Extract a snippet around the first match
                    first_match_idx = -1
                    for term in query_terms:
                        idx = content_lower.find(term)
                        if idx != -1 and (first_match_idx == -1 or idx < first_match_idx):
                            first_match_idx = idx

                    snippet_start = max(0, first_match_idx - 100)
                    snippet_end = min(len(content), first_match_idx + 200)
                    snippet = content[snippet_start:snippet_end].strip()
                    if snippet_start > 0:
                        snippet = "..." + snippet
                    if snippet_end < len(content):
                        snippet = snippet + "..."

                    results.append({
                        "title": f"[LOCAL WIKI] {title}",
                        "content": snippet,
                        "url": f"file://{os.path.abspath(filepath)}",
                        "score": score
                    })
            except Exception as e:
                logger.error("Error reading wiki file %s: %s", filepath, e)

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:num_results]
