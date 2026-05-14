# Security notes

This document describes trust boundaries and the hardening added around **untrusted URLs**, **Lemonade HTTP endpoints**, and **sensitive data** (screen captures, page text, audio) sent to inference backends.

## Lemonade and transport (HTTP vs TLS)

**Lemonade does not provide HTTPS.** Defaults in this repository use **HTTP** to `localhost`, which is appropriate for a single-machine setup.

If you point clients (MAGE, MASHA, lore-cli, etc.) at Lemonade over a **non-loopback** host using **HTTP**, anyone who can observe or modify traffic on that path can read or tamper with requests and responses (including screenshots, translated text, and API payloads). Mitigations you control:

- Keep Lemonade bound to **loopback** when possible.
- For remote access, use a **VPN**, **SSH tunnel**, or a **reverse proxy with TLS** in front of Lemonade, then configure the client with an `https://…` base URL if your proxy presents HTTPS.

MAGE and MASHA show a **warning** when you save an **HTTP** URL whose host is **not** localhost / `127.0.0.1` / `::1`. You can still save after acknowledging the risk.

## Configurable backend URL

Treat the Lemonade base URL like a **sensitive capability**: whoever controls that URL receives whatever the client sends (images, batch text, metadata). Do not paste untrusted URLs into settings; verify the hostname before saving.

## Web search enrichment (MAGE / `xian` `WebSearcher`)

Search results can contain arbitrary links. **Enrichment** (fetching result pages to expand snippets) uses:

- **No automatic HTTP redirects** on the enrichment client.
- A **manual redirect loop** where **every hop** is checked with the same policy as the initial URL (scheme, DNS, and blocked address ranges such as loopback, RFC1918, link-local, CGNAT, and cloud metadata hostnames).

If any hop is disallowed, **that result is skipped for enrichment only**; the search itself still returns hits.

## LORE Playwright scraper

The scraper validates the **initial** URL the same way as enrichment. In addition, a **Playwright route handler** runs on **`**/*`**: every navigation and subresource request must pass the same safety check. Redirects and in-page fetches to private or loopback targets are **aborted**, so the browser should not complete a navigation chain to an unsafe host.

**Limitation:** Client-side redirects implemented purely in JavaScript (without a separate HTTP request Playwright classifies as a document request) are not modeled here; untrusted scraping targets remain risky for other reasons (drive-by, malicious content).

## Generated Markdown (wiki / LORE)

Links written into Markdown from web sources are restricted to **`http:`** and **`https:`** schemes. Other schemes (for example `javascript:`) are omitted from the clickable URL; the title is kept with a short note that the URL was omitted.

## Logging

User chat messages are logged at **INFO** with **length only**, not full content, to reduce accidental disclosure of secrets in log files.

## MASHA browser extension

The extension requests **broad host access** by design (`<all_urls>`) so it can translate pages you visit. Only configure a Lemonade URL you **fully trust**. The background script rejects image translation requests whose `imageUrl` uses a scheme other than **http**, **https**, **data**, or **blob**.

Message handlers also verify `sender.id === chrome.runtime.id` as a small defense-in-depth check.
