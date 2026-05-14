from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from readability import Document
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class PlaywrightScraper:
    """Scrapes content using Playwright and cleans it using Readability.
    """
    
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
    async def scrape(self, url: str) -> dict | None:
        """Fetches URL and returns cleaned content and metadata.
        """
        logger.info("Scraping URL: %s", url)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            logger.error("Refusing to scrape non-http(s) URL or missing host: %s", url)
            return None

        async with async_playwright() as p:
            # Launch headless browser
            browser = await p.chromium.launch(headless=True)
            
            context = await browser.new_context(
                user_agent=self.user_agent,
                extra_http_headers={
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Referer": "https://www.google.com/",
                }
            )
            
            page = await context.new_page()
            
            try:
                # Set a reasonable timeout
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Get the full HTML content
                html_content = await page.content()
                title = await page.title()
                
                logger.info("Successfully fetched content from %s", url)
                
                # Clean content using Readability
                doc = Document(html_content)
                summary_html = doc.summary()
                readable_title = doc.title()
                
                # Use BeautifulSoup to convert HTML summary to clean text or markdown-ready structure
                soup = BeautifulSoup(summary_html, "lxml")
                
                # Remove common noise if Readability missed it
                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()
                
                clean_text = soup.get_text(separator="\n", strip=True)
                
                # Also keep the HTML for structure if needed
                return {
                    "url": url,
                    "title": readable_title or title,
                    "content": clean_text,
                    "html": str(soup),
                    "raw_html": html_content
                }
                
            except Exception as e:
                logger.error("Failed to scrape %s: %s", url, e)
                return None
                
            finally:
                await browser.close()

    def extract_infobox(self, html_content: str) -> dict:
        """Heuristic to extract 'Baike-style' infobox data if present.
        Often found in Baidu Baike or interactive wikis.
        """
        soup = BeautifulSoup(html_content, "lxml")
        infobox = {}
        
        # Look for common Baidu Baike infobox structures
        # Class names like 'basic-info' or 'l-summary'
        base_info = soup.find(class_="basic-info")
        if base_info:
            names = base_info.find_all(class_="basicInfo-item name")
            values = base_info.find_all(class_="basicInfo-item value")
            
            for n, v in zip(names, values):
                key = n.get_text(strip=True).replace("\xa0", "")
                val = v.get_text(strip=True)
                infobox[key] = val
                
        return infobox
