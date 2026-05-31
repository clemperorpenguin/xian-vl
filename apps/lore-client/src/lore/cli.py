import asyncio
import json
import logging
import os
import sys
import uuid
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import base64
from xian.omni_router import OmniModelRouter
from xian.lemonade_client import LemonadeClient

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.status import Status

from shared_types.constants import DEFAULT_API_URL, DEFAULT_MODEL

from openai import AsyncOpenAI

from xian.searcher import WebSearcher
from lore.scraper import PlaywrightScraper
from xian.compiler import WikiCompiler

app = typer.Typer(help="LORE — Interactive CLI tool for researching Chinese entities.")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("playwright").setLevel(logging.WARNING)


def _coerce_infobox_for_wiki(raw: object) -> dict[str, str]:
    """Normalize MASHA export metadata into flat string values for WikiCompiler."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in raw.items():
        sk = str(key).strip()
        if not sk:
            continue
        if isinstance(val, str):
            out[sk] = val
        elif val is None:
            out[sk] = ""
        else:
            try:
                out[sk] = json.dumps(val, ensure_ascii=False)
            except (TypeError, ValueError):
                out[sk] = str(val)
    return out


async def translate_content(text: str, entity_name: str) -> str:
    """Passes content to xian-vl via Lemonade for translation.
    """
    client = AsyncOpenAI(base_url=DEFAULT_API_URL, api_key="not-needed")
    
    system_prompt = (
        "You are an expert Chinese-to-English translator specializing in game lore, "
        "wuxia/xianxia terms, and Chinese encyclopedias (Baike). "
        "Translate the following content accurately and naturally. "
        "Preserve proper nouns. Use Obsidian-style links [[Entity]] for key concepts "
        "if they appear in the text. Output ONLY the translated content."
    )
    
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Entity: {entity_name}\n\nContent:\n{text}"},
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content or text
    except Exception as e:
        logger.error("Translation failed: %s", e)
        return text


async def extract_and_translate_infobox(raw_html: str, entity_name: str) -> dict:
    """Uses LLM or heuristics to extract and translate infobox data.
    """
    scraper = PlaywrightScraper()
    heuristic_infobox = scraper.extract_infobox(raw_html)
    
    if not heuristic_infobox:
        return {}
        
    # Translate the infobox keys and values
    client = AsyncOpenAI(base_url=DEFAULT_API_URL, api_key="not-needed")
    
    infobox_str = "\n".join([f"{k}: {v}" for k, v in heuristic_infobox.items()])
    
    system_prompt = (
        "You are an expert translator. Translate the following key-value pairs "
        "from a Chinese encyclopedia infobox into English. "
        "Return the result as a valid JSON object. "
        "Example: {\"Name\": \"Value\"}"
    )
    
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": infobox_str},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as e:
        logger.error("Infobox translation failed: %s", e)
        # Return untranslated or best effort
        return heuristic_infobox


async def research_entity(entity_name: str):
    """Main research loop: Search -> Select -> Ingest.
    """
    searcher = WebSearcher()
    try:
        scraper = PlaywrightScraper()
        wiki_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "wiki")
        compiler = WikiCompiler(wiki_dir=wiki_dir)

        console.print(f"[bold green]Starting research for entity:[/bold green] {entity_name}")

        # 1. Search
        with console.status(f"[bold blue]Searching for '{entity_name}'...[/bold blue]"):
            try:
                results = await searcher.search(entity_name, num_results=10)
            except Exception as e:
                console.print(f"[bold red]Search failed:[/bold red] {e}")
                return

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        # 3. Present Results
        table = Table(title=f"Search Results for '{entity_name}'")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Title", style="magenta")
        table.add_column("URL", style="green")

        for i, res in enumerate(results):
            table.add_row(str(i + 1), res.get("title", ""), res.get("url", ""))

        console.print(table)

        # 4. Selection
        selection = Prompt.ask(
            "Enter the ID to ingest, or 'all' to ingest top 3, or 'q' to quit",
            default="1"
        )

        if selection.lower() == 'q':
            return

        to_ingest = []
        if selection.lower() == 'all':
            to_ingest = results[:3]
        else:
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(results):
                    to_ingest = [results[idx]]
                else:
                    console.print("[red]Invalid selection.[/red]")
                    return
            except ValueError:
                console.print("[red]Invalid input.[/red]")
                return

        # 5. Ingest and Process
        all_content = []
        sources = []
        translated_infobox = {}

        for item in to_ingest:
            url = item.get("url")
            title = item.get("title")

            with console.status(f"[bold blue]Scraping {url}...[/bold blue]"):
                scraped_data = await scraper.scrape(url)

            if not scraped_data:
                console.print(f"[red]Failed to scrape {url}[/red]")
                continue

            console.print(f"[green]Successfully scraped {url}[/green]")

            # Extract and translate infobox from the first successful scrape (usually best source)
            if not translated_infobox and scraped_data.get("raw_html"):
                with console.status("[bold blue]Extracting and translating infobox...[/bold blue]"):
                    translated_infobox = await extract_and_translate_infobox(
                        scraped_data["raw_html"], entity_name
                    )

            # Translate main content
            with console.status(f"[bold blue]Translating content from {url}...[/bold blue]"):
                translated_text = await translate_content(scraped_data["content"], entity_name)

            all_content.append(translated_text)
            sources.append({"title": title, "url": url})

        if not all_content:
            console.print("[bold red]No content was successfully processed.[/bold red]")
            return

        # 6. Compile
        with console.status("[bold blue]Compiling Wiki file...[/bold blue]"):
            combined_content = "\n\n---\n\n".join(all_content)

            metadata = {
                "original_names": [entity_name],  # In real usage, could extract from infobox
                "sources": sources,
                "related_entities": []  # Could be filled by LLM extraction
            }

            filepath = compiler.compile(
                entity_name=entity_name,
                content=combined_content,
                metadata=metadata,
                infobox=translated_infobox
            )

        console.print(f"[bold green]Wiki compilation complete![/bold green] File saved to: {filepath}")

    finally:
        await searcher.close()


@app.command()
def research(entity: str = typer.Argument(..., help="The Chinese entity name to research")):
    """Research a Chinese entity and compile a localized English Wiki page."""
    asyncio.run(research_entity(entity))

@app.command()
def ingest(filepath: str = typer.Argument(..., help="Path to the JSON payload exported from MASHA")):
    """Ingest a site context exported from the MASHA extension into the LORE Wiki."""
    wiki_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "wiki")
    compiler = WikiCompiler(wiki_dir=wiki_dir)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Failed to read or parse file:[/bold red] {e}")
        return
        
    entity_name = data.get("title", "Imported_Entity").replace(" ", "_")
    content = data.get("translated_content", data.get("content", ""))
    metadata = {
        "url": data.get("url", ""),
        "context": data.get("context", ""),
        "sources": [{"title": data.get("title", "Imported Source"), "url": data.get("url", "")}]
    }
    
    with console.status(f"[bold blue]Ingesting '{entity_name}' from MASHA export...[/bold blue]"):
        out_path = compiler.compile(
            entity_name=entity_name,
            content=content,
            metadata=metadata,
            infobox=_coerce_infobox_for_wiki(data.get("metadata", {})),
        )
        
    console.print(f"[bold green]MASHA export successfully ingested![/bold green] File saved to: {out_path}")

async def download_images_and_replace(html: str, base_url: str, wiki_dir: str, router: OmniModelRouter, translate_images: bool) -> str:
    soup = BeautifulSoup(html, "lxml")
    images_dir = os.path.join(wiki_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    async with httpx.AsyncClient() as client:
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
                
            img_url = urljoin(base_url, src)
            try:
                if not img_url.startswith(("http://", "https://")):
                    continue
                    
                resp = await client.get(img_url, timeout=10.0)
                if resp.status_code == 200:
                    ext = os.path.splitext(urlparse(img_url).path)[1]
                    if not ext:
                        ext = ".jpg"
                    filename = f"{uuid.uuid4().hex}{ext}"
                    local_path = os.path.join(images_dir, filename)
                    
                    image_bytes = resp.content
                    
                    if translate_images:
                        try:
                            async with LemonadeClient() as l_client:
                                edit_resp = await l_client.edit_image(
                                    image_bytes=image_bytes,
                                    prompt="Translate all Chinese text in this image to English, keeping the original background and style.",
                                    model=router.edit(),
                                    response_format="b64_json"
                                )
                                data = edit_resp.get("data", [])
                                if data and "b64_json" in data[0]:
                                    image_bytes = base64.b64decode(data[0]["b64_json"])
                        except Exception as e:
                            logger.warning(f"Failed to translate image {img_url}: {e}")
                    
                    with open(local_path, "wb") as f:
                        f.write(image_bytes)
                        
                    img["src"] = f"images/{filename}"
            except Exception as e:
                logger.warning(f"Failed to download image {img_url}: {e}")
                
    return str(soup)

async def translate_html_to_markdown(html_content: str, title: str, router: OmniModelRouter) -> str:
    client = AsyncOpenAI(base_url=DEFAULT_API_URL, api_key="not-needed")
    
    system_prompt = (
        "You are an expert Chinese-to-English translator specializing in game lore, "
        "wuxia/xianxia terms, and Chinese encyclopedias. "
        "Your task is to translate the provided HTML content accurately into English, "
        "and format the output as clean Markdown. "
        "Preserve all images by converting them to Markdown image syntax `![alt text](src)`. "
        "Output ONLY the translated Markdown content."
    )
    
    try:
        response = await client.chat.completions.create(
            model=router.llm(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\\n\\nContent HTML:\\n{html_content}"},
            ],
            max_tokens=4096,
        )
        content = response.choices[0].message.content
        if not content:
            console.print("\\n[bold red]Error:[/bold red] The model returned an empty response.")
            return ""
        return content
    except Exception as e:
        console.print(f"\\n[bold red]Translation API Error:[/bold red] {e}")
        return ""

async def process_url(url: str, user_agent: str | None, translate_images: bool):
    scraper = PlaywrightScraper(user_agent=user_agent)
    wiki_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "wiki")
    compiler = WikiCompiler(wiki_dir=wiki_dir)
    
    router = OmniModelRouter(DEFAULT_API_URL)
    with console.status("[bold blue]Discovering Lemonade Models...[/bold blue]"):
        await router.discover_async()
    
    console.print(f"[bold green]Processing URL:[/bold green] {url}")
    
    with console.status("[bold blue]Scraping webpage...[/bold blue]"):
        scraped_data = await scraper.scrape(url)
        
    if not scraped_data:
        console.print("[bold red]Failed to scrape URL.[/bold red]")
        return
        
    html_content = scraped_data.get("html")
    title = scraped_data.get("title", "Untitled")
    
    if not html_content:
        console.print("[bold red]No HTML content extracted.[/bold red]")
        return
        
    with console.status("[bold blue]Downloading images...[/bold blue]"):
        local_html = await download_images_and_replace(html_content, url, wiki_dir, router, translate_images)
        
    with console.status("[bold blue]Translating and converting to Markdown...[/bold blue]"):
        translated_md = await translate_html_to_markdown(local_html, title, router)
        
    if not translated_md:
        console.print("[bold red]Failed to translate content.[/bold red]")
        return
        
    with console.status("[bold blue]Compiling Wiki file...[/bold blue]"):
        metadata = {
            "sources": [{"title": title, "url": url}],
        }
        
        filepath = compiler.compile(
            entity_name=title,
            content=translated_md,
            metadata=metadata,
            infobox={}
        )
        
    console.print(f"[bold green]Wiki compilation complete![/bold green] File saved to: {filepath}")

@app.command()
def url(
    target_url: str = typer.Argument(..., help="The URL to fetch, translate, and convert to Markdown"),
    user_agent: str = typer.Option(None, "--user-agent", "-a", help="Override the user agent used by the browser"),
    translate_images: bool = typer.Option(False, "--translate-images", help="Translate Chinese text within downloaded images using Lemonade API")
):
    """Fetch a URL, download its images, convert it to Markdown, and translate it to English."""
    asyncio.run(process_url(target_url, user_agent, translate_images))

if __name__ == "__main__":
    app()
