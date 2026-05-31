import asyncio
from lore.scraper import PlaywrightScraper

async def main():
    s = PlaywrightScraper()
    res = await s.scrape("https://www.jx3box.com/community/6220")
    print("HTML Content length:", len(res.get("html", "")))
    print("HTML Content snippet:", res.get("html", "")[:500])

if __name__ == "__main__":
    asyncio.run(main())
