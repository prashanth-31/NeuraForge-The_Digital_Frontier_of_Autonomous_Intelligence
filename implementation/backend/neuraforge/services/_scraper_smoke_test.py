from __future__ import annotations

import asyncio

from .scraper import scrape_url


async def main():
    # Simple public HTML page test
    url = "https://example.com/"
    print("Fetching:", url)
    text = await scrape_url(url, include_links=True, max_chars=800)
    print(text[:500])


if __name__ == "__main__":
    asyncio.run(main())
