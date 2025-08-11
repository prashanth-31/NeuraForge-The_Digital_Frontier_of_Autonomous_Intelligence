"""Simple, respectful web scraper with robots.txt check and HTML extraction."""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import httpx
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 12.0
DEFAULT_UA = "NeuraForgeBot/1.0 (+https://example.com/bot)"


class ScrapeError(Exception):
    pass


_robots_cache: Dict[str, robotparser.RobotFileParser] = {}


def _get_robots(domain: str) -> robotparser.RobotFileParser:
    if domain in _robots_cache:
        return _robots_cache[domain]
    rp = robotparser.RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots cannot be fetched, default to allowing
        rp = robotparser.RobotFileParser()
        rp.parse("")
    _robots_cache[domain] = rp
    return rp


def _allowed_by_robots(url: str, user_agent: str = DEFAULT_UA) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc
    if not domain:
        return False
    rp = _get_robots(domain)
    # Try exact UA, then generic '*'
    return rp.can_fetch(user_agent, url) and rp.can_fetch("*", url)


async def _fetch(url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[str, str, str]:
    """Return (final_url, content_type, text)"""
    used_headers = {"User-Agent": DEFAULT_UA, "Accept": "text/html,application/xhtml+xml"}
    if headers:
        used_headers.update(headers)
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, follow_redirects=True, headers=used_headers) as client:
        resp = await client.get(url)
        ct = resp.headers.get("content-type", "")
        # Only handle textual HTML here
        if resp.status_code >= 400:
            raise ScrapeError(f"GET {url} -> {resp.status_code}")
        text = resp.text
        return (str(resp.url), ct, text)


def _extract_text(html: str, selector: Optional[str] = None, max_chars: int = 4000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style/nav/footer
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "form", "aside"]):
        tag.decompose()

    node = None
    if selector:
        try:
            node = soup.select_one(selector)
        except Exception:
            node = None

    # Heuristics for main content
    if node is None:
        node = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup

    # Gather text from headings and paragraphs
    parts: List[str] = []
    for el in node.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
        if sum(len(p) for p in parts) > max_chars * 1.5:  # soft cap to stop early
            break
    text = "\n".join(parts)

    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text[:max_chars].rstrip()
    return text


def _extract_links(html: str, base_url: str, limit: int = 20) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(base_url, href)
        text = a.get_text(" ", strip=True)[:120]
        links.append(f"- {text} -> {full}")
        if len(links) >= limit:
            break
    return links


async def scrape_url(url: str, selector: Optional[str] = None, include_links: bool = False, max_chars: int = 4000) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ScrapeError("URL must start with http:// or https://")
    if not _allowed_by_robots(url):
        return "Blocked by robots.txt. Please provide a different URL."

    final_url, content_type, html = await _fetch(url)
    if "text/html" not in content_type:
        return f"The URL does not appear to be an HTML page (content-type: {content_type})."

    text = _extract_text(html, selector=selector, max_chars=max_chars)
    out = [f"Source: {final_url}", "", text]
    if include_links:
        links = _extract_links(html, base_url=final_url)
        if links:
            out.extend(["", "Links:", *links])
    return "\n".join(out)
