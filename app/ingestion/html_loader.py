# app/ingestion/html_loader.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup, Tag


class HtmlLoadError(Exception):
    pass


MIN_EXTRACTED_CHARS = 150

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-IE,en;q=0.9",
    "Connection": "keep-alive",
}

NOISE_TAG_NAMES = [
    "nav",
    "header",
    "footer",
    "script",
    "style",
    "aside",
    "noscript",
    "iframe",
    "form",
]

NOISE_CLASS_FRAGMENTS = (
    "breadcrumb",
    "cookie-banner",
    "cookie-notice",
    "social-share",
    "sidebar-menu",
    "site-header",
    "site-footer",
    "page-footer",
    "main-nav",
    "site-nav",
    "navbar",
    "skip-link",
    "alert-info",
)

NOISE_ID_FRAGMENTS = (
    "sidebar",
    "cookie",
    "navbar",
)


@dataclass
class LoadedPage:
    url: str
    canonical_url: str
    title: str
    text: str
    content_hash: str
    fetched_at: str


def canonicalize_url(url: str) -> str:
    """Strip tracking params and fragments so doc_id stays stable."""
    parsed = urlparse(url.strip())
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/") + "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            "",
            "",
        )
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _remove_noise_tags(root: Tag) -> None:
    for tag_name in NOISE_TAG_NAMES:
        for tag in root.find_all(tag_name):
            tag.decompose()

    for tag in list(root.find_all(True)):
        if not isinstance(tag, Tag) or tag.attrs is None:
            continue
        classes = " ".join(tag.get("class", [])).lower()
        tag_id = (tag.get("id") or "").lower()
        if any(fragment in classes for fragment in NOISE_CLASS_FRAGMENTS):
            tag.decompose()
            continue
        if tag_id and any(fragment in tag_id for fragment in NOISE_ID_FRAGMENTS):
            tag.decompose()


def _normalize_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("you are here"):
            continue
        if cleaned.lower() in {"open", "close", "page navigation", "home"}:
            continue
        lines.append(cleaned)
    return "\n".join(lines)


def _first_matching(root: BeautifulSoup | Tag, selectors: list[str]) -> Optional[Tag]:
    for selector in selectors:
        match = root.select_one(selector)
        if match is not None:
            return match
    return None


def extract_page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"\s*-\s*(Citizens Information|Residential Tenancies Board|HSE\.ie|Leap Card).*$", "", title, flags=re.I)
        if title:
            return title

    heading = soup.find("h1")
    if heading:
        return heading.get_text(" ", strip=True)
    return "Untitled document"


def _extract_citizensinformation(soup: BeautifulSoup) -> str:
    root = _first_matching(soup, ["main[role='main']", "main", "[role='main']", ".content"])
    if root is None:
        root = soup.body or soup
    _remove_noise_tags(root)
    return root.get_text("\n")


def _extract_rtb(soup: BeautifulSoup) -> str:
    root = _first_matching(soup, ["main", "#main-content", ".content"])
    if root is None:
        root = soup.body or soup
    _remove_noise_tags(root)
    return root.get_text("\n")


def _extract_hse(soup: BeautifulSoup) -> str:
    root = _first_matching(soup, ["main", "article", "[role='main']"])
    if root is None:
        root = soup.body or soup
    _remove_noise_tags(root)
    return root.get_text("\n")


def _extract_leapcard(soup: BeautifulSoup) -> str:
    root = _first_matching(soup, ["#maincontent", ".content-c", "main"])
    if root is None:
        root = soup.body or soup
    _remove_noise_tags(root)
    return root.get_text("\n")


def _extract_garda(soup: BeautifulSoup) -> str:
    root = _first_matching(soup, [".page-content", ".main-content", "main"])
    if root is None:
        root = soup.body or soup
    _remove_noise_tags(root)
    return root.get_text("\n")


def _extract_default(soup: BeautifulSoup) -> str:
    root = _first_matching(soup, ["main", "article", "[role='main']", "#main-content", ".content"])
    if root is None:
        root = soup.body or soup
    _remove_noise_tags(root)
    return root.get_text("\n")


def _extractor_for_url(url: str) -> Callable[[BeautifulSoup], str]:
    host = urlparse(url).netloc.lower()
    if host.endswith("citizensinformation.ie"):
        return _extract_citizensinformation
    if host.endswith("rtb.ie"):
        return _extract_rtb
    if host.endswith("hse.ie"):
        return _extract_hse
    if host.endswith("leapcard.ie"):
        return _extract_leapcard
    if host.endswith("garda.ie"):
        return _extract_garda
    return _extract_default


def extract_main_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    extractor = _extractor_for_url(url)
    text = _normalize_text(extractor(soup))
    if len(text) < MIN_EXTRACTED_CHARS:
        raise HtmlLoadError(
            f"Extracted only {len(text)} characters from {url}; "
            f"expected at least {MIN_EXTRACTED_CHARS}."
        )
    return text


def fetch_html(url: str, timeout: float = 30.0) -> str:
    """
    Fetch raw HTML from a URL using httpx, with browser-like headers.

    Raises HtmlLoadError if the request fails.
    """
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=DEFAULT_HEADERS,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if not resp.text.strip():
                raise HtmlLoadError(f"Empty response body from {url}")
            return resp.text
    except HtmlLoadError:
        raise
    except Exception as e:
        raise HtmlLoadError(f"Failed to fetch {url}: {e}") from e


def load_page(url: str) -> LoadedPage:
    """
    Fetch a page and return cleaned text plus provenance metadata.
    """
    canonical_url = canonicalize_url(url)
    html = fetch_html(canonical_url)
    soup = BeautifulSoup(html, "lxml")
    title = extract_page_title(soup)
    text = extract_main_text(html, canonical_url)
    fetched_at = datetime.now(timezone.utc).isoformat()

    return LoadedPage(
        url=url,
        canonical_url=canonical_url,
        title=title,
        text=text,
        content_hash=_content_hash(text),
        fetched_at=fetched_at,
    )


def load_and_clean(url: str) -> str:
    """Backward-compatible helper that returns cleaned page text only."""
    return load_page(url).text
