# app/ingestion/html_loader.py
import httpx
from bs4 import BeautifulSoup


class HtmlLoadError(Exception):
    pass


# ✅ Pretend to be a real browser (reasonable, honest UA string)
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


def fetch_html(url: str, timeout: float = 20.0) -> str:
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
            return resp.text
    except Exception as e:
        raise HtmlLoadError(f"Failed to fetch {url}: {e}") from e


def extract_main_text(html: str) -> str:
    """
    Very simple HTML -> text cleaner.

    - Parses the HTML
    - Tries to use <main> or <article> if present, falls back to <body>
    - Removes nav/header/footer/script/style tags
    - Returns cleaned newline-separated text
    """
    soup = BeautifulSoup(html, "lxml")

    main = soup.find("main")
    if main is None:
        main = soup.find("article")
    if main is None:
        main = soup.body or soup

    # remove noisy elements
    for tag_name in ["nav", "header", "footer", "script", "style", "aside"]:
        for tag in main.find_all(tag_name):
            tag.decompose()

    # get text with newlines
    text = main.get_text("\n")

    # normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def load_and_clean(url: str) -> str:
    """
    High-level helper:
      - fetches HTML from URL
      - extracts main text
    """
    html = fetch_html(url)
    text = extract_main_text(html)
    return text
