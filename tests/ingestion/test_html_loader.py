from app.ingestion.html_loader import (
    canonicalize_url,
    extract_main_text,
    extract_page_title,
    load_page,
)
from bs4 import BeautifulSoup


def test_canonicalize_url_strips_tracking_params():
    raw = (
        "https://rtb.ie/renting/tenant-rights/?"
        "_gl=1*abc&_ga=123#section"
    )
    assert canonicalize_url(raw) == "https://rtb.ie/renting/tenant-rights/"


def test_extract_citizensinformation_removes_breadcrumb_noise():
    html = """
    <html><head><title>PPS number - Citizens Information</title></head>
    <body><main role="main">
      <nav class="breadcrumb">You are here: Home > Social Welfare</nav>
      <h1>Personal Public Service (PPS) number</h1>
      <p>You need a PPS number to work or access public services in Ireland.</p>
      <p>Apply through your local Intreo centre with proof of identity.</p>
    </main></body></html>
    """
    text = extract_main_text(html, "https://www.citizensinformation.ie/en/example/")
    assert "You are here" not in text
    assert "PPS number" in text
    assert len(text) >= 150


def test_extract_leapcard_uses_maincontent():
    html = """
    <html><head><title>About TFI Leap Card - Leap Card</title></head>
    <body>
      <div class="sidebar-menu">Open Close page navigation About Fares</div>
      <div id="maincontent" class="content-c">
        <h1>About TFI Leap Card</h1>
        <p>The TFI Leap Card is a prepaid travel card for public transport in Ireland.</p>
        <p>It offers discounted fares on Dublin Bus, Luas, DART, and commuter services.</p>
      </div>
    </body></html>
    """
    text = extract_main_text(html, "https://about.leapcard.ie/about")
    assert "sidebar-menu" not in text.lower()
    assert "prepaid travel card" in text


def test_extract_page_title_prefers_document_title():
    html = "<html><head><title>Rent increases - Citizens Information</title></head><body><h1>Other</h1></body></html>"
    soup = BeautifulSoup(html, "lxml")
    assert extract_page_title(soup) == "Rent increases"


def test_load_page_returns_provenance_metadata():
    page = load_page(
        "https://www.citizensinformation.ie/en/health/health-overview/"
    )
    assert page.canonical_url.startswith("https://www.citizensinformation.ie/")
    assert page.title
    assert len(page.text) >= 150
    assert len(page.content_hash) == 64
    assert page.fetched_at.endswith("+00:00")
