# app/ingestion/legal_discovery.py
"""
DEV-TIME discovery utility (NOT part of runtime ingestion).

Crawls trusted Irish public-information section hubs to discover the real,
working article URLs within each topic, verifies each one loads through the
production `html_loader`, and emits verified entries that are baked into
`legal_sources.py` as a curated static list.

This keeps runtime ingestion simple (an explicit, reviewable list) while
avoiding hand-guessed URLs that 404/403.

Run:
    python3 -m app.ingestion.legal_discovery
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from app.ingestion.html_loader import (
    HtmlLoadError,
    canonicalize_url,
    fetch_html,
    load_page,
)

MIN_ARTICLE_CHARS = 700
CRAWL_DELAY_SECONDS = 0.3

CACHE_PATH = "app/ingestion/_discovered_sources.json"
SOURCES_PATH = "app/ingestion/legal_sources.py"


@dataclass
class Seed:
    hub_url: str
    topic: str
    provider: str
    authority_tier: str
    path_prefix: str


# Newcomer-essential pages that answer the concrete "arriving as a student"
# journey (border, registration/IRP, Stamp 2 work rules, PPSN, bank account,
# health insurance, address proof, renewal). These are added explicitly so the
# most important pages are guaranteed present regardless of crawl link changes.
#
# NOTE: The official source for IRP/Stamp 2 is irishimmigration.ie (ISD), but it
# sits behind a Cloudflare JS challenge and cannot be fetched over plain HTTP.
# Citizens Information is the official government information service and mirrors
# ISD rules, so it is used as the authoritative-accessible proxy here.
CURATED_EXTRA: List[dict] = [
    # Border / permission to enter
    {"url": "https://www.citizensinformation.ie/en/moving-country/visas-for-ireland/permission-to-land-in-ireland/",
     "topic": "immigration", "subtopic": "permission_to_enter", "authority_tier": "guidance"},
    # Study permission + Stamp 2 rules (work hours, registration, renewal)
    {"url": "https://www.citizensinformation.ie/en/moving-country/visas-for-ireland/student-visas/",
     "topic": "immigration", "subtopic": "student_visas", "authority_tier": "guidance"},
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/studying-in-ireland/",
     "topic": "immigration", "subtopic": "studying_in_ireland", "authority_tier": "guidance"},
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/studying-in-ireland/immigration-nonEEA-students/",
     "topic": "immigration", "subtopic": "immigration_rules_nonEEA_students", "authority_tier": "guidance"},
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/studying-in-ireland/third-level-education/",
     "topic": "immigration", "subtopic": "third_level_education", "authority_tier": "guidance"},
    # Registration / IRP / residence permission
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/rights-of-residence-in-ireland/registration-of-non-eea-nationals-in-ireland/",
     "topic": "immigration", "subtopic": "registration_of_non_eea_nationals", "authority_tier": "guidance"},
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/rights-of-residence-in-ireland/types-residence-permission-non-eea-nationals/",
     "topic": "immigration", "subtopic": "types_of_residence_permission", "authority_tier": "guidance"},
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/rights-of-residence-in-ireland/residence-rights-of-non-eea-nationals-in-ireland/",
     "topic": "immigration", "subtopic": "residence_rights_non_eea", "authority_tier": "guidance"},
    # Settling in
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/coming-to-live-in-ireland/",
     "topic": "immigration", "subtopic": "coming_to_live_in_ireland", "authority_tier": "guidance"},
    {"url": "https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/coming-to-live-in-ireland/support-services-for-foreign-nationals/",
     "topic": "immigration", "subtopic": "support_services_foreign_nationals", "authority_tier": "guidance"},
    # PPSN
    {"url": "https://www.citizensinformation.ie/en/social-welfare/irish-social-welfare-system/personal-public-service-number/",
     "topic": "social_welfare", "subtopic": "pps_number", "authority_tier": "guidance"},
    # Bank account
    {"url": "https://www.citizensinformation.ie/en/money-and-tax/personal-finance/banking/opening-a-bank-account/",
     "topic": "banking", "subtopic": "opening_a_bank_account", "authority_tier": "guidance"},
    # Private medical insurance (required for student visa/IRP)
    {"url": "https://www.citizensinformation.ie/en/health/health-system/private-health-insurance/",
     "topic": "health", "subtopic": "private_health_insurance", "authority_tier": "guidance"},
]


# Curated to the sub-sections most relevant to migrants, students and workers
# in Ireland. Two levels deep within each prefix.
SEEDS: List[Seed] = [
    # Immigration
    Seed("https://www.citizensinformation.ie/en/moving-country/visas-for-ireland/",
         "immigration", "citizensinformation", "guidance",
         "/en/moving-country/visas-for-ireland/"),
    Seed("https://www.citizensinformation.ie/en/moving-country/moving-to-ireland/",
         "immigration", "citizensinformation", "guidance",
         "/en/moving-country/moving-to-ireland/"),
    # Employment
    Seed("https://www.citizensinformation.ie/en/employment/employment-rights-and-conditions/",
         "employment", "citizensinformation", "guidance",
         "/en/employment/employment-rights-and-conditions/"),
    Seed("https://www.citizensinformation.ie/en/employment/starting-work-and-changing-job/",
         "employment", "citizensinformation", "guidance",
         "/en/employment/starting-work-and-changing-job/"),
    Seed("https://www.citizensinformation.ie/en/employment/types-of-employment/",
         "employment", "citizensinformation", "guidance",
         "/en/employment/types-of-employment/"),
    # Tax
    Seed("https://www.citizensinformation.ie/en/money-and-tax/tax/",
         "tax", "citizensinformation", "guidance",
         "/en/money-and-tax/tax/"),
    # Social welfare
    Seed("https://www.citizensinformation.ie/en/social-welfare/irish-social-welfare-system/",
         "social_welfare", "citizensinformation", "guidance",
         "/en/social-welfare/irish-social-welfare-system/"),
    # Transport
    Seed("https://www.citizensinformation.ie/en/travel-and-recreation/motoring/driver-licensing/",
         "transport", "citizensinformation", "guidance",
         "/en/travel-and-recreation/motoring/driver-licensing/"),
    # Health
    Seed("https://www.citizensinformation.ie/en/health/medical-cards-and-gp-visit-cards/",
         "health", "citizensinformation", "guidance",
         "/en/health/medical-cards-and-gp-visit-cards/"),
    Seed("https://www.citizensinformation.ie/en/health/entitlement-to-health-services/",
         "health", "citizensinformation", "guidance",
         "/en/health/entitlement-to-health-services/"),
    # Housing
    Seed("https://www.citizensinformation.ie/en/housing/renting-a-home/",
         "housing", "citizensinformation", "guidance",
         "/en/housing/renting-a-home/"),
    # Providers (non-CI, official/operational)
    Seed("https://rtb.ie/renting/",
         "housing", "rtb", "official", "/renting/"),
    Seed("https://www2.hse.ie/services/",
         "health", "hse", "official", "/services/"),
    Seed("https://vetting.garda.ie/",
         "vetting", "garda", "official", "/"),
    Seed("https://about.leapcard.ie/",
         "transport", "leapcard", "operational", "/"),
]

# Path fragments we never want to ingest.
EXCLUDE_FRAGMENTS = (
    "/ga/",
    "/about/",
    "/cookie",
    "/privacy",
    "/disclaimer",
    "/feedback",
    "/accessibility",
    "/search",
    "/contact",
    "login",
    "sitemap",
    ".pdf",
)


def _same_host(url: str, host: str) -> bool:
    return urlparse(url).netloc.lower().endswith(host)


def _discover_links(hub_url: str, path_prefix: str) -> List[str]:
    """Return same-domain links under the given path prefix."""
    host = urlparse(hub_url).netloc.lower()
    try:
        html = fetch_html(canonicalize_url(hub_url))
    except HtmlLoadError:
        return []

    soup = BeautifulSoup(html, "lxml")
    found: Set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(hub_url, href)
        canonical = canonicalize_url(absolute)
        if not _same_host(canonical, host):
            continue
        path = urlparse(canonical).path
        if not path.startswith(path_prefix):
            continue
        if any(fragment in canonical.lower() for fragment in EXCLUDE_FRAGMENTS):
            continue
        found.add(canonical)
    return sorted(found)


def _slug_to_subtopic(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    slug = path.split("/")[-1] if path else "overview"
    return slug.replace("-", "_") or "overview"


def discover() -> List[dict]:
    verified: Dict[str, dict] = {}

    for seed in SEEDS:
        print(f"\n[SEED] {seed.topic} :: {seed.hub_url}")
        # Level 1: direct children of the hub
        level1 = _discover_links(seed.hub_url, seed.path_prefix)
        # Level 2: children of each level-1 page
        level2: Set[str] = set()
        for child in level1:
            level2.update(_discover_links(child, seed.path_prefix))
            time.sleep(CRAWL_DELAY_SECONDS)

        candidates = {seed.hub_url, *level1, *level2}
        print(f"  discovered {len(candidates)} candidate URLs")

        for url in sorted(candidates):
            canonical = canonicalize_url(url)
            if canonical in verified:
                continue
            try:
                page = load_page(canonical)
            except HtmlLoadError:
                continue
            if len(page.text) < MIN_ARTICLE_CHARS:
                continue
            verified[canonical] = {
                "url": canonical,
                "provider": seed.provider,
                "topic": seed.topic,
                "subtopic": _slug_to_subtopic(canonical),
                "title": page.title,
                "authority_tier": seed.authority_tier,
                "chars": len(page.text),
            }
            time.sleep(CRAWL_DELAY_SECONDS)

        print(f"  verified running total: {len(verified)}")

    return sorted(verified.values(), key=lambda r: (r["topic"], r["url"]))


# ---------------------------------------------------------------------------
# Curation: drop location/branch/duplicate noise so the stored list stays
# focused on substantive guidance rather than directory listings.
# ---------------------------------------------------------------------------

# HSE: keep informational service/scheme pages, drop individual branches.
_HSE_DROP_SUBSTRINGS = (
    "/activity-performance-data/",
    "dashboard",
    "/civil-registration-service-",
    "urgent-emergency-care-report",
    "urgent-emergency-care-weekly-update",
)
_HSE_DROP_INSTANCE_PREFIXES = (
    "/services/hospitals/",
    "/services/primary-care-centres/",
    "/services/find-urgent-emergency-care/",
)

# Leap Card: operational site with many per-city/image/duplicate pages.
# Curate to genuinely informational pages only.
_LEAPCARD_KEEP_SLUGS = {
    "about",
    "faqs",
    "fare-capping",
    "tfi-90-minute-fare",
    "how-to-renew-your-card",
    "card-replacement-refunds",
    "autotop-up",
    "leap-visitor-card",
    "young-adult-and-student-card-launch",
    "top-tips",
    "accepted-non-standard-documents",
    "dart",
    "dublin-bus",
    "luas",
    "bus-eireann",
    "private-operators",
    "student",
}


def _keep_row(row: dict) -> bool:
    provider = row["provider"]
    path = urlparse(row["url"]).path.rstrip("/")

    if provider == "hse":
        lower = row["url"].lower()
        if any(fragment in lower for fragment in _HSE_DROP_SUBSTRINGS):
            return False
        for prefix in _HSE_DROP_INSTANCE_PREFIXES:
            if path.startswith(prefix.rstrip("/")) and path != prefix.rstrip("/"):
                return False
        return True

    if provider == "leapcard":
        if "wp-content" in row["url"] or row["url"].lower().endswith(".png"):
            return False
        slug = path.split("/")[-1]
        return slug in _LEAPCARD_KEEP_SLUGS

    # citizensinformation, rtb, garda: keep all verified pages.
    return True


def _camel_to_words(text: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip()


def _garda_title(url: str) -> str:
    segments = [s for s in urlparse(url).path.split("/") if s]
    if not segments:
        return "Garda Vetting"
    readable = ": ".join(_camel_to_words(seg) for seg in segments[-2:])
    return f"Garda Vetting — {readable}" if readable else "Garda Vetting"


def _garda_subtopic(url: str) -> str:
    segments = [s for s in urlparse(url).path.split("/") if s]
    if not segments:
        return "vetting_overview"
    return re.sub(r"[^a-z0-9]+", "_", _camel_to_words(segments[-1]).lower()).strip("_")


def _force_https(url: str) -> str:
    return url.replace("http://", "https://", 1)


def curate(rows: List[dict]) -> List[dict]:
    seen: Set[str] = set()
    curated: List[dict] = []
    for row in rows:
        row = dict(row)
        row["url"] = _force_https(row["url"])
        if row["url"] in seen or not _keep_row(row):
            continue
        seen.add(row["url"])
        if row["provider"] == "garda":
            row["title"] = _garda_title(row["url"])
            row["subtopic"] = _garda_subtopic(row["url"])
        curated.append(row)
    return sorted(curated, key=lambda r: (r["topic"], r["provider"], r["url"]))


def _py_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_sources_file(rows: List[dict]) -> None:
    lines: List[str] = [
        "from dataclasses import dataclass",
        "from typing import List, Literal",
        "",
        'AuthorityTier = Literal["primary", "official", "guidance", "operational"]',
        "",
        "",
        "@dataclass",
        "class LegalSource:",
        "    url: str",
        "    provider: str",
        "    topic: str",
        "    subtopic: str",
        "    title: str",
        "    authority_tier: AuthorityTier",
        '    description: str = ""',
        "",
        "",
        "# This list is generated by app/ingestion/legal_discovery.py from a",
        "# curated crawl of trusted Irish public-information domains. Every URL is",
        "# verified to load with substantial content. Re-run the discovery tool to",
        "# refresh it; do not hand-edit individual entries.",
        "LEGAL_SOURCES: List[LegalSource] = [",
    ]

    current_topic = None
    for row in rows:
        if row["topic"] != current_topic:
            current_topic = row["topic"]
            lines.append(f"    # ===== {current_topic.upper()} =====")
        lines.append("    LegalSource(")
        lines.append(f"        url={_py_str(row['url'])},")
        lines.append(f"        provider={_py_str(row['provider'])},")
        lines.append(f"        topic={_py_str(row['topic'])},")
        lines.append(f"        subtopic={_py_str(row['subtopic'])},")
        lines.append(f"        title={_py_str(row['title'])},")
        lines.append(f"        authority_tier={_py_str(row['authority_tier'])},")
        lines.append(f"        description={_py_str(row['title'])},")
        lines.append("    ),")
    lines.append("]")
    lines.append("")

    with open(SOURCES_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _provider_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.endswith("citizensinformation.ie"):
        return "citizensinformation"
    if host.endswith("rtb.ie"):
        return "rtb"
    if host.endswith("hse.ie"):
        return "hse"
    if host.endswith("leapcard.ie"):
        return "leapcard"
    if host.endswith("garda.ie"):
        return "garda"
    return host


def load_curated_extras() -> List[dict]:
    """Fetch + verify the explicit newcomer-essential pages."""
    rows: List[dict] = []
    for entry in CURATED_EXTRA:
        url = canonicalize_url(entry["url"])
        try:
            page = load_page(url)
        except HtmlLoadError as e:
            print(f"[WARN] curated extra failed to load, skipping: {url} ({e})")
            continue
        rows.append({
            "url": url,
            "provider": _provider_for_url(url),
            "topic": entry["topic"],
            "subtopic": entry["subtopic"],
            "title": page.title,
            "authority_tier": entry["authority_tier"],
            "chars": len(page.text),
        })
        time.sleep(CRAWL_DELAY_SECONDS)
    return rows


def _print_summary(rows: List[dict], label: str) -> None:
    by_topic: Dict[str, int] = {}
    for row in rows:
        by_topic[row["topic"]] = by_topic.get(row["topic"], 0) + 1
    print(f"\n=== {label} ===")
    for topic, count in sorted(by_topic.items()):
        print(f"  {topic:16} {count}")
    print(f"  {'TOTAL':16} {len(rows)}")


def main() -> None:
    if os.path.exists(CACHE_PATH):
        print(f"[INFO] Using cached crawl: {CACHE_PATH}")
        with open(CACHE_PATH, encoding="utf-8") as fh:
            results = json.load(fh)
    else:
        results = discover()
        with open(CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"[INFO] Wrote crawl cache: {CACHE_PATH}")

    _print_summary(results, "Discovered (raw)")

    print("\n[INFO] Loading curated newcomer-essential pages...")
    extras = load_curated_extras()
    print(f"[INFO] Verified {len(extras)} curated extra pages")

    curated = curate(results + extras)
    _print_summary(curated, "Curated (written to legal_sources.py)")
    write_sources_file(curated)
    print(f"\nWrote {SOURCES_PATH} with {len(curated)} sources")


if __name__ == "__main__":
    main()
