"""
scrape_central_banks.py
-----------------------
Scrapes meeting minutes and policy statements from:
  - Bank of England (BoE)
  - Federal Reserve (Fed / FOMC)
  - European Central Bank (ECB)

Each scraper follows the same pattern:
  1. Fetch the index page to find available documents
  2. Extract links to individual minutes/transcripts
  3. Follow each link and extract clean text
  4. Save to /data/raw/minutes/BANK_YYYY-MM-DD.txt

Usage:
  python scripts/scrape_central_banks.py --bank BOE
  python scripts/scrape_central_banks.py --bank FED
  python scripts/scrape_central_banks.py --bank ECB
  python scripts/scrape_central_banks.py  # runs all three
"""

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Setup ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
MINUTES_DIR = ROOT / "data" / "raw" / "minutes"
MINUTES_DIR.mkdir(parents=True, exist_ok=True)

# Mimic a real browser — some sites block plain requests
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ── Shared Utilities ───────────────────────────────────────────────────────────

def fetch_page(
    url: str,
    timeout: int = 30,
    retries: int = 2,
    session: requests.Session | None = None,
    referer: str | None = None,
) -> BeautifulSoup | None:
    """
    Fetch a URL and return a BeautifulSoup object.

    Uses a session with browser-like headers. Retries once on timeout
    since some central bank sites are slow to respond.

    Args:
        url: Full URL to fetch
        timeout: Request timeout in seconds
        retries: Number of retry attempts on timeout

    Returns:
        BeautifulSoup object, or None if request fails
    """
    session = session or SESSION

    for attempt in range(retries + 1):
        try:
            headers = {"Referer": referer} if referer else None
            response = session.get(url, timeout=timeout, headers=headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.Timeout:
            if attempt < retries:
                logger.info(f"  Timeout on attempt {attempt + 1}, retrying in 3s...")
                time.sleep(3)
            else:
                logger.warning(f"  Failed after {retries + 1} attempts (timeout): {url}")
                return None
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", "unknown")
            logger.warning(f"  Failed to fetch {url}: HTTP {status}")
            return None
        except Exception as e:
            logger.warning(f"  Failed to fetch {url}: {e}")
            return None


def clean_text(soup: BeautifulSoup, remove_tags: list[str] = None) -> str:
    """
    Extract clean text from a BeautifulSoup object.

    Removes navigation, headers, footers, and script/style tags.
    Collapses whitespace and strips empty lines.

    Args:
        soup: Parsed HTML
        remove_tags: Additional HTML tags to remove before extracting text

    Returns:
        Clean plain text string
    """
    # Always remove these
    for tag in ["script", "style", "nav", "header", "footer",
                "aside", "noscript", "button", "form"]:
        for element in soup.find_all(tag):
            element.decompose()

    # Remove any extra tags the caller specifies
    if remove_tags:
        for tag in remove_tags:
            for element in soup.find_all(tag):
                element.decompose()

    text = soup.get_text(separator="\n")

    # Collapse multiple blank lines into one
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _looks_like_ecb_account_text(text: str) -> bool:
    """
    Distinguish true ECB accounts from short policy statements.

    Valid account pages typically contain either the standard
    "Account of the monetary policy meeting" title or variants such as the
    strategy-assessment account that still begin with "Account of ...".
    """
    head = text[:4000]
    if "Account of the monetary policy meeting" in head:
        return True
    return "Account of" in head and "Governing Council" in head


def already_scraped(bank: str, date_str: str) -> bool:
    """
    Check whether a minutes file already exists for this bank and date.

    Args:
        bank: "BOE", "FED", or "ECB"
        date_str: Date in YYYY-MM-DD format

    Returns:
        True if file exists and has content
    """
    filepath = MINUTES_DIR / f"{bank}_{date_str}.txt"
    if not filepath.exists() or filepath.stat().st_size <= 100:
        return False

    if bank != "ECB":
        return True

    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return _looks_like_ecb_account_text(text)


def save_minutes(bank: str, date_str: str, text: str) -> Path:
    """
    Save scraped minutes text to disk.

    Args:
        bank: "BOE", "FED", or "ECB"
        date_str: Date in YYYY-MM-DD format (meeting date, not publication date)
        text: Clean text content

    Returns:
        Path to saved file
    """
    filepath = MINUTES_DIR / f"{bank}_{date_str}.txt"
    filepath.write_text(text, encoding="utf-8")
    logger.info(f"  Saved → {filepath.name} ({len(text):,} chars)")
    return filepath


def get_latest_minutes(bank: str) -> str | None:
    """
    Return the text of the most recently scraped document for a given bank.

    Args:
        bank: "BOE", "FED", or "ECB"

    Returns:
        Text content of most recent file, or None if no files exist
    """
    files = sorted(MINUTES_DIR.glob(f"{bank}_*.txt"), reverse=True)
    if not files:
        logger.warning(f"No scraped minutes found for {bank}")
        return None
    latest = files[0]
    logger.info(f"Latest {bank} minutes: {latest.name}")
    return latest.read_text(encoding="utf-8")

# ── Bank of England ────────────────────────────────────────────────────────────

# Month name → number and name for URL construction
MONTH_MAP = {
    "january": ("01", "january"), "february": ("02", "february"),
    "march": ("03", "march"), "april": ("04", "april"),
    "may": ("05", "may"), "june": ("06", "june"),
    "july": ("07", "july"), "august": ("08", "august"),
    "september": ("09", "september"), "october": ("10", "october"),
    "november": ("11", "november"), "december": ("12", "december"),
}

# BoE MPC meeting months — they meet 8 times per year
# These are the months meetings actually occur
BOE_MEETING_MONTHS = [
    ("2026", "february"), ("2025", "december"), ("2025", "november"),
    ("2025", "september"), ("2025", "august"), ("2025", "june"),
    ("2025", "may"), ("2025", "march"), ("2025", "february"),
]


def _boe_candidate_dates(include_historical: bool = False) -> list[tuple[str, str]]:
    """
    Produce (year, month_name) candidates to try. For the recent-only mode we
    use the hand-curated list; for the backfill we iterate every year × month
    from 2016 onwards and let 404s filter out non-meeting months.
    """
    if not include_historical:
        return BOE_MEETING_MONTHS

    current_year = datetime.now().year
    candidates = []
    for year in range(current_year, 2015, -1):
        for month_name in (
            "december", "november", "september", "august", "june",
            "may", "march", "february", "october", "january",
            "july", "april",
        ):
            candidates.append((str(year), month_name))
    return candidates


def scrape_boe(limit: int = 3, include_historical: bool = False) -> list[Path]:
    """
    Scrape Bank of England MPC minutes by constructing URLs directly.

    BoE minutes URL pattern:
        bankofengland.co.uk/monetary-policy-summary-and-minutes/YYYY/month-YYYY

    The index page is JavaScript-rendered so we construct URLs and let 404s
    filter non-meeting months. Meets ~8x/year so ~4 URLs per year will 404.

    Args:
        limit: Max new documents to save this run (most recent first).
        include_historical: Iterate every month 2016-present for the 10-year
            backfill. Default False keeps regular runs fast.
    """
    logger.info(
        f"\nScraping BoE (limit={limit}, historical={include_historical})..."
    )
    saved = []
    candidates = _boe_candidate_dates(include_historical=include_historical)

    for year, month_name in candidates:
        if len(saved) >= limit:
            break

        month_num = MONTH_MAP[month_name][0]
        date_str = f"{year}-{month_num}-01"

        if already_scraped("BOE", date_str):
            logger.info(f"  Skipping BOE {date_str} — already scraped")
            continue

        url = (
            f"https://www.bankofengland.co.uk"
            f"/monetary-policy-summary-and-minutes/{year}/{month_name}-{year}"
        )
        page = fetch_page(url)

        if not page:
            # Non-meeting month or unpublished — silent in historical mode to
            # avoid 40+ warnings per run.
            if not include_historical:
                logger.warning(f"  BoE {date_str}: page not found")
            continue

        article = page.find("main") or page
        text = clean_text(article)

        if len(text) < 2000:
            # 2000-char threshold filters out BoE's generic "page not found"
            # pages which render a short template rather than 404.
            if not include_historical:
                logger.warning(
                    f"  BoE {date_str}: text too short ({len(text)} chars)"
                )
            continue

        saved.append(save_minutes("BOE", date_str, text))
        time.sleep(1)

    return saved


# ── European Central Bank ──────────────────────────────────────────────────────

ECB_ACCOUNTS_INDEX = "https://www.ecb.europa.eu/press/accounts/html/index.en.html"
ECB_HOME = "https://www.ecb.europa.eu/"


def _parse_ecb_meeting_date(text: str) -> str | None:
    """
    Parse meeting dates from titles such as:
      - "Meeting of 17-18 December 2025"
      - "Meeting of 4-5 February 2026"

    We use the final meeting day as the canonical date, matching the
    convention already used elsewhere in the project.
    """
    match = re.search(
        r"Meeting of (?:(\d{1,2})-)?(\d{1,2}) ([A-Za-z]+) (\d{4})",
        text,
    )
    if not match:
        return None

    day = match.group(2)
    month = match.group(3)
    year = match.group(4)
    try:
        return datetime.strptime(
            f"{day} {month} {year}",
            "%d %B %Y",
        ).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_ecb_publication_date(text: str) -> str | None:
    """Parse index publication dates like '16 April 2026'."""
    cleaned = " ".join(text.split())
    try:
        return datetime.strptime(cleaned, "%d %B %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _find_ecb_account_links_live() -> list[tuple[str, str]]:
    """
    Parse the official ECB accounts index for live account URLs.

    The page currently exposes a <dl> list where each <dt> is a publication
    date and the following <dd> contains one or more language links.
    """
    index_page = fetch_page(
        ECB_ACCOUNTS_INDEX,
        session=SESSION,
        referer=ECB_HOME,
    )
    if not index_page:
        return []

    results = []
    seen_urls = set()
    language_labels = {"english", "french", "francais", "français", "german", "deutsch"}

    for dd in index_page.find_all("dd"):
        dt = dd.find_previous_sibling("dt")
        publication_date = None
        if dt is not None:
            publication_date = _parse_ecb_publication_date(
                dt.get_text(" ", strip=True)
            )

        anchors = dd.find_all("a", href=True)
        candidate_anchors = [
            anchor
            for anchor in anchors
            if anchor.get_text(" ", strip=True).lower() not in language_labels
        ]
        if not candidate_anchors:
            candidate_anchors = anchors

        for anchor in candidate_anchors:
            href = urljoin(ECB_ACCOUNTS_INDEX, anchor["href"])
            if "/press/accounts/" not in href or not href.endswith(".en.html"):
                continue

            if href in seen_urls:
                continue

            title = anchor.get_text(" ", strip=True)
            date_str = _parse_ecb_meeting_date(title) or publication_date
            if not date_str:
                logger.info(f"  ECB: couldn't parse date from {title!r} — skipping")
                continue

            seen_urls.add(href)
            results.append((date_str, href))

    results.sort(key=lambda item: item[0], reverse=True)
    logger.info(f"  ECB accounts index: found {len(results)} documents")
    return results


def _merge_ecb_candidates(
    primary: list[tuple[str, str]],
    fallback: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """
    Merge live and curated ECB candidates, preferring live URLs when dates
    overlap.
    """
    merged = []
    seen = set()
    for date_str, url in primary + fallback:
        if date_str in seen:
            continue
        seen.add(date_str)
        merged.append((date_str, url))
    return merged


def _extract_ecb_account_body(page) -> object:
    """
    Locate the main content for an ECB monetary policy account.

    ECB pages include many generic ``div.section`` wrappers before the actual
    article, so we anchor on the article heading or fall back to <main>.
    """
    heading = page.find(
        ["h1", "h2", "h3"],
        string=re.compile(r"Account of the monetary policy meeting", re.I),
    ) or page.find(
        ["h1", "h2", "h3"],
        string=re.compile(r"Meeting of \d{1,2}(?:-\d{1,2})? [A-Za-z]+ \d{4}", re.I),
    )
    if heading is not None:
        for ancestor in heading.parents:
            if getattr(ancestor, "name", None) in {"main", "article"}:
                return ancestor

    return page.find("main") or page.find("article") or page


# Complete lookup of ECB Monetary Policy Accounts (the minutes-equivalent).
# The live scraper should prefer the official index page, but we keep a curated
# fallback list because ECB occasionally changes markup and link hashes.
# IMPORTANT: the left-hand date is the filename key used by the existing
# pipeline. Newer rows mostly use meeting dates; some older backfill rows still
# use publication-date aliases and should be normalised later if we want fully
# exact cross-bank alignment.
_ECB_ACCOUNTS: list[tuple[str, str]] = [
    # ── 2026 ──
    ("2026-03-19", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260416~6a27b0c258.en.html"),
    ("2026-02-05", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260305~4a9b7afe1c.en.html"),
    # ── 2025 ──
    ("2025-12-18", "https://www.ecb.europa.eu/press/accounts/2026/html/ecb.mg260122~5ca84e0f51.en.html"),
    ("2025-10-30", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg251127~dc88fc4bec.en.html"),
    ("2025-09-11", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg251009~eec3e95eb5.en.html"),
    ("2025-07-24", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250828~071d6cc9c7.en.html"),
    ("2025-06-25", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250828_1~e9f77119ce.en.html"),
    ("2025-06-05", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250703~07feaceb60.en.html"),
    ("2025-04-17", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250522~31b2c664d4.en.html"),
    ("2025-03-06", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250403~e578b5dbea.en.html"),
    ("2025-01-30", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250227~5a2b6faa14.en.html"),
    # ── 2024 ──
    ("2024-12-12", "https://www.ecb.europa.eu/press/accounts/2025/html/ecb.mg250116~2f8f2a2ad3.en.html"),
    ("2024-10-17", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg241114~c0e6f53cf7.en.html"),
    ("2024-09-12", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg241010~1036884a9a.en.html"),
    ("2024-07-18", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240822~d49b920824.en.html"),
    ("2024-06-06", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240704~fbde4f46aa.en.html"),
    ("2024-04-11", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240510~6505e9dac3.en.html"),
    ("2024-03-07", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240404~b79e424115.en.html"),
    ("2024-01-25", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240222~1af5fcd5f9.en.html"),
    # ── 2023 ──
    ("2023-12-14", "https://www.ecb.europa.eu/press/accounts/2024/html/ecb.mg240118~57d24ff18f.en.html"),
    ("2023-10-26", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg231123~40c9631bc7.en.html"),
    ("2023-09-14", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg231012~2f3d803d32.en.html"),
    ("2023-07-27", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230831~b04764f45f.en.html"),
    ("2023-06-15", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230713~f7e54fdb87.en.html"),
    ("2023-05-04", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230601~9d35f80dee.en.html"),
    ("2023-03-16", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230420~e8043d2d3d.en.html"),
    ("2023-02-02", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230302~009d06dd5a.en.html"),
    ("2022-12-15", "https://www.ecb.europa.eu/press/accounts/2023/html/ecb.mg230119~e522ad4e37.en.html"),
    # ── 2022 ──
    ("2022-10-27", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg221124~3527764024.en.html"),
    ("2022-09-08", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg221006~a5f7fb03f3.en.html"),
    ("2022-07-21", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220825~162cfabae9.en.html"),
    ("2022-06-09", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220707~d5c3246061.en.html"),
    ("2022-04-14", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220519~c9200dba08.en.html"),
    ("2022-03-10", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220407~8e7069ffa0.en.html"),
    ("2022-02-03", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220303~7ac13bacbe.en.html"),
    ("2021-12-16", "https://www.ecb.europa.eu/press/accounts/2022/html/ecb.mg220120~7ed187b5b1.en.html"),
    # ── 2021 ──
    ("2021-10-28", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg211125~ca9833f9a9.en.html"),
    ("2021-09-09", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg211007~1c2f4db595.en.html"),
    ("2021-07-22", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210826~16a0691c87.en.html"),
    ("2021-07-07", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210729~b83737e3b5.en.html"),
    ("2021-06-10", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210709~8d7a056036.en.html"),
    ("2021-04-22", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210514~4b2606bff9.en.html"),
    ("2021-03-11", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210408~46b9deaa4a.en.html"),
    ("2021-01-21", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210218~9dab5cb5f7.en.html"),
    # ── 2020 ──
    ("2020-12-10", "https://www.ecb.europa.eu/press/accounts/2021/html/ecb.mg210114~14ef04b8bd.en.html"),
    ("2020-10-29", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg201126~20e838e857.en.html"),
    ("2020-09-10", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg201008~49aeff32e1.en.html"),
    ("2020-07-16", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200820~c30e2e26b9.en.html"),
    ("2020-06-04", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200625~fd97330d5f.en.html"),
    ("2020-04-30", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200522~f0355619ae.en.html"),
    ("2020-03-18", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200409_1~baf4b2ad06.en.html"),  # emergency COVID meeting
    ("2020-03-12", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200409~0026941ce4.en.html"),
    ("2020-02-20", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200220~c4d71ec138.en.html"),
    ("2020-01-16", "https://www.ecb.europa.eu/press/accounts/2020/html/ecb.mg200116~973b558e59.en.html"),
    # ── 2019 ──
    ("2019-11-21", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg191121~b1d36734d7.en.html"),
    ("2019-10-10", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg191010~d8086505d0.en.html"),
    ("2019-08-22", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190822~63660ecd81.en.html"),
    ("2019-07-11", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190711~16eb146254.en.html"),
    ("2019-05-23", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190523~3e19e27fb7.en.html"),
    ("2019-04-11", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190404~edc605830b.en.html"),
    ("2019-02-21", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190221~0f3dd919fa.en.html"),
    ("2019-01-10", "https://www.ecb.europa.eu/press/accounts/2019/html/ecb.mg190110.en.html"),
    # ── 2018 ──
    ("2018-11-22", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg181122.en.html"),
    ("2018-10-11", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg181011.en.html"),
    ("2018-08-23", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180823.en.html"),
    ("2018-07-12", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180712.en.html"),
    ("2018-05-24", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180524.en.html"),
    ("2018-04-12", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180412.en.html"),
    ("2018-02-22", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180222.en.html"),
    ("2018-01-11", "https://www.ecb.europa.eu/press/accounts/2018/html/ecb.mg180111.en.html"),
    # ── 2017 ──
    ("2017-11-23", "https://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg171123.en.html"),
    ("2017-10-05", "https://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg171005.en.html"),
    ("2017-08-17", "https://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg170817.en.html"),
    ("2017-07-06", "https://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg170706.en.html"),
    ("2017-05-18", "https://www.ecb.europa.eu/press/accounts/2017/html/ecb.mg170518.en.html"),
    ("2017-04-06", "https://www.ecb.europa.eu/press/accounts/2017/html/mg170406.en.html"),
    ("2017-02-16", "https://www.ecb.europa.eu/press/accounts/2017/html/mg170216.en.html"),
    ("2017-01-12", "https://www.ecb.europa.eu/press/accounts/2017/html/mg170112.en.html"),
    # ── 2016 ──
    ("2016-11-17", "https://www.ecb.europa.eu/press/accounts/2016/html/mg161117.en.html"),
    ("2016-10-06", "https://www.ecb.europa.eu/press/accounts/2016/html/mg161006.en.html"),
    ("2016-08-18", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160818.en.html"),
    ("2016-07-07", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160707.en.html"),
    ("2016-05-19", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160519.en.html"),
    ("2016-04-07", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160407.en.html"),
    ("2016-02-18", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160218.en.html"),
    ("2016-01-14", "https://www.ecb.europa.eu/press/accounts/2016/html/mg160114.en.html"),
    # ── 2015 ──
    ("2015-11-19", "https://www.ecb.europa.eu/press/accounts/2015/html/mg151119.en.html"),
    ("2015-10-08", "https://www.ecb.europa.eu/press/accounts/2015/html/mg151008.en.html"),
    ("2015-08-13", "https://www.ecb.europa.eu/press/accounts/2015/html/mg150813.en.html"),
    ("2015-07-02", "https://www.ecb.europa.eu/press/accounts/2015/html/mg150702.en.html"),
    ("2015-05-21", "https://www.ecb.europa.eu/press/accounts/2015/html/mg150521.en.html"),
    ("2015-04-02", "https://www.ecb.europa.eu/press/accounts/2015/html/mg150402.en.html"),
    ("2015-02-19", "https://www.ecb.europa.eu/press/accounts/2015/html/mg150219.en.html"),
]


def scrape_ecb(limit: int = 3, include_historical: bool = False) -> list[Path]:
    """
    Scrape ECB Monetary Policy Accounts (the minutes-equivalent).

    Prefers the official accounts index, using a persistent session so the ECB
    site sees a normal browsing flow (index page first, account page second).
    Falls back to the curated lookup if the index page is unavailable or if the
    site markup changes.

    Args:
        limit: Max new documents to save this run (most recent first).
        include_historical: If True, walk the entire list back to 2015.
            If False, only attempt the most recent ``limit`` entries.
    """
    logger.info(
        f"\nScraping ECB accounts "
        f"(limit={limit}, historical={include_historical})..."
    )

    live_candidates = _find_ecb_account_links_live()
    if live_candidates:
        candidates = _merge_ecb_candidates(live_candidates, _ECB_ACCOUNTS)
        logger.info(f"  ECB candidate pool: {len(candidates)} documents")
    else:
        logger.warning("  ECB accounts index unavailable — using curated lookup")
        candidates = _ECB_ACCOUNTS

    if not include_historical:
        candidates = candidates[: max(limit * 4, 12)]

    saved = []
    new_count = 0
    for date_str, url in candidates:
        if new_count >= limit:
            break

        if already_scraped("ECB", date_str):
            logger.info(f"  Skipping ECB {date_str} — already scraped")
            continue

        logger.info(f"  Fetching ECB account: {date_str} → {url}")
        page = fetch_page(
            url,
            session=SESSION,
            referer=ECB_ACCOUNTS_INDEX,
        )
        if not page:
            logger.warning(f"  ECB {date_str}: fetch failed")
            continue

        article = _extract_ecb_account_body(page)
        text = clean_text(article)

        if len(text) < 2000 or not _looks_like_ecb_account_text(text):
            preview = text[:160].replace("\n", " ")
            logger.warning(
                f"  ECB {date_str}: unexpected page content ({len(text)} chars) — "
                f"skipping. Preview: {preview!r}"
            )
            continue

        saved.append(save_minutes("ECB", date_str, text))
        new_count += 1
        logger.info(f"  ECB {date_str}: saved ({len(text):,} chars)")
        time.sleep(1)

    return saved

# ── Federal Reserve ────────────────────────────────────────────────────────────

FED_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

# For the 10-year backfill, Fed publishes separate historical pages per year
# (e.g. fomchistorical2016.htm). fomccalendars.htm covers the current + recent
# years. We walk all of them to find every fomcminutesYYYYMMDD.htm link.
FED_HISTORICAL_PAGES = [
    f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
    for y in range(2016, 2022)  # 2016-2021 are archived to historical pages
]


def scrape_fed(limit: int = 8, include_historical: bool = False) -> list[Path]:
    """
    Scrape Federal Reserve FOMC meeting MINUTES (not statements).

    FOMC minutes are the detailed record of the meeting, released ~3 weeks
    after each meeting, and typically 20+ pages long. They live at URLs
    matching /monetarypolicy/fomcminutesYYYYMMDD.htm.

    Args:
        limit: Max new documents to save this run (most recent first).
        include_historical: Walk 2016-2021 historical index pages too. Use
            this for the 10-year backfill.
    """
    logger.info(
        f"\nScraping Fed FOMC minutes "
        f"(limit={limit}, historical={include_historical})..."
    )
    minutes_links = _find_fed_minutes_links(include_historical=include_historical)

    if not minutes_links:
        logger.warning("  No minutes found on calendar page — trying URL fallback")
        minutes_links = _fallback_fed_minutes_urls()

    minutes_links.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Candidate minutes: {len(minutes_links)}")

    saved = []
    attempted = 0
    for date_str, url in minutes_links:
        if attempted >= limit:
            break
        if already_scraped("FED", date_str):
            logger.info(f"  Skipping FED {date_str} — already scraped")
            attempted += 1
            continue

        logger.info(f"  Scraping Fed: {url}")
        page = fetch_page(url)
        if not page:
            continue

        # Fed minutes are in a div with id="article" or class containing "col-xs-12"
        article = (
            page.find("div", id="article") or
            page.find("div", class_=re.compile(r"col-xs-12|content-detail", re.I)) or
            page.find("main") or
            page
        )

        text = clean_text(article)

        if len(text) < 500:
            logger.warning(f"  Fed {date_str}: text too short ({len(text)} chars) — skipping")
            continue

        saved.append(save_minutes("FED", date_str, text))
        attempted += 1
        time.sleep(1)

    return saved


def _find_fed_minutes_links(include_historical: bool = False) -> list[tuple[str, str]]:
    """
    Walk FOMC calendar/history pages to collect all fomcminutes links.

    Args:
        include_historical: Also walk fomchistorical{YEAR}.htm for 2016-2021,
            which are needed for the 10-year backfill.

    Returns:
        List of (YYYY-MM-DD, full_url) tuples.
    """
    pages_to_check = [FED_CALENDAR]
    if include_historical:
        pages_to_check.extend(FED_HISTORICAL_PAGES)

    minutes_links: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    for page_url in pages_to_check:
        logger.info(f"  Walking: {page_url}")
        soup = fetch_page(page_url)
        if not soup:
            logger.warning(f"  Failed to fetch {page_url}")
            continue

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # FOMC minutes URLs look like /monetarypolicy/fomcminutesYYYYMMDD.htm
            m = re.search(r"fomcminutes(\d{4})(\d{2})(\d{2})", href)
            if not m:
                continue
            full_url = (
                f"https://www.federalreserve.gov{href}"
                if href.startswith("/") else href
            )
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            minutes_links.append((date_str, full_url))

    logger.info(f"  Total minutes links found: {len(minutes_links)}")
    return minutes_links


def _fallback_fed_minutes_urls() -> list[tuple[str, str]]:
    """
    Generate plausible FOMC minutes URLs for recent FOMC meeting dates
    as a fallback when the calendar page is unavailable.
    """
    # 8 most recent FOMC meetings — update periodically
    known_dates = [
        "20260129", "20251217", "20251029", "20250917",
        "20250730", "20250618", "20250507", "20250319",
    ]
    base = "https://www.federalreserve.gov/monetarypolicy/fomcminutes"
    return [
        (
            f"{d[:4]}-{d[4:6]}-{d[6:]}",
            f"{base}{d}.htm",
        )
        for d in known_dates
    ]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape central bank meeting minutes"
    )
    parser.add_argument(
        "--bank",
        choices=["BOE", "FED", "ECB"],
        help="Scrape a specific bank only. Omit to scrape all three."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of recent documents to scrape per bank (default: 3). "
             "Set high (e.g. 200) with --historical for backfill."
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Include historical archive pages for the 10-year backfill."
    )
    args = parser.parse_args()

    all_saved = []

    if args.bank in (None, "BOE"):
        all_saved.extend(scrape_boe(
            limit=args.limit, include_historical=args.historical))

    if args.bank in (None, "FED"):
        all_saved.extend(scrape_fed(
            limit=args.limit, include_historical=args.historical))

    if args.bank in (None, "ECB"):
        all_saved.extend(scrape_ecb(
            limit=args.limit, include_historical=args.historical))

    logger.info(f"\n── Complete ──────────────────────────────────────────")
    logger.info(f"  Total documents saved: {len(all_saved)}")
    for path in all_saved:
        logger.info(f"  ✓ {path.name}")

    if not all_saved:
        logger.info("  No new documents (all already scraped, or scraping failed)")


if __name__ == "__main__":
    main()