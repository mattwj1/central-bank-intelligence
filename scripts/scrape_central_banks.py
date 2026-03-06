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


# ── Shared Utilities ───────────────────────────────────────────────────────────

def fetch_page(url: str, timeout: int = 30, retries: int = 2) -> BeautifulSoup | None:
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
    session = requests.Session()
    session.headers.update(HEADERS)

    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.Timeout:
            if attempt < retries:
                logger.info(f"  Timeout on attempt {attempt + 1}, retrying in 3s...")
                time.sleep(3)
            else:
                logger.warning(f"  Failed after {retries + 1} attempts (timeout): {url}")
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
    return filepath.exists() and filepath.stat().st_size > 100


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


def scrape_boe(limit: int = 3) -> list[Path]:
    """
    Scrape Bank of England MPC minutes by constructing URLs directly.

    BoE minutes URL pattern:
    bankofengland.co.uk/monetary-policy-summary-and-minutes/YYYY/month-YYYY

    The index page is JavaScript-rendered so we construct URLs directly
    from known MPC meeting months rather than scraping the index.

    Args:
        limit: Maximum number of documents to scrape (most recent first)

    Returns:
        List of paths to saved files
    """
    logger.info(f"\nScraping BoE (last {limit} documents)...")
    saved = []

    for year, month_name in BOE_MEETING_MONTHS[:limit + 3]:
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
        logger.info(f"  Trying BoE: {url}")
        page = fetch_page(url)

        if not page:
            logger.warning(f"  BoE {date_str}: page not found — minutes may not be published yet")
            continue

        # Minutes text is in the main content area
        article = page.find("main") or page

#        article = (
 #               page.find("div", class_="col9") or
  #              page.find("main") or
   #             page
    #    )
        # DEBUG
        logger.info(f"  DEBUG col9 raw length before cleaning: {len(article.get_text())}")
        logger.info(
            f"  DEBUG col9 child tags: {[c.name for c in article.children if hasattr(c, 'name') and c.name][:10]}")
        text = clean_text(article)

        if len(text) < 200:
            logger.warning(f"  BoE {date_str}: text too short ({len(text)} chars) — page may not be published yet")
            continue

        saved.append(save_minutes("BOE", date_str, text))
        time.sleep(1)

    return saved


# ── European Central Bank ──────────────────────────────────────────────────────

# ECB publishes press conference transcripts as PDFs.
# URLs must be obtained manually from the ECB press conference page
# as they contain a unique hash per document.
# Add new entries after each ECB Governing Council meeting.
ECB_KNOWN_PDFS = [
    ("2026-02-05", "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/shared/pdf/ecb.ds260205~0c74e825a7.en.pdf"),
    ("2025-12-18", "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/shared/pdf/ecb.ds251218~f264376788.en.pdf"),
    ("2025-10-30", "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/shared/pdf/ecb.ds251030~25d99d5b1c.en.pdf"),
]


def scrape_ecb(limit: int = 3) -> list[Path]:
    """
    Fetch ECB press conference transcripts from official PDF publications.

    The ECB index page is JavaScript-rendered so HTML scraping is not
    reliable. PDFs are fetched directly and text extracted with pdfplumber.
    After each ECB meeting, add the new PDF URL to ECB_KNOWN_PDFS.

    Args:
        limit: Maximum number of documents to scrape (most recent first)

    Returns:
        List of paths to saved text files
    """
    try:
        import pdfplumber
        import io
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return []

    logger.info(f"\nScraping ECB (last {limit} documents via PDF)...")
    saved = []

    for date_str, pdf_url in ECB_KNOWN_PDFS[:limit]:
        if already_scraped("ECB", date_str):
            logger.info(f"  Skipping ECB {date_str} — already scraped")
            continue

        logger.info(f"  Fetching ECB PDF: {pdf_url}")
        try:
            response = requests.get(pdf_url, headers=HEADERS, timeout=30)
            response.raise_for_status()

            # Extract text from PDF bytes
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)

            text = "\n".join(pages_text)

            # Clean up common PDF extraction artefacts
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            text = "\n".join(lines)

            if len(text) < 500:
                logger.warning(f"  ECB {date_str}: extracted text too short ({len(text)} chars)")
                continue

            logger.info(f"  ECB {date_str}: extracted {len(text):,} chars from {len(pdf.pages)} pages")
            saved.append(save_minutes("ECB", date_str, text))
            time.sleep(1)

        except Exception as e:
            logger.warning(f"  ECB {date_str}: failed — {e}")

    return saved

# ── Federal Reserve ────────────────────────────────────────────────────────────

FED_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"


def scrape_fed(limit: int = 3) -> list[Path]:
    """
    Scrape Federal Reserve FOMC meeting minutes.

    Parses the FOMC calendar page to find HTML minutes links,
    then extracts text from each minutes document.

    Args:
        limit: Maximum number of documents to scrape (most recent first)

    Returns:
        List of paths to saved files
    """
    logger.info(f"\nScraping Fed FOMC (last {limit} documents)...")
    soup = fetch_page(FED_CALENDAR)
    if not soup:
        return []

    # Find all links to minutes HTML pages
    # Pattern: /newsevents/pressreleases/monetaryYYYYMMDDa.htm
    minutes_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"monetary\d{8}a\.htm", href) and a.get_text(strip=True) == "HTML":
            full_url = f"https://www.federalreserve.gov{href}" if href.startswith("/") else href
            # Extract date from URL: monetary20260129a.htm → 2026-01-29
            date_match = re.search(r"monetary(\d{4})(\d{2})(\d{2})a", href)
            if date_match:
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                minutes_links.append((date_str, full_url))

    # Sort newest first
    minutes_links.sort(key=lambda x: x[0], reverse=True)
    logger.info(f"  Found {len(minutes_links)} Fed minutes links")

    saved = []
    for date_str, url in minutes_links[:limit]:
        if already_scraped("FED", date_str):
            logger.info(f"  Skipping FED {date_str} — already scraped")
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
        time.sleep(1)

    return saved


# ── European Central Bank ──────────────────────────────────────────────────────


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
        help="Number of recent documents to scrape per bank (default: 3)"
    )
    args = parser.parse_args()

    all_saved = []

    if args.bank in (None, "BOE"):
        all_saved.extend(scrape_boe(limit=args.limit))

    if args.bank in (None, "FED"):
        all_saved.extend(scrape_fed(limit=args.limit))

    if args.bank in (None, "ECB"):
        all_saved.extend(scrape_ecb(limit=args.limit))

    logger.info(f"\n── Complete ──────────────────────────────────────────")
    logger.info(f"  Total documents saved: {len(all_saved)}")
    for path in all_saved:
        logger.info(f"  ✓ {path.name}")

    if not all_saved:
        logger.info("  No new documents (all already scraped, or scraping failed)")


if __name__ == "__main__":
    main()
