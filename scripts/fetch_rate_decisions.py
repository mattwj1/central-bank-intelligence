"""
fetch_rate_decisions.py
-----------------------
For every meeting date in data/processed/sentiment_history.json, look up
the policy rate before and after the decision and compute the change in
basis points. Writes data/processed/rate_decisions.json, consumed by the
dashboard to overlay hike/cut markers on the sentiment chart.

All three banks now use hardcoded rate-change lookup tables. These are
well-documented public facts and more reliable than FRED API calls (which
suffer from effective-date lags for ECB, monthly granularity for BoE,
and network dependency for all).

Usage:
    python scripts/fetch_rate_decisions.py
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
HISTORY     = ROOT / "data" / "processed" / "sentiment_history.json"
OUTPUT      = ROOT / "data" / "processed" / "rate_decisions.json"

# ── Fed Funds Rate history ────────────────────────────────────────────────────
# Decision date → new Fed Funds Target Rate, Upper Limit (%).
# Source: federalreserve.gov/monetarypolicy/openmarket.htm
# Pre-2015 entries anchor the lookup for early meeting dates.

_FED_RATE_CHANGES = [
    ("2008-12-16", 0.25),   # ZIRP (floor)
    ("2015-12-16", 0.50),   # first hike off ZLB
    ("2016-12-14", 0.75),
    ("2017-03-15", 1.00),
    ("2017-06-14", 1.25),
    ("2017-12-13", 1.50),
    ("2018-03-21", 1.75),
    ("2018-06-13", 2.00),
    ("2018-09-26", 2.25),
    ("2018-12-19", 2.50),   # peak of first cycle
    ("2019-07-31", 2.25),   # first cut
    ("2019-09-18", 2.00),
    ("2019-10-30", 1.75),
    ("2020-03-03", 1.25),   # emergency cut (COVID)
    ("2020-03-15", 0.25),   # emergency cut to ZLB
    ("2022-03-16", 0.50),   # first hike — tightening cycle
    ("2022-05-04", 1.00),
    ("2022-06-15", 1.75),
    ("2022-07-27", 2.50),
    ("2022-09-21", 3.25),
    ("2022-11-02", 4.00),
    ("2022-12-14", 4.50),
    ("2023-02-01", 4.75),
    ("2023-03-22", 5.00),
    ("2023-05-03", 5.25),
    ("2023-07-26", 5.50),   # peak
    ("2024-09-18", 5.00),   # first cut
    ("2024-11-07", 4.75),
    ("2024-12-18", 4.50),
    ("2025-01-29", 4.50),   # hold
]

# ── ECB Deposit Facility Rate history ────────────────────────────────────────
# Decision date → new deposit facility rate (%).
# Source: ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates
# FRED's ECBDFR series uses the *effective* date (weeks after the decision),
# so the before/after window around the meeting date always shows 0 bps.
# This lookup maps the actual Governing Council decision dates instead.

_ECB_RATE_CHANGES = [
    ("2014-01-01", -0.10),   # rate at the start of our window (Sep 2014 cut)
    ("2014-09-04",  -0.20),  # cut
    ("2015-12-03",  -0.30),  # cut
    ("2016-03-10",  -0.40),  # cut
    ("2019-09-12",  -0.50),  # cut (tiering introduced)
    ("2022-07-21",   0.00),  # first hike — exit from negative rates
    ("2022-09-08",   0.75),  # hike +75 bps
    ("2022-10-27",   1.50),  # hike +75 bps
    ("2022-12-15",   2.00),  # hike +50 bps
    ("2023-02-02",   2.50),  # hike +50 bps
    ("2023-03-16",   3.00),  # hike +50 bps
    ("2023-05-04",   3.25),  # hike +25 bps
    ("2023-06-15",   3.50),  # hike +25 bps
    ("2023-07-27",   3.75),  # hike +25 bps
    ("2023-09-14",   4.00),  # hike +25 bps — peak
    ("2024-06-06",   3.75),  # first cut
    ("2024-09-12",   3.50),  # cut
    ("2024-10-17",   3.25),  # cut
    ("2024-12-12",   3.00),  # cut
    ("2025-01-30",   2.75),  # cut
    ("2025-03-06",   2.50),  # cut
    ("2025-04-17",   2.25),  # cut
    ("2025-06-05",   2.00),  # cut — current level
]


def _ecb_rate_on(date_str: str) -> float | None:
    """Look up ECB Deposit Facility Rate on (or just before) the given date."""
    target = date_str
    rate = None
    for effective_date, r in _ECB_RATE_CHANGES:
        if effective_date <= target:
            rate = r
        else:
            break
    return rate


def _ecb_decisions(entries: list[dict]) -> list[dict]:
    """Compute before/after rates for every ECB meeting from the lookup."""
    results = []
    for entry in entries:
        meeting = entry["date"]
        before_date = (pd.Timestamp(meeting) - timedelta(days=1)).strftime("%Y-%m-%d")
        after_date  = (pd.Timestamp(meeting) + timedelta(days=1)).strftime("%Y-%m-%d")

        rate_before = _ecb_rate_on(before_date)
        rate_after  = _ecb_rate_on(after_date)

        if rate_before is not None and rate_after is not None:
            change_bps = round((rate_after - rate_before) * 100, 1)
        else:
            change_bps = None

        results.append({
            "date":        meeting,
            "score":       entry.get("score"),
            "rate_before": rate_before,
            "rate_after":  rate_after,
            "change_bps":  change_bps,
        })
        label = (
            f"  {meeting}: {rate_before}% → {rate_after}% ({change_bps:+.0f} bps)"
            if change_bps is not None
            else f"  {meeting}: no data"
        )
        logger.info(label)
    return results

def _fed_rate_on(date_str: str) -> float | None:
    """Look up Fed Funds Target Rate (upper) on or just before the given date."""
    rate = None
    for effective_date, r in _FED_RATE_CHANGES:
        if effective_date <= date_str:
            rate = r
        else:
            break
    return rate


def _fed_decisions(entries: list[dict]) -> list[dict]:
    """Compute before/after rates for every Fed meeting from the lookup."""
    results = []
    for entry in entries:
        meeting = entry["date"]
        before_date = (pd.Timestamp(meeting) - timedelta(days=1)).strftime("%Y-%m-%d")
        after_date  = (pd.Timestamp(meeting) + timedelta(days=1)).strftime("%Y-%m-%d")

        rate_before = _fed_rate_on(before_date)
        rate_after  = _fed_rate_on(after_date)

        if rate_before is not None and rate_after is not None:
            change_bps = round((rate_after - rate_before) * 100, 1)
        else:
            change_bps = None

        results.append({
            "date":        meeting,
            "score":       entry.get("score"),
            "rate_before": rate_before,
            "rate_after":  rate_after,
            "change_bps":  change_bps,
        })
        label = (
            f"  {meeting}: {rate_before}% → {rate_after}% ({change_bps:+.0f} bps)"
            if change_bps is not None
            else f"  {meeting}: no data"
        )
        logger.info(label)
    return results


# ── BoE Bank Rate history ─────────────────────────────────────────────────────
# Effective date → new rate.  Source: bankofengland.co.uk/boeapps/database/
# Only need entries where the rate *changed*; the lookup forward-fills.

_BOE_RATE_CHANGES = [
    ("2016-01-01", 0.50),   # rate at the start of our window
    ("2016-08-04", 0.25),   # cut — post-Brexit referendum
    ("2017-11-02", 0.50),   # hike
    ("2018-08-02", 0.75),   # hike
    ("2020-03-11", 0.25),   # emergency cut (COVID)
    ("2020-03-19", 0.10),   # second emergency cut
    ("2021-12-16", 0.25),   # hike — start of tightening cycle
    ("2022-02-03", 0.50),
    ("2022-03-17", 0.75),
    ("2022-05-05", 1.00),
    ("2022-06-16", 1.25),
    ("2022-08-04", 1.75),
    ("2022-09-22", 2.25),
    ("2022-11-03", 3.00),
    ("2022-12-15", 3.50),
    ("2023-02-02", 4.00),
    ("2023-03-23", 4.25),
    ("2023-05-11", 4.50),
    ("2023-06-22", 5.00),
    ("2023-08-03", 5.25),   # peak
    ("2024-08-01", 5.00),   # first cut
    ("2024-11-07", 4.75),
    ("2025-02-06", 4.50),
    ("2025-05-08", 4.25),
    ("2025-08-01", 4.00),   # cut
    ("2025-12-01", 3.75),   # cut
]


def _boe_rate_on(date_str: str) -> float | None:
    """Look up Bank Rate on (or just before) the given date."""
    target = date_str
    rate = None
    for effective_date, r in _BOE_RATE_CHANGES:
        if effective_date <= target:
            rate = r
        else:
            break
    return rate


def _boe_decisions(entries: list[dict]) -> list[dict]:
    """Compute before/after rates for every BoE meeting from the lookup."""
    results = []
    for entry in entries:
        meeting = entry["date"]
        # "before" = rate the day before the meeting
        before_date = (pd.Timestamp(meeting) - timedelta(days=1)).strftime("%Y-%m-%d")
        # "after"  = rate a few days after (captures same-day decision)
        after_date  = (pd.Timestamp(meeting) + timedelta(days=5)).strftime("%Y-%m-%d")

        rate_before = _boe_rate_on(before_date)
        rate_after  = _boe_rate_on(after_date)

        if rate_before is not None and rate_after is not None:
            change_bps = round((rate_after - rate_before) * 100, 1)
        else:
            change_bps = None

        results.append({
            "date":        meeting,
            "score":       entry.get("score"),
            "rate_before": rate_before,
            "rate_after":  rate_after,
            "change_bps":  change_bps,
        })
        label = (
            f"  {meeting}: {rate_before}% → {rate_after}% ({change_bps:+.0f} bps)"
            if change_bps is not None
            else f"  {meeting}: no data"
        )
        logger.info(label)
    return results


# ── Main builder ──────────────────────────────────────────────────────────────

def build_decisions() -> dict:
    """Build rate decisions for all three banks using hardcoded lookups."""
    with open(HISTORY) as f:
        history = json.load(f)

    decisions: dict[str, list[dict]] = {}

    logger.info("Computing BOE rate decisions from lookup table...")
    decisions["BOE"] = _boe_decisions(history["banks"].get("BOE", []))

    logger.info("Computing FED rate decisions from lookup table...")
    decisions["FED"] = _fed_decisions(history["banks"].get("FED", []))

    logger.info("Computing ECB rate decisions from lookup table...")
    decisions["ECB"] = _ecb_decisions(history["banks"].get("ECB", []))

    if not any(decisions.values()):
        logger.error(
            "No decisions computed for any bank — leaving existing "
            f"{OUTPUT.name} untouched."
        )
        return {}

    output = {
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "BOE": "Hardcoded lookup (bankofengland.co.uk)",
            "FED": "Hardcoded lookup (federalreserve.gov)",
            "ECB": "Hardcoded lookup (ecb.europa.eu/stats)",
        },
        "banks": decisions,
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote → {OUTPUT}")
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build_decisions()
