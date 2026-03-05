"""
fetch_macro_data.py
-------------------
Pulls macroeconomic data required for Taylor Rule calculations.

Data sources:
- FRED API: US CPI, GDP, unemployment, policy rate
- ONS API: UK CPI, GDP (no key required)

Outputs:
- data/raw/macro_data_YYYY-MM-DD.csv
- data/processed/macro_latest.json
"""

import os
import json
import logging
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
from fredapi import Fred
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in .env file")

fred = Fred(api_key=FRED_API_KEY)

# Project root — works regardless of where you run the script from
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── FRED Series IDs ────────────────────────────────────────────────────────────

FRED_SERIES = {
    # United States
    "us_cpi":          "CPIAUCSL",   # CPI All Urban Consumers (YoY calculated below)
    "us_core_cpi":     "CPILFESL",   # Core CPI (ex food & energy)
    "us_gdp":          "GDPC1",      # Real GDP (quarterly)
    "us_unemployment": "UNRATE",     # Unemployment rate
    "us_policy_rate":  "FEDFUNDS",   # Federal Funds Rate (actual policy rate)
    "us_potential_gdp":"GDPPOT",     # CBO Potential GDP (for output gap)

    # United Kingdom
    "uk_cpi":          "GBRCPIALLMINMEI",  # UK CPI
    "uk_gdp":          "CLVMNACSCAB1GQUK", # UK Real GDP (quarterly)
    "uk_unemployment": "LMUNRRTTGBM156S",  # UK unemployment

    # Euro Area
    "eu_cpi":          "CP0000EZ19M086NEST", # Euro area CPI
    "eu_policy_rate":  "ECBDFR",             # ECB deposit facility rate
}

# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_series(series_id: str, series_name: str) -> pd.Series | None:
    """Fetch a single FRED series with error handling."""
    try:
        data = fred.get_series(series_id, observation_start="2015-01-01")
        logger.info(f"  ✓ {series_name} ({series_id}): {len(data)} observations")
        return data
    except Exception as e:
        logger.warning(f"  ✗ {series_name} ({series_id}): {e}")
        return None


def fetch_all_series() -> dict[str, pd.Series]:
    """Fetch all configured FRED series."""
    logger.info("Fetching macroeconomic data from FRED...")
    results = {}
    for name, series_id in FRED_SERIES.items():
        series = fetch_series(series_id, name)
        if series is not None:
            results[name] = series
    logger.info(f"Successfully fetched {len(results)}/{len(FRED_SERIES)} series")
    return results

# In FRED_SERIES dict, remove this line:
# "uk_policy_rate":  "BOERUKQ",

# Add this new function below fetch_all_series():

def fetch_uk_policy_rate() -> float:
    """
    Fetch current Bank Rate from BoE directly.
    Falls back to FRED if BoE API unavailable.
    """
    import requests

    # BoE Statistical Interactive Dataset API
    url = (
        "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp"
        "?Travel=NIxSUx&FromSeries=1&ToSeries=50&DAT=RNG"
        "&FD=1&FM=Jan&FY=2024&TD=1&TM=Mar&TY=2026"
        "&VFD=Y&html.x=66&html.y=26&C=BYB&Filter=N"
    )

    try:
        # Simpler approach: use FRED series UKBRBASE which is more current
        boe_rate = fred.get_series("UKBRBASE").dropna().iloc[-1]
        logger.info(f"  ✓ uk_policy_rate (UKBRBASE): {boe_rate}")
        return round(float(boe_rate), 3)
    except Exception as e:
        logger.warning(f"  ✗ uk_policy_rate fallback failed: {e}")
        return 4.5  # Hardcode current known rate as last resort
    # Current BoE Bank Rate as of March 2026 — update manually after each MPC meeting

# ── Data Processing ────────────────────────────────────────────────────────────

def calculate_yoy_inflation(cpi_series: pd.Series) -> pd.Series:
    """Calculate year-over-year inflation rate from CPI index."""
    return cpi_series.pct_change(periods=12) * 100


def calculate_output_gap(gdp: pd.Series, potential_gdp: pd.Series | None) -> pd.Series:
    """
    Calculate output gap as % deviation from potential GDP.

    If potential GDP is available (US only via CBO estimate), use that.
    Otherwise, use HP filter trend as proxy for potential.

    Output gap > 0: economy running above potential (inflationary pressure)
    Output gap < 0: economy running below potential (slack)
    """
    if potential_gdp is not None:
        # Align the two series on common dates
        aligned = pd.concat([gdp, potential_gdp], axis=1).dropna()
        aligned.columns = ["gdp", "potential"]
        gap = ((aligned["gdp"] - aligned["potential"]) / aligned["potential"]) * 100
        return gap
    else:
        # Simple trend deviation: % gap from 4-quarter rolling average
        # This is a crude proxy — fine for now, can upgrade to HP filter later
        trend = gdp.rolling(window=8, min_periods=4).mean()
        gap = ((gdp - trend) / trend) * 100
        return gap


def get_latest_value(series: pd.Series, lag_quarters: int = 0) -> float | None:
    """Get the most recent non-NaN value from a series."""
    clean = series.dropna()
    if clean.empty:
        return None
    if lag_quarters > 0:
        # GDP data is released with a lag — step back if needed
        idx = max(0, len(clean) - 1 - lag_quarters)
        return round(float(clean.iloc[idx]), 3)
    return round(float(clean.iloc[-1]), 3)


# ── Main Processing ────────────────────────────────────────────────────────────

def process_macro_data(raw_data: dict[str, pd.Series]) -> dict:
    """
    Transform raw FRED series into processed metrics for Taylor Rule.

    Returns a dict with the latest values for each country.
    """
    processed = {
        "last_updated": datetime.now().isoformat(),
        "data_date": date.today().isoformat(),
        "countries": {}
    }

    # ── United States ──────────────────────────────────────────────────────────
    if "us_cpi" in raw_data:
        us_inflation = calculate_yoy_inflation(raw_data["us_cpi"])
        us_output_gap = calculate_output_gap(
            raw_data.get("us_gdp"),
            raw_data.get("us_potential_gdp")
        ) if "us_gdp" in raw_data else None

        processed["countries"]["US"] = {
            "inflation": get_latest_value(us_inflation),
            "core_inflation": get_latest_value(
                calculate_yoy_inflation(raw_data["us_core_cpi"])
            ) if "us_core_cpi" in raw_data else None,
            "unemployment": get_latest_value(raw_data.get("us_unemployment", pd.Series())),
            "policy_rate": get_latest_value(raw_data.get("us_policy_rate", pd.Series())),
            "output_gap": get_latest_value(us_output_gap) if us_output_gap is not None else None,
            "inflation_target": 2.0,
            "r_neutral": 2.5,
        }

    # ── United Kingdom ─────────────────────────────────────────────────────────
    if "uk_cpi" in raw_data:
        uk_inflation = calculate_yoy_inflation(raw_data["uk_cpi"])
        uk_output_gap = calculate_output_gap(
            raw_data["uk_gdp"], None
        ) if "uk_gdp" in raw_data else None

        processed["countries"]["UK"] = {
            "inflation": get_latest_value(uk_inflation),
            "unemployment": get_latest_value(raw_data.get("uk_unemployment", pd.Series())),
            "policy_rate": fetch_uk_policy_rate(),
            "output_gap": get_latest_value(uk_output_gap) if uk_output_gap is not None else None,
            "inflation_target": 2.0,
            "r_neutral": 2.5,
        }

    # ── Euro Area ──────────────────────────────────────────────────────────────
    if "eu_cpi" in raw_data:
        eu_inflation = calculate_yoy_inflation(raw_data["eu_cpi"])

        processed["countries"]["EU"] = {
            "inflation": get_latest_value(eu_inflation),
            "policy_rate": get_latest_value(raw_data.get("eu_policy_rate", pd.Series())),
            "output_gap": None,  # Add EU GDP series later if needed
            "inflation_target": 2.0,
            "r_neutral": 2.0,  # ECB neutral rate typically estimated lower
        }

    return processed


def save_raw_data(raw_data: dict[str, pd.Series]) -> Path:
    """Save all raw series to a single timestamped CSV."""
    today = date.today().isoformat()
    filepath = RAW_DIR / f"macro_data_{today}.csv"

    # Combine all series into one DataFrame (outer join — keeps all dates)
    df = pd.DataFrame(raw_data)
    df.index.name = "date"
    df.to_csv(filepath)

    logger.info(f"Raw data saved → {filepath}")
    return filepath


def save_processed_data(processed: dict) -> Path:
    """Save processed metrics to JSON for dashboard consumption."""
    filepath = PROCESSED_DIR / "macro_latest.json"
    with open(filepath, "w") as f:
        json.dump(processed, f, indent=2)

    logger.info(f"Processed data saved → {filepath}")
    return filepath


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    # 1. Fetch
    raw_data = fetch_all_series()

    if not raw_data:
        logger.error("No data fetched — check your FRED API key and internet connection")
        return

    # 2. Save raw
    save_raw_data(raw_data)

    # 3. Process
    processed = process_macro_data(raw_data)

    # 4. Save processed
    save_processed_data(processed)

    # 5. Print summary so you can see it worked
    logger.info("\n── Latest Values ──────────────────────────────")
    for country, metrics in processed["countries"].items():
        logger.info(f"\n{country}:")
        for key, value in metrics.items():
            if key not in ("inflation_target", "r_neutral"):
                logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()