"""
fetch_yield_data.py
-------------------
Pulls government bond yield curve data for UK, US, and EU.

Data sources:
- FRED API: US Treasuries (most reliable), UK Gilts, EU Bunds
- yfinance: fallback for any series FRED doesn't have

Tenors captured: 3M, 2Y, 5Y, 10Y, 30Y
These four points are sufficient to characterise:
  - Level  (10Y yield)
  - Slope  (10Y minus 2Y — the classic recession indicator)
  - Curvature (2*5Y minus 2Y minus 10Y)

Outputs:
- data/raw/yields_YYYY-MM-DD.csv        (full history from 2015)
- data/processed/yields_latest.json     (latest curve + volatility)
"""

import os
import json
import logging
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf
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

ROOT         = Path(__file__).resolve().parent.parent
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

START_DATE = "2015-01-01"

# ── FRED Series for each country ───────────────────────────────────────────────

# Tenors in years → FRED series ID
US_YIELD_SERIES = {
    "3M":  "DTB3",    # 3-Month Treasury Bill
    "2Y":  "DGS2",    # 2-Year Treasury
    "5Y":  "DGS5",    # 5-Year Treasury
    "10Y": "DGS10",   # 10-Year Treasury
    "30Y": "DGS30",   # 30-Year Treasury
}

UK_YIELD_SERIES = {
    "2Y":  "IRLTST01GBM156N",   # UK 2Y Gilt
    "5Y":  "IRLTST05GBM156N",   # UK 5Y Gilt  (may need fallback)
    "10Y": "IRLTLT01GBM156N",   # UK 10Y Gilt
    "30Y": "GBGBOND30YD",       # UK 30Y Gilt
}

EU_YIELD_SERIES = {
    "2Y":  "IRLTST01EZM156N",   # EA 2Y Bund
    "10Y": "IRLTLT01EZM156N",   # EA 10Y Bund
}

# yfinance fallback tickers if FRED fails
YFINANCE_FALLBACK = {
    "US": {
        "3M":  "^IRX",
        "5Y":  "^FVX",
        "10Y": "^TNX",
        "30Y": "^TYX",
    }
}

# ── Fetching ───────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str, name: str) -> pd.Series | None:
    """Fetch a single FRED yield series."""
    try:
        data = fred.get_series(series_id, observation_start=START_DATE)
        data = data.dropna()
        logger.info(f"  ✓ {name} ({series_id}): {len(data)} observations, "
                    f"latest = {data.iloc[-1]:.3f}%")
        return data
    except Exception as e:
        logger.warning(f"  ✗ {name} ({series_id}): {e}")
        return None


def fetch_yfinance_series(ticker: str, name: str) -> pd.Series | None:
    """Fetch yield data from yfinance as fallback."""
    try:
        data = yf.download(ticker, start=START_DATE, progress=False)["Close"]
        data = data.dropna()
        # yfinance returns yields as percentages already for these tickers
        logger.info(f"  ✓ {name} ({ticker}) via yfinance: {len(data)} observations, "
                    f"latest = {float(data.iloc[-1]):.3f}%")
        return data
    except Exception as e:
        logger.warning(f"  ✗ {name} ({ticker}) via yfinance: {e}")
        return None


def fetch_country_yields(
    country: str,
    fred_series: dict[str, str],
    yf_fallback: dict[str, str] | None = None
) -> dict[str, pd.Series]:
    """
    Fetch all tenors for a single country.
    Falls back to yfinance if FRED series fails.
    """
    logger.info(f"\nFetching {country} yields...")
    results = {}

    for tenor, series_id in fred_series.items():
        series = fetch_fred_series(series_id, f"{country} {tenor}")

        # Try yfinance fallback if FRED failed
        if series is None and yf_fallback and tenor in yf_fallback:
            logger.info(f"  Trying yfinance fallback for {country} {tenor}...")
            series = fetch_yfinance_series(yf_fallback[tenor], f"{country} {tenor}")

        if series is not None:
            results[tenor] = series

    logger.info(f"  {country}: {len(results)}/{len(fred_series)} tenors fetched")
    return results


# ── Volatility ─────────────────────────────────────────────────────────────────

def calculate_historical_volatility(
    series: pd.Series,
    window_days: int = 90
) -> float | None:
    """
    Calculate annualised rolling volatility of yield changes.

    Uses daily changes (not log returns, since yields can go negative).
    Annualised by multiplying by sqrt(252).

    This feeds into the confidence bands on the yield curve chart.
    """
    if series is None or len(series) < window_days:
        return None

    # Daily changes in yield (in percentage points)
    daily_changes = series.diff().dropna()
    rolling_vol = daily_changes.rolling(window=window_days).std()
    latest_vol = rolling_vol.dropna().iloc[-1]

    # Annualise
    annualised_vol = float(latest_vol) * np.sqrt(252)
    return round(annualised_vol, 4)


# ── Processing ─────────────────────────────────────────────────────────────────

def get_latest_yields(tenor_series: dict[str, pd.Series]) -> dict[str, float]:
    """Extract the most recent yield for each tenor."""
    latest = {}
    for tenor, series in tenor_series.items():
        if series is not None and not series.empty:
            latest[tenor] = round(float(series.dropna().iloc[-1]), 3)
    return latest


def calculate_curve_metrics(yields: dict[str, float]) -> dict:
    """
    Calculate summary metrics that characterise the yield curve shape.

    Slope:     10Y minus 2Y (positive = normal, negative = inverted)
    Curvature: 2*5Y minus 2Y minus 10Y (positive = humped middle)
    Level:     10Y yield as proxy for overall rate level
    """
    metrics = {}

    if "10Y" in yields and "2Y" in yields:
        metrics["slope_10y_2y"] = round(yields["10Y"] - yields["2Y"], 3)
        metrics["inverted"] = metrics["slope_10y_2y"] < 0

    if "5Y" in yields and "10Y" in yields and "2Y" in yields:
        metrics["curvature"] = round(
            2 * yields["5Y"] - yields["2Y"] - yields["10Y"], 3
        )

    if "10Y" in yields:
        metrics["level_10y"] = yields["10Y"]

    if "3M" in yields and "10Y" in yields:
        metrics["slope_10y_3m"] = round(yields["10Y"] - yields["3M"], 3)

    return metrics


def process_country_yields(
    country: str,
    tenor_series: dict[str, pd.Series]
) -> dict:
    """Package latest yields, metrics, and volatility for one country."""
    latest = get_latest_yields(tenor_series)
    metrics = calculate_curve_metrics(latest)

    volatility = {}
    for tenor, series in tenor_series.items():
        vol = calculate_historical_volatility(series)
        if vol is not None:
            volatility[tenor] = vol

    return {
        "yields": latest,
        "curve_metrics": metrics,
        "volatility_annualised": volatility,
    }


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_raw_yields(all_series: dict[str, dict[str, pd.Series]]) -> Path:
    """
    Combine all yield series into a single flat CSV.
    Column names formatted as COUNTRY_TENOR (e.g. US_10Y).
    """
    frames = {}
    for country, tenor_dict in all_series.items():
        for tenor, series in tenor_dict.items():
            frames[f"{country}_{tenor}"] = series

    if not frames:
        logger.warning("No yield data to save")
        return None

    df = pd.DataFrame(frames)
    df.index.name = "date"

    today = date.today().isoformat()
    filepath = RAW_DIR / f"yields_{today}.csv"
    df.to_csv(filepath)
    logger.info(f"\nRaw yields saved → {filepath}")
    return filepath


def save_processed_yields(processed: dict) -> Path:
    """Save processed yield data for dashboard consumption."""
    filepath = PROCESSED_DIR / "yields_latest.json"
    with open(filepath, "w") as f:
        json.dump(processed, f, indent=2)
    logger.info(f"Processed yields saved → {filepath}")
    return filepath


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Fetch all countries
    us_series = fetch_country_yields("US", US_YIELD_SERIES, YFINANCE_FALLBACK["US"])
    uk_series = fetch_country_yields("UK", UK_YIELD_SERIES)
    eu_series = fetch_country_yields("EU", EU_YIELD_SERIES)

    all_series = {
        "US": us_series,
        "UK": uk_series,
        "EU": eu_series,
    }

    # Save raw
    save_raw_yields(all_series)

    # Process and save
    processed = {
        "last_updated": datetime.now().isoformat(),
        "data_date": date.today().isoformat(),
        "countries": {}
    }

    for country, series_dict in all_series.items():
        if series_dict:
            processed["countries"][country] = process_country_yields(
                country, series_dict
            )

    save_processed_yields(processed)

    # Print summary
    logger.info("\n── Current Yield Curves ─────────────────────────────")
    for country, data in processed["countries"].items():
        yields = data["yields"]
        metrics = data["curve_metrics"]
        logger.info(f"\n{country}:")
        for tenor, rate in sorted(yields.items()):
            logger.info(f"  {tenor:>3}: {rate:.3f}%")
        if "slope_10y_2y" in metrics:
            inverted = " ⚠ INVERTED" if metrics.get("inverted") else ""
            logger.info(f"  Slope (10Y-2Y): {metrics['slope_10y_2y']:+.3f}%{inverted}")


if __name__ == "__main__":
    main()