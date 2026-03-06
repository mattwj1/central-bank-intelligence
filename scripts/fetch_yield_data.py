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

UK_YIELD_SERIES = {}


EU_YIELD_SERIES = {}

# yfinance fallback tickers if FRED fails
YFINANCE_FALLBACK = {
    "US": {
        "3M":  "^IRX",
        "5Y":  "^FVX",
        "10Y": "^TNX",
        "30Y": "^TYX",
    },
    "UK": {
        # Intentionally empty: common Yahoo UK gilt symbols are unreliable/non-existent.
    },
    "EU": {
        "2Y":  "^DE2YY",
        "5Y":  "^DE5YY",
        "10Y": "^DE10YY",
        "30Y": "^DE30YY",
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
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        data = data.dropna()
        if data.empty:
            return None
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

def fetch_uk_yields_boe() -> dict[str, pd.Series]:
    """
    Fetch UK nominal zero-coupon yields from BoE Database CSV endpoint.

    Notes:
    - BoE publishes 5Y/10Y/20Y nominal zero-coupon series directly.
    - We map 20Y into the 30Y slot as a long-end proxy when 30Y is unavailable.
    """
    import requests
    from io import StringIO

    boe_series = {
        "5Y": "IUMASNZC",   # Monthly average, 5Y nominal zero coupon
        "10Y": "IUMAMNZC",  # Monthly average, 10Y nominal zero coupon
        "20Y": "IUMALNZC",  # Monthly average, 20Y nominal zero coupon
    }

    logger.info("\nFetching UK yields via BoE Database CSV API...")
    results: dict[str, pd.Series] = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for tenor, code in boe_series.items():
        url = (
            "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
            f"?csv.x=yes&Datefrom=01/Jan/2015&Dateto=now&SeriesCodes={code}&UsingCodes=Y&VPD=Y&VFD=N"
        )
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()

            df = pd.read_csv(StringIO(r.text))
            if df.empty:
                logger.warning(f"  ? UK {tenor} (BoE {code}): empty CSV")
                continue

            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            value_col = code if code in df.columns else next(
                (c for c in df.columns if c != date_col and "notes" not in c.lower()),
                None,
            )
            if not date_col or not value_col:
                logger.warning(f"  ? UK {tenor} (BoE {code}): unexpected CSV columns {list(df.columns)}")
                continue

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            series = df.dropna(subset=[date_col, value_col]).set_index(date_col)[value_col].sort_index()

            if series.empty:
                logger.warning(f"  ? UK {tenor} (BoE {code}): empty after parse")
                continue

            results[tenor] = series
            logger.info(
                f"  ? UK {tenor} (BoE {code}): {len(series)} obs, latest = {series.iloc[-1]:.3f}%"
            )

        except Exception as e:
            logger.warning(f"  ? UK {tenor} (BoE {code}): {e}")

    # Populate 30Y slot with 20Y proxy if no direct 30Y series is available.
    if "20Y" in results and "30Y" not in results:
        results["30Y"] = results["20Y"].copy()
        logger.info("  ? UK 30Y populated using BoE 20Y nominal zero-coupon proxy")

    if "20Y" in results:
        results.pop("20Y", None)

    logger.info(f"  UK (BoE API): {len(results)}/4 tenors fetched")
    return results


def interpolate_uk_2y(uk_series: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """
    Interpolate UK 2Y yield from BoE 5Y and short-end data.

    BoE doesn't publish a 2Y zero-coupon series directly.
    We estimate it by linear interpolation between the policy rate
    (treated as the 0Y anchor) and the 5Y spot rate.

    This is a proxy — flagged as such in methodology docs.
    """
    if "5Y" not in uk_series:
        logger.warning("  Cannot interpolate UK 2Y — 5Y series unavailable")
        return uk_series

    # Policy rate as short-end anchor (0Y proxy)
    # 4.5% as of March 2026 — update after each MPC meeting
    POLICY_RATE = 4.5

    five_year = uk_series["5Y"]

    # Linear interpolation: 2Y is 2/5 of the way from 0Y to 5Y
    two_year = POLICY_RATE + (2 / 5) * (five_year - POLICY_RATE)
    two_year.name = "2Y_interpolated"

    uk_series["2Y"] = two_year
    logger.info(
        f"  ✓ UK 2Y (interpolated from policy rate + BoE 5Y): "
        f"latest = {two_year.dropna().iloc[-1]:.3f}%"
    )
    return uk_series

def fetch_eu_yields_ecb() -> dict[str, pd.Series]:
    """
    Fetch AAA euro area bond yields from ECB Statistical Data Warehouse.
    Uses the correct SDMX REST API format.
    """
    import requests
    from io import StringIO

    # Correct ECB SDW format: dataflow/agency/id/version
    # YC = yield curves, B.U2 = euro area, spot rates
    ECB_SERIES = {
        "2Y":  "SR_2Y",
        "5Y":  "SR_5Y",
        "10Y": "SR_10Y",
        "30Y": "SR_30Y",
    }

    logger.info("\nFetching EU yields via ECB API...")
    results = {}

    headers = {"Accept": "text/csv"}

    for tenor, maturity_code in ECB_SERIES.items():
        try:
            # ECB SDMX 2.1 REST API — correct URL structure
            url = (
                f"https://data-api.ecb.europa.eu/service/data/"
                f"ECB,YC,1.0/B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity_code}"
                f"?startPeriod=2015-01-01&format=csvdata"
            )
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()

            df = pd.read_csv(StringIO(r.text))

            # Find time and value columns (names vary slightly)
            time_col  = next(c for c in df.columns if "TIME" in c.upper())
            value_col = next(c for c in df.columns if "OBS_VALUE" in c.upper())

            df[time_col]  = pd.to_datetime(df[time_col], errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df = df.dropna(subset=[time_col, value_col])
            series = df.set_index(time_col)[value_col].sort_index()

            if not series.empty:
                logger.info(f"  ✓ EU {tenor} (ECB): {len(series)} obs, latest = {series.iloc[-1]:.3f}%")
                results[tenor] = series
            else:
                logger.warning(f"  ✗ EU {tenor}: empty after parsing")

        except Exception as e:
            logger.warning(f"  ✗ EU {tenor} (ECB): {e}")

    logger.info(f"  EU: {len(results)}/4 tenors fetched")
    return results

def main():
    # US via FRED (most reliable for Treasuries)
    us_series = fetch_country_yields("US", US_YIELD_SERIES, YFINANCE_FALLBACK["US"])

    # UK from FRED baseline, then BoE API enrichment for fuller curve
    uk_series = fetch_uk_yields_boe()
    uk_series = interpolate_uk_2y(uk_series)

    eu_series = fetch_eu_yields_ecb()

    all_series = {"US": us_series, "UK": uk_series, "EU": eu_series}

    save_raw_yields(all_series)

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

    logger.info("\n── Current Yield Curves ─────────────────────────────")
    for country, data in processed["countries"].items():
        yields = data["yields"]
        metrics = data["curve_metrics"]
        logger.info(f"\n{country}:")
        for tenor in ["3M", "2Y", "5Y", "10Y", "30Y"]:
            if tenor in yields:
                logger.info(f"  {tenor:>3}: {yields[tenor]:.3f}%")
        if "slope_10y_2y" in metrics:
            inverted = " ⚠ INVERTED" if metrics.get("inverted") else ""
            logger.info(f"  Slope (10Y-2Y): {metrics['slope_10y_2y']:+.3f}%{inverted}")

if __name__ == "__main__":
    main()


