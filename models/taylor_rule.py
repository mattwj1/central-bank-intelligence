"""
taylor_rule.py
--------------
Implements the Taylor Rule for calculating implied policy rates.

The Taylor Rule (Taylor, 1993) provides a systematic framework for
determining appropriate interest rate levels based on inflation and
output conditions.

Formula: r = r* + π + 0.5(π - π*) + 0.5(y - y*)

Where:
    r  = implied policy rate
    r* = neutral real interest rate
    π  = current inflation rate
    π* = inflation target
    y - y* = output gap (% deviation from potential GDP)

Citation: Taylor, John B. (1993). "Discretion versus policy rules in
practice." Carnegie-Rochester Conference Series on Public Policy, 39,
195-214.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class TaylorRuleInputs:
    """All inputs required for a Taylor Rule calculation."""
    country: str
    inflation: float
    inflation_target: float
    output_gap: float
    r_neutral: float
    actual_policy_rate: float


@dataclass
class TaylorRuleResult:
    """Output from a Taylor Rule calculation."""
    country: str
    implied_rate: float
    actual_rate: float
    gap_bps: int                  # actual minus implied, in basis points
    policy_stance: str            # "too tight", "too loose", "approximately neutral"
    interpretation: str           # plain-English summary
    inputs: TaylorRuleInputs


# ── Core Model ─────────────────────────────────────────────────────────────────

def calculate_taylor_rate(
    r_neutral: float,
    inflation: float,
    inflation_target: float,
    output_gap: float,
) -> float:
    """
    Calculate the Taylor Rule implied policy rate.

    Args:
        r_neutral: Neutral real interest rate (%)
        inflation: Current inflation rate (%)
        inflation_target: Central bank inflation target (%)
        output_gap: Output gap as % of potential GDP

    Returns:
        Implied policy rate (%)

    Note: Output gap of None is handled by the caller — this function
    requires all four inputs to be valid floats.
    """
    implied = (
        r_neutral
        + inflation
        + 0.5 * (inflation - inflation_target)
        + 0.5 * output_gap
    )
    return round(implied, 2)


def calculate_policy_gap(actual_rate: float, implied_rate: float) -> int:
    """
    Calculate deviation of actual policy from Taylor Rule implied rate.

    Positive = policy too tight (actual rate above model)
    Negative = policy too loose (actual rate below model)

    Returns gap in basis points (1% = 100bps).
    """
    gap_pct = actual_rate - implied_rate
    return round(gap_pct * 100)


def classify_policy_stance(gap_bps: int) -> str:
    """
    Classify policy stance based on deviation from Taylor Rule.

    Thresholds are judgement calls — ±50bps is a reasonable
    'approximately neutral' band for developed market central banks.
    """
    if gap_bps > 100:
        return "significantly too tight"
    elif gap_bps > 50:
        return "moderately too tight"
    elif gap_bps >= -50:
        return "approximately neutral"
    elif gap_bps >= -100:
        return "moderately too loose"
    else:
        return "significantly too loose"


def generate_interpretation(result_data: dict) -> str:
    """
    Generate a plain-English interpretation of the Taylor Rule output.
    Used in the dashboard and LinkedIn posts.
    """
    country = result_data["country"]
    gap = result_data["gap_bps"]
    implied = result_data["implied_rate"]
    actual = result_data["actual_rate"]
    stance = result_data["policy_stance"]

    direction = "above" if gap > 0 else "below"
    abs_gap = abs(gap)

    return (
        f"{country} policy rate ({actual}%) is {abs_gap}bps {direction} "
        f"the Taylor Rule implied rate ({implied}%). "
        f"Policy is {stance}."
    )


# ── Country Runner ─────────────────────────────────────────────────────────────

def analyse_country(country: str, metrics: dict) -> dict | None:
    """
    Run full Taylor Rule analysis for a single country.

    Handles missing output gap gracefully — uses 0.0 as neutral
    assumption and flags it in the interpretation.
    """
    inflation = metrics.get("inflation")
    actual_rate = metrics.get("policy_rate")

    # Minimum data requirements
    if inflation is None or actual_rate is None:
        logger.warning(f"{country}: missing inflation or policy rate — skipping")
        return None

    output_gap = metrics.get("output_gap")
    gap_assumed = False

    if output_gap is None:
        logger.warning(f"{country}: output gap unavailable — assuming 0.0 (neutral)")
        output_gap = 0.0
        gap_assumed = True

    implied_rate = calculate_taylor_rate(
        r_neutral=metrics["r_neutral"],
        inflation=inflation,
        inflation_target=metrics["inflation_target"],
        output_gap=output_gap,
    )

    gap_bps = calculate_policy_gap(actual_rate, implied_rate)
    stance = classify_policy_stance(gap_bps)

    result = {
        "country": country,
        "implied_rate": implied_rate,
        "actual_rate": actual_rate,
        "gap_bps": gap_bps,
        "policy_stance": stance,
        "inputs": {
            "inflation": inflation,
            "inflation_target": metrics["inflation_target"],
            "output_gap": output_gap,
            "output_gap_assumed": gap_assumed,
            "r_neutral": metrics["r_neutral"],
        },
    }

    result["interpretation"] = generate_interpretation(result)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run_taylor_rule_analysis() -> dict:
    """
    Load latest macro data and run Taylor Rule for all countries.
    Saves results to data/processed/taylor_rule_latest.json.
    """
    # Load macro data
    macro_path = PROCESSED_DIR / "macro_latest.json"
    if not macro_path.exists():
        raise FileNotFoundError(
            "macro_latest.json not found — run fetch_macro_data.py first"
        )

    with open(macro_path) as f:
        macro_data = json.load(f)

    results = {
        "last_updated": macro_data["last_updated"],
        "data_date": macro_data["data_date"],
        "countries": {}
    }

    # Run analysis for each country
    for country, metrics in macro_data["countries"].items():
        logger.info(f"Analysing {country}...")
        result = analyse_country(country, metrics)
        if result:
            results["countries"][country] = result
            logger.info(f"  → {result['interpretation']}")

    # Save results
    output_path = PROCESSED_DIR / "taylor_rule_latest.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved → {output_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_taylor_rule_analysis()