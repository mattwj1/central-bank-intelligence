"""
test_taylor_rule.py
-------------------
Unit tests for the Taylor Rule model.

Tests verify:
1. Core formula produces correct outputs for known inputs
2. Edge cases are handled gracefully
3. Policy stance classification is correct
4. Interpretation text is generated correctly

Run with: pytest tests/test_taylor_rule.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path so we can import models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.taylor_rule import (
    calculate_taylor_rate,
    calculate_policy_gap,
    classify_policy_stance,
    generate_interpretation,
    analyse_country,
)


# ── Core Formula Tests ─────────────────────────────────────────────────────────

class TestCalculateTaylorRate:

    def test_neutral_conditions(self):
        """
        When inflation = target and output gap = 0,
        implied rate should equal r_neutral + inflation_target.
        i.e. r = 2.5 + 2.0 + 0.5(0) + 0.5(0) = 4.5%
        """
        result = calculate_taylor_rate(
            r_neutral=2.5,
            inflation=2.0,
            inflation_target=2.0,
            output_gap=0.0,
        )
        assert result == 4.5, f"Expected 4.5, got {result}"

    def test_above_target_inflation(self):
        """
        Inflation 2% above target should add 0.5 * 2 = 1% to implied rate.
        r = 2.5 + 4.0 + 0.5(2.0) + 0.5(0) = 7.5%
        """
        result = calculate_taylor_rate(
            r_neutral=2.5,
            inflation=4.0,
            inflation_target=2.0,
            output_gap=0.0,
        )
        assert result == 7.5, f"Expected 7.5, got {result}"

    def test_positive_output_gap(self):
        """
        Positive output gap (economy above potential) raises implied rate.
        r = 2.5 + 2.0 + 0.5(0) + 0.5(2.0) = 5.5%
        """
        result = calculate_taylor_rate(
            r_neutral=2.5,
            inflation=2.0,
            inflation_target=2.0,
            output_gap=2.0,
        )
        assert result == 5.5, f"Expected 5.5, got {result}"

    def test_negative_output_gap(self):
        """
        Negative output gap (recession) lowers implied rate.
        r = 2.5 + 2.0 + 0.5(0) + 0.5(-2.0) = 3.5%
        """
        result = calculate_taylor_rate(
            r_neutral=2.5,
            inflation=2.0,
            inflation_target=2.0,
            output_gap=-2.0,
        )
        assert result == 3.5, f"Expected 3.5, got {result}"

    def test_combined_inflationary_pressures(self):
        """
        High inflation + positive output gap = significantly elevated rate.
        r = 2.5 + 5.0 + 0.5(3.0) + 0.5(2.0) = 10.0%
        """
        result = calculate_taylor_rate(
            r_neutral=2.5,
            inflation=5.0,
            inflation_target=2.0,
            output_gap=2.0,
        )
        assert result == 10.0, f"Expected 10.0, got {result}"

    def test_lower_neutral_rate(self):
        """
        ECB uses lower neutral rate of 2.0%.
        r = 2.0 + 2.0 + 0.5(0) + 0.5(0) = 4.0%
        """
        result = calculate_taylor_rate(
            r_neutral=2.0,
            inflation=2.0,
            inflation_target=2.0,
            output_gap=0.0,
        )
        assert result == 4.0, f"Expected 4.0, got {result}"

    def test_returns_float(self):
        """Result should always be a float."""
        result = calculate_taylor_rate(2.5, 2.0, 2.0, 0.0)
        assert isinstance(result, float)

    def test_result_is_rounded(self):
        """Result should be rounded to 2 decimal places."""
        result = calculate_taylor_rate(2.5, 2.333, 2.0, 0.0)
        assert result == round(result, 2)


# ── Policy Gap Tests ───────────────────────────────────────────────────────────

class TestCalculatePolicyGap:

    def test_too_tight(self):
        """Actual rate above implied = positive gap = too tight."""
        gap = calculate_policy_gap(actual_rate=5.0, implied_rate=4.0)
        assert gap == 100, f"Expected 100bps, got {gap}"

    def test_too_loose(self):
        """Actual rate below implied = negative gap = too loose."""
        gap = calculate_policy_gap(actual_rate=3.0, implied_rate=4.0)
        assert gap == -100, f"Expected -100bps, got {gap}"

    def test_neutral(self):
        """Equal rates = zero gap."""
        gap = calculate_policy_gap(actual_rate=4.5, implied_rate=4.5)
        assert gap == 0

    def test_gap_in_basis_points(self):
        """Gap should be in basis points — 1% = 100bps."""
        gap = calculate_policy_gap(actual_rate=4.75, implied_rate=4.5)
        assert gap == 25

    def test_returns_integer(self):
        """Gap should be returned as integer basis points."""
        gap = calculate_policy_gap(4.5, 4.0)
        assert isinstance(gap, int)


# ── Policy Stance Classification Tests ────────────────────────────────────────

class TestClassifyPolicyStance:

    def test_significantly_too_tight(self):
        assert classify_policy_stance(150) == "significantly too tight"

    def test_moderately_too_tight(self):
        assert classify_policy_stance(75) == "moderately too tight"

    def test_approximately_neutral_positive(self):
        assert classify_policy_stance(25) == "approximately neutral"

    def test_approximately_neutral_zero(self):
        assert classify_policy_stance(0) == "approximately neutral"

    def test_approximately_neutral_negative(self):
        assert classify_policy_stance(-25) == "approximately neutral"

    def test_moderately_too_loose(self):
        assert classify_policy_stance(-75) == "moderately too loose"

    def test_significantly_too_loose(self):
        assert classify_policy_stance(-150) == "significantly too loose"

    def test_boundary_tight(self):
        """Exactly 100bps should be moderately tight, not significantly."""
        assert classify_policy_stance(100) == "moderately too tight"

    def test_boundary_loose(self):
        """Exactly -100bps should be moderately loose, not significantly."""
        assert classify_policy_stance(-100) == "moderately too loose"


# ── Interpretation Tests ───────────────────────────────────────────────────────

class TestGenerateInterpretation:

    def test_contains_country(self):
        result = generate_interpretation({
            "country": "UK",
            "gap_bps": 100,
            "implied_rate": 4.0,
            "actual_rate": 5.0,
            "policy_stance": "moderately too tight",
        })
        assert "UK" in result

    def test_contains_gap(self):
        result = generate_interpretation({
            "country": "US",
            "gap_bps": -200,
            "implied_rate": 5.5,
            "actual_rate": 3.5,
            "policy_stance": "significantly too loose",
        })
        assert "200" in result

    def test_tight_says_above(self):
        result = generate_interpretation({
            "country": "US",
            "gap_bps": 100,
            "implied_rate": 4.0,
            "actual_rate": 5.0,
            "policy_stance": "moderately too tight",
        })
        assert "above" in result

    def test_loose_says_below(self):
        result = generate_interpretation({
            "country": "US",
            "gap_bps": -100,
            "implied_rate": 5.0,
            "actual_rate": 4.0,
            "policy_stance": "moderately too loose",
        })
        assert "below" in result


# ── Integration Tests ──────────────────────────────────────────────────────────

class TestAnalyseCountry:

    def test_full_analysis_us_like(self):
        """Test complete pipeline with US-like inputs."""
        metrics = {
            "inflation": 2.4,
            "inflation_target": 2.0,
            "output_gap": 1.0,
            "r_neutral": 2.5,
            "policy_rate": 3.5,
        }
        result = analyse_country("US", metrics)

        assert result is not None
        assert result["country"] == "US"
        assert "implied_rate" in result
        assert "gap_bps" in result
        assert "policy_stance" in result
        assert "interpretation" in result
        # With these inputs, policy should be too loose
        assert result["gap_bps"] < 0

    def test_missing_inflation_returns_none(self):
        """Should return None gracefully if inflation is missing."""
        metrics = {
            "inflation": None,
            "inflation_target": 2.0,
            "output_gap": 0.0,
            "r_neutral": 2.5,
            "policy_rate": 4.5,
        }
        result = analyse_country("UK", metrics)
        assert result is None

    def test_missing_output_gap_uses_zero(self):
        """Missing output gap should default to 0.0, not crash."""
        metrics = {
            "inflation": 2.0,
            "inflation_target": 2.0,
            "output_gap": None,
            "r_neutral": 2.0,
            "policy_rate": 2.0,
        }
        result = analyse_country("EU", metrics)
        assert result is not None
        assert result["inputs"]["output_gap_assumed"] is True