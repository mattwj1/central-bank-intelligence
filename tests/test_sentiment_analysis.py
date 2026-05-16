"""
Tests for the pure helper functions in models/sentiment_analysis.py.
API-dependent code is not tested here — that requires live credentials
and costs money. Those paths are exercised manually when running the
full pipeline.
"""

import pytest

from models.sentiment_analysis import (
    classify_stance,
    extract_json,
    parse_filename,
)


# ── parse_filename ─────────────────────────────────────────────────────────────

def test_parse_filename_boe():
    assert parse_filename("BOE_2026-02-01.txt") == ("BOE", "2026-02-01")


def test_parse_filename_fed():
    assert parse_filename("FED_2025-12-10.txt") == ("FED", "2025-12-10")


def test_parse_filename_ecb():
    assert parse_filename("ECB_2025-10-30.txt") == ("ECB", "2025-10-30")


def test_parse_filename_invalid():
    with pytest.raises(ValueError):
        parse_filename("random_file.txt")


# ── classify_stance ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0,   "extremely dovish"),
    (15,  "extremely dovish"),
    (20,  "extremely dovish"),
    (21,  "dovish"),
    (40,  "dovish"),
    (50,  "neutral"),
    (60,  "neutral"),
    (65,  "hawkish"),
    (80,  "hawkish"),
    (85,  "extremely hawkish"),
    (100, "extremely hawkish"),
])
def test_classify_stance(score, expected):
    assert classify_stance(score) == expected


# ── extract_json ───────────────────────────────────────────────────────────────

def test_extract_json_clean():
    result = extract_json('{"score": 55, "summary": "balanced"}')
    assert result["score"] == 55


def test_extract_json_with_markdown_fences():
    wrapped = '```json\n{"score": 70, "summary": "hawkish"}\n```'
    result = extract_json(wrapped)
    assert result["score"] == 70


def test_extract_json_with_prose_preamble():
    messy = 'Here is the analysis:\n{"score": 35, "summary": "dovish"}\nThanks!'
    result = extract_json(messy)
    assert result["score"] == 35


def test_extract_json_invalid():
    with pytest.raises(ValueError):
        extract_json("no json here")
