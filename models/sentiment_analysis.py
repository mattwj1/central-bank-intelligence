"""
sentiment_analysis.py
---------------------
Scores central bank meeting minutes for hawkish/dovish language
using the Anthropic Claude API.

Scoring scale:
    0-20   : Extremely dovish
    21-40  : Dovish
    41-60  : Neutral
    61-80  : Hawkish
    81-100 : Extremely hawkish

Each minute file in /data/raw/minutes/ is scored once and the result
cached in /data/processed/sentiment/{BANK}_{DATE}.json. An aggregate
time series is written to /data/processed/sentiment_history.json for
consumption by the dashboard.

Usage:
    python models/sentiment_analysis.py              # score everything new
    python models/sentiment_analysis.py --rescore    # force rescore all
    python models/sentiment_analysis.py --file BOE_2026-02-01.txt
"""

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic, APIError, APIStatusError
from dotenv import load_dotenv

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional — fall back to a no-op wrapper
    def tqdm(iterable, **kwargs):
        return iterable

load_dotenv()

logger = logging.getLogger(__name__)

# ── Paths & Config ─────────────────────────────────────────────────────────────

ROOT           = Path(__file__).resolve().parent.parent
MINUTES_DIR    = ROOT / "data" / "raw" / "minutes"
SENTIMENT_DIR  = ROOT / "data" / "processed" / "sentiment"
HISTORY_PATH   = ROOT / "data" / "processed" / "sentiment_history.json"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL              = "claude-sonnet-4-5-20250929"   # balanced accuracy vs cost
MAX_TOKENS         = 1500
MAX_INPUT_CHARS    = 80_000        # safety cap; BoE minutes run ~30KB
SLEEP_BETWEEN_CALLS = 2.0          # seconds — be polite to the API

BANK_FULL_NAMES = {
    "BOE": "Bank of England Monetary Policy Committee",
    "FED": "Federal Reserve Open Market Committee (FOMC)",
    "ECB": "European Central Bank Governing Council",
}


# ── Data Class ─────────────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    """A single scored minute."""
    bank: str
    date: str                   # YYYY-MM-DD
    score: int                  # 0-100
    stance: str                 # "dovish" | "neutral" | "hawkish" etc.
    hawkish_phrases: list[str]
    dovish_phrases: list[str]
    summary: str
    model: str
    scored_at: str              # ISO timestamp
    source_file: str
    char_count: int


# ── Prompt ─────────────────────────────────────────────────────────────────────

SENTIMENT_PROMPT = """You are a macro strategist analysing {bank_full_name} meeting minutes for monetary policy sentiment. Your score will be used in a time series to track policy shifts, so meaningful differentiation between meetings matters — avoid clustering everything near 50.

SCORING SCALE (0 = maximally dovish, 100 = maximally hawkish):

   0-15  Extremely dovish     Aggressive cuts (50bp+) or emergency easing; strong growth/employment concern dominates
  16-30  Dovish               25bp cut OR explicit guidance toward cuts; downside risks dominate
  31-45  Mildly dovish        Hold with clear easing bias, OR cut with hawkish dissent
  46-55  Neutral              Genuinely balanced: no directional bias, no meaningful guidance either way
  56-70  Mildly hawkish       Hold with tightening bias, OR hold with hawkish rhetoric, OR cut with strong hawkish dissent
  71-85  Hawkish              25bp hike OR hold with explicit guidance toward further tightening
  86-100 Extremely hawkish    50bp+ hike, or strong signal of sustained further tightening

WEIGHTING (apply in order):

1. ACTUAL DECISION carries the most weight. A cut anchors you below 50; a hike above 50; a hold can sit anywhere from 30-70 depending on guidance.
2. FORWARD GUIDANCE: phrases like "further cuts likely", "remain restrictive", "closer call" shift the score 5-15 points.
3. VOTE SPLIT: a unanimous decision reinforces the score; meaningful dissent pulls the score 5-10 points toward the dissenters. A 5-4 cut is less dovish than a unanimous cut.
4. RISK BALANCE: explicit emphasis on inflation persistence = hawkish; emphasis on labour market weakness or demand = dovish.

CALIBRATION EXAMPLES (use these as anchors):
- Fed cut with dovish labour-market emphasis, mixed dissents     → ~30
- BoE 5-4 cut with hawkish dissent concerned about persistence   → ~35
- BoE 7-2 hold emphasising inflation risks                       → ~65
- ECB unanimous hold with genuinely balanced outlook             → ~50
- Fed hold with two dissenters wanting cuts                      → ~42

DO NOT default to the 45-60 range when the content is directional. If you find yourself scoring every meeting near neutral, you are under-differentiating.

Extract:
1. Up to 5 key hawkish phrases (short verbatim quotes)
2. Up to 5 key dovish phrases (short verbatim quotes)
3. A one-sentence summary that states the decision, the vote split (if available), and the directional bias.

Text to analyse:
---
{text}
---

Return ONLY valid JSON, no markdown fences, no prose:
{{
  "score": <integer 0-100>,
  "hawkish_phrases": [<strings>],
  "dovish_phrases": [<strings>],
  "summary": "<one sentence>"
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> tuple[str, str]:
    """Parse BANK_YYYY-MM-DD.txt → (bank, date)."""
    match = re.match(r"(BOE|FED|ECB)_(\d{4}-\d{2}-\d{2})\.txt$", filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group(1), match.group(2)


def classify_stance(score: int) -> str:
    """Map numeric score to a stance label."""
    if score <= 20:
        return "extremely dovish"
    if score <= 40:
        return "dovish"
    if score <= 60:
        return "neutral"
    if score <= 80:
        return "hawkish"
    return "extremely hawkish"


def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from model output. Handles the case
    where the model wraps output in markdown fences despite instructions.
    """
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Find the first JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response:\n{text[:300]}")
    return json.loads(text[start:end + 1])


# ── Core API Call ──────────────────────────────────────────────────────────────

def score_text(client: Anthropic, text: str, bank: str) -> dict:
    """
    Send text to Claude and parse the structured sentiment response.

    Raises on API error or malformed JSON — caller decides how to handle.
    """
    prompt = SENTIMENT_PROMPT.format(
        bank_full_name=BANK_FULL_NAMES.get(bank, bank),
        text=text[:MAX_INPUT_CHARS],
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    parsed = extract_json(raw)

    # Validate required fields
    for field in ("score", "hawkish_phrases", "dovish_phrases", "summary"):
        if field not in parsed:
            raise ValueError(f"Missing field '{field}' in response: {parsed}")

    score = int(parsed["score"])
    if not 0 <= score <= 100:
        raise ValueError(f"Score {score} out of range 0-100")

    return parsed


# ── Per-File Scoring ───────────────────────────────────────────────────────────

def score_file(client: Anthropic, path: Path, force: bool = False) -> SentimentResult | None:
    """
    Score a single minute file, using cache unless --force.
    Returns None if skipped (already cached).
    """
    bank, date = parse_filename(path.name)
    output_path = SENTIMENT_DIR / f"{bank}_{date}.json"

    if output_path.exists() and not force:
        logger.info(f"  [cached] {path.name}")
        with open(output_path) as f:
            return SentimentResult(**json.load(f))

    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text.strip()) < 200:
        logger.warning(f"  [skip] {path.name}: too short ({len(text)} chars)")
        return None

    logger.info(f"  [scoring] {path.name} ({len(text):,} chars)...")

    try:
        parsed = score_text(client, text, bank)
    except (APIError, APIStatusError) as e:
        logger.error(f"  [api error] {path.name}: {e}")
        return None
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"  [parse error] {path.name}: {e}")
        return None

    result = SentimentResult(
        bank=bank,
        date=date,
        score=int(parsed["score"]),
        stance=classify_stance(int(parsed["score"])),
        hawkish_phrases=parsed["hawkish_phrases"],
        dovish_phrases=parsed["dovish_phrases"],
        summary=parsed["summary"],
        model=MODEL,
        scored_at=datetime.now().isoformat(timespec="seconds"),
        source_file=path.name,
        char_count=len(text),
    )

    with open(output_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info(f"  [done] {path.name} → score={result.score} ({result.stance})")
    return result


# ── Aggregation ────────────────────────────────────────────────────────────────

def build_history() -> dict:
    """
    Collect all per-file sentiment JSONs into a single time-series file
    used by the dashboard. Sorted by date per bank.
    """
    history: dict[str, list[dict]] = {"BOE": [], "FED": [], "ECB": []}

    for path in sorted(SENTIMENT_DIR.glob("*.json")):
        with open(path) as f:
            entry = json.load(f)
        bank = entry["bank"]
        if bank in history:
            history[bank].append(entry)

    for bank in history:
        history[bank].sort(key=lambda e: e["date"])

    output = {
        "last_updated": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "banks": history,
    }

    with open(HISTORY_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"History written → {HISTORY_PATH}")
    return output


# ── Main ───────────────────────────────────────────────────────────────────────

def analyse_all(rescore: bool = False, only_file: str | None = None) -> None:
    """Score every minute file (or just one), then rebuild history."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — check your .env file")

    client = Anthropic(api_key=api_key)

    files = sorted(MINUTES_DIR.glob("*.txt"))
    if only_file:
        files = [f for f in files if f.name == only_file]
        if not files:
            raise FileNotFoundError(f"{only_file} not in {MINUTES_DIR}")

    logger.info(f"Scoring {len(files)} minute file(s)...\n")

    # Fast-path: if nothing new needs scoring and we're not forcing, skip
    # straight to the history rebuild so the user gets instant feedback.
    if not rescore:
        pending = [
            p for p in files
            if not (SENTIMENT_DIR / f"{p.stem}.json").exists()
        ]
        if not pending:
            logger.info("All files already cached. Rebuilding history only...")
            build_history()
            return

    scored = 0
    skipped = 0
    errored = 0
    # tqdm shows ETA so an 80-file run doesn't feel like it's hung
    for path in tqdm(files, desc="Scoring minutes", unit="file"):
        try:
            result = score_file(client, path, force=rescore)
        except KeyboardInterrupt:
            logger.warning("Interrupted — partial results cached. Re-run to resume.")
            break
        if result is None:
            errored += 1
        elif (SENTIMENT_DIR / f"{path.stem}.json").stat().st_mtime > time.time() - 10:
            scored += 1
        else:
            skipped += 1
        time.sleep(SLEEP_BETWEEN_CALLS)

    logger.info(
        f"\nScored: {scored}  ·  Already cached: {skipped}  ·  Errored: {errored}"
    )
    logger.info("Rebuilding history...")
    build_history()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true",
                        help="Force rescore even if cached")
    parser.add_argument("--file", type=str, default=None,
                        help="Score just one file (e.g. BOE_2026-02-01.txt)")
    args = parser.parse_args()

    analyse_all(rescore=args.rescore, only_file=args.file)
