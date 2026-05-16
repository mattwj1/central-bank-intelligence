# 10-Year Sentiment Backfill — Run Instructions

All of this runs from your PyCharm terminal. The Cowork sandbox can't reach
FRED or do long-running API work, so execution stays on your machine.

---

## Before you start

1. **Rotate the FRED key.** The current one has been screenshotted. Grab a new
   key at https://fred.stlouisfed.org/docs/api/api_key.html and update `.env`:

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   FRED_API_KEY=<new-key>
   ```

2. **Install the progress-bar dependency** (optional but highly recommended
   for an 80-file run):

   ```
   pip install tqdm
   ```

3. **Check the working directory** — every command below assumes you're in
   the repo root:

   ```
   cd C:\Users\matth\PycharmProjects\central-bank-intelligence
   ```

---

## Step 1 — Regenerate rate_decisions.json from FRED

Replaces the hand-populated JSON with a real FRED pull. Verifies your key
works before you spend money on the backfill.

```
python scripts/fetch_rate_decisions.py
```

**Expected output:** INFO lines showing each meeting date with before → after
rates. Any `ERROR` line means the FRED call failed; the script's safety guard
means it *won't* wipe your existing good JSON in that case — you'll need to
fix the key / network and rerun.

---

## Step 2 — Backfill the scrapes (BoE + Fed)

The `--historical` flag walks year × month combos back to 2016. Expect ~80
Fed minutes + ~80 BoE summaries. Runtime: 10-15 minutes. No API cost (just
HTTP).

```
python scripts/scrape_central_banks.py --bank FED --limit 200 --historical
python scripts/scrape_central_banks.py --bank BOE --limit 200 --historical
```

**What you'll see:** for BoE, a lot of silent 404s (expected — only 8 months
per year are actual meetings). For Fed, the calendar + 6 historical pages
get parsed and every `fomcminutesYYYYMMDD.htm` link gets fetched.

Already-scraped files are skipped — safe to rerun if interrupted.

**ECB is NOT in this step.** The current ECB scraper uses hand-collected PDF
URLs because the ECB index is JavaScript-rendered. Scaling that to 10 years
needs a separate pass (either scrape the static accounts index, or enumerate
press-conference PDFs with their unique hashes). Defer until the BoE/Fed
backfill is in and the chart tells a story — then decide if ECB is worth the
work for this phase or a later polish.

---

## Step 3 — Score the newly-scraped minutes

The scorer is resumable: interrupt it with Ctrl-C and rerun to pick up where
you left off. Each file is cached individually in
`data/processed/sentiment/{BANK}_{DATE}.json` as it completes.

```
python models/sentiment_analysis.py
```

**Expected runtime:** ~160 new files × (API call + 2s sleep) ≈ 10-15 minutes
of wall-clock time, plus ~$5-8 in Anthropic credit.

**What you'll see:** a tqdm progress bar with ETA. At the end, counts of
scored / cached / errored, then the aggregated `sentiment_history.json`
rebuild.

---

## Step 4 — Refresh rate-change markers

Once sentiment_history has the new entries, regenerate rate_decisions.json so
the chart markers cover the whole 10-year span:

```
python scripts/fetch_rate_decisions.py
```

---

## Step 5 — Smoke-test the dashboard

```
streamlit run dashboard/app.py
```

The sentiment chart should now span 2016-2026 with ▲/▼ markers through the
entire tightening-then-easing cycle. Eyeball the 2020 COVID period — you
should see an immediate dovish spike on both BoE and Fed.

---

## When things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `FRED_API_KEY missing` | `.env` didn't load | Check you're in the repo root; check the key isn't wrapped in quotes |
| `403 Forbidden` on FRED | Key revoked or IP blocked | Generate new key; check you haven't hit the (generous) rate limit |
| BoE scraper returns 0 files in historical mode | URL pattern changed upstream | Inspect one failing URL in a browser; BoE occasionally restructures archive URLs |
| Fed scraper returns "text too short" on historical entries | Older minutes use a different DOM structure | `_extract_fed_minutes_body` has a list of fallback selectors; add another if the page's actual tag differs |
| API rate limit errors mid-run | Anthropic tier caps exceeded | Scorer already has 2s between calls; if hitting limits, bump `SLEEP_BETWEEN_CALLS` in `models/sentiment_analysis.py` |
| Scoring errored on a handful of files | Model returned malformed JSON | Paste the bad file and re-run with `--file BANK_YYYY-MM-DD.txt` to retry just that one |

---

## After the backfill — sanity checks

A few things to eyeball in the resulting chart before declaring it done:

- **2020-Q1 (COVID)** — every bank should drop sharply dovish. If BoE or Fed
  scores stay above 40 here, something's wrong with the scoring.
- **2022-2023 hiking cycle** — should show a clear hawkish plateau. If it
  looks flat/neutral, the prompt may need another calibration pass.
- **2024-2025 pivot** — scores should trend downward as cuts begin.

If any of those look off, the fix is usually in the prompt, not the data. We
can tune on the full 10-year dataset without re-scraping.
