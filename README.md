# Central Bank Intelligence Dashboard

A portfolio project combining LLM sentiment analysis, Taylor Rule modelling, 
and interactive yield curve visualisation to analyse monetary policy across 
the Bank of England, Federal Reserve, and ECB.

**[Live Dashboard →](https://share.streamlit.io/deploy?repository=https://github.com/mattwj1/central-bank-intelligence&mainModule=dashboard/app.py)** |
**[Methodology →](docs/methodology.md)**

---

## What It Does

| Module | Description |
|--------|-------------|
| 🦅 Hawk/Dove Tracker | Uses Claude API to score central bank minutes 0–100 (dovish→hawkish) |
| 📐 Taylor Rule Analyser | Calculates where rates *should* be vs where they are |
| 📈 Yield Curve Lab | Interactive stress-testing with sliders and scenario presets |

## Why It Matters

Central banks move markets. This dashboard tracks what they're *saying* 
(sentiment), what they *should* be doing (Taylor Rule fair value), and what 
markets are *pricing* (yield curves) — the same analytical framework used 
by professional macro investors.

## Tech Stack

`Python` · `Streamlit` · `Plotly` · `Anthropic Claude API` · `FRED API` · `pandas`

## Setup
```bash
git clone https://github.com/mattwj1/central-bank-intelligence.git
cd central-bank-intelligence
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

Run the dashboard locally:

```bash
streamlit run dashboard/app.py
```

## Deploy A Demo

This is a Streamlit app, so GitHub Pages cannot run it directly. The simplest
Git-backed demo is Streamlit Community Cloud:

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Create a new app from `mattwj1/central-bank-intelligence`.
4. Set the main file path to `dashboard/app.py`.

The dashboard reads the checked-in processed JSON files, so the public demo can
render without API keys. API keys are only needed when refreshing the data.

## Project Structure
```
├── models/          # Taylor Rule, sentiment analysis, yield curve logic
├── dashboard/       # Streamlit app and pages
├── scripts/         # Data fetching and processing pipelines
├── data/            # Raw and processed data (gitignored)
└── docs/            # Methodology documentation
```

## Data Sources

- **FRED API** — US macroeconomic data (CPI, GDP, yields)
- **ONS API** — UK macroeconomic data  
- **yfinance** — Government bond yields
- **BoE/Fed/ECB websites** — Meeting minutes (public domain)

---

*Built by Matthew Wynne-Jones | MSc Finance, University of Exeter*
