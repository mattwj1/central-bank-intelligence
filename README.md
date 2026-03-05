# Central Bank Intelligence Dashboard

A portfolio project combining LLM sentiment analysis, Taylor Rule modelling, 
and interactive yield curve visualisation to analyse monetary policy across 
the Bank of England, Federal Reserve, and ECB.

**[Live Dashboard →](https://YOUR-URL.streamlit.app)** | 
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
git clone https://github.com/YOUR_USERNAME/central-bank-intelligence.git
cd central-bank-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

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