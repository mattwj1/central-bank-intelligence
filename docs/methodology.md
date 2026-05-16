# Methodology

This dashboard combines three lightweight macro signals for the Bank of
England, Federal Reserve, and European Central Bank.

## Sentiment Tracker

Meeting minutes are scraped from official central bank sources and scored with
Claude on a 0-100 hawk/dove scale. Lower scores indicate dovish or easing-biased
language; higher scores indicate hawkish or tightening-biased language. The
dashboard plots the time series and overlays rate hike/cut markers.

## Taylor Rule Analyser

The Taylor Rule module compares actual policy rates with an implied fair-value
rate from inflation, target inflation, a neutral-rate assumption, and an output
gap estimate. The displayed gap is measured in basis points.

## Yield Curve Lab

The yield module displays current sovereign curves for the UK, US, and euro
area, then applies scenario shocks to show how curve level, slope, and front-end
pricing can move under different macro regimes.

The checked-in processed JSON files are intended for the public demo. API keys
are only required when refreshing the underlying data and sentiment scores.
