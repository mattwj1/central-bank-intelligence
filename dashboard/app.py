"""
Central Bank Intelligence Dashboard
-----------------------------------
Streamlit entry point. Single-page layout with tabs for each module.

Run locally:
    streamlit run dashboard/app.py

Pages:
    1. Sentiment Tracker  — hawk/dove scores from Claude analysis
    2. Taylor Rule        — fair-value policy rate vs actual (placeholder)
    3. Yield Curve Lab    — interactive curves with stress tests (placeholder)
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

BANK_META = {
    "BOE": {"name": "Bank of England",    "emoji": "🇬🇧", "colour": "#CC0000"},
    "FED": {"name": "Federal Reserve",    "emoji": "🇺🇸", "colour": "#003F87"},
    "ECB": {"name": "European Central Bank", "emoji": "🇪🇺", "colour": "#FFD700"},
}


st.set_page_config(
    page_title="Central Bank Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Data Loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_sentiment_history() -> dict:
    """Load aggregated sentiment time series."""
    path = PROCESSED / "sentiment_history.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_taylor_rule() -> dict:
    """Load latest Taylor Rule outputs."""
    path = PROCESSED / "taylor_rule_latest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_rate_decisions() -> dict:
    """Load hike/hold/cut decisions keyed by bank + meeting date."""
    path = PROCESSED / "rate_decisions.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_yields() -> dict:
    """Load latest yield curve data."""
    path = PROCESSED / "yields_latest.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Sentiment Helpers ──────────────────────────────────────────────────────────

def sentiment_stance_label(score: int) -> str:
    if score <= 20:   return "Extremely dovish"
    if score <= 40:   return "Dovish"
    if score <= 55:   return "Mildly dovish"
    if score <= 60:   return "Neutral"
    if score <= 70:   return "Mildly hawkish"
    if score <= 85:   return "Hawkish"
    return "Extremely hawkish"


def build_sentiment_chart(history: dict, decisions: dict | None = None) -> go.Figure:
    """
    Stacked subplot per bank — each gets its own panel with shaded hawk/dove
    bands, rate-change markers, and a shared x-axis with rangeslider for zoom.
    Inspired by Nelson-Siegel parameter charts (stacked, clean, comparable).
    """
    bank_order = ["BOE", "FED", "ECB"]
    subplot_titles = [
        f"{BANK_META[b]['emoji']} {BANK_META[b]['name']}" for b in bank_order
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    # Shaded background bands — applied to each subplot
    band_colours = [
        (0,  30, "rgba(30, 144, 255, 0.10)"),
        (30, 55, "rgba(30, 144, 255, 0.04)"),
        (55, 70, "rgba(220, 20, 60, 0.04)"),
        (70, 100, "rgba(220, 20, 60, 0.10)"),
    ]

    # Build decision lookup: (bank, date) → event
    decision_lookup: dict[tuple[str, str], dict] = {}
    if decisions:
        for bank, events in decisions.get("banks", {}).items():
            for ev in events:
                decision_lookup[(bank, ev["date"])] = ev

    for row_idx, bank in enumerate(bank_order, start=1):
        # Bands + neutral line per subplot
        for y0, y1, colour in band_colours:
            fig.add_hrect(
                y0=y0, y1=y1, fillcolor=colour, line_width=0,
                layer="below", row=row_idx, col=1,
            )
        fig.add_hline(
            y=50, line_dash="dot", line_color="grey", line_width=0.8,
            row=row_idx, col=1,
        )

        entries = history.get("banks", {}).get(bank, [])
        if not entries:
            continue

        dates  = [e["date"] for e in entries]
        scores = [e["score"] for e in entries]
        hover  = [
            f"<b>{BANK_META[bank]['name']}</b><br>"
            f"{e['date']}<br>"
            f"Score: {e['score']} ({e['stance']})<br>"
            f"<br>{e['summary'][:120]}..."
            for e in entries
        ]

        # Main score line
        fig.add_trace(
            go.Scatter(
                x=dates, y=scores,
                mode="lines+markers",
                name=BANK_META[bank]["name"],
                line=dict(color=BANK_META[bank]["colour"], width=2),
                marker=dict(size=6, line=dict(width=0.5, color="white")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
                showlegend=False,
            ),
            row=row_idx, col=1,
        )

        # Rate-change markers for this bank
        hike_dates, hike_y, hike_sizes, hike_text = [], [], [], []
        cut_dates,  cut_y,  cut_sizes,  cut_text  = [], [], [], []

        for e in entries:
            ev = decision_lookup.get((bank, e["date"]))
            if not ev:
                continue
            bps = ev.get("change_bps")
            if bps is None or bps == 0:
                continue
            size = max(10, min(22, 8 + abs(bps) / 5))
            label = (
                f"<b>{BANK_META[bank]['name']}</b><br>"
                f"{ev['date']}<br>"
                f"{ev['rate_before']}% → {ev['rate_after']}% "
                f"({bps:+.0f} bps)"
            )
            if bps > 0:
                hike_dates.append(e["date"])
                hike_y.append(e["score"] + 4)
                hike_sizes.append(size)
                hike_text.append(label)
            else:
                cut_dates.append(e["date"])
                cut_y.append(e["score"] - 4)
                cut_sizes.append(size)
                cut_text.append(label)

        if hike_dates:
            fig.add_trace(
                go.Scatter(
                    x=hike_dates, y=hike_y, mode="markers",
                    marker=dict(symbol="triangle-up", color="#D62728",
                                size=hike_sizes, line=dict(width=1, color="white")),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=hike_text,
                    showlegend=(row_idx == 1),
                    name="Rate hike",
                ),
                row=row_idx, col=1,
            )
        if cut_dates:
            fig.add_trace(
                go.Scatter(
                    x=cut_dates, y=cut_y, mode="markers",
                    marker=dict(symbol="triangle-down", color="#2CA02C",
                                size=cut_sizes, line=dict(width=1, color="white")),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=cut_text,
                    showlegend=(row_idx == 1),
                    name="Rate cut",
                ),
                row=row_idx, col=1,
            )

    # Global layout
    for i in range(1, 4):
        fig.update_yaxes(
            range=[0, 100], gridcolor="rgba(0,0,0,0.05)",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0\nDovish", "25", "50\nNeutral", "75", "100\nHawkish"],
            row=i, col=1,
        )

    fig.update_xaxes(
        gridcolor="rgba(0,0,0,0.05)",
        row=3, col=1,
    )

    fig.update_layout(
        height=820,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=50, r=30, t=60, b=40),
        plot_bgcolor="white",
    )
    return fig


def render_latest_metrics(history: dict) -> None:
    """Three-column callout with each bank's latest score."""
    cols = st.columns(3)
    for i, bank in enumerate(["BOE", "FED", "ECB"]):
        entries = history.get("banks", {}).get(bank, [])
        meta = BANK_META[bank]
        with cols[i]:
            if not entries:
                st.metric(f"{meta['emoji']} {meta['name']}", "—", "No data")
                continue
            latest = entries[-1]
            prev   = entries[-2] if len(entries) > 1 else None
            delta  = (latest["score"] - prev["score"]) if prev else None

            st.metric(
                label=f"{meta['emoji']} {meta['name']}  ·  {latest['date']}",
                value=f"{latest['score']}  ({latest['stance']})",
                delta=f"{delta:+d} vs prev" if delta is not None else None,
                delta_color="off",  # direction is context-dependent, don't colour
            )


def render_latest_detail(history: dict) -> None:
    """Expanders showing summary + hawkish/dovish phrase extracts."""
    for bank in ["BOE", "FED", "ECB"]:
        entries = history.get("banks", {}).get(bank, [])
        if not entries:
            continue
        latest = entries[-1]
        meta = BANK_META[bank]

        with st.expander(
            f"{meta['emoji']} **{meta['name']}** — latest meeting ({latest['date']})",
            expanded=False,
        ):
            st.markdown(f"**Summary:** {latest['summary']}")
            left, right = st.columns(2)
            with left:
                st.markdown("**🔺 Hawkish signals**")
                if latest["hawkish_phrases"]:
                    for p in latest["hawkish_phrases"]:
                        st.markdown(f"- _{p}_")
                else:
                    st.caption("None identified.")
            with right:
                st.markdown("**🔻 Dovish signals**")
                if latest["dovish_phrases"]:
                    for p in latest["dovish_phrases"]:
                        st.markdown(f"- _{p}_")
                else:
                    st.caption("None identified.")


# ── Tabs ───────────────────────────────────────────────────────────────────────

def render_sentiment_tab() -> None:
    history = load_sentiment_history()
    if not history or not history.get("banks"):
        st.warning(
            "No sentiment data found. Run: `python models/sentiment_analysis.py`"
        )
        return

    st.caption(
        f"Data as of {history.get('last_updated', '—')[:10]} · "
        f"Scored with {history.get('model', '—')}"
    )

    render_latest_metrics(history)
    st.markdown("---")

    st.subheader("Sentiment trajectory")
    st.caption(
        "Each point is one meeting. Higher = hawkish rhetoric / tighter policy bias; "
        "lower = dovish / easing bias. Scores come from Claude analysis of the full minutes."
    )
    decisions = load_rate_decisions()
    st.plotly_chart(
        build_sentiment_chart(history, decisions),
        width="stretch",
    )
    if decisions:
        st.caption(
            "▲ = rate hike, ▼ = rate cut (size scales with magnitude in bps). "
            "Meetings without a marker held rates unchanged."
        )

    st.markdown("---")
    st.subheader("Latest meeting detail")
    render_latest_detail(history)


COUNTRY_META = {
    "US": {"name": "United States", "emoji": "🇺🇸", "bank": "Federal Reserve"},
    "UK": {"name": "United Kingdom", "emoji": "🇬🇧", "bank": "Bank of England"},
    "EU": {"name": "Euro Area",      "emoji": "🇪🇺", "bank": "European Central Bank"},
}


def build_gap_gauge(country: str, data: dict) -> go.Figure:
    """
    Gauge showing policy gap in basis points.
    Centred at 0 (neutral); blue = too loose; red = too tight.
    """
    gap = data["gap_bps"]
    meta = COUNTRY_META[country]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gap,
        number={"suffix": " bps", "font": {"size": 28}},
        title={"text": f"{meta['emoji']} {meta['name']}<br>"
                       f"<span style='font-size:0.75em;color:grey'>"
                       f"vs Taylor Rule implied</span>",
               "font": {"size": 14}},
        gauge={
            "axis": {"range": [-300, 300], "tickwidth": 1,
                     "tickcolor": "grey", "tickfont": {"size": 10}},
            "bar": {"color": "rgba(0,0,0,0.85)", "thickness": 0.18},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "lightgrey",
            "steps": [
                {"range": [-300, -100], "color": "rgba(30,144,255,0.35)"},
                {"range": [-100,  -50], "color": "rgba(30,144,255,0.15)"},
                {"range": [ -50,   50], "color": "rgba(128,128,128,0.08)"},
                {"range": [  50,  100], "color": "rgba(220,20,60,0.15)"},
                {"range": [ 100,  300], "color": "rgba(220,20,60,0.35)"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.85,
                "value": 0,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_taylor_metrics(data: dict) -> None:
    """Three-column header: actual rate, implied rate, gap for each country."""
    cols = st.columns(3)
    for i, country in enumerate(["UK", "US", "EU"]):
        entry = data["countries"].get(country)
        meta  = COUNTRY_META[country]
        with cols[i]:
            if not entry:
                st.metric(f"{meta['emoji']} {meta['name']}", "—", "No data")
                continue
            gap  = entry["gap_bps"]
            direction = "too tight" if gap > 50 else "too loose" if gap < -50 else "neutral"
            st.metric(
                label=f"{meta['emoji']} {meta['name']} ({meta['bank']})",
                value=f"{entry['actual_rate']}% actual",
                delta=f"{gap:+d} bps {direction} · implied {entry['implied_rate']}%",
                delta_color="off",
            )


def render_taylor_inputs(data: dict) -> None:
    """Table showing the inputs driving each country's implied rate."""
    rows = []
    for country in ["UK", "US", "EU"]:
        entry = data["countries"].get(country)
        if not entry:
            continue
        inp = entry["inputs"]
        rows.append({
            "Country": f"{COUNTRY_META[country]['emoji']} {country}",
            "Inflation (%)": inp["inflation"],
            "Inflation target (%)": inp["inflation_target"],
            "Output gap (%)": (
                f"{inp['output_gap']}  ⚠️ assumed"
                if inp.get("output_gap_assumed") else f"{inp['output_gap']}"
            ),
            "Neutral rate r* (%)": inp["r_neutral"],
            "→ Implied rate (%)": entry["implied_rate"],
            "Actual rate (%)": entry["actual_rate"],
            "Gap (bps)": entry["gap_bps"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch")


def render_taylor_tab() -> None:
    data = load_taylor_rule()
    if not data or not data.get("countries"):
        st.warning(
            "No Taylor Rule data found. Run: "
            "`python scripts/fetch_macro_data.py && python models/taylor_rule.py`"
        )
        return

    st.caption(
        f"Data as of {data.get('data_date', '—')} · "
        f"Formula: r = r* + π + 0.5(π − π*) + 0.5(y − y*)  "
        f"(Taylor, 1993)"
    )

    render_taylor_metrics(data)
    st.markdown("---")

    st.subheader("Policy gap vs Taylor Rule")
    st.caption(
        "Each gauge shows how far the actual policy rate sits from the model-implied rate. "
        "Negative (blue) = policy is looser than the rule suggests; "
        "positive (red) = tighter. Within ±50 bps is treated as approximately neutral."
    )
    gauge_cols = st.columns(3)
    for i, country in enumerate(["UK", "US", "EU"]):
        entry = data["countries"].get(country)
        if not entry:
            continue
        with gauge_cols[i]:
            st.plotly_chart(build_gap_gauge(country, entry), width="stretch")
            st.markdown(f"_{entry['interpretation']}_")

    st.markdown("---")
    st.subheader("Model inputs")
    st.caption(
        "The inputs driving each implied rate. Flagged (⚠️) values are assumed "
        "due to missing source data — these should be sourced properly before "
        "drawing strong conclusions."
    )
    render_taylor_inputs(data)


# ── Yield Curve helpers ────────────────────────────────────────────────────────

TENOR_ORDER = ["3M", "2Y", "5Y", "10Y", "30Y"]
TENOR_YEARS = {"3M": 0.25, "2Y": 2, "5Y": 5, "10Y": 10, "30Y": 30}

# Scenario presets — each maps to (parallel_bps, steepener_bps, front_end_bps)
#   parallel:   shifts every tenor by the same amount
#   steepener:  rotates curve; positive = long-end up vs short-end down (bear steepener)
#   front_end:  extra shock to 2Y-and-shorter (independent of parallel)
YIELD_SCENARIOS = {
    "Baseline (no shock)":        (0, 0, 0),
    "BoE/Fed cut 100bps":         (-75, -20, -100),  # front drops more than back
    "Hard hike 75bps":            (50, 10, 75),      # front rises more than back
    "Recession (bull steepener)": (-100, -50, -150), # sharp front cut, curve steepens
    "Reflation (bear steepener)": (30, 60, 0),       # long end sells off, front anchored
    "Stagflation":                (75, -30, 50),     # curve flattens, level rises
}


def _apply_shock(
    yields: dict[str, float],
    parallel_bps: float,
    steepener_bps: float,
    front_end_bps: float,
) -> dict[str, float]:
    """Apply a shock to every tenor in a yield curve. Returns new yields."""
    shocked = {}
    for tenor, y in yields.items():
        years = TENOR_YEARS.get(tenor, 10)
        # Steepener weight: -1 at front, +1 at long end, linear in log-maturity
        steepener_weight = (years - 5) / 25.0  # roughly -0.19 at 3M, +1.0 at 30Y
        front_weight = 1.0 if years <= 2 else 0.0

        shock_bps = (
            parallel_bps
            + steepener_bps * steepener_weight
            + front_end_bps * front_weight
        )
        shocked[tenor] = y + shock_bps / 100.0   # bps → pct points
    return shocked


def _sorted_curve_points(yields: dict[str, float]) -> tuple[list[float], list[float], list[str]]:
    """Return (x_years, y_pct, labels) sorted by maturity for plotting."""
    pairs = sorted(
        ((TENOR_YEARS[t], yields[t], t) for t in yields if t in TENOR_YEARS),
        key=lambda p: p[0],
    )
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]
    return xs, ys, labels


def build_yield_chart(
    country_data: dict,
    country_code: str,
    shock: tuple[float, float, float],
    show_confidence: bool,
) -> go.Figure:
    """Single-country curve with baseline, shocked, and optional ±1σ band."""
    yields = country_data["yields"]
    vols   = country_data.get("volatility_annualised", {})

    baseline_x, baseline_y, labels = _sorted_curve_points(yields)
    shocked = _apply_shock(yields, *shock)
    _, shocked_y, _ = _sorted_curve_points(shocked)

    fig = go.Figure()

    # Confidence band (±1σ, capped at ±100bps for display sanity)
    if show_confidence and vols:
        upper, lower = [], []
        for tenor, y in zip(labels, baseline_y):
            vol_pct_pts = min(vols.get(tenor, 0.5), 1.0)   # cap at 100bps
            upper.append(y + vol_pct_pts)
            lower.append(y - vol_pct_pts)
        fig.add_trace(go.Scatter(
            x=baseline_x + baseline_x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(128,128,128,0.12)",
            line=dict(width=0),
            name="±1σ historical range",
            hoverinfo="skip",
        ))

    colour = BANK_META.get({"UK": "BOE", "US": "FED", "EU": "ECB"}[country_code], {}).get("colour", "#333")

    # Baseline curve
    fig.add_trace(go.Scatter(
        x=baseline_x, y=baseline_y,
        mode="lines+markers+text",
        name="Current curve",
        line=dict(color=colour, width=2.5),
        marker=dict(size=9),
        text=[f"{y:.2f}%" for y in baseline_y],
        textposition="top center",
        textfont=dict(size=11),
        hovertemplate="%{text} at %{customdata}<extra>Current</extra>",
        customdata=labels,
    ))

    # Shocked curve (only if any shock applied)
    if shock != (0, 0, 0):
        fig.add_trace(go.Scatter(
            x=baseline_x, y=shocked_y,
            mode="lines+markers",
            name="Shocked curve",
            line=dict(color=colour, width=2.5, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="%{y:.2f}% at %{customdata}<extra>Shocked</extra>",
            customdata=labels,
        ))

    fig.update_layout(
        xaxis=dict(title="Maturity (years)", type="log",
                   tickvals=[0.25, 2, 5, 10, 30],
                   ticktext=["3M", "2Y", "5Y", "10Y", "30Y"]),
        yaxis=dict(title="Yield (%)", gridcolor="rgba(128,128,128,0.15)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def build_multicurve_chart(data: dict) -> go.Figure:
    """Overlay all three country curves for comparison."""
    fig = go.Figure()
    for country in ["UK", "US", "EU"]:
        entry = data["countries"].get(country)
        if not entry:
            continue
        xs, ys, labels = _sorted_curve_points(entry["yields"])
        bank_code = {"UK": "BOE", "US": "FED", "EU": "ECB"}[country]
        meta = BANK_META[bank_code]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines+markers",
            name=f"{meta['emoji']} {country}",
            line=dict(color=meta["colour"], width=2.5),
            marker=dict(size=9),
            hovertemplate=f"<b>{country}</b> %{{customdata}}: %{{y:.2f}}%<extra></extra>",
            customdata=labels,
        ))

    fig.update_layout(
        xaxis=dict(title="Maturity (years)", type="log",
                   tickvals=[0.25, 2, 5, 10, 30],
                   ticktext=["3M", "2Y", "5Y", "10Y", "30Y"]),
        yaxis=dict(title="Yield (%)", gridcolor="rgba(128,128,128,0.15)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def render_curve_metrics(country_data: dict) -> None:
    """Top-row metrics for a single country's curve."""
    m = country_data.get("curve_metrics", {})
    cols = st.columns(4)
    cols[0].metric("10Y yield",     f"{m.get('level_10y', '—')}%")
    cols[1].metric("10Y-2Y slope",  f"{m.get('slope_10y_2y', '—')*100:+.0f} bps"
                                    if "slope_10y_2y" in m else "—")
    cols[2].metric("Curvature",     f"{m.get('curvature', '—')}")
    cols[3].metric("Inverted?",     "Yes ⚠️" if m.get("inverted") else "No")


def render_yield_tab() -> None:
    data = load_yields()
    if not data or not data.get("countries"):
        st.warning(
            "No yield data found. Run: `python scripts/fetch_yield_data.py`"
        )
        return

    st.caption(
        f"Data as of {data.get('data_date', '—')} · "
        f"Government bond yields for UK Gilts, US Treasuries, and EU Bunds"
    )

    mode = st.radio(
        "Mode",
        ["📊 Compare all", "🇬🇧 Stress test UK", "🇺🇸 Stress test US", "🇪🇺 Stress test EU"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode.startswith("📊"):
        st.subheader("Current yield curves")
        st.caption(
            "All three curves on log-maturity scale. The spread between the UK, US, and "
            "EU curves reflects differences in expected policy rates, term premia, and risk."
        )
        st.plotly_chart(build_multicurve_chart(data), width="stretch")
        return

    # Single-country stress test mode
    country_code = {"🇬🇧": "UK", "🇺🇸": "US", "🇪🇺": "EU"}[mode.split()[0]]
    country_data = data["countries"].get(country_code)
    if not country_data:
        st.warning(f"No data for {country_code}.")
        return

    st.subheader(f"{country_code} stress-test lab")
    render_curve_metrics(country_data)

    st.markdown("**Scenario presets**")
    preset_cols = st.columns(len(YIELD_SCENARIOS))
    for i, (name, vals) in enumerate(YIELD_SCENARIOS.items()):
        if preset_cols[i].button(name, width="stretch", key=f"preset_{name}"):
            st.session_state["yield_shock"] = vals

    # Default shock state
    shock = st.session_state.get("yield_shock", (0, 0, 0))

    st.markdown("**Manual shock (basis points)**")
    slider_cols = st.columns(3)
    parallel  = slider_cols[0].slider("Parallel shift",    -200, 200, int(shock[0]), step=5)
    steepener = slider_cols[1].slider("Steepener (long−short)", -200, 200, int(shock[1]), step=5)
    front_end = slider_cols[2].slider("Front-end only",    -200, 200, int(shock[2]), step=5)

    current_shock = (parallel, steepener, front_end)
    show_conf = st.checkbox("Show ±1σ historical range", value=True)

    st.plotly_chart(
        build_yield_chart(country_data, country_code, current_shock, show_conf),
        width="stretch",
    )

    if current_shock != (0, 0, 0):
        st.caption(
            f"**Applied shock:** {parallel:+d} bps parallel, "
            f"{steepener:+d} bps steepener, {front_end:+d} bps front-end. "
            f"Click **Baseline** to reset."
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📊 Central Bank Intelligence Dashboard")
    st.caption(
        "Tracking monetary policy sentiment, fair-value rates, and yield curves "
        "across the Bank of England, Federal Reserve, and ECB."
    )

    tab_sent, tab_taylor, tab_yield = st.tabs([
        "🦅 Sentiment Tracker",
        "📐 Taylor Rule Analyser",
        "📈 Yield Curve Lab",
    ])

    with tab_sent:   render_sentiment_tab()
    with tab_taylor: render_taylor_tab()
    with tab_yield:  render_yield_tab()

    st.markdown("---")
    st.caption(
        f"Built by Matthew Wynne-Jones · MSc Finance, University of Exeter · "
        f"Rendered {datetime.now():%Y-%m-%d %H:%M}"
    )


if __name__ == "__main__":
    main()
