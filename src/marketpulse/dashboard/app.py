"""Streamlit BI dashboard for MarketPulse AI.

Run with: streamlit run src/marketpulse/dashboard/app.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

# Allow `streamlit run` to find the package without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketpulse.config import settings
from marketpulse.db import (
    count_qa_usage_today,
    fetch_filing_sentiment,
    fetch_filings,
    fetch_price_history,
    init_db,
    log_qa_usage,
)
from marketpulse.rag.pipeline import answer_question, has_ticker_documents
from marketpulse.refresh import index_ticker_filings, refresh_ticker
from marketpulse.snapshot import load_snapshot, read_snapshot

# Live refreshes hit SEC EDGAR and Yahoo Finance. They're free, but SEC
# throttles or blocks clients that hammer it, and every request carries this
# app's User-Agent - so one visitor (or bot) mashing the button shouldn't be
# able to get the whole deployment blocked. One refresh per ticker per
# window, shared across all visitors.
REFRESH_COOLDOWN_SECONDS = 10 * 60

GREEN, RED, GRAY = "#059669", "#DC2626", "#94A3B8"

st.set_page_config(page_title="MarketPulse AI", page_icon=":material/monitoring:", layout="wide")


@st.cache_resource(show_spinner=False)
def _bootstrap() -> list[str]:
    """Runs once per server process: create tables and seed any ticker that
    has no data from the bundled snapshot. On a free host the database is
    wiped whenever the app restarts or wakes from sleep, so this is what
    keeps the first page a visitor sees from being empty."""
    init_db()
    return load_snapshot()


@st.cache_resource(show_spinner=False)
def _last_refreshed() -> dict[str, float]:
    """Ticker -> time.time() of its last live refresh, shared by every session."""
    return {}


@st.cache_data(show_spinner=False)
def _snapshot_date() -> str | None:
    data = read_snapshot()
    return data["generated_at"][:10] if data else None


def _net_tone(label: str, score: float) -> float:
    """Collapse (label, confidence) into one signed number in [-1, 1]."""
    return {"positive": score, "negative": -score}.get(label, 0.0)


def _fmt_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d:%b} {d.day}, {d.year}"


seeded_tickers = _bootstrap()
last_refreshed = _last_refreshed()

if flash := st.session_state.pop("flash", None):
    st.toast(flash, icon=":material/check_circle:")

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    tickers = list(settings.default_tickers) or ["AAPL"]
    ticker = st.selectbox("Ticker", options=tickers, key="ticker", bind="query-params")

    st.divider()
    st.markdown("**Live data**")
    since = time.time() - last_refreshed.get(ticker, 0.0)
    cooling_down = since < REFRESH_COOLDOWN_SECONDS
    st.caption(
        "Pull the latest prices and 10-K filings from Yahoo Finance and SEC EDGAR, "
        "then re-score and re-index them. Takes about 20-40 seconds."
    )
    if st.button(
        f"Refresh {ticker}",
        icon=":material/refresh:",
        disabled=cooling_down,
        width="stretch",
    ):
        last_refreshed[ticker] = time.time()
        with st.status(f"Refreshing {ticker}...", expanded=True) as status:
            try:
                counts = refresh_ticker(ticker, log=status.write)
            except Exception as e:  # noqa: BLE001 - surface any ingestion failure in the UI, don't crash the app
                status.update(label=f"Couldn't refresh {ticker}", state="error")
                st.error(str(e))
            else:
                st.session_state.flash = (
                    f"{ticker} refreshed: {counts['prices']} prices, {counts['filings']} filings indexed."
                )
                st.rerun()
    if cooling_down:
        minutes = int((REFRESH_COOLDOWN_SECONDS - since) // 60) + 1
        st.caption(f":material/schedule: {ticker} was just refreshed. Available again in ~{minutes} min.")

    st.divider()
    st.caption(
        "Built with Streamlit, ChromaDB, and Claude. "
        "[Source on GitHub](https://github.com/H-yz2022/marketpulse-ai)"
    )

# --- Data --------------------------------------------------------------------

price_rows = fetch_price_history(ticker)
filing_rows = fetch_filings(ticker)
tone_rows = fetch_filing_sentiment(ticker)

prices = pd.DataFrame([dict(r) for r in price_rows])
filings = pd.DataFrame([dict(r) for r in filing_rows])
tone = pd.DataFrame([dict(r) for r in tone_rows])
if not tone.empty:
    tone["net_tone"] = [_net_tone(lbl, s) for lbl, s in zip(tone["label"], tone["score"])]

# --- Header ------------------------------------------------------------------

st.title("MarketPulse AI")
st.caption(
    "Market and SEC-filing research assistant: price history, 10-K risk-factor sentiment, "
    "and grounded Q&A over the filings with Claude."
)

from_snapshot = ticker in seeded_tickers and ticker not in last_refreshed
if from_snapshot and (snap := _snapshot_date()):
    st.info(
        f"Showing a bundled snapshot of real data captured {_fmt_date(snap)}, so this page works "
        "even right after the app wakes up. Use **Refresh** in the sidebar to pull live data.",
        icon=":material/inventory_2:",
    )

# --- KPI row -----------------------------------------------------------------

if prices.empty:
    st.warning(
        f"No data for {ticker} yet. Use **Refresh {ticker}** in the sidebar to fetch it.",
        icon=":material/database_off:",
    )
else:
    closes = prices["close"].tolist()
    first, prev, last = closes[0], closes[-2] if len(closes) > 1 else closes[0], closes[-1]
    kpi_price, kpi_return, kpi_tone, kpi_filings = st.columns(4)
    kpi_price.metric(
        "Last close",
        f"${last:,.2f}",
        f"{(last / prev - 1) * 100:+.2f}% day",
        border=True,
        height="stretch",
        chart_data=closes,
        chart_type="line",
        help=f"Close on {_fmt_date(prices['trade_date'].iloc[-1])}",
    )
    kpi_return.metric(
        "Period return",
        f"{(last / first - 1) * 100:+.1f}%",
        f"since {_fmt_date(prices['trade_date'].iloc[0])}",
        delta_color="off",
        border=True,
        height="stretch",
        help=f"Range ${prices['low'].min():,.2f} - ${prices['high'].max():,.2f}",
    )
    if not tone.empty:
        latest = tone.iloc[-1]
        change = latest["net_tone"] - tone.iloc[-2]["net_tone"] if len(tone) > 1 else None
        kpi_tone.metric(
            "Latest 10-K tone",
            latest["label"].title(),
            f"{change:+.2f} vs prior 10-K" if change is not None else None,
            border=True,
            height="stretch",
            help="Net tone of the risk-factors section, from -1 (negative) to +1 (positive).",
        )
    kpi_filings.metric(
        "10-K filings indexed",
        len(filings),
        f"latest {_fmt_date(filings['filed_date'].max())}" if not filings.empty else None,
        delta_color="off",
        border=True,
        height="stretch",
    )

    # --- Charts --------------------------------------------------------------

    col_price, col_tone = st.columns([2, 1])
    with col_price:
        with st.container(border=True):
            st.subheader(f"{ticker} price history")
            fig = go.Figure(
                go.Candlestick(
                    x=prices["trade_date"],
                    open=prices["open"],
                    high=prices["high"],
                    low=prices["low"],
                    close=prices["close"],
                    increasing_line_color=GREEN,
                    decreasing_line_color=RED,
                    name=ticker,
                )
            )
            fig.update_layout(
                height=360, margin=dict(l=0, r=0, t=0, b=0), xaxis_rangeslider_visible=False, showlegend=False
            )
            st.plotly_chart(fig, config={"displayModeBar": False})

    with col_tone:
        with st.container(border=True):
            st.subheader("Tone by 10-K")
            if tone.empty:
                st.caption("No scored filings yet.")
            else:
                chart = (
                    alt.Chart(tone)
                    .mark_bar(cornerRadius=3)
                    .encode(
                        x=alt.X(
                            "filed_date:N",
                            title=None,
                            axis=alt.Axis(labelAngle=0, labelExpr="slice(datum.value, 0, 4)"),  # "2025-10-31" -> "2025"
                        ),
                        y=alt.Y("net_tone:Q", title="Net tone", scale=alt.Scale(domain=[-1, 1])),
                        color=alt.Color(
                            "label:N",
                            scale=alt.Scale(domain=["positive", "neutral", "negative"], range=[GREEN, GRAY, RED]),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("filed_date:N", title="Filed"),
                            alt.Tooltip("label:N", title="Tone"),
                            alt.Tooltip("score:Q", title="Confidence", format=".2f"),
                        ],
                    )
                    .properties(height=330)
                )
                st.altair_chart(chart)

# --- Filings table -----------------------------------------------------------

with st.container(border=True):
    st.subheader("Recent 10-K filings")
    if filings.empty:
        st.caption("No filings indexed yet.")
    else:
        table = filings[["filing_id", "filed_date", "form_type", "url", "excerpt"]].copy()
        if not tone.empty:
            table = table.merge(tone[["filing_id", "label", "net_tone"]], on="filing_id", how="left")
        else:
            table["label"], table["net_tone"] = None, None
        table["filed_date"] = pd.to_datetime(table["filed_date"])
        table["excerpt"] = table["excerpt"].str.slice(0, 240) + "…"
        st.dataframe(
            table[["filed_date", "form_type", "label", "net_tone", "excerpt", "url"]],
            hide_index=True,
            column_config={
                "filed_date": st.column_config.DateColumn("Filed", format="MMM D, YYYY"),
                "form_type": st.column_config.TextColumn("Form", width="small"),
                "label": st.column_config.TextColumn("Tone", width="small"),
                "net_tone": st.column_config.NumberColumn("Net tone", format="%+.2f", width="small"),
                "excerpt": st.column_config.TextColumn("Risk factors (opening)", width="large"),
                "url": st.column_config.LinkColumn("Document", display_text="SEC.gov", width="small"),
            },
        )

# --- Q&A ---------------------------------------------------------------------

with st.container(border=True):
    st.subheader(f"Ask about {ticker}'s risk factors")
    st.caption(
        "Answers come from Claude, grounded only in the indexed 10-K excerpts above, with sources. "
        f"To keep costs bounded this is capped at {settings.max_session_questions} questions per visit "
        f"and {settings.max_daily_questions} per day across all visitors."
    )

    # Two independent caps, both enforced *before* calling answer_question()
    # so a blocked question never reaches (and never bills) the Anthropic API:
    #   - a per-session cap, tracked in st.session_state (private to this
    #     visitor's browser tab, resets if they reload it)
    #   - a shared daily cap, tracked in the qa_usage table (shared across
    #     every visitor, resets at UTC midnight - and on a free host whose
    #     database is wiped on restart, whenever the app restarts, so the
    #     Anthropic Console spend limit remains the real backstop)
    st.session_state.setdefault("qa_session_count", 0)
    session_limit_hit = st.session_state.qa_session_count >= settings.max_session_questions
    daily_limit_hit = count_qa_usage_today() >= settings.max_daily_questions
    blocked = session_limit_hit or daily_limit_hit or filings.empty

    if session_limit_hit:
        st.warning(
            f"You've used this visit's {settings.max_session_questions} questions. Reload the page to start over.",
            icon=":material/block:",
        )
    elif daily_limit_hit:
        st.warning(
            "This demo has reached its shared daily question limit. Please check back tomorrow "
            "(it resets at midnight UTC).",
            icon=":material/block:",
        )

    with st.form("qa", border=False, enter_to_submit=True):
        question = st.text_input(
            "Question",
            placeholder="What supply-chain or geopolitical risks does the company flag?",
            disabled=blocked,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", icon=":material/send:", type="primary", disabled=blocked)

    if submitted and question.strip():
        try:
            if not has_ticker_documents(ticker):
                with st.spinner(f"Indexing {ticker}'s filings for search (first question only)..."):
                    index_ticker_filings(ticker)
            with st.spinner("Retrieving relevant excerpts and generating an answer..."):
                result = answer_question(question.strip(), where={"ticker": ticker.upper()})
        except Exception as e:  # noqa: BLE001 - a failed answer shouldn't take the page down
            st.error(f"Couldn't answer that: {e}", icon=":material/error:")
        else:
            if result["sources"]:  # only a retrieval hit actually calls (and bills) the API
                st.session_state.qa_session_count += 1
                log_qa_usage(ticker)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"Sources ({len(result['sources'])} excerpts)"):
                    for s in result["sources"]:
                        meta = s["metadata"]
                        filed = meta.get("filed_date")
                        st.caption(
                            f"{meta.get('form_type', 'Filing')}"
                            + (f" filed {_fmt_date(filed)}" if filed else "")
                            + f" · chunk {meta.get('chunk_index', '?')}"
                            + (f" · distance {s['distance']:.3f}" if s["distance"] is not None else "")
                        )
                        st.text(s["text"][:400])
