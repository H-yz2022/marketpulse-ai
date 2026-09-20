"""Streamlit BI dashboard for MarketPulse AI.

Run with: streamlit run src/marketpulse/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run` to find the package without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketpulse.config import settings
from marketpulse.db import (
    delete_filings_for_ticker,
    delete_sentiment_scores_for_ticker,
    fetch_filings,
    fetch_price_history,
    fetch_sentiment_scores,
    init_db,
)
from marketpulse.ingestion.filings import ingest_filings_for_ticker
from marketpulse.ingestion.market_data import ingest_price_history
from marketpulse.nlp.sentiment import score_and_store
from marketpulse.rag.pipeline import answer_question, delete_ticker_documents, index_document

st.set_page_config(page_title="MarketPulse AI", layout="wide")
st.title("MarketPulse AI — Market & Filing Research Assistant")

tickers = list(settings.default_tickers) or ["AAPL"]
ticker = st.sidebar.selectbox("Ticker", options=tickers)


def _fetch_live_data(ticker: str) -> int:
    """Run the same ingestion steps as scripts/run_pipeline.py, from inside
    the running app. This lets a freshly deployed dashboard (empty database,
    e.g. on Streamlit Community Cloud) populate itself with one click,
    instead of requiring shell access to run the CLI script.

    This is a *refresh*, not an append: this ticker's existing filings,
    sentiment rows, and indexed chunks are cleared first. Every ingestion
    write elsewhere in this app is an upsert-by-ID, which never removes a
    filing that stops being re-fetched (an older 10-K displaced by a newer
    one) or a sentiment row re-scored on a later run - those would otherwise
    accumulate forever and dilute retrieval/analysis with stale data. Wiping
    this ticker's data before rebuilding it is what makes clicking the
    button repeatedly safe, with no manual "reboot the app" step required.
    """
    init_db()
    delete_filings_for_ticker(ticker)
    delete_sentiment_scores_for_ticker(ticker, source_type="filing")
    delete_ticker_documents(ticker)

    ingest_price_history(ticker)
    ingest_filings_for_ticker(ticker)
    n_indexed = 0
    for filing in fetch_filings(ticker):
        text = filing["excerpt"] or filing["title"] or ""
        if not text:
            continue
        score_and_store(ticker, text, source_type="filing", source_id=filing["filing_id"])
        index_document(
            filing["filing_id"], text, metadata={"ticker": ticker.upper(), "form_type": filing["form_type"] or ""}
        )
        n_indexed += 1
    return n_indexed


with st.sidebar:
    st.caption(
        "No data for this ticker yet? Fetch live price history and SEC filings below "
        "(takes ~10-20s; longer on the very first click while dependencies warm up)."
    )
    if st.button(f"Fetch/refresh live data for {ticker}"):
        with st.spinner(f"Fetching price history and SEC filings for {ticker}..."):
            try:
                n_indexed = _fetch_live_data(ticker)
                st.success(f"Indexed {n_indexed} filing(s) for {ticker}.")
                st.rerun()
            except Exception as e:  # noqa: BLE001 - surface any ingestion failure in the UI, don't crash the app
                st.error(f"Couldn't fetch live data for {ticker}: {e}")

price_rows = fetch_price_history(ticker)
sentiment_rows = fetch_sentiment_scores(ticker)
filing_rows = fetch_filings(ticker)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{ticker} price history")
    if price_rows:
        df = pd.DataFrame([dict(r) for r in price_rows])
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["trade_date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                )
            ]
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price history yet.")

with col2:
    st.subheader("Sentiment trend")
    if sentiment_rows:
        sdf = pd.DataFrame([dict(r) for r in sentiment_rows])
        st.line_chart(sdf.set_index("scored_date")["score"])
        st.dataframe(sdf[["scored_date", "label", "score", "source_type"]], hide_index=True)
    else:
        st.info("No sentiment scores yet.")

st.subheader("Recent filings")
if filing_rows:
    fdf = pd.DataFrame([dict(r) for r in filing_rows])
    st.dataframe(fdf[["filed_date", "form_type", "title", "url"]], hide_index=True)
else:
    st.info("No filings indexed yet.")

st.subheader("Ask a question about this company's filings")
question = st.text_input(
    "Question", placeholder="What risk factors did the company flag this year?"
)
if st.button("Ask") and question:
    with st.spinner("Retrieving and generating an answer..."):
        try:
            result = answer_question(question)
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.caption(f"{s['metadata']} (distance={s['distance']})")
                        st.text(s["text"][:300])
        except RuntimeError as e:
            st.error(str(e))