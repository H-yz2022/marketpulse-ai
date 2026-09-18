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
from marketpulse.db import fetch_filings, fetch_price_history, fetch_sentiment_scores
from marketpulse.rag.pipeline import answer_question

st.set_page_config(page_title="MarketPulse AI", layout="wide")
st.title("MarketPulse AI — Market & Filing Research Assistant")

tickers = list(settings.default_tickers) or ["AAPL"]
ticker = st.sidebar.selectbox("Ticker", options=tickers)
st.sidebar.caption(
    "No data for this ticker yet? Run `python scripts/run_pipeline.py --ticker "
    f"{ticker}` first."
)

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
