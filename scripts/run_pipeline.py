"""End-to-end pipeline: ingest price history + filings, score sentiment, index for RAG.

Usage:
    python scripts/run_pipeline.py --ticker AAPL
    python scripts/run_pipeline.py --ticker MSFT --period 1y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketpulse.db import (  # noqa: E402
    delete_filings_for_ticker,
    delete_sentiment_scores_for_ticker,
    fetch_filings,
    init_db,
)
from marketpulse.ingestion.filings import ingest_filings_for_ticker  # noqa: E402
from marketpulse.ingestion.market_data import ingest_price_history  # noqa: E402
from marketpulse.nlp.sentiment import score_and_store  # noqa: E402
from marketpulse.rag.pipeline import delete_ticker_documents, index_document  # noqa: E402


def run(ticker: str, period: str = "6mo") -> None:
    init_db()

    # Refresh, not append: clear this ticker's existing filings, sentiment
    # rows, and indexed chunks first. Every ingestion write below is an
    # upsert-by-ID, which never removes a filing that stops being re-fetched
    # or a sentiment row re-scored on a later run - those would otherwise
    # accumulate across repeated runs and dilute retrieval with stale data.
    print(f"[0/3] Clearing existing data for {ticker}...")
    delete_filings_for_ticker(ticker)
    delete_sentiment_scores_for_ticker(ticker, source_type="filing")
    delete_ticker_documents(ticker)

    print(f"[1/3] Fetching price history for {ticker}...")
    n_prices = ingest_price_history(ticker, period=period)
    print(f"      wrote {n_prices} rows")

    print(f"[2/3] Searching SEC filings for {ticker}...")
    n_filings = ingest_filings_for_ticker(ticker)
    print(f"      wrote {n_filings} filings")

    print("[3/3] Scoring sentiment and indexing filing text for RAG...")
    for filing in fetch_filings(ticker):
        text = filing["excerpt"] or filing["title"] or ""
        if not text:
            continue
        result = score_and_store(
            ticker, text, source_type="filing", source_id=filing["filing_id"]
        )
        index_document(
            filing["filing_id"],
            text,
            metadata={"ticker": ticker.upper(), "form_type": filing["form_type"] or ""},
        )
        print(f"      {filing['filing_id']}: {result.label} ({result.method})")

    print("\nDone. Launch the dashboard with:")
    print("      streamlit run src/marketpulse/dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period", default="6mo", help="yfinance history period, e.g. 6mo, 1y")
    args = parser.parse_args()
    run(args.ticker, args.period)