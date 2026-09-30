"""The ingest -> score -> index steps for one ticker.

Shared by the CLI (scripts/run_pipeline.py), the dashboard's refresh button,
and the demo-snapshot builder (scripts/build_snapshot.py), so all three
always run exactly the same pipeline.
"""
from __future__ import annotations

from typing import Callable, Optional

from marketpulse.db import (
    delete_filings_for_ticker,
    delete_sentiment_scores_for_ticker,
    fetch_filings,
    init_db,
)
from marketpulse.ingestion.filings import ingest_filings_for_ticker
from marketpulse.ingestion.market_data import ingest_price_history
from marketpulse.nlp.sentiment import score_and_store
from marketpulse.rag.pipeline import delete_ticker_documents, index_document


def _filing_text(filing) -> str:
    return filing["excerpt"] or filing["title"] or ""


def index_ticker_filings(ticker: str, db_path: Optional[str] = None) -> int:
    """Chunk and embed every stored filing for a ticker into the vector store.
    Returns the number of filings indexed."""
    n_indexed = 0
    for filing in fetch_filings(ticker, db_path=db_path):
        text = _filing_text(filing)
        if not text:
            continue
        metadata = {
            "ticker": ticker.upper(),
            "form_type": filing["form_type"] or "",
            "filed_date": filing["filed_date"] or "",
        }
        index_document(filing["filing_id"], text, metadata=metadata)
        n_indexed += 1
    return n_indexed


def refresh_ticker(
    ticker: str,
    period: str = "6mo",
    index: bool = True,
    db_path: Optional[str] = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> dict:
    """Rebuild one ticker's data from live sources. Returns per-step counts.

    This is a *refresh*, not an append: the ticker's existing filings,
    sentiment rows, and indexed chunks are cleared first. Every ingestion
    write is an upsert-by-ID, which never removes a filing that stops being
    re-fetched (an older 10-K displaced by a newer one) or a sentiment row
    re-scored on a later run - those would otherwise accumulate forever and
    dilute retrieval/analysis with stale data.

    Pass `index=False` to skip the vector store entirely (the snapshot
    builder only needs the SQL tables).
    """
    init_db(db_path)
    log(f"Clearing existing data for {ticker}...")
    delete_filings_for_ticker(ticker, db_path=db_path)
    delete_sentiment_scores_for_ticker(ticker, source_type="filing", db_path=db_path)
    if index:
        delete_ticker_documents(ticker)

    log(f"Fetching price history for {ticker}...")
    n_prices = ingest_price_history(ticker, period=period, db_path=db_path)
    log(f"  wrote {n_prices} rows")

    log(f"Fetching SEC filings for {ticker}...")
    n_filings = ingest_filings_for_ticker(ticker, db_path=db_path)
    log(f"  wrote {n_filings} filings")

    log("Scoring filing sentiment...")
    n_scored = 0
    for filing in fetch_filings(ticker, db_path=db_path):
        text = _filing_text(filing)
        if not text:
            continue
        result = score_and_store(ticker, text, source_type="filing", source_id=filing["filing_id"], db_path=db_path)
        log(f"  {filing['filed_date']} {filing['form_type']}: {result.label} ({result.method}, {len(text):,} chars)")
        n_scored += 1

    n_indexed = 0
    if index:
        log("Indexing filing text for Q&A...")
        n_indexed = index_ticker_filings(ticker, db_path=db_path)

    return {"prices": n_prices, "filings": n_filings, "scored": n_scored, "indexed": n_indexed}
