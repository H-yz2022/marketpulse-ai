import json

from marketpulse.db import (
    fetch_filings,
    fetch_price_history,
    fetch_sentiment_scores,
    insert_sentiment_score,
    upsert_filing,
    upsert_price_history,
)
from marketpulse.snapshot import export_snapshot, load_snapshot


def _populate(db_path, ticker, close):
    price = {"ticker": ticker, "trade_date": "2026-01-02", "open": 1, "high": 2, "low": 0.5, "close": close}
    upsert_price_history([dict(price, volume=10)], db_path=db_path)
    upsert_filing(
        {
            "filing_id": f"{ticker}-10k",
            "ticker": ticker,
            "form_type": "10-K",
            "filed_date": "2025-10-31",
            "title": "Annual report",
            "url": "https://example.com",
            "excerpt": "risk text",
        },
        db_path=db_path,
    )
    insert_sentiment_score(
        {
            "ticker": ticker,
            "source_type": "filing",
            "source_id": f"{ticker}-10k",
            "scored_date": "2026-09-29",
            "label": "negative",
            "score": 0.8,
            "text_snippet": "risk text",
        },
        db_path=db_path,
    )


def test_export_then_load_roundtrip(tmp_path):
    source_db, target_db = str(tmp_path / "source.db"), str(tmp_path / "target.db")
    snapshot = str(tmp_path / "snapshot.json")
    _populate(source_db, "AAPL", close=1.5)
    _populate(source_db, "MSFT", close=9.0)

    counts = export_snapshot(["aapl", "msft"], path=snapshot, db_path=source_db)
    assert counts == {"price_history": 2, "filings": 2, "sentiment_scores": 2}
    assert json.loads(open(snapshot, encoding="utf-8").read())["tickers"] == ["AAPL", "MSFT"]

    assert load_snapshot(path=snapshot, db_path=target_db) == ["AAPL", "MSFT"]
    assert fetch_price_history("MSFT", db_path=target_db)[0]["close"] == 9.0
    assert fetch_filings("AAPL", db_path=target_db)[0]["excerpt"] == "risk text"
    assert fetch_sentiment_scores("AAPL", db_path=target_db)[0]["label"] == "negative"


def test_load_is_idempotent_and_skips_tickers_with_live_data(tmp_path):
    source_db, target_db = str(tmp_path / "source.db"), str(tmp_path / "target.db")
    snapshot = str(tmp_path / "snapshot.json")
    _populate(source_db, "AAPL", close=1.5)
    _populate(source_db, "MSFT", close=9.0)
    export_snapshot(["AAPL", "MSFT"], path=snapshot, db_path=source_db)

    _populate(target_db, "AAPL", close=222.0)  # e.g. a live refresh already ran for AAPL

    assert load_snapshot(path=snapshot, db_path=target_db) == ["MSFT"]
    assert load_snapshot(path=snapshot, db_path=target_db) == []  # second startup: nothing to do
    assert fetch_price_history("AAPL", db_path=target_db)[0]["close"] == 222.0
    assert len(fetch_sentiment_scores("MSFT", db_path=target_db)) == 1  # not duplicated


def test_load_without_snapshot_file_is_a_no_op(tmp_path):
    assert load_snapshot(path=str(tmp_path / "missing.json"), db_path=str(tmp_path / "t.db")) == []
