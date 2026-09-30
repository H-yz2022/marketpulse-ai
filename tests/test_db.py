from marketpulse.db import (
    count_qa_usage_today,
    delete_filings_for_ticker,
    delete_sentiment_scores_for_ticker,
    fetch_filing_sentiment,
    fetch_filings,
    fetch_price_history,
    fetch_sentiment_scores,
    insert_sentiment_score,
    log_qa_usage,
    tickers_with_prices,
    upsert_filing,
    upsert_price_history,
)


def test_price_history_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    rows = [
        {
            "ticker": "AAPL",
            "trade_date": "2026-01-02",
            "open": 1,
            "high": 2,
            "low": 0.5,
            "close": 1.5,
            "volume": 1000,
        }
    ]
    written = upsert_price_history(rows, db_path=db_path)
    assert written == 1

    fetched = fetch_price_history("aapl", db_path=db_path)
    assert len(fetched) == 1
    assert fetched[0]["close"] == 1.5


def test_price_history_upsert_updates_existing_row(tmp_path):
    db_path = str(tmp_path / "test.db")
    row = {
        "ticker": "AAPL",
        "trade_date": "2026-01-02",
        "open": 1,
        "high": 2,
        "low": 0.5,
        "close": 1.5,
        "volume": 1000,
    }
    upsert_price_history([row], db_path=db_path)
    upsert_price_history([dict(row, close=9.99)], db_path=db_path)

    fetched = fetch_price_history("AAPL", db_path=db_path)
    assert len(fetched) == 1  # same primary key -> updated, not duplicated
    assert fetched[0]["close"] == 9.99


def test_filing_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    upsert_filing(
        {
            "filing_id": "f1",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filed_date": "2026-01-01",
            "title": "Annual report",
            "url": "https://example.com",
            "excerpt": "some risk text",
        },
        db_path=db_path,
    )
    filings = fetch_filings("AAPL", db_path=db_path)
    assert len(filings) == 1
    assert filings[0]["form_type"] == "10-K"


def test_sentiment_score_roundtrip(tmp_path):
    db_path = str(tmp_path / "test.db")
    row_id = insert_sentiment_score(
        {
            "ticker": "AAPL",
            "source_type": "filing",
            "source_id": "f1",
            "scored_date": "2026-01-01",
            "label": "positive",
            "score": 0.9,
            "text_snippet": "great quarter",
        },
        db_path=db_path,
    )
    assert row_id > 0

    scores = fetch_sentiment_scores("AAPL", db_path=db_path)
    assert len(scores) == 1
    assert scores[0]["label"] == "positive"


def test_delete_filings_for_ticker(tmp_path):
    db_path = str(tmp_path / "test.db")
    upsert_filing(
        {
            "filing_id": "f1",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filed_date": "2026-01-01",
            "title": "Annual report",
            "url": "https://example.com",
            "excerpt": "some risk text",
        },
        db_path=db_path,
    )
    upsert_filing(
        {
            "filing_id": "f2",
            "ticker": "MSFT",
            "form_type": "10-K",
            "filed_date": "2026-01-01",
            "title": "Annual report",
            "url": "https://example.com",
            "excerpt": "other risk text",
        },
        db_path=db_path,
    )

    deleted = delete_filings_for_ticker("aapl", db_path=db_path)
    assert deleted == 1
    assert fetch_filings("AAPL", db_path=db_path) == []
    # a different ticker's filings are untouched
    assert len(fetch_filings("MSFT", db_path=db_path)) == 1


def test_delete_sentiment_scores_for_ticker_scoped_to_source_type(tmp_path):
    db_path = str(tmp_path / "test.db")
    insert_sentiment_score(
        {
            "ticker": "AAPL",
            "source_type": "filing",
            "source_id": "f1",
            "scored_date": "2026-01-01",
            "label": "positive",
            "score": 0.9,
            "text_snippet": "great quarter",
        },
        db_path=db_path,
    )
    insert_sentiment_score(
        {
            "ticker": "AAPL",
            "source_type": "news",
            "source_id": "n1",
            "scored_date": "2026-01-01",
            "label": "neutral",
            "score": 0.5,
            "text_snippet": "some headline",
        },
        db_path=db_path,
    )

    deleted = delete_sentiment_scores_for_ticker("AAPL", source_type="filing", db_path=db_path)
    assert deleted == 1

    remaining = fetch_sentiment_scores("AAPL", db_path=db_path)
    assert len(remaining) == 1
    assert remaining[0]["source_type"] == "news"

def test_qa_usage_starts_at_zero(tmp_path):
    db_path = str(tmp_path / "test.db")
    assert count_qa_usage_today(db_path=db_path) == 0


def test_qa_usage_counts_logged_questions(tmp_path):
    db_path = str(tmp_path / "test.db")
    log_qa_usage(ticker="AAPL", db_path=db_path)
    log_qa_usage(ticker="MSFT", db_path=db_path)
    log_qa_usage(db_path=db_path)  # ticker is optional
    assert count_qa_usage_today(db_path=db_path) == 3


def test_fetch_filing_sentiment_dates_scores_by_filing(tmp_path):
    db_path = str(tmp_path / "test.db")
    for filing_id, filed in [("f2025", "2025-10-31"), ("f2024", "2024-11-01")]:
        upsert_filing(
            {
                "filing_id": filing_id,
                "ticker": "AAPL",
                "form_type": "10-K",
                "filed_date": filed,
                "title": "Annual report",
                "url": "https://example.com",
                "excerpt": "risk text",
            },
            db_path=db_path,
        )
        # Both scored on the same day - the join should still spread them out by filing date.
        insert_sentiment_score(
            {
                "ticker": "AAPL",
                "source_type": "filing",
                "source_id": filing_id,
                "scored_date": "2026-09-29",
                "label": "negative",
                "score": 0.7,
                "text_snippet": "risk text",
            },
            db_path=db_path,
        )

    rows = fetch_filing_sentiment("aapl", db_path=db_path)
    assert [(r["filing_id"], r["filed_date"]) for r in rows] == [("f2024", "2024-11-01"), ("f2025", "2025-10-31")]


def test_tickers_with_prices(tmp_path):
    db_path = str(tmp_path / "test.db")
    assert tickers_with_prices(db_path=db_path) == set()
    upsert_price_history(
        [{"ticker": "MSFT", "trade_date": "2026-01-02", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}],
        db_path=db_path,
    )
    assert tickers_with_prices(db_path=db_path) == {"MSFT"}
