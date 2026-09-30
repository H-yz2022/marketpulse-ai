from marketpulse import refresh
from marketpulse.db import fetch_filings, fetch_sentiment_scores, upsert_filing


def _stub_ingestion(monkeypatch, excerpt="Revenue growth was strong."):
    def fake_prices(ticker, period="6mo", db_path=None):
        return 3

    def fake_filings(ticker, db_path=None):
        upsert_filing(
            {
                "filing_id": "new-10k",
                "ticker": ticker,
                "form_type": "10-K",
                "filed_date": "2026-01-15",
                "title": "Annual report",
                "url": "https://example.com",
                "excerpt": excerpt,
            },
            db_path=db_path,
        )
        return 1

    monkeypatch.setattr(refresh, "ingest_price_history", fake_prices)
    monkeypatch.setattr(refresh, "ingest_filings_for_ticker", fake_filings)
    monkeypatch.setattr("marketpulse.nlp.sentiment._load_finbert_pipeline", lambda: None)


def test_refresh_replaces_stale_filings_and_skips_index_when_asked(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    _stub_ingestion(monkeypatch)
    upsert_filing(
        {
            "filing_id": "stale-10k",
            "ticker": "AAPL",
            "form_type": "10-K",
            "filed_date": "2016-10-26",
            "title": "Old annual report",
            "url": "https://example.com/old",
            "excerpt": "old text",
        },
        db_path=db_path,
    )

    def no_vector_store(*_args, **_kwargs):
        raise AssertionError("index=False must not touch the vector store")

    monkeypatch.setattr(refresh, "delete_ticker_documents", no_vector_store)
    monkeypatch.setattr(refresh, "index_document", no_vector_store)

    counts = refresh.refresh_ticker("AAPL", index=False, db_path=db_path)

    assert counts == {"prices": 3, "filings": 1, "scored": 1, "indexed": 0}
    assert [f["filing_id"] for f in fetch_filings("AAPL", db_path=db_path)] == ["new-10k"]
    assert [s["source_id"] for s in fetch_sentiment_scores("AAPL", db_path=db_path)] == ["new-10k"]


def test_index_ticker_filings_records_filing_date(monkeypatch, tmp_path):
    db_path = str(tmp_path / "test.db")
    _stub_ingestion(monkeypatch)
    refresh.ingest_filings_for_ticker("AAPL", db_path=db_path)

    indexed = []
    monkeypatch.setattr(refresh, "index_document", lambda doc_id, text, metadata: indexed.append(metadata))

    assert refresh.index_ticker_filings("AAPL", db_path=db_path) == 1
    assert indexed == [{"ticker": "AAPL", "form_type": "10-K", "filed_date": "2026-01-15"}]
