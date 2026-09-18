from marketpulse.db import fetch_filings
from marketpulse.ingestion import filings


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


def test_get_cik_for_ticker(monkeypatch):
    fake_data = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }

    def fake_get(url, headers=None, timeout=None, params=None):
        return _FakeResponse(fake_data)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    cik = filings.get_cik_for_ticker("aapl")
    assert cik == "0000320193"


def test_search_filings(monkeypatch):
    fake_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "abc",
                    "_source": {
                        "adsh": "0000320193-26-000010",
                        "cik": "320193",
                        "form": "10-K",
                        "file_date": "2026-01-15",
                        "display_names": ["Apple Inc. (AAPL)"],
                    },
                    "highlight": {"text": ["...supply chain risk..."]},
                }
            ]
        }
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(fake_hits)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    hits = filings.search_filings("supply chain", forms="10-K")
    assert len(hits) == 1
    assert hits[0]["_source"]["form"] == "10-K"


def test_ingest_filings_for_ticker(monkeypatch, tmp_path):
    fake_cik_data = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    fake_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "abc",
                    "_source": {
                        "adsh": "0000320193-26-000010",
                        "cik": "320193",
                        "form": "10-K",
                        "file_date": "2026-01-15",
                        "display_names": ["Apple Inc. (AAPL)"],
                    },
                    "highlight": {"text": ["...supply chain risk..."]},
                }
            ]
        }
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        if "company_tickers" in url:
            return _FakeResponse(fake_cik_data)
        return _FakeResponse(fake_hits)

    monkeypatch.setattr(filings.requests, "get", fake_get)

    db_path = str(tmp_path / "test.db")
    n = filings.ingest_filings_for_ticker("AAPL", db_path=db_path)
    assert n == 1

    rows = fetch_filings("AAPL", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["form_type"] == "10-K"
