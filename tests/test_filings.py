from marketpulse.db import fetch_filings
from marketpulse.ingestion import filings


class _FakeResponse:
    def __init__(self, json_data=None, text_data="", status_code=200):
        self._json = json_data
        self.text = text_data
        self.status_code = status_code

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
        return _FakeResponse(json_data=fake_data)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    cik = filings.get_cik_for_ticker("aapl")
    assert cik == "0000320193"


def test_search_filings(monkeypatch):
    fake_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "0000320193-26-000010:aapl10k.htm",
                    "_source": {
                        "adsh": "0000320193-26-000010",
                        "ciks": ["0000320193"],
                        "form": "10-K",
                        "file_date": "2026-01-15",
                        "display_names": ["Apple Inc. (AAPL)"],
                    },
                }
            ]
        }
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(json_data=fake_hits)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    hits = filings.search_filings("risk factors", forms="10-K")
    assert len(hits) == 1
    assert hits[0]["_source"]["form"] == "10-K"


def test_fetch_document_excerpt_extracts_risk_factors(monkeypatch):
    fake_html = (
        "<html><body><p>Item 1A. Risk Factors</p>"
        "<p>Our business faces risks related to supply chain and competition.</p>"
        "<p>Item 1B. Unresolved Staff Comments</p></body></html>"
    )

    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(text_data=fake_html)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    excerpt = filings.fetch_document_excerpt("0000320193", "0000320193-26-000010", "aapl10k.htm")
    assert "supply chain" in excerpt.lower()
    assert "unresolved staff comments" not in excerpt.lower()


def test_fetch_document_excerpt_skips_table_of_contents_entry(monkeypatch):
    # Real 10-Ks repeat "Item 1A. Risk Factors" in the table of contents
    # (immediately followed by a page number and "Item 1B") before the real
    # section appears later in the document. The excerpt should come from
    # the real section, not the two-word ToC line.
    fake_html = (
        "<html><body>"
        "<p>Item 1A. Risk Factors 12 Item 1B. Unresolved Staff Comments 14</p>"
        "<p>... table of contents continues ...</p>"
        "<p>Item 1A. Risk Factors</p>"
        "<p>Our business faces risks related to supply chain and competition.</p>"
        "<p>Item 1B. Unresolved Staff Comments</p>"
        "</body></html>"
    )

    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(text_data=fake_html)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    excerpt = filings.fetch_document_excerpt("0000320193", "0000320193-26-000010", "aapl10k.htm")
    assert "supply chain" in excerpt.lower()
    assert "table of contents continues" not in excerpt.lower()
    assert "unresolved staff comments" not in excerpt.lower()


def test_fetch_document_excerpt_skips_non_html(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        raise AssertionError("should not fetch a non-HTML document at all")

    monkeypatch.setattr(filings.requests, "get", fake_get)
    excerpt = filings.fetch_document_excerpt("0000320193", "0000320193-26-000010", "R1.xml")
    assert excerpt == ""


def test_ingest_filings_for_ticker(monkeypatch, tmp_path):
    fake_cik_data = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    fake_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "0000320193-26-000010:aapl10k.htm",
                    "_source": {
                        "adsh": "0000320193-26-000010",
                        "ciks": ["0000320193"],
                        "form": "10-K",
                        "file_date": "2026-01-15",
                        "display_names": ["Apple Inc. (AAPL)"],
                    },
                }
            ]
        }
    }
    fake_html = (
        "<html><body><p>Item 1A. Risk Factors</p>"
        "<p>Our business faces risks related to supply chain and competition.</p>"
        "<p>Item 1B. Unresolved Staff Comments</p></body></html>"
    )

    def fake_get(url, params=None, headers=None, timeout=None):
        if "company_tickers" in url:
            return _FakeResponse(json_data=fake_cik_data)
        if "efts.sec.gov" in url:
            return _FakeResponse(json_data=fake_hits)
        return _FakeResponse(text_data=fake_html)  # the filing document itself

    monkeypatch.setattr(filings.requests, "get", fake_get)

    db_path = str(tmp_path / "test.db")
    n = filings.ingest_filings_for_ticker("AAPL", db_path=db_path)
    assert n == 1

    rows = fetch_filings("AAPL", db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["form_type"] == "10-K"
    assert "supply chain" in rows[0]["excerpt"].lower()

def _fake_document(monkeypatch, html):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(text_data=html)

    monkeypatch.setattr(filings.requests, "get", fake_get)
    return filings.fetch_document_excerpt("0000789019", "0000789019-26-000001", "msft10k.htm")


def test_fetch_document_excerpt_small_caps_header_and_trailing_index(monkeypatch):
    # Microsoft-style 10-K: the real heading is styled in small caps, so its
    # text comes out as "RIS K FACTORS", and a cross-reference index at the
    # *end* of the document repeats "Item 1A. Risk Factors <page>". Earlier
    # versions matched only the ToC/index lines and stored "14" as the excerpt.
    excerpt = _fake_document(
        monkeypatch,
        "<html><body>"
        "<p>Item 1A. Risk Factors 14 Item 1B. Unresolved Staff Comments 29</p>"
        "<p>ITEM 1A. RIS K FACTORS</p>"
        "<p>Our operations and financial results are subject to various risks and uncertainties.</p>"
        "<p>ITEM 1B. UNRESOLVED STAFF COMMENTS</p>"
        "<p>Index: Item 1A. Risk Factors 14 Item 1B. Unresolved Staff Comments 29</p>"
        "</body></html>",
    )
    assert excerpt.startswith("Our operations and financial results")
    assert "unresolved" not in excerpt.lower()


def test_fetch_document_excerpt_strips_header_punctuation(monkeypatch):
    excerpt = _fake_document(
        monkeypatch,
        "<html><body><p>Item 1A. Risk Factors.</p>"
        "<p>The following discussion sets forth the material risk factors.</p>"
        "<p>Item 1B. Unresolved Staff Comments.</p></body></html>",
    )
    assert excerpt.startswith("The following discussion")


def test_fetch_document_excerpt_falls_back_when_only_toc_matches(monkeypatch):
    # Only the ToC line matches (the real heading says just "Risk factors"),
    # so the excerpt should be the document start, not the page range "8-18".
    excerpt = _fake_document(
        monkeypatch,
        "<html><body><p>Annual report cover page</p>"
        "<p>Item 1A Risk factors 8-18 Item 1B Unresolved SEC Staff comments 18</p>"
        "<p>Risk factors: our business is exposed to credit and market risk.</p></body></html>",
    )
    assert excerpt.startswith("Annual report cover page")


def test_primary_documents_drops_exhibits_dedupes_and_sorts_newest_first():
    hits = [
        {"_id": "acc-2022:jpm-20211231.htm", "_source": {"file_date": "2022-02-22", "file_type": "10-K"}},
        {"_id": "acc-2026:jpm-20251231.htm", "_source": {"file_date": "2026-02-13", "file_type": "10-K"}},
        {"_id": "acc-2026:exhibit1017.htm", "_source": {"file_date": "2026-02-13", "file_type": "EX-10.17"}},
        {"_id": "acc-2024:jpm-20231231.htm", "_source": {"file_date": "2024-02-16", "file_type": "10-K"}},
        {"_id": "acc-2024:jpm-20231231.htm", "_source": {"file_date": "2024-02-16", "file_type": "10-K"}},
    ]
    kept = filings._primary_documents(hits, "10-K")
    assert [h["_source"]["file_date"] for h in kept] == ["2026-02-13", "2024-02-16", "2022-02-22"]


def test_get_retries_transient_server_errors(monkeypatch):
    responses = [_FakeResponse(status_code=500), _FakeResponse(json_data={"ok": True})]
    monkeypatch.setattr(filings.requests, "get", lambda *a, **kw: responses.pop(0))
    monkeypatch.setattr(filings.time, "sleep", lambda _s: None)

    assert filings._get("https://efts.sec.gov/LATEST/search-index").json() == {"ok": True}
    assert responses == []
