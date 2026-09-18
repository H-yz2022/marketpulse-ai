"""SEC filing ingestion via the EDGAR full-text search API (efts.sec.gov).

Free, no API key required, but the SEC requires a descriptive User-Agent
header identifying you (see SEC_USER_AGENT in .env.example) and enforces a
rate limit of ~10 requests/second. See:
https://www.sec.gov/edgar/search/ (the human-facing UI this API powers)
"""
from __future__ import annotations

from typing import Optional

import requests

from marketpulse.config import settings
from marketpulse.db import upsert_filing

FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _headers() -> dict:
    return {"User-Agent": settings.sec_user_agent}


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Look up a company's 10-digit zero-padded CIK from its ticker symbol."""
    resp = requests.get(COMPANY_TICKERS_URL, headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()  # dict of {"0": {"cik_str": ..., "ticker": "AAPL", "title": ...}, ...}
    ticker = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None


def search_filings(
    query: str,
    forms: str = "10-K,10-Q",
    ciks: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    size: int = 10,
) -> list[dict]:
    """Search SEC EDGAR full-text search for filings matching a query.

    `query` searches the filing text itself (e.g. "supply chain risk").
    Pass `ciks` (10-digit, zero-padded) to restrict to one company - use
    `get_cik_for_ticker` to resolve a ticker to a CIK first.
    """
    params: dict = {"q": query, "forms": forms, "size": size}
    if ciks:
        params["ciks"] = ciks
    if start_date and end_date:
        params.update({"dateRange": "custom", "startdt": start_date, "enddt": end_date})

    resp = requests.get(FULL_TEXT_SEARCH_URL, params=params, headers=_headers(), timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("hits", {}).get("hits", [])


def ingest_filings_for_ticker(
    ticker: str,
    query: str = "risk factors",
    forms: str = "10-K,10-Q",
    size: int = 5,
    db_path: Optional[str] = None,
) -> int:
    """Look up a ticker's CIK, search recent filings, and persist metadata + excerpt."""
    cik = get_cik_for_ticker(ticker)
    hits = search_filings(query, forms=forms, ciks=cik, size=size)

    count = 0
    for hit in hits:
        source = hit.get("_source", {})
        filing_id = hit.get("_id", "")
        accession_no = source.get("adsh", "").replace("-", "")
        cik_no_pad = source.get("cik", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_no_pad}/{accession_no}/"
            if accession_no and cik_no_pad
            else ""
        )
        upsert_filing(
            {
                "filing_id": filing_id or f"{ticker}-{source.get('adsh', '')}",
                "ticker": ticker.upper(),
                "form_type": source.get("form", ""),
                "filed_date": source.get("file_date", ""),
                "title": source.get("display_names", [""])[0] if source.get("display_names") else "",
                "url": url,
                "excerpt": " ".join(hit.get("highlight", {}).get("text", []))
                if hit.get("highlight")
                else "",
            },
            db_path=db_path,
        )
        count += 1
    return count
