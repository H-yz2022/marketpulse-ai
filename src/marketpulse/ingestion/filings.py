"""SEC filing ingestion via the EDGAR full-text search API (efts.sec.gov).

Free, no API key required, but the SEC requires a descriptive User-Agent
header identifying you (see SEC_USER_AGENT in .env.example) and enforces a
rate limit of ~10 requests/second. See:
https://www.sec.gov/edgar/search/ (the human-facing UI this API powers)

Note: EDGAR's full-text search API only returns *metadata* about which
filings matched a query - despite "highlight" being a common convention for
this kind of search API, EDGAR doesn't return one. To get real filing text,
we fetch the matched document itself (using the "_id" field, which encodes
the accession number and filename, e.g. "0001628280-16-020309:a10-k.htm")
and try to isolate the "Item 1A. Risk Factors" section with a simple regex,
falling back to the start of the document body if that section can't be
found (e.g. this is a 10-Q, which doesn't always include one).
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

from marketpulse.config import settings
from marketpulse.db import upsert_filing

FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _loose(phrase: str) -> str:
    """Regex for `phrase` that tolerates a stray space inside any word.

    Some filers style headings in small caps by wrapping the first letter of
    each word in its own <span>, so the extracted text of Microsoft's
    "ITEM 1A. RISK FACTORS" heading comes out as "ITEM 1A. RIS K FACTORS".
    A plain "risk factors" regex never matches that, and the only hit left
    is the table-of-contents line - which is how MSFT excerpts used to end
    up as a bare page number like "14".
    """
    return r"\s*".join(r"\s?".join(re.escape(ch) for ch in word) for word in phrase.split())


_RISK_FACTORS_HEADER_RE = re.compile(_loose("item 1a") + r"\.?\s*" + _loose("risk factors"), re.IGNORECASE)
_ITEM_1B_RE = re.compile(_loose("item 1b") + r"\.?", re.IGNORECASE)


def _headers() -> dict:
    return {"User-Agent": settings.sec_user_agent}


def _get(url: str, *, params: Optional[dict] = None, timeout: int = 15, retries: int = 2) -> requests.Response:
    """GET with a short retry on transient failures.

    EDGAR's full-text search endpoint intermittently answers 500 for a
    request that succeeds a second later, which used to surface as a hard
    failure of the dashboard's "Fetch/refresh" button.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if resp.status_code < 500 or attempt == retries:
                resp.raise_for_status()
                return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
        time.sleep(1.0 * (attempt + 1))
    raise AssertionError("unreachable")  # pragma: no cover - loop always returns or raises


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """Look up a company's 10-digit zero-padded CIK from its ticker symbol."""
    resp = _get(COMPANY_TICKERS_URL)
    data = resp.json()  # dict of {"0": {"cik_str": ..., "ticker": "AAPL", "title": ...}, ...}
    ticker = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None


def search_filings(
    query: str,
    forms: str = "10-K",
    ciks: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    size: int = 10,
) -> list[dict]:
    """Search SEC EDGAR full-text search for filings matching a query.

    `query` searches the filing text itself (e.g. "risk factors"). Pass
    `ciks` (10-digit, zero-padded) to restrict to one company - use
    `get_cik_for_ticker` to resolve a ticker to a CIK first.

    Note this only returns *metadata* about matching filings (which
    document matched, its form type, filing date, etc) - not the matching
    text itself. See `fetch_document_excerpt` for that.
    """
    params: dict = {"q": query, "forms": forms, "size": size}
    if ciks:
        params["ciks"] = ciks
    if start_date and end_date:
        params.update({"dateRange": "custom", "startdt": start_date, "enddt": end_date})

    resp = _get(FULL_TEXT_SEARCH_URL, params=params)
    payload = resp.json()
    return payload.get("hits", {}).get("hits", [])


# If even the longest "section" is shorter than this, every match was a
# table-of-contents or cross-reference line ("Item 1A. Risk Factors 14 Item
# 1B...", i.e. just a page number or range) and the real header uses some
# other wording - fall back to the document start rather than store "14".
_MIN_SECTION_CHARS = 40


def fetch_document_excerpt(cik: str, accession_no: str, filename: str, max_chars: int = 12000) -> str:
    """Fetch a filing document and return a plain-text excerpt from it.

    Skips non-HTML documents (a filing's matched file can be a raw XBRL
    ".xml" data file instead of the readable ".htm" filing itself - those
    aren't prose, can be huge, and aren't worth trying to parse as text).
    Tries to isolate the "Item 1A. Risk Factors" section; falls back to the
    start of the document body if that section isn't found.

    `max_chars` defaults to ~12k characters (roughly the first 15 retrieval
    chunks): enough to reach well past the section's boilerplate intro into
    the company-specific risks, without making every filing a huge index.
    """
    if not filename.lower().endswith((".htm", ".html")):
        return ""

    cik_num = str(int(cik))  # EDGAR's Archives path wants no leading zeros
    accession_nodash = accession_no.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accession_nodash}/{filename}"
    resp = _get(url, timeout=30)

    # lxml is a compiled parser and handles large real-world filing HTML
    # (some run several MB) far faster and more reliably than the pure
    # Python stdlib "html.parser".
    soup = BeautifulSoup(resp.text, "lxml")
    body_text = soup.get_text(separator=" ", strip=True)
    body_text = re.sub(r"\s+", " ", body_text)

    section = _longest_risk_factors_section(body_text)
    if len(section) >= _MIN_SECTION_CHARS:
        return section[:max_chars]
    return body_text[:max_chars]


def _longest_risk_factors_section(body_text: str) -> str:
    """Return the text between an "Item 1A. Risk Factors" header and the next
    "Item 1B", choosing the longest such span in the document.

    A 10-K mentions "Item 1A. Risk Factors" in more than one place: the table
    of contents up front, sometimes a cross-reference index at the *end*
    (Microsoft, JPMorgan), and the real section header. Every place except
    the real header is followed almost immediately by "Item 1B", so the
    real section is simply the longest span. (Picking the first or last
    occurrence instead - as earlier versions did - lands on a ToC or index
    line and yields an "excerpt" that's just a page number.) Spans that
    actually reach an "Item 1B" win over ones that run off the end of the
    document, so a stray cross-reference late in the filing can't beat the
    real, properly terminated section.
    """
    best: tuple[bool, int, str] = (False, 0, "")
    for header in _RISK_FACTORS_HEADER_RE.finditer(body_text):
        end_match = _ITEM_1B_RE.search(body_text, header.end())
        end = end_match.start() if end_match else len(body_text)
        # lstrip: some headers end "Risk Factors." - don't start the excerpt with ". "
        section = body_text[header.end():end].strip().lstrip(".:;-–— ")
        candidate = (end_match is not None, len(section), section)
        if candidate[:2] > best[:2]:
            best = candidate
    return best[2]


def ingest_filings_for_ticker(
    ticker: str,
    query: str = "risk factors",
    forms: str = "10-K",
    size: int = 5,
    lookback_years: int = 6,
    db_path: Optional[str] = None,
) -> int:
    """Look up a ticker's CIK, fetch its most recent filings' excerpt text, and persist.

    Defaults to 10-K filings only. A 10-Q's "Item 1A. Risk Factors" section is
    usually just a pointer ("see Part I, Item 1A of the year's Form 10-K")
    plus a short list of *changes* since then - often "None." - so indexing
    10-Qs alongside 10-Ks mostly adds boilerplate noise that crowds out the
    actual, detailed risk-factor text a 10-K contains. Pass forms="10-K,10-Q"
    explicitly if you specifically want quarter-over-quarter change language.

    EDGAR full-text search ranks hits by *relevance*, not date, and returns
    exhibits (e.g. an "EX-10.17" attached to a 10-K) as separate hits. Asked
    for 5 results unbounded, it used to hand back 10-Ks from 2003-2016. So
    this searches a recent `lookback_years` window, over-fetches, keeps only
    each filing's primary document, and takes the newest `size` by date.
    """
    cik = get_cik_for_ticker(ticker)
    today = date.today()
    start = today - timedelta(days=365 * lookback_years)
    raw_hits = search_filings(
        query,
        forms=forms,
        ciks=cik,
        start_date=start.isoformat(),
        end_date=today.isoformat(),
        size=max(size * 4, 20),
    )
    hits = _primary_documents(raw_hits, forms)[:size]

    count = 0
    for hit in hits:
        source = hit.get("_source", {})
        raw_id = hit.get("_id", "")
        accession_no, _, filename = raw_id.partition(":")
        filer_cik = cik or next(iter(source.get("ciks", [])), None)

        url = ""
        excerpt = ""
        if filer_cik and accession_no and filename:
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(filer_cik)}/"
                f"{accession_no.replace('-', '')}/{filename}"
            )
            try:
                excerpt = fetch_document_excerpt(filer_cik, accession_no, filename)
            except Exception:  # noqa: BLE001 - network/parsing hiccups shouldn't kill the whole run
                excerpt = ""

        display_name = source.get("display_names", [""])[0] if source.get("display_names") else ""
        if not excerpt:
            excerpt = display_name

        upsert_filing(
            {
                "filing_id": raw_id or f"{ticker}-{accession_no}",
                "ticker": ticker.upper(),
                "form_type": source.get("form", ""),
                "filed_date": source.get("file_date", ""),
                "title": display_name,
                "url": url,
                "excerpt": excerpt,
            },
            db_path=db_path,
        )
        count += 1
    return count


def _primary_documents(hits: list[dict], forms: str) -> list[dict]:
    """Keep one hit per filing - its main document, not an exhibit - newest first."""
    wanted = {f.strip() for f in forms.split(",") if f.strip()}
    seen: set[str] = set()
    kept = []
    for hit in hits:
        source = hit.get("_source", {})
        # `file_type` is the document's own type within the filing: "10-K" for
        # the main document, "EX-10.17" etc. for exhibits. Missing means the
        # API didn't say, so give it the benefit of the doubt.
        file_type = source.get("file_type")
        if file_type is not None and file_type not in wanted:
            continue
        accession_no = hit.get("_id", "").partition(":")[0]
        if accession_no in seen:
            continue
        seen.add(accession_no)
        kept.append(hit)
    return sorted(kept, key=lambda h: h.get("_source", {}).get("file_date", ""), reverse=True)