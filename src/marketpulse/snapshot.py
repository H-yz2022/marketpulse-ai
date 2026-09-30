"""A committed JSON snapshot of real data, so the dashboard is never empty.

Free hosts like Streamlit Community Cloud put an idle app to sleep and give
it a fresh container when it wakes up, which wipes the SQLite file. Without
this, every visitor after a restart landed on a page of "No data yet" boxes
until someone clicked the refresh button and waited for live ingestion.

`scripts/build_snapshot.py` runs the real pipeline and calls
`export_snapshot`; the dashboard calls `load_snapshot` once per process,
which fills in any ticker the database doesn't have yet (and leaves alone
any ticker that has live data already).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from marketpulse.config import settings
from marketpulse.db import (
    fetch_filings,
    fetch_price_history,
    fetch_sentiment_scores,
    init_db,
    insert_sentiment_score,
    tickers_with_prices,
    upsert_filing,
    upsert_price_history,
)

SNAPSHOT_VERSION = 1


def export_snapshot(tickers: Iterable[str], path: Optional[str] = None, db_path: Optional[str] = None) -> dict:
    """Write every stored row for `tickers` to a JSON snapshot file. Returns row counts."""
    tickers = [t.upper() for t in tickers]
    data = {
        "version": SNAPSHOT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tickers": tickers,
        "price_history": [],
        "filings": [],
        "sentiment_scores": [],
    }
    for ticker in tickers:
        data["price_history"] += [dict(r) for r in fetch_price_history(ticker, db_path=db_path)]
        data["filings"] += [dict(r) for r in fetch_filings(ticker, db_path=db_path)]
        # Drop the autoincrement id: it's local to the database the row came from.
        data["sentiment_scores"] += [
            {k: v for k, v in dict(r).items() if k != "id"} for r in fetch_sentiment_scores(ticker, db_path=db_path)
        ]

    out = Path(path or settings.snapshot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {key: len(data[key]) for key in ("price_history", "filings", "sentiment_scores")}


def read_snapshot(path: Optional[str] = None) -> Optional[dict]:
    """Parse the snapshot file, or return None if there isn't one."""
    p = Path(path or settings.snapshot_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_snapshot(path: Optional[str] = None, db_path: Optional[str] = None) -> list[str]:
    """Seed the database from the snapshot for every ticker it has no price data for.

    Returns the tickers that were seeded (empty if there was nothing to do).
    Tickers that already have data - e.g. from a live refresh - are left
    untouched, so this is safe to call on every startup.
    """
    data = read_snapshot(path)
    if data is None:
        return []

    init_db(db_path)
    existing = tickers_with_prices(db_path=db_path)
    to_seed = {t for t in data.get("tickers", []) if t not in existing}
    if not to_seed:
        return []

    upsert_price_history([r for r in data["price_history"] if r["ticker"] in to_seed], db_path=db_path)
    for row in data["filings"]:
        if row["ticker"] in to_seed:
            upsert_filing(row, db_path=db_path)
    for row in data["sentiment_scores"]:
        if row["ticker"] in to_seed:
            insert_sentiment_score(row, db_path=db_path)
    return sorted(to_seed)
