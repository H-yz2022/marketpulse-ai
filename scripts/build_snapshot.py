"""Rebuild demo_data/snapshot.json from live data.

The dashboard loads this snapshot into an empty database on startup, so the
deployed app shows real charts right after a cold start or wake-from-sleep
(see src/marketpulse/snapshot.py). Re-run this whenever you want the demo's
data to be more current, then commit the updated JSON.

It builds into a throwaway temporary database, so your local
data/marketpulse.db and Chroma index are never touched.

Usage:
    python scripts/build_snapshot.py                    # MARKETPULSE_TICKERS (default AAPL,MSFT,JPM)
    python scripts/build_snapshot.py --tickers AAPL NVDA
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketpulse.config import settings  # noqa: E402
from marketpulse.refresh import refresh_ticker  # noqa: E402
from marketpulse.snapshot import export_snapshot  # noqa: E402


def main(tickers: list[str], period: str, out: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "snapshot.db")
        for ticker in tickers:
            print(f"\n=== {ticker} ===")
            refresh_ticker(ticker, period=period, index=False, db_path=db_path, log=print)
        counts = export_snapshot(tickers, path=out, db_path=db_path)
    print(f"\nWrote {out}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(settings.default_tickers))
    parser.add_argument("--period", default="6mo", help="yfinance history period, e.g. 6mo, 1y")
    parser.add_argument("--out", default=settings.snapshot_path)
    args = parser.parse_args()
    main([t.upper() for t in args.tickers], args.period, args.out)
