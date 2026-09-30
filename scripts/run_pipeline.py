"""End-to-end pipeline: ingest price history + filings, score sentiment, index for RAG.

Usage:
    python scripts/run_pipeline.py --ticker AAPL
    python scripts/run_pipeline.py --ticker MSFT --period 1y
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketpulse.refresh import refresh_ticker  # noqa: E402


def run(ticker: str, period: str = "6mo") -> None:
    counts = refresh_ticker(ticker, period=period, log=print)
    print(f"\nDone: {counts}. Launch the dashboard with:")
    print("      streamlit run src/marketpulse/dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--period", default="6mo", help="yfinance history period, e.g. 6mo, 1y")
    args = parser.parse_args()
    run(args.ticker, args.period)
