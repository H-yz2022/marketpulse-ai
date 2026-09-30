"""Configuration for MarketPulse AI, loaded from environment variables.

Copy `.env.example` to `.env` and fill in real values. Every setting has a
sensible default so the app runs (in a degraded/offline mode) even with no
`.env` file at all.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; env vars still work without it
    pass

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _parse_tickers(raw: str) -> tuple[str, ...]:
    return tuple(t.strip().upper() for t in raw.split(",") if t.strip())


@dataclass(frozen=True)
class Settings:
    db_path: str = field(default_factory=lambda: os.getenv("MARKETPULSE_DB_PATH", str(DATA_DIR / "marketpulse.db")))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"))
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("SEC_USER_AGENT", "MarketPulse AI research@example.com")
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
    )
    default_tickers: tuple = field(
        default_factory=lambda: _parse_tickers(os.getenv("MARKETPULSE_TICKERS", "AAPL,MSFT,JPM"))
    )
    # A committed JSON snapshot of real price/filing/sentiment data (built by
    # scripts/build_snapshot.py). The dashboard loads it into an empty
    # database on startup, so a freshly deployed or just-woken-up instance -
    # whose SQLite file is wiped on every restart - shows real charts
    # immediately instead of a page of "No data yet" boxes.
    snapshot_path: str = field(
        default_factory=lambda: os.getenv("MARKETPULSE_SNAPSHOT_PATH", str(BASE_DIR / "demo_data" / "snapshot.json"))
    )
    # Only the "Ask a question" button calls the billed Anthropic API - every
    # other feature (price history, filing ingestion, sentiment scoring) is
    # free/local. These two caps exist so a public deployment (e.g. on
    # Streamlit Community Cloud, where anyone with the link can click Ask)
    # can't run up an unbounded API bill if a lot of people - or a script -
    # hit the button many times. Defaults are deliberately generous for a
    # single demo visitor but low enough to bound worst-case spend.
    max_session_questions: int = field(
        default_factory=lambda: int(os.getenv("MARKETPULSE_MAX_SESSION_QUESTIONS", "5"))
    )
    max_daily_questions: int = field(
        default_factory=lambda: int(os.getenv("MARKETPULSE_MAX_DAILY_QUESTIONS", "30"))
    )


settings = Settings()
