"""Headless smoke test: the dashboard renders real data on a brand-new, empty database.

That's exactly the state a free host leaves the app in after a restart or a
wake-from-sleep, so this guards the "never show an empty page" behavior.
"""
import dataclasses
import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

import marketpulse.config
import marketpulse.db
import marketpulse.rag.pipeline
import marketpulse.snapshot

APP = Path(__file__).resolve().parents[1] / "src" / "marketpulse" / "dashboard" / "app.py"
SNAPSHOT = Path(__file__).resolve().parents[1] / "demo_data" / "snapshot.json"


def test_dashboard_renders_snapshot_on_empty_database(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot.json"
    shutil.copy(SNAPSHOT, snapshot)
    isolated = dataclasses.replace(
        marketpulse.config.settings,
        db_path=str(tmp_path / "marketpulse.db"),
        chroma_persist_dir=str(tmp_path / "chroma"),
        snapshot_path=str(snapshot),
        default_tickers=("AAPL", "MSFT", "JPM"),
    )
    # `settings` is bound at import time in each module, so patch every copy
    # - otherwise the app would read and write the developer's real database.
    for module in (marketpulse.config, marketpulse.db, marketpulse.snapshot, marketpulse.rag.pipeline):
        monkeypatch.setattr(module, "settings", isolated)

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()

    assert not at.exception
    assert "bundled snapshot" in at.info[0].value
    assert [m.label for m in at.metric] == ["Last close", "Period return", "Latest 10-K tone", "10-K filings indexed"]
    assert at.metric[3].value == "5"
    assert not at.warning  # no "No data yet" and no rate-limit banners on a fresh visit
