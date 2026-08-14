"""Smoke test for the Streamlit bake-off explorer (ticket 27)."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecast_pipeline.snapshot import RESULTS_SUBDIR


@pytest.fixture
def minimal_snapshot(tmp_path, monkeypatch):
    results = tmp_path / RESULTS_SUBDIR
    results.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["catboost", "chronos2", "catboost", "chronos2"],
            "crps": [9.0, 9.5, 10.0, 9.7],
            "mae": [12.0, 10.0, 13.0, 11.0],
            "mtu_minutes": [60, 60, 15, 15],
        }
    ).to_csv(results / "results.csv", index=False)
    monkeypatch.setenv("FORECAST_SNAPSHOT_DIR", str(tmp_path))
    return tmp_path


def test_app_renders_headline_table(minimal_snapshot):
    app = AppTest.from_file(
        str(Path(__file__).parent.parent / "app.py"), default_timeout=30
    )
    app.run()

    assert not app.exception, f"app raised: {app.exception}"
    # The ranked headline table renders the two snapshot models.
    assert app.title[0].value == "Nordic electricity price forecasting"
    assert app.header[0].value == "Overview"
    assert app.header[1].value == "Pareto frontier"
    assert app.header[2].value == "Ranked headline table"
