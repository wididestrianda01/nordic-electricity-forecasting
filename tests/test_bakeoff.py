"""Tests for the bake-off runner (ticket 16)."""

from datetime import date

import numpy as np
import pandas as pd

from forecast_pipeline import bakeoff
from forecast_pipeline.backtest import DEFAULT_SPECS, RESULT_COLUMNS


def _spec(name: str):
    return next(s for s in DEFAULT_SPECS if s.name == name)


def test_run_bakeoff_smoke_end_to_end(monkeypatch, tmp_path):
    """An all-hourly window (end before the MTU switch) runs end-to-end."""
    hourly = pd.date_range("2024-06-01", "2024-06-30", freq="h", tz="UTC")
    frame = pd.DataFrame({"price": np.linspace(40, 60, len(hourly))}, index=hourly)
    frame["load_forecast"] = 1000.0
    frame["wind_forecast"] = 200.0

    monkeypatch.setattr(bakeoff, "assemble_data", lambda *a, **k: frame)

    result = bakeoff.run_bakeoff(
        ["SE3"],
        date(2024, 6, 1),
        date(2024, 6, 30),
        specs=[_spec("lear"), _spec("lgbm")],
        test_start=date(2024, 6, 28),
        out_dir=tmp_path / "reports",
        log=False,
    )

    assert list(result.columns) == RESULT_COLUMNS
    assert len(result) == 2  # two models, one cutoff
    assert (result["mtu_minutes"] == 60).all()
    assert bool(np.isfinite(result["crps"]).all())
    assert (tmp_path / "reports" / "results.csv").exists()
    assert (tmp_path / "reports" / "results.parquet").exists()


def test_run_bakeoff_assembles_two_regime_windows(monkeypatch, tmp_path):
    """Two windows are assembled with the correct MTU-switch boundaries."""
    assemble_calls: list[tuple] = []
    monkeypatch.setattr(
        bakeoff,
        "assemble_data",
        lambda zones, s, e, **kw: assemble_calls.append((s, e)) or pd.DataFrame(),
    )
    monkeypatch.setattr(bakeoff, "generate_folds", lambda frame, **kw: [date(2024, 1, 1)])

    run_calls: list[int] = []

    def fake_run(hist, specs, cutoffs, *, tracking_uri, log, **kw):
        run_calls.append(len(hist))
        return pd.DataFrame(
            [
                {
                    "model": "lear",
                    "family": "ml",
                    "feature_set": "price-only",
                    "cutoff": date(2024, 1, 1),
                    "mtu_minutes": 60,
                    "crps": 1.0,
                    "pinball_p10": 0.0,
                    "pinball_p50": 0.0,
                    "pinball_p90": 0.0,
                    "mae": 1.0,
                    "skill_score_crps": 0.0,
                    "train_wall_clock": 0.0,
                    "inference_wall_clock": 0.0,
                }
            ]
        )

    monkeypatch.setattr(bakeoff, "run_backtest", fake_run)

    result = bakeoff.run_bakeoff(
        ["SE3"],
        date(2021, 1, 1),
        date(2026, 8, 13),
        specs=[_spec("lear")],
        out_dir=tmp_path / "reports",
        log=False,
    )

    assert assemble_calls == [
        (date(2021, 1, 1), date(2025, 9, 30)),
        (date(2025, 10, 1), date(2026, 8, 13)),
    ]
    assert len(run_calls) == 2  # one backtest per regime
    assert len(result) == 2
