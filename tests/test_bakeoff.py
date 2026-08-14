"""Tests for the bake-off runner (ticket 16)."""

from datetime import date

import numpy as np
import pandas as pd

from forecast_pipeline import bakeoff
from forecast_pipeline.backtest import DEFAULT_SPECS, RESULT_COLUMNS


def _spec(name: str):
    return next(s for s in DEFAULT_SPECS if s.name == name)


def _two_regime_frame() -> pd.DataFrame:
    """Hourly before the MTU switch, 15-minute from 2025-10-01 onward."""
    hourly = pd.date_range("2024-06-01", "2024-06-30", freq="h", tz="UTC")
    quarter = pd.date_range("2025-10-15", "2026-02-28", freq="15min", tz="UTC")
    index = hourly.append(quarter)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.normal(50, 10, len(index)),
            "load_forecast": rng.normal(1000, 100, len(index)),
            "wind_forecast": rng.normal(200, 50, len(index)),
        },
        index=index,
    )


def test_split_regimes_partitions_hourly_and_15min():
    frame = _two_regime_frame()
    regimes = bakeoff.split_regimes(frame)

    assert set(regimes) == {60, 15}
    assert (regimes[60].index.diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (regimes[15].index.diff().dropna() == pd.Timedelta(minutes=15)).all()
    assert len(regimes[60]) + len(regimes[15]) == len(frame)


def test_split_regimes_single_regime():
    hourly = _two_regime_frame().loc[:"2024-06-30 23:00:00+00:00"]
    regimes = bakeoff.split_regimes(hourly)
    assert set(regimes) == {60}


def test_run_bakeoff_smoke_end_to_end(monkeypatch, tmp_path):
    """One hourly regime runs assemble -> folds -> run_backtest through the runner."""
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


def test_run_bakeoff_combines_both_regimes(monkeypatch, tmp_path):
    """The runner runs each regime separately and concatenates their results."""
    frame = _two_regime_frame()
    monkeypatch.setattr(bakeoff, "assemble_data", lambda *a, **k: frame)

    def fake_run_backtest(historical_data, specs, cutoffs, *, tracking_uri, log, **kw):
        mtu = 60 if (historical_data.index.diff().dropna() == pd.Timedelta(hours=1)).all() else 15
        return pd.DataFrame(
            {
                "model": [s.name for s in specs],
                "family": [s.family for s in specs],
                "feature_set": [s.feature_set for s in specs],
                "cutoff": [cutoffs[0]] * len(specs),
                "mtu_minutes": [mtu] * len(specs),
                "crps": [1.0] * len(specs),
                "pinball_p10": [0.0] * len(specs),
                "pinball_p50": [0.0] * len(specs),
                "pinball_p90": [0.0] * len(specs),
                "mae": [1.0] * len(specs),
                "skill_score_crps": [0.0] * len(specs),
                "train_wall_clock": [0.0] * len(specs),
                "inference_wall_clock": [0.0] * len(specs),
            }
        )

    monkeypatch.setattr(bakeoff, "run_backtest", fake_run_backtest)

    specs = [_spec("lear"), _spec("lgbm")]
    result = bakeoff.run_bakeoff(
        ["SE3"],
        date(2024, 6, 1),
        date(2025, 11, 15),
        specs=specs,
        test_start=date(2023, 1, 1),
        out_dir=tmp_path / "reports",
        log=False,
    )

    assert set(result["mtu_minutes"]) == {60, 15}
    assert len(result) == 4  # two specs x two regimes
