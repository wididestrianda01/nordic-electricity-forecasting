"""Tests for the walk-forward backtest harness (ticket 15)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline import backtest
from forecast_pipeline.backtest import (
    DEFAULT_SPECS,
    RESULT_COLUMNS,
    generate_folds,
    run_backtest,
)
from forecast_pipeline.pipeline import _mtu_minutes_for
from forecast_pipeline.regime_boundaries import REGIME_BOUNDARIES


def _prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic MTU-indexed frame: price + the two day-ahead covariates."""
    rng = np.random.default_rng(42)
    n = len(index)
    return pd.DataFrame(
        {
            "price": rng.normal(40.0, 10.0, n),
            "load_forecast": rng.normal(6000.0, 500.0, n),
            "wind_forecast": rng.normal(1500.0, 300.0, n),
        },
        index=index,
    )


def _hourly_frame_between(start: date, end: date) -> pd.DataFrame:
    index = pd.date_range(
        pd.Timestamp(start),
        pd.Timestamp(end) + pd.Timedelta(hours=23),
        freq="h",
        tz="UTC",
    )
    return _prices(index)


def _15min_frame_between(start: date, end: date) -> pd.DataFrame:
    index = pd.date_range(
        pd.Timestamp(start),
        pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=45),
        freq="15min",
        tz="UTC",
    )
    return _prices(index)


def _two_regime_frame() -> pd.DataFrame:
    """Hourly up to the MTU switch, 15-minute from 2025-10-01 onward."""
    hourly = _hourly_frame_between(date(2024, 10, 1), date(2025, 9, 30))
    quarter = _15min_frame_between(date(2025, 10, 1), date(2025, 11, 15))
    return pd.concat([hourly, quarter])


def _spec(name: str):
    return next(s for s in DEFAULT_SPECS if s.name == name)


def _assert_clear_of_boundaries(cutoffs: list[date], purge_days: int = 7) -> None:
    """Every cutoff is >= purge_days from both boundaries; no horizon straddle."""
    for cutoff in cutoffs:
        for boundary in REGIME_BOUNDARIES:
            assert abs((cutoff - boundary).days) >= purge_days
            assert not (cutoff <= boundary < cutoff + timedelta(days=1))


# --- generate_folds invariants -------------------------------------------------


def test_generate_folds_yearly_cadence():
    history = _hourly_frame_between(date(2023, 1, 1), date(2024, 6, 30))
    cutoffs = generate_folds(history, test_start=date(2023, 1, 1))
    assert cutoffs == [date(2023, 1, 1), date(2024, 1, 1)]
    assert all(_mtu_minutes_for(c) == 60 for c in cutoffs)
    _assert_clear_of_boundaries(cutoffs)


def test_generate_folds_purges_near_boundary():
    history = _hourly_frame_between(date(2023, 11, 1), date(2024, 12, 31))
    cutoffs = generate_folds(history, test_start=date(2023, 11, 1))
    # 2024-11-01 sits 3 days before the 2024-11-04 boundary -> purged.
    assert cutoffs == [date(2023, 11, 1)]
    _assert_clear_of_boundaries(cutoffs)


def test_generate_folds_keeps_clear_cutoffs():
    history = _hourly_frame_between(date(2023, 11, 1), date(2024, 12, 31))
    cutoffs = generate_folds(history, test_start=date(2023, 11, 25))
    # 2024-11-25 is 21 days past the 2024-11-04 boundary -> kept.
    assert cutoffs == [date(2023, 11, 25), date(2024, 11, 25)]
    _assert_clear_of_boundaries(cutoffs)


def test_generate_folds_drops_boundary_day_cutoff():
    history = _hourly_frame_between(date(2024, 11, 4), date(2024, 12, 31))
    # purge_days=0 isolates the no-straddle rule: the boundary-day horizon
    # contains the boundary, so the cutoff is dropped.
    cutoffs = generate_folds(history, test_start=date(2024, 11, 4), purge_days=0)
    assert cutoffs == []


def test_generate_folds_keeps_day_after_boundary():
    history = _hourly_frame_between(date(2024, 11, 5), date(2024, 12, 31))
    cutoffs = generate_folds(history, test_start=date(2024, 11, 5), purge_days=0)
    assert cutoffs == [date(2024, 11, 5)]


def test_generate_folds_rejects_unknown_cadence():
    history = _hourly_frame_between(date(2023, 1, 1), date(2024, 1, 1))
    try:
        generate_folds(history, test_start=date(2023, 1, 1), refit_cadence="daily")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unsupported refit_cadence")


def test_generate_folds_attributes_mtu():
    hourly = _hourly_frame_between(date(2024, 1, 1), date(2024, 12, 31))
    cutoffs = generate_folds(hourly, test_start=date(2024, 1, 1))
    assert cutoffs
    assert [_mtu_minutes_for(c) for c in cutoffs] == [60] * len(cutoffs)
    _assert_clear_of_boundaries(cutoffs)


def test_generate_folds_15min_regime():
    quarter = _15min_frame_between(date(2025, 10, 15), date(2026, 1, 31))
    cutoffs = generate_folds(quarter, test_start=date(2025, 10, 15))
    assert cutoffs
    assert [_mtu_minutes_for(c) for c in cutoffs] == [15] * len(cutoffs)
    _assert_clear_of_boundaries(cutoffs)


def test_generate_folds_rejects_mixed_frequency():
    history = _two_regime_frame()
    with pytest.raises(ValueError, match="single-frequency"):
        generate_folds(history, test_start=date(2024, 10, 15))


# --- run_backtest end-to-end smoke ---------------------------------------------


def test_run_backtest_smoke(monkeypatch, tmp_path):
    history = _hourly_frame_between(date(2024, 1, 1), date(2024, 1, 30))
    specs = [_spec("lear"), _spec("lgbm")]
    cutoffs = [date(2024, 1, 28), date(2024, 1, 29)]

    calls: list[dict] = []
    real_build_model = backtest.build_model

    def spy_build_model(spec):
        arm = real_build_model(spec)

        class Spy:
            def __init__(self) -> None:
                self.features = None
                self.future_features = None

            def fit(self, target, features=None):
                self.features = features
                arm.fit(target, features)
                return self

            def predict_quantiles(self, horizon, future_features=None):
                self.future_features = future_features
                predictions = arm.predict_quantiles(horizon, future_features)
                calls.append(
                    {
                        "name": spec.name,
                        "features": self.features,
                        "future_features": self.future_features,
                        "predictions": predictions,
                    }
                )
                return predictions

        return Spy()

    monkeypatch.setattr(backtest, "build_model", spy_build_model)

    result = run_backtest(
        history,
        specs,
        cutoffs,
        tracking_uri=str(tmp_path / "mlruns"),
        log=True,
    )

    assert list(result.columns) == RESULT_COLUMNS
    assert len(result) == len(cutoffs) * len(specs) == 4
    assert result.groupby("model").size().to_dict() == {"lear": 2, "lgbm": 2}

    assert bool(np.isfinite(result["crps"]).all())
    assert bool(np.isfinite(result["mae"]).all())
    assert bool(np.isfinite(result["skill_score_crps"]).all())
    assert (result["train_wall_clock"] >= 0).all()
    assert (result["inference_wall_clock"] >= 0).all()

    # price-only arms (lear) receive None; full-features arms (lgbm) receive frames.
    for call in calls:
        if call["name"] == "lear":
            assert call["features"] is None
            assert call["future_features"] is None
        else:
            assert call["features"] is not None
            assert call["future_features"] is not None

    # p10 <= p50 <= p90 held in every logged prediction.
    for call in calls:
        predictions = call["predictions"]
        assert (predictions["p10"] <= predictions["p50"]).all()
        assert (predictions["p50"] <= predictions["p90"]).all()

    # MLflow wrote nested parent/fold runs to the temp tracking uri.
    assert (tmp_path / "mlruns").exists()


def test_run_backtest_no_logging_skips_mlflow(monkeypatch, tmp_path):
    history = _hourly_frame_between(date(2024, 1, 1), date(2024, 1, 30))
    result = run_backtest(
        history,
        [_spec("lear")],
        [date(2024, 1, 28)],
        tracking_uri=str(tmp_path / "mlruns"),
        log=False,
    )
    assert len(result) == 1
    assert list(result.columns) == RESULT_COLUMNS
    assert not (tmp_path / "mlruns").exists()


def test_run_backtest_15min_regime_e2e(tmp_path):
    """A post-switch 15-min frame runs end-to-end (lear + lgbm) with horizon 96."""
    history = _15min_frame_between(date(2025, 10, 1), date(2025, 11, 1))
    specs = [_spec("lear"), _spec("lgbm")]
    cutoffs = [date(2025, 10, 28), date(2025, 10, 29)]

    result = run_backtest(
        history,
        specs,
        cutoffs,
        tracking_uri=str(tmp_path / "mlruns"),
        log=False,
    )

    assert len(result) == len(cutoffs) * len(specs) == 4
    assert (result["mtu_minutes"] == 15).all()
    assert bool(np.isfinite(result["crps"]).all())
    assert bool(np.isfinite(result["mae"]).all())


def test_run_backtest_rejects_mixed_frequency():
    history = _two_regime_frame()
    with pytest.raises(ValueError, match="single-frequency"):
        run_backtest(
            history, [_spec("lear")], [date(2025, 10, 28)], log=False
        )


def test_run_backtest_rejects_regime_mismatched_cutoff():
    history = _15min_frame_between(date(2025, 10, 1), date(2025, 11, 1))
    with pytest.raises(ValueError, match="single-frequency frame"):
        run_backtest(
            history, [_spec("lear")], [date(2024, 10, 28)], log=False
        )
