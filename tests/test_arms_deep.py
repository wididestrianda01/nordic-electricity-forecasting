"""Tests for the deep-learning arms (ticket 13): NbeatsArm and TftArm.

CPU-only smoke tests on tiny synthetic series with the arms' pinned small
budgets; no GPU and no weight downloads are assumed.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.arms_deep import NbeatsArm, TftArm

QUANTILE_COLUMNS = ["p10", "p50", "p90"]


def _history(n: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range("2024-06-15", periods=n, freq=freq, tz="UTC")


def _future(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """One MTU step past ``index[-1]``, then ``horizon`` steps at the same freq."""
    freq = index.freq or pd.infer_freq(index)
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]


def _price(index: pd.DatetimeIndex) -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(40.0, 8.0, len(index)), index=index, name="price")


def _features(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = len(index)
    return pd.DataFrame(
        {
            "regime": rng.integers(0, 3, n).astype(float),
            "lag_1d": rng.normal(0.0, 1.0, n),
            "load_forecast": rng.normal(6000.0, 500.0, n),
            "wind_forecast": rng.normal(1500.0, 300.0, n),
        },
        index=index,
    )


def _assert_ordered_quantiles(result: pd.DataFrame, expected_rows: int, grid: pd.DatetimeIndex) -> None:
    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert result.index.equals(grid)
    assert not result[QUANTILE_COLUMNS].isna().any().any()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


# --- feature-set routing ---


def test_deep_arms_expose_feature_set_routing():
    assert NbeatsArm.feature_set == "price-only"
    assert TftArm.feature_set == "full-features"


# --- N-BEATS (price-only) ---


@pytest.mark.parametrize(
    "freq,horizon",
    [("h", 24), ("15min", 96)],
    ids=["hourly", "15min"],
)
def test_nbeats_predict_returns_ordered_quantiles(freq, horizon):
    index = _history(48, freq)
    target = _price(index)
    grid = _future(index, horizon)

    arm = NbeatsArm().fit(target)

    result = arm.predict_quantiles(horizon=horizon)
    _assert_ordered_quantiles(result, horizon, grid)


def test_nbeats_ignores_features():
    index = _history(48, "h")
    target = _price(index)
    grid = _future(index, 24)

    # Features are passed but must be ignored by the price-only arm.
    arm = NbeatsArm().fit(target, features=_features(index))
    result = arm.predict_quantiles(horizon=24, future_features=_features(grid))

    assert len(result) == 24
    assert result.index.equals(grid)


def test_nbeats_requires_fit():
    with pytest.raises(ValueError, match="fit"):
        NbeatsArm().predict_quantiles(horizon=24)


def test_nbeats_fit_is_chainable():
    target = _price(_history(48, "h"))
    arm = NbeatsArm()
    assert arm.fit(target) is arm


# --- TFT (full-features) ---


@pytest.mark.parametrize(
    "freq,horizon",
    [("h", 24), ("15min", 96)],
    ids=["hourly", "15min"],
)
def test_tft_predict_returns_ordered_quantiles(freq, horizon):
    index = _history(48, freq)
    target = _price(index)
    grid = _future(index, horizon)

    arm = TftArm().fit(target, _features(index))

    result = arm.predict_quantiles(horizon=horizon, future_features=_features(grid))
    _assert_ordered_quantiles(result, horizon, grid)


def test_tft_excludes_load_wind_covariates():
    arm = TftArm()
    cov = arm._covariates(_features(_history(48, "h")))

    assert "load_forecast" not in cov.columns
    assert "wind_forecast" not in cov.columns
    assert "regime" in cov.columns
    assert "lag_1d" in cov.columns


def test_tft_requires_fit_and_features():
    index = _history(48, "h")
    with pytest.raises(ValueError, match="full-features"):
        TftArm().fit(_price(index), None)
    with pytest.raises(ValueError, match="fit"):
        TftArm().predict_quantiles(horizon=24, future_features=_features(_future(index, 24)))
    with pytest.raises(ValueError, match="future_features"):
        TftArm().fit(_price(index), _features(index)).predict_quantiles(horizon=24)


def test_tft_fit_is_chainable():
    index = _history(48, "h")
    arm = TftArm()
    assert arm.fit(_price(index), _features(index)) is arm


# --- registry integration ---


def test_build_model_returns_deep_arms():
    from forecast_pipeline.registry import ModelSpec, build_model

    nbeats = build_model(
        ModelSpec(name="nbeats", family="deep", feature_set="price-only", hyperparams={}, seed=42)
    )
    assert isinstance(nbeats, NbeatsArm)

    tft = build_model(
        ModelSpec(name="tft", family="deep", feature_set="full-features", hyperparams={}, seed=42)
    )
    assert isinstance(tft, TftArm)
