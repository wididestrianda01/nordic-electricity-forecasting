"""Tests for the classical darts arms (ticket 11): SarimaArm and EtsArm."""

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.arms_classical import EtsArm, SarimaArm
from forecast_pipeline.registry import ModelSpec, build_model

QUANTILE_COLUMNS = ["p10", "p50", "p90"]
ARMS = [SarimaArm, EtsArm]

#: Seconds in one day, used to infer the one-day seasonal period (in MTU steps).
_DAY_SECONDS = 24 * 60 * 60


def _synthetic_price(n: int, freq: str, seed: int = 42) -> pd.Series:
    """Deterministic daily-seasonal price series with small noise."""
    idx = pd.date_range("2024-06-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    steps_per_day = round(_DAY_SECONDS / (idx[1] - idx[0]).total_seconds())
    daily = 6.0 * np.sin(2 * np.pi * t / steps_per_day) + 2.5 * np.sin(
        4 * np.pi * t / steps_per_day
    )
    noise = rng.normal(0.0, 1.0, size=n)
    return pd.Series(40.0 + daily + noise, index=idx, name="price")


def _expected_grid(target: pd.Series, horizon: int) -> pd.DatetimeIndex:
    step = target.index[-1] - target.index[-2]
    return pd.date_range(target.index[-1] + step, periods=horizon, freq=step)


@pytest.mark.parametrize("arm_cls", ARMS)
def test_hourly_contract(arm_cls):
    target = _synthetic_price(24 * 14, "h")
    arm = arm_cls()
    assert arm.fit(target) is arm
    out = arm.predict_quantiles(24)

    assert out.shape == (24, 3)
    assert list(out.columns) == QUANTILE_COLUMNS
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()
    assert out.index.equals(_expected_grid(target, 24))
    assert not out.isna().any().any()


@pytest.mark.parametrize("arm_cls", ARMS)
def test_15min_contract(arm_cls):
    target = _synthetic_price(200, "15min")
    out = arm_cls().fit(target).predict_quantiles(96)

    assert out.shape == (96, 3)
    assert list(out.columns) == QUANTILE_COLUMNS
    assert (out["p10"] <= out["p50"]).all()
    assert (out["p50"] <= out["p90"]).all()
    assert out.index.equals(_expected_grid(target, 96))
    assert not out.isna().any().any()


@pytest.mark.parametrize("arm_cls", ARMS)
def test_reproducible(arm_cls):
    """Same target + pinned seed -> identical quantiles across two runs."""
    target = _synthetic_price(24 * 14, "h")
    first = arm_cls().fit(target).predict_quantiles(24)
    second = arm_cls().fit(target).predict_quantiles(24)
    pd.testing.assert_frame_equal(first, second)


def test_feature_set_routing():
    assert SarimaArm.feature_set == "price-only"
    assert EtsArm.feature_set == "price-only"


def test_requires_fit():
    with pytest.raises(ValueError, match="fit"):
        SarimaArm().predict_quantiles(24)
    with pytest.raises(ValueError, match="fit"):
        EtsArm().predict_quantiles(24)


def test_build_model_dispatches_classical_arms():
    sarima = build_model(
        ModelSpec(
            name="sarima",
            family="classical",
            feature_set="price-only",
            hyperparams={},
            seed=42,
        )
    )
    ets = build_model(
        ModelSpec(
            name="ets",
            family="classical",
            feature_set="price-only",
            hyperparams={},
            seed=42,
        )
    )
    assert isinstance(sarima, SarimaArm)
    assert isinstance(ets, EtsArm)
