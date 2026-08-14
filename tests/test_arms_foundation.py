"""Tests for the foundation-model arms (ticket 14)."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

from forecast_pipeline import arms_foundation
from forecast_pipeline.arms_foundation import ChronosArm, TimesfmArm
from forecast_pipeline.features import build_features, build_horizon_features

QUANTILE_COLUMNS = ["p10", "p50", "p90"]

# MTU granularity per scenario mirrors tests/conftest.py.
SCENARIOS = [
    ("pre_nov_2024_scenario", 24),
    ("post_oct_2025_scenario", 96),
]


class _FakeModel:
    """Stand-in for a darts foundation model; no network, no weights.

    ``predict`` returns a stochastic ``TimeSeries`` whose ``i``-th sample is
    ``arange(n) + i``, so the empirical quantiles are deterministic and the
    spread is visibly native (not a synthetic band).
    """

    instances: ClassVar[list["_FakeModel"]] = []

    def __init__(self, input_chunk_length=None, output_chunk_length=None, likelihood=None, **kwargs):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.likelihood = likelihood
        self.predict_calls: list[dict] = []
        _FakeModel.instances.append(self)

    def predict(
        self,
        n,
        series=None,
        past_covariates=None,
        future_covariates=None,
        num_samples=1,
        random_state=None,
        **kwargs,
    ):
        self.predict_calls.append(
            {
                "n": n,
                "series": series,
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
                "num_samples": num_samples,
                "random_state": random_state,
            }
        )
        start = series.end_time() + series.freq
        times = pd.date_range(start=start, periods=n, freq=series.freq)
        base = np.arange(n, dtype=float)
        values = np.empty((n, 1, num_samples), dtype=float)
        for i in range(num_samples):
            values[:, 0, i] = base + i
        return TimeSeries.from_times_and_values(times, values)


def _patch_model(monkeypatch, model_attr: str) -> None:
    _FakeModel.instances = []
    monkeypatch.setattr(arms_foundation, model_attr, _FakeModel)


def test_arm_routing():
    assert ChronosArm.feature_set == "full-features"
    assert TimesfmArm.feature_set == "price-only"


def test_chronos_model_loaded_lazily_on_first_predict(monkeypatch, pre_nov_2024_scenario):
    _patch_model(monkeypatch, "Chronos2Model")
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)

    arm = ChronosArm()
    assert arm._model is None
    assert _FakeModel.instances == []

    arm.fit(history["price"], train)
    assert _FakeModel.instances == []  # fit is a no-op

    arm.predict_quantiles(horizon=24, future_features=horizon)
    assert len(_FakeModel.instances) == 1

    arm.predict_quantiles(horizon=24, future_features=horizon)
    assert len(_FakeModel.instances) == 1  # model is reused, not rebuilt


def test_timesfm_model_loaded_lazily_on_first_predict(monkeypatch, pre_nov_2024_scenario):
    _patch_model(monkeypatch, "TimesFM2p5Model")
    _, history = pre_nov_2024_scenario

    arm = TimesfmArm()
    assert arm._model is None
    assert _FakeModel.instances == []

    arm.fit(history["price"])
    assert _FakeModel.instances == []

    arm.predict_quantiles(horizon=24)
    assert len(_FakeModel.instances) == 1

    arm.predict_quantiles(horizon=24)
    assert len(_FakeModel.instances) == 1


@pytest.mark.parametrize("scenario_name,expected_rows", SCENARIOS)
def test_chronos_returns_ordered_quantiles_on_horizon_grid(
    monkeypatch, scenario_name, expected_rows, request
):
    _patch_model(monkeypatch, "Chronos2Model")
    as_of, history = request.getfixturevalue(scenario_name)
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)

    arm = ChronosArm().fit(history["price"], train)
    result = arm.predict_quantiles(horizon=expected_rows, future_features=horizon)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert result.index.equals(horizon.index)
    assert not result[QUANTILE_COLUMNS].isna().any().any()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


@pytest.mark.parametrize("scenario_name,expected_rows", SCENARIOS)
def test_timesfm_returns_ordered_quantiles_on_derived_grid(
    monkeypatch, scenario_name, expected_rows, request
):
    _patch_model(monkeypatch, "TimesFM2p5Model")
    _, history = request.getfixturevalue(scenario_name)

    arm = TimesfmArm().fit(history["price"])
    result = arm.predict_quantiles(horizon=expected_rows)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert not result[QUANTILE_COLUMNS].isna().any().any()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()

    step = history.index[-1] - history.index[-2]
    expected = pd.date_range(start=history.index[-1] + step, periods=expected_rows, freq=step)
    assert result.index.equals(expected)


def test_chronos_passes_past_covariates_excluding_load_wind(monkeypatch, pre_nov_2024_scenario):
    _patch_model(monkeypatch, "Chronos2Model")
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)

    arm = ChronosArm().fit(history["price"], train)
    arm.predict_quantiles(horizon=24, future_features=horizon)

    model = _FakeModel.instances[0]
    assert model.likelihood is not None
    assert list(model.likelihood.quantiles) == [0.1, 0.5, 0.9]

    call = model.predict_calls[0]
    cov = call["past_covariates"]
    assert "load_forecast" not in cov.components
    assert "wind_forecast" not in cov.components
    assert "price" not in cov.components
    assert call["future_covariates"] is None
    assert call["num_samples"] == arms_foundation.NUM_SAMPLES
    assert call["random_state"] == arms_foundation.SEED


def test_chronos_passes_future_covariates_for_15min(monkeypatch, post_oct_2025_scenario):
    _patch_model(monkeypatch, "Chronos2Model")
    as_of, history = post_oct_2025_scenario
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)
    assert len(horizon) == 96  # 15-min regime spans the output chunk length

    arm = ChronosArm().fit(history["price"], train)
    arm.predict_quantiles(horizon=96, future_features=horizon)

    call = _FakeModel.instances[0].predict_calls[0]
    future_cov = call["future_covariates"]
    assert future_cov is not None
    assert "load_forecast" not in future_cov.components
    assert "wind_forecast" not in future_cov.components

def test_native_quantiles_come_from_model_samples(monkeypatch, pre_nov_2024_scenario):
    _patch_model(monkeypatch, "Chronos2Model")
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)

    arm = ChronosArm().fit(history["price"], train)
    result = arm.predict_quantiles(horizon=24, future_features=horizon)

    # Sample ``i`` is ``arange(24) + i``; the emitted quantiles must be the
    # empirical quantiles of those samples -- i.e. native, not synthetic.
    samples = np.arange(24)[:, None] + np.arange(arm.NUM_SAMPLES)[None, :]
    np.testing.assert_allclose(result["p10"].to_numpy(), np.quantile(samples, 0.1, axis=1))
    np.testing.assert_allclose(result["p50"].to_numpy(), np.quantile(samples, 0.5, axis=1))
    np.testing.assert_allclose(result["p90"].to_numpy(), np.quantile(samples, 0.9, axis=1))
    # A real, non-degenerate spread (not p10 == p50 == p90).
    assert (result["p10"] < result["p50"]).all()
    assert (result["p50"] < result["p90"]).all()


def test_fit_is_chainable(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)

    chronos = ChronosArm()
    assert chronos.fit(history["price"], train) is chronos
    timesfm = TimesfmArm()
    assert timesfm.fit(history["price"], None) is timesfm


@pytest.mark.parametrize("arm_class", [ChronosArm, TimesfmArm])
def test_requires_fit_before_predict(arm_class):
    with pytest.raises(ValueError):
        arm_class().predict_quantiles(horizon=24)


def test_chronos_requires_features_and_future_features(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario

    with pytest.raises(ValueError):
        ChronosArm().fit(history["price"], None)
    with pytest.raises(ValueError):
        ChronosArm().fit(history["price"], build_features(as_of, history)[1]).predict_quantiles(
            horizon=24, future_features=None
        )


def test_timesfm_ignores_features(pre_nov_2024_scenario, monkeypatch):
    _patch_model(monkeypatch, "TimesFM2p5Model")
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)

    arm = TimesfmArm().fit(history["price"], train)
    result = arm.predict_quantiles(horizon=24)

    call = _FakeModel.instances[0].predict_calls[0]
    assert call["past_covariates"] is None
    assert call["future_covariates"] is None
    assert len(result) == 24
