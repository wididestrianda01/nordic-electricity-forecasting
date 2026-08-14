"""Tests for the model registry (ticket 05, ticket 07 contract)."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from forecast_pipeline.lear import lear_forecast
from forecast_pipeline.lgbm import lgbm_quantile_forecast
from forecast_pipeline.pipeline import _mtu_minutes_for
from forecast_pipeline.registry import LearAdapter, LgbmAdapter, ModelSpec

QUANTILE_COLUMNS = ["p10", "p50", "p90"]

# MTU granularity per scenario mirrors tests/conftest.py.
SCENARIOS = [
    ("pre_nov_2024_scenario", 24),
    ("post_oct_2025_scenario", 96),
]


def test_model_spec_has_five_fields():
    spec = ModelSpec(
        name="lear",
        family="linear",
        feature_set="price-only",
        hyperparams={"lag_days": (1, 2, 3, 7)},
        seed=42,
    )
    assert spec.name == "lear"
    assert spec.family == "linear"
    assert spec.feature_set == "price-only"
    assert spec.hyperparams == {"lag_days": (1, 2, 3, 7)}
    assert spec.seed == 42


def test_model_spec_is_frozen():
    spec = ModelSpec("lear", "linear", "price-only", {}, 0)
    with pytest.raises(FrozenInstanceError):
        spec.name = "lgbm"  # type: ignore[misc]


@pytest.mark.parametrize("scenario_name,expected_rows", SCENARIOS)
def test_lear_adapter_returns_degenerate_quantiles(scenario_name, expected_rows, request):
    as_of, history = request.getfixturevalue(scenario_name)
    result = LearAdapter(as_of, history).predict_quantiles(horizon=expected_rows)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert (result["p10"] == result["p50"]).all()
    assert (result["p50"] == result["p90"]).all()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


def test_lear_adapter_matches_lear_forecast(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    result = LearAdapter(as_of, history).predict_quantiles(horizon=24)
    point = lear_forecast(as_of, history)

    assert (result["p50"] == point).all()
    assert result.index.equals(point.index)


def test_lear_adapter_fit_is_chainable_noop(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    adapter = LearAdapter(as_of, history)
    assert adapter.fit(target=None, features=None) is adapter
    # fit must not disturb the forecast.
    assert (adapter.predict_quantiles(horizon=24)["p10"] == adapter.predict_quantiles(horizon=24)["p10"]).all()


@pytest.mark.parametrize("scenario_name,expected_rows", SCENARIOS)
def test_lgbm_adapter_returns_ordered_quantiles(scenario_name, expected_rows, request):
    as_of, history = request.getfixturevalue(scenario_name)
    result = LgbmAdapter(as_of, history).predict_quantiles(horizon=expected_rows)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert not result[QUANTILE_COLUMNS].isna().any().any()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


def test_lgbm_adapter_matches_direct_forecast(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    result = LgbmAdapter(as_of, history).predict_quantiles(horizon=24)
    direct = lgbm_quantile_forecast(as_of, history, _mtu_minutes_for(as_of))

    np.testing.assert_allclose(result["p10"].to_numpy(), direct["p10"].to_numpy())
    np.testing.assert_allclose(result["p50"].to_numpy(), direct["p50"].to_numpy())
    np.testing.assert_allclose(result["p90"].to_numpy(), direct["p90"].to_numpy())


def test_lgbm_adapter_fit_is_chainable_noop(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    adapter = LgbmAdapter(as_of, history)
    assert adapter.fit(target=None, features=None) is adapter


def test_adapters_expose_feature_set_routing():
    assert LearAdapter.feature_set == "price-only"
    assert LgbmAdapter.feature_set == "full-features"


def test_adapters_accept_unused_protocol_args(pre_nov_2024_scenario):
    """features/future_features are accepted-but-unused (Phase 2 darts arms)."""
    as_of, history = pre_nov_2024_scenario
    dummy_features = history[["load_forecast", "wind_forecast"]]

    assert LearAdapter(as_of, history).predict_quantiles(horizon=24, future_features=dummy_features) is not None
    assert LgbmAdapter(as_of, history).predict_quantiles(horizon=24, future_features=dummy_features) is not None
