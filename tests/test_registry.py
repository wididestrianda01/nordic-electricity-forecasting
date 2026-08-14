"""Tests for the model registry (ticket 05, ticket 07, ticket 10 contract)."""


from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from forecast_pipeline.features import build_features, build_horizon_features
from forecast_pipeline.lear import lear_forecast
from forecast_pipeline.registry import (
    LearAdapter,
    LgbmAdapter,
    ModelSpec,
    build_model,
)

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
    _as_of, history = request.getfixturevalue(scenario_name)
    result = LearAdapter().fit(history["price"]).predict_quantiles(horizon=expected_rows)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert (result["p10"] == result["p50"]).all()
    assert (result["p50"] == result["p90"]).all()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


def test_lear_adapter_matches_lear_forecast(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    result = LearAdapter().fit(history["price"]).predict_quantiles(horizon=24)
    # LearAdapter feeds price-only history (no load/wind covariates).
    point = lear_forecast(as_of, history[["price"]])

    assert (result["p50"] == point).all()
    assert result.index.equals(point.index)


def test_lear_adapter_fit_is_chainable(pre_nov_2024_scenario):
    _as_of, history = pre_nov_2024_scenario
    adapter = LearAdapter()
    assert adapter.fit(history["price"]) is adapter
    # fit must not disturb the forecast.
    assert (adapter.predict_quantiles(horizon=24)["p10"] == adapter.predict_quantiles(horizon=24)["p10"]).all()


def test_lear_adapter_requires_fit():
    with pytest.raises(ValueError):
        LearAdapter().predict_quantiles(horizon=24)


def _fit_lgbm_arm(as_of, history):
    """Fit the reworked LgbmAdapter on canonical full-features."""
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)
    arm = LgbmAdapter().fit(history["price"], train)
    return arm, horizon


@pytest.mark.parametrize("scenario_name,expected_rows", SCENARIOS)
def test_lgbm_adapter_returns_ordered_quantiles(scenario_name, expected_rows, request):
    as_of, history = request.getfixturevalue(scenario_name)
    arm, horizon = _fit_lgbm_arm(as_of, history)
    result = arm.predict_quantiles(horizon=expected_rows, future_features=horizon)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert result.index.equals(horizon.index)
    assert not result[QUANTILE_COLUMNS].isna().any().any()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


def test_lgbm_adapter_excludes_load_wind_covariates(pre_nov_2024_scenario):
    """The arm drops load/wind from covariates -- no mean-fill train/serve skew."""
    as_of, history = pre_nov_2024_scenario
    arm, horizon = _fit_lgbm_arm(as_of, history)

    assert "load_forecast" not in arm._covariate_columns
    assert "wind_forecast" not in arm._covariate_columns

    # Perturbing the day-ahead covariates must leave predictions untouched.
    baseline = arm.predict_quantiles(horizon=24, future_features=horizon)
    perturbed = horizon.copy()
    perturbed["load_forecast"] = perturbed["load_forecast"] * 1e4
    perturbed["wind_forecast"] = perturbed["wind_forecast"] * 1e4
    after = arm.predict_quantiles(horizon=24, future_features=perturbed)
    np.testing.assert_allclose(after["p50"].to_numpy(), baseline["p50"].to_numpy())


def test_lgbm_adapter_fit_is_chainable(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)
    arm = LgbmAdapter()
    assert arm.fit(history["price"], train) is arm


def test_lgbm_adapter_requires_fit_and_future_features(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    horizon = build_horizon_features(as_of, history)

    with pytest.raises(ValueError):
        LgbmAdapter().predict_quantiles(horizon=24, future_features=horizon)  # not fitted
    with pytest.raises(ValueError):
        LgbmAdapter().fit(history["price"], None)


def test_adapters_expose_feature_set_routing():
    assert LearAdapter.feature_set == "price-only"
    assert LgbmAdapter.feature_set == "full-features"


def test_lear_adapter_accepts_unused_protocol_args(pre_nov_2024_scenario):
    """LearAdapter accepts-and-ignores future_features (Phase 2 darts arms)."""
    _as_of, history = pre_nov_2024_scenario
    dummy_features = history[["load_forecast", "wind_forecast"]]
    assert (
        LearAdapter()
        .fit(history["price"])
        .predict_quantiles(horizon=24, future_features=dummy_features)
        is not None
    )


# --- build_model dispatch (ticket 10) ---


def test_build_model_returns_lear_arm():
    arm = build_model(ModelSpec(name="lear", family="linear", feature_set="price-only", hyperparams={}, seed=0))
    assert isinstance(arm, LearAdapter)


def test_build_model_returns_lgbm_arm():
    arm = build_model(ModelSpec(name="lgbm", family="gbdt", feature_set="full-features", hyperparams={}, seed=0))
    assert isinstance(arm, LgbmAdapter)


def test_build_model_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown model name"):
        build_model(ModelSpec(name="not-an-arm", family="?", feature_set="price-only", hyperparams={}, seed=0))


def test_build_model_feature_set_mismatch_raises():
    with pytest.raises(ValueError, match="feature_set mismatch"):
        build_model(ModelSpec(name="lear", family="linear", feature_set="full-features", hyperparams={}, seed=0))
    with pytest.raises(ValueError, match="feature_set mismatch"):
        build_model(ModelSpec(name="lgbm", family="gbdt", feature_set="price-only", hyperparams={}, seed=0))


@pytest.mark.parametrize(
    "name,feature_set,class_name",
    [
        ("sarima", "price-only", "SarimaArm"),
        ("ets", "price-only", "EtsArm"),
        ("xgboost", "full-features", "XgboostArm"),
        ("catboost", "full-features", "CatboostArm"),
        ("nbeats", "price-only", "NbeatsArm"),
        ("tft", "full-features", "TftArm"),
        ("chronos2", "full-features", "ChronosArm"),
        ("timesfm", "price-only", "TimesfmArm"),
    ],
)
def test_build_model_dispatches_darts_arms(name, feature_set, class_name):
    """build_model lazily imports and constructs each darts arm by name."""
    arm = build_model(
        ModelSpec(name=name, family="?", feature_set=feature_set, hyperparams={}, seed=42)
    )
    assert type(arm).__name__ == class_name
