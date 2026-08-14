"""Tests for the native gradient-boosted arms (ticket 12)."""

import numpy as np
import pytest

from forecast_pipeline.arms_gbdt import CatboostArm, XgboostArm
from forecast_pipeline.features import build_features, build_horizon_features

QUANTILE_COLUMNS = ["p10", "p50", "p90"]

# MTU granularity per scenario mirrors tests/conftest.py.
SCENARIOS = [
    ("pre_nov_2024_scenario", 24),
    ("post_oct_2025_scenario", 96),
]

ARMS = [XgboostArm, CatboostArm]


def _fit_arm(arm_class, as_of, history):
    """Fit one gradient-boosted arm on canonical full-features."""
    _, train = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)
    arm = arm_class().fit(history["price"], train)
    return arm, horizon


@pytest.mark.parametrize("arm_class", ARMS)
@pytest.mark.parametrize("scenario_name,expected_rows", SCENARIOS)
def test_arm_returns_ordered_quantiles(arm_class, scenario_name, expected_rows, request):
    as_of, history = request.getfixturevalue(scenario_name)
    arm, horizon = _fit_arm(arm_class, as_of, history)
    result = arm.predict_quantiles(horizon=expected_rows, future_features=horizon)

    assert list(result.columns) == QUANTILE_COLUMNS
    assert len(result) == expected_rows
    assert result.index.equals(horizon.index)
    assert not result[QUANTILE_COLUMNS].isna().any().any()
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


@pytest.mark.parametrize("arm_class", ARMS)
def test_arm_excludes_load_wind_covariates(arm_class, pre_nov_2024_scenario):
    """Dropping load/wind leaves predictions unchanged under perturbation."""
    as_of, history = pre_nov_2024_scenario
    arm, horizon = _fit_arm(arm_class, as_of, history)

    assert "load_forecast" not in arm._covariate_columns
    assert "wind_forecast" not in arm._covariate_columns

    baseline = arm.predict_quantiles(horizon=24, future_features=horizon)
    perturbed = horizon.copy()
    perturbed["load_forecast"] = perturbed["load_forecast"] * 1e4
    perturbed["wind_forecast"] = perturbed["wind_forecast"] * 1e4
    after = arm.predict_quantiles(horizon=24, future_features=perturbed)
    np.testing.assert_allclose(after["p50"].to_numpy(), baseline["p50"].to_numpy())


@pytest.mark.parametrize("arm_class", ARMS)
def test_arm_fit_is_chainable(arm_class, pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    _, train = build_features(as_of, history)
    arm = arm_class()
    assert arm.fit(history["price"], train) is arm


@pytest.mark.parametrize("arm_class", ARMS)
def test_arm_requires_fit_and_future_features(arm_class, pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    horizon = build_horizon_features(as_of, history)

    with pytest.raises(ValueError):
        arm_class().predict_quantiles(horizon=24, future_features=horizon)  # not fitted
    with pytest.raises(ValueError):
        arm_class().fit(history["price"], None)


@pytest.mark.parametrize("arm_class", ARMS)
def test_arm_exposes_full_features_routing(arm_class):
    assert arm_class.feature_set == "full-features"
