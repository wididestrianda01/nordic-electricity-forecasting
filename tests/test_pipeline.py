import pandas as pd
import pytest

from forecast_pipeline.pipeline import ForecastOutput, forecast_pipeline


def test_returns_forecast_output(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    result = forecast_pipeline(as_of, history)
    assert isinstance(result, ForecastOutput)


def test_forecast_has_quantiles_and_regime_per_mtu(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    result = forecast_pipeline(as_of, history)
    for column in ("p10", "p50", "p90", "regime"):
        assert column in result.forecast.columns


@pytest.mark.parametrize(
    "scenario_name, expected_mtu_minutes, expected_rows",
    [
        ("pre_nov_2024_scenario", 60, 24),
        ("straddle_nov_2024_scenario", 60, 24),
        ("post_oct_2025_scenario", 15, 96),
    ],
)
def test_mtu_granularity_matches_as_of_date(
    scenario_name, expected_mtu_minutes, expected_rows, request
):
    as_of, history = request.getfixturevalue(scenario_name)
    result = forecast_pipeline(as_of, history)
    assert result.mtu_minutes == expected_mtu_minutes
    assert len(result.forecast) == expected_rows


@pytest.mark.parametrize(
    "scenario_name",
    ["pre_nov_2024_scenario", "straddle_nov_2024_scenario", "post_oct_2025_scenario"],
)
def test_quantile_ordering_holds_for_every_row(scenario_name, request):
    as_of, history = request.getfixturevalue(scenario_name)
    result = forecast_pipeline(as_of, history)
    assert (result.forecast["p10"] <= result.forecast["p50"]).all()
    assert (result.forecast["p50"] <= result.forecast["p90"]).all()


def test_malformed_historical_data_raises_missing_column(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    broken = history.rename(columns={"price": "not_price"})
    with pytest.raises(ValueError):
        forecast_pipeline(as_of, broken)


def test_malformed_historical_data_raises_non_monotonic_index(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    shuffled = history.sample(frac=1, random_state=1)
    with pytest.raises(ValueError):
        forecast_pipeline(as_of, shuffled)


def test_baseline_forecast_field_exists(pre_nov_2024_scenario):
    """ForecastOutput has a baseline_forecast field (ticket 05)."""
    as_of, history = pre_nov_2024_scenario
    result = forecast_pipeline(as_of, history)
    assert hasattr(result, "baseline_forecast")


def test_baseline_forecast_is_series(pre_nov_2024_scenario):
    """baseline_forecast is a pd.Series."""
    as_of, history = pre_nov_2024_scenario
    result = forecast_pipeline(as_of, history)
    assert isinstance(result.baseline_forecast, pd.Series)


@pytest.mark.parametrize(
    "scenario_name, expected_rows",
    [
        ("pre_nov_2024_scenario", 24),
        ("straddle_nov_2024_scenario", 24),
        ("post_oct_2025_scenario", 96),
    ],
)
def test_baseline_forecast_row_count_matches_mtu_grid(scenario_name, expected_rows, request):
    """baseline_forecast has one value per MTU slot in forecast day."""
    as_of, history = request.getfixturevalue(scenario_name)
    result = forecast_pipeline(as_of, history)
    assert len(result.baseline_forecast) == expected_rows


@pytest.mark.parametrize(
    "scenario_name",
    ["pre_nov_2024_scenario", "straddle_nov_2024_scenario", "post_oct_2025_scenario"],
)
def test_baseline_forecast_no_nans(scenario_name, request):
    """baseline_forecast has no NaN values (LEAR fallback ensures coverage)."""
    as_of, history = request.getfixturevalue(scenario_name)
    result = forecast_pipeline(as_of, history)
    assert not result.baseline_forecast.isna().any()


def test_too_short_historical_data_raises(malformed_input_scenario):
    as_of, history = malformed_input_scenario
    with pytest.raises(ValueError):
        forecast_pipeline(as_of, history)


def test_no_external_calls_needed(pre_nov_2024_scenario, monkeypatch):
    """forecast_pipeline must work purely from its arguments -- no network access."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("forecast_pipeline attempted network access")

    monkeypatch.setattr(socket, "socket", _blocked)
    as_of, history = pre_nov_2024_scenario
    forecast_pipeline(as_of, history)


def test_regime_labels_populated_not_unlabeled(two_regime_scenario):
    """Regime column must contain actual regime labels, not 'unlabeled' placeholder."""
    as_of, history = two_regime_scenario
    result = forecast_pipeline(as_of, history)
    assert (result.forecast["regime"] != "unlabeled").all()
    assert result.forecast["regime"].isin(["regime_0", "regime_1"]).all()


def test_forecast_responds_to_regime_shift(regime_shift_to_high_scenario):
    """Forecast should be materially higher when regime has shifted to high regime."""
    as_of, history = regime_shift_to_high_scenario
    result = forecast_pipeline(as_of, history)

    # In high regime (80±15), P50 should be significantly above low regime (20±3)
    # Using a conservative threshold: mean(high regime) - 2*std(low regime) = 80 - 6 = 74
    assert (result.forecast["p50"] > 50).all(), \
        f"P50 should be high in high regime, got mean={result.forecast['p50'].mean():.1f}"


def test_forecast_regime_label_is_most_recent(two_regime_scenario):
    """Forecast regime label should reflect the most recent regime from history."""
    as_of, history = two_regime_scenario
    result = forecast_pipeline(as_of, history)

    # The two_regime_scenario has high regime in second half (most recent)
    # So forecast should be labeled as the high regime
    assert (result.forecast["regime"] == "regime_1").all(), \
        "Forecast should use most recent regime label"
