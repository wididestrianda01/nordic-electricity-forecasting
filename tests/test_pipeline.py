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
