from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.lear import lear_forecast


@pytest.mark.parametrize(
    "scenario_name, expected_rows",
    [
        ("pre_nov_2024_scenario", 24),
        ("straddle_nov_2024_scenario", 24),
        ("post_oct_2025_scenario", 96),
    ],
)
def test_forecast_matches_mtu_grid_for_as_of_date(scenario_name, expected_rows, request):
    as_of, history = request.getfixturevalue(scenario_name)
    forecast = lear_forecast(as_of, history)
    assert len(forecast) == expected_rows
    assert not forecast.isna().any()


def test_forecast_values_stay_within_plausible_range(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    forecast = lear_forecast(as_of, history)
    lo = history["price"].min() - 3 * history["price"].std()
    hi = history["price"].max() + 3 * history["price"].std()
    assert forecast.between(lo, hi).all()


def test_malformed_historical_data_raises_missing_column(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    broken = history.rename(columns={"price": "not_price"})
    with pytest.raises(ValueError):
        lear_forecast(as_of, broken)


def test_too_short_historical_data_raises(malformed_input_scenario):
    as_of, history = malformed_input_scenario
    with pytest.raises(ValueError):
        lear_forecast(as_of, history)


def test_deterministic_across_calls(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    first = lear_forecast(as_of, history)
    second = lear_forecast(as_of, history)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_missing_lag_date_raises_clear_error():
    """Forecast should raise clear ValueError if a lag date is missing from pivot."""
    rng = np.random.default_rng(seed=42)

    # Create first 10 days of hourly data covering all 24 slots
    prices1 = rng.normal(loc=40.0, scale=10.0, size=10 * 24)
    index1 = pd.date_range(
        pd.Timestamp(date(2024, 6, 1), tz="UTC"),
        periods=len(prices1),
        freq="h",
    )

    # Create last 5 days (skip 2 days - creates a GAP in the pivot dates!)
    prices2 = rng.normal(loc=40.0, scale=10.0, size=5 * 24)
    index2 = pd.date_range(
        pd.Timestamp(date(2024, 6, 18), tz="UTC"),
        periods=len(prices2),
        freq="h",
    )

    history = pd.DataFrame(
        {"price": np.concatenate([prices1, prices2])},
        index=index1.append(index2)
    )

    # Try to forecast for 2024-06-20. The pivot will have dates:
    # 2024-06-01 to 2024-06-10 and 2024-06-18 to 2024-06-22.
    # With lag_days=(1,2,3,7), we need dates: 2024-06-19, 2024-06-18, 2024-06-17, 2024-06-13.
    # Dates 2024-06-13 and 2024-06-17 are missing from the pivot (in the gap).
    as_of_2 = date(2024, 6, 20)
    with pytest.raises(ValueError, match=r"(?i)missing.*lag"):
        lear_forecast(as_of_2, history)


def test_empty_training_data_raises_clear_error():
    """Forecast should raise clear ValueError if training data for a slot is empty after dropna."""
    rng = np.random.default_rng(seed=43)

    # Create 30 days of data but only for hours 6-18 to trigger validation (> 14 days)
    # while still omitting midnight-6AM and 7PM-midnight slots.
    # When lear_forecast tries to process slot 0 (00:00), it won't be in the pivot.
    prices_list: list[float] = []
    index_list: list[pd.Timestamp] = []

    for day_offset in range(30):
        for hour in range(6, 19):  # Only 6 AM to 6 PM = 13 hours/day
            ts = pd.Timestamp(date(2024, 6, 1), tz="UTC") + pd.Timedelta(
                days=day_offset, hours=hour
            )
            prices_list.append(rng.normal(loc=40.0, scale=10.0))
            index_list.append(ts)

    history = pd.DataFrame(
        {"price": prices_list},
        index=pd.DatetimeIndex(index_list)
    )

    # Try to forecast - should fail when processing slot 0 (00:00)
    # which does not exist in the pivot, causing the KeyError -> ValueError we added.
    as_of_2 = date(2024, 7, 1)
    with pytest.raises(ValueError, match=r"(?i)(no training data|missing lag)"):
        lear_forecast(as_of_2, history)
