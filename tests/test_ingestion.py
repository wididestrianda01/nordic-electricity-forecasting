"""Ticket 01: ENTSO-E data ingestion tests.

Mocks the entsoe-py client via dependency injection -- no live API calls.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.ingestion import fetch_market_data
from forecast_pipeline.pipeline import _mtu_minutes_for, _validate


class _FakeEntsoeClient:
    """Stand-in for entsoe.EntsoePandasClient -- returns recorded-shape fixtures."""
    def __init__(self, price: pd.Series, load: pd.DataFrame, wind: pd.DataFrame):
        self._price = price
        self._load = load
        self._wind = wind
        self.zones: list[str] = []

    def query_day_ahead_prices(self, zone, start, end):
        self.zones.append(zone)
        return self._price

    def query_load_forecast(self, zone, start, end):
        self.zones.append(zone)
        return self._load

    def query_wind_and_solar_forecast(self, zone, start, end, **kwargs):
        self.zones.append(zone)
        return self._wind


def _fixture(start: date, periods: int, freq: str, tz="Europe/Stockholm", seed=1):
    index = pd.date_range(pd.Timestamp(start, tz=tz), periods=periods, freq=freq)
    rng = np.random.default_rng(seed=seed)
    price = pd.Series(rng.normal(40.0, 10.0, len(index)), index=index)
    load = pd.DataFrame({"Forecasted Load": rng.normal(8000.0, 500.0, len(index))}, index=index)
    wind = pd.DataFrame(
        {
            "Wind Onshore": rng.normal(1200.0, 300.0, len(index)),
            "Wind Offshore": rng.normal(300.0, 100.0, len(index)),
        },
        index=index,
    )
    return price, load, wind


def test_fetch_market_data_returns_expected_shape_pre_switch():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    client = _FakeEntsoeClient(price, load, wind)

    df = fetch_market_data("SE_3", start, end, client=client)

    assert list(df.columns) == ["price", "load_forecast", "wind_forecast"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any()


def test_fetch_market_data_sums_multiple_wind_psr_types():
    start, end = date(2025, 10, 1), date(2025, 10, 10)
    price, load, wind = _fixture(start, periods=9 * 96, freq="15min", seed=2)
    client = _FakeEntsoeClient(price, load, wind)

    df = fetch_market_data("SE_3", start, end, client=client)

    expected_wind = (wind["Wind Onshore"] + wind["Wind Offshore"]).tz_convert("UTC")
    pd.testing.assert_series_equal(df["wind_forecast"], expected_wind, check_names=False)


def test_fetch_market_data_upsamples_hourly_to_15min_across_mtu_switch():
    start, end = date(2025, 9, 20), date(2025, 10, 5)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    client = _FakeEntsoeClient(price, load, wind)

    df = fetch_market_data("SE_3", start, end, client=client)

    assert _mtu_minutes_for(end) == 15
    diffs = df.index.to_series().diff().dropna().unique()
    assert list(diffs) == [pd.Timedelta(minutes=15)]


def test_fetch_market_data_output_passes_pipeline_validate():
    start, end = date(2024, 10, 4), date(2024, 11, 1)
    price, load, wind = _fixture(start, periods=28 * 24, freq="h")
    client = _FakeEntsoeClient(price, load, wind)

    df = fetch_market_data("SE_3", start, end, client=client)

    _validate(df, _mtu_minutes_for(end))


def test_fetch_market_data_requires_api_key_or_client():
    with pytest.raises(ValueError):
        fetch_market_data("SE_3", date(2024, 10, 4), date(2024, 10, 20))

def test_fetch_market_data_raises_on_missing_forecasted_load_column():
    """Raises ValueError if load forecast response lacks 'Forecasted Load' column."""
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    
    # Rename the load column to simulate API response change
    load = load.rename(columns={"Forecasted Load": "Load_Forecast"})
    client = _FakeEntsoeClient(price, load, wind)

    with pytest.raises(ValueError, match="Expected column 'Forecasted Load'"):
        fetch_market_data("SE_3", start, end, client=client)


def test_fetch_market_data_raises_on_missing_wind_column_marker():
    """Raises ValueError if wind forecast response has no column containing 'Wind'."""
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    
    # Rename wind columns to remove 'Wind' marker
    wind = wind.rename(columns={"Wind Onshore": "Onshore", "Wind Offshore": "Offshore"})
    client = _FakeEntsoeClient(price, load, wind)

    with pytest.raises(ValueError, match="No columns containing 'Wind'"):
        fetch_market_data("SE_3", start, end, client=client)


@pytest.mark.parametrize(
    ("domain_zone", "entsoe_code"),
    [("SE1", "SE_1"), ("SE2", "SE_2"), ("SE3", "SE_3"), ("SE4", "SE_4")],
)
def test_fetch_market_data_maps_domain_zone_to_entsoe_code(domain_zone, entsoe_code):
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    client = _FakeEntsoeClient(price, load, wind)

    fetch_market_data(domain_zone, start, end, client=client)

    assert client.zones == [entsoe_code] * 3


def test_fetch_market_data_accepts_entsoe_code_directly():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    client = _FakeEntsoeClient(price, load, wind)

    fetch_market_data("SE_3", start, end, client=client)

    assert client.zones == ["SE_3"] * 3


@pytest.mark.parametrize("zone", ["", "SE5", "DE", "SE", "se3"])
def test_fetch_market_data_raises_on_unknown_zone(zone):
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    price, load, wind = _fixture(start, periods=16 * 24, freq="h")
    client = _FakeEntsoeClient(price, load, wind)

    with pytest.raises(ValueError, match="Unknown ENTSO-E zone"):
        fetch_market_data(zone, start, end, client=client)
