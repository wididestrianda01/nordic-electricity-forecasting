"""ENTSO-E Transparency fetcher tests (hydro + cross-border).

Mocks the entsoe-py client via dependency injection -- no live API calls.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.ingestion_transparency import fetch_cross_border, fetch_hydro
from forecast_pipeline.pipeline import _mtu_minutes_for


class _FakeEntsoeClient:
    """Stand-in for entsoe.EntsoePandasClient -- returns recorded fixtures."""

    def __init__(self):
        self.hydro = {}
        self.net_position = {}
        self.scheduled = {}
        self.day_ahead = {}
        self.hydro_zones = []
        self.net_zones = []
        self.scheduled_borders = []
        self.price_zones = []

    def query_aggregate_water_reservoirs_and_hydro_storage(self, zone, start, end):
        self.hydro_zones.append(zone)
        return self.hydro[zone]

    def query_net_position(self, zone, start, end, dayahead=True, **kwargs):
        self.net_zones.append(zone)
        return self.net_position[zone]

    def query_scheduled_exchanges(
        self, country_code_from, country_code_to, start, end, dayahead=False, **kwargs
    ):
        self.scheduled_borders.append((country_code_from, country_code_to))
        return self.scheduled[(country_code_from, country_code_to)]

    def query_day_ahead_prices(self, zone, start, end, **kwargs):
        self.price_zones.append(zone)
        return self.day_ahead[zone]


# External borders of the Swedish zones (entsoe-code form), with a distinct
# scheduled-exchange value per border so aggregation is observable.
_ALL_BORDERS = {
    ("SE_1", "FI"): 10.0,
    ("SE_1", "NO_4"): 20.0,
    ("SE_2", "NO_3"): 30.0,
    ("SE_2", "NO_4"): 40.0,
    ("SE_3", "NO_1"): 50.0,
    ("SE_3", "DK_1"): 60.0,
    ("SE_3", "FI"): 70.0,
    ("SE_4", "DK_2"): 80.0,
    ("SE_4", "DE_LU"): 90.0,
    ("SE_4", "PL"): 100.0,
}

_NEIGHBOURS = {
    "FI": 50.0,
    "NO_1": 40.0,
    "NO_3": 42.0,
    "NO_4": 38.0,
    "DK_1": 55.0,
    "DK_2": 58.0,
    "DE_LU": 60.0,
    "PL": 70.0,
}
_NEIGHBOUR_MEAN = sum(_NEIGHBOURS.values()) / len(_NEIGHBOURS)


def _weekly(start, values):
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="7D")
    return pd.Series(values, index=index, dtype=float)


def _hourly(start, periods, value, tz="Europe/Stockholm"):
    index = pd.date_range(pd.Timestamp(start, tz=tz), periods=periods, freq="h")
    return pd.Series([float(value)] * periods, index=index)


def _fake_cross_border_client(start, periods):
    client = _FakeEntsoeClient()
    for zone in ("SE_1", "SE_2", "SE_3", "SE_4"):
        client.net_position[zone] = _hourly(start, periods, 100.0)
    for border, value in _ALL_BORDERS.items():
        client.scheduled[border] = _hourly(start, periods, value)
    for zone, value in _NEIGHBOURS.items():
        client.day_ahead[zone] = _hourly(start, periods, value)
    return client


# --- hydro ----------------------------------------------------------------


def test_fetch_hydro_forward_fills_weekly_to_mtu():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _FakeEntsoeClient()
    client.hydro["SE_3"] = _weekly("2024-10-06 22:00", [100.0, 120.0])

    df = fetch_hydro(["SE3"], start, end, client=client)

    assert list(df.columns) == ["hydro_storage_mwh"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any()

    # First non-NaN MTU is the first weekly observation; values forward-fill
    # across each week and the last value extends to the end of the range.
    assert df.index[0] == pd.Timestamp("2024-10-06 22:00", tz="UTC")
    assert (df.loc[: "2024-10-13 21:45", "hydro_storage_mwh"] == 100.0).all()
    assert (df.loc["2024-10-13 22:00":, "hydro_storage_mwh"] == 120.0).all()


def test_fetch_hydro_grid_excludes_end_timestamp():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _FakeEntsoeClient()
    client.hydro["SE_3"] = _weekly("2024-10-06 22:00", [100.0, 120.0])

    df = fetch_hydro(["SE3"], start, end, client=client)

    end_ts = pd.Timestamp(end, tz="UTC")
    assert end_ts not in df.index
    assert df.index.max() < end_ts


def test_fetch_hydro_sums_across_zones():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _FakeEntsoeClient()
    client.hydro["SE_1"] = _weekly("2024-10-06 22:00", [100.0])
    client.hydro["SE_2"] = _weekly("2024-10-06 22:00", [50.0])

    df = fetch_hydro(["SE1", "SE2"], start, end, client=client)

    assert (df["hydro_storage_mwh"] == 150.0).all()
    assert client.hydro_zones == ["SE_1", "SE_2"]


@pytest.mark.parametrize(
    ("domain_zone", "entsoe_code"),
    [("SE1", "SE_1"), ("SE2", "SE_2"), ("SE3", "SE_3"), ("SE4", "SE_4")],
)
def test_fetch_hydro_maps_domain_zone_to_entsoe_code(domain_zone, entsoe_code):
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _FakeEntsoeClient()
    client.hydro[entsoe_code] = _weekly("2024-10-06 22:00", [100.0])

    fetch_hydro([domain_zone], start, end, client=client)

    assert client.hydro_zones == [entsoe_code]


def test_fetch_hydro_accepts_entsoe_code_directly():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _FakeEntsoeClient()
    client.hydro["SE_3"] = _weekly("2024-10-06 22:00", [100.0])

    fetch_hydro(["SE_3"], start, end, client=client)

    assert client.hydro_zones == ["SE_3"]


def test_fetch_hydro_requires_api_key_or_client():
    with pytest.raises(ValueError):
        fetch_hydro(["SE3"], date(2024, 10, 4), date(2024, 10, 20))


@pytest.mark.parametrize("zone", ["", "SE5", "DE", "SE", "se3"])
def test_fetch_hydro_raises_on_unknown_zone(zone):
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    with pytest.raises(ValueError, match="Unknown ENTSO-E zone"):
        fetch_hydro([zone], start, end, client=_FakeEntsoeClient())


# --- cross-border ---------------------------------------------------------


def test_fetch_cross_border_returns_expected_shape():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _fake_cross_border_client(start, 16 * 24)

    df = fetch_cross_border(["SE3"], start, end, client=client)

    assert list(df.columns) == [
        "net_position_mwh",
        "scheduled_exchange_mwh",
        "neighbour_price_eur_mwh",
    ]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any()

    # SE3 external borders: NO_1 (50) + DK_1 (60) + FI (70) = 180.
    assert (df["net_position_mwh"] == 100.0).all()
    assert (df["scheduled_exchange_mwh"] == 180.0).all()
    assert np.allclose(df["neighbour_price_eur_mwh"], _NEIGHBOUR_MEAN)

    assert client.scheduled_borders == [
        ("SE_3", "NO_1"),
        ("SE_3", "DK_1"),
        ("SE_3", "FI"),
    ]
    assert client.price_zones == list(_NEIGHBOURS)
    assert client.net_zones == ["SE_3"]


def test_fetch_cross_border_sums_net_position_across_zones():
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    client = _fake_cross_border_client(start, 16 * 24)
    client.net_position["SE_1"] = _hourly(start, 16 * 24, 200.0)
    client.net_position["SE_2"] = _hourly(start, 16 * 24, -50.0)

    df = fetch_cross_border(["SE1", "SE2"], start, end, client=client)

    assert (df["net_position_mwh"] == 150.0).all()
    assert client.net_zones == ["SE_1", "SE_2"]
    assert client.scheduled_borders == [
        ("SE_1", "FI"),
        ("SE_1", "NO_4"),
        ("SE_2", "NO_3"),
        ("SE_2", "NO_4"),
    ]


def test_fetch_cross_border_upsamples_hourly_to_15min_across_mtu_switch():
    start, end = date(2025, 9, 20), date(2025, 10, 5)
    client = _fake_cross_border_client(start, 16 * 24)

    df = fetch_cross_border(["SE3"], start, end, client=client)

    assert _mtu_minutes_for(end) == 15
    diffs = df.index.to_series().diff().dropna().dt.total_seconds()
    assert (diffs == 900.0).all()


def test_fetch_cross_border_requires_api_key_or_client():
    with pytest.raises(ValueError):
        fetch_cross_border(["SE3"], date(2024, 10, 4), date(2024, 10, 20))


@pytest.mark.parametrize("zone", ["", "SE5", "DE", "SE", "se3"])
def test_fetch_cross_border_raises_on_unknown_zone(zone):
    start, end = date(2024, 10, 4), date(2024, 10, 20)
    with pytest.raises(ValueError, match="Unknown ENTSO-E zone"):
        fetch_cross_border([zone], start, end, client=_FakeEntsoeClient())
