"""Tests for the no-key external ingestion fetchers (weather, FX, carbon).

HTTP is mocked via the injected `fetch` callable -- no live API calls.
"""

import io
import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.ingestion_external import (
    _EEX_PRICE_COLUMN,
    _EEX_SHEET,
    _WEATHER_VARIABLES,
    _open_meteo_url,
    fetch_carbon,
    fetch_fx,
    fetch_weather,
)
from forecast_pipeline.pipeline import _mtu_minutes_for


def _weather_payload(start: pd.Timestamp, hours: int, seed: int = 1) -> bytes:
    """A recorded Open-Meteo `hourly` response over `hours` hours."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    hourly = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": rng.normal(0.0, 5.0, hours).tolist(),
        "wind_speed_10m": rng.normal(5.0, 2.0, hours).tolist(),
        "wind_speed_100m": rng.normal(8.0, 3.0, hours).tolist(),
        "shortwave_radiation": rng.normal(200.0, 100.0, hours).tolist(),
        "precipitation": rng.normal(0.1, 0.2, hours).tolist(),
        "snowfall": rng.normal(0.0, 0.1, hours).tolist(),
    }
    return json.dumps({"hourly": hourly}).encode("utf-8")


def _ecb_csv(pairs: dict[str, float]) -> bytes:
    """A recorded ECB EXR csvdata response."""
    header = "KEY,FREQ,CURRENCY,CURRENCY_DENOM,EXR_TYPE,EXR_SUFFIX,TIME_PERIOD,OBS_VALUE,OBS_STATUS"
    lines = [header]
    for period, value in pairs.items():
        lines.append(f"EXR.D.SEK.EUR.SP00.A,D,SEK,EUR,SP00,A,{period},{value},A")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _eex_xlsx(pairs: dict[str, float]) -> bytes:
    """A recorded EEX EUA auction xlsx mirroring the header-on-row-5 layout."""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(list(pairs)),
            _EEX_PRICE_COLUMN: list(pairs.values()),
            "Status": ["successful"] * len(pairs),
        }
    )
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=_EEX_SHEET, index=False, startrow=5)
    return buffer.getvalue()


def test_fetch_weather_returns_expected_shape():
    start, end = date(2024, 1, 1), date(2024, 1, 3)
    payload = _weather_payload(pd.Timestamp("2024-01-01", tz="UTC"), hours=48)

    df = fetch_weather(["SE1", "SE2", "SE3", "SE4"], start, end, fetch=lambda url: payload)

    assert isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns.names) == ["zone", "variable"]
    assert list(df.columns.get_level_values(0).unique()) == ["SE1", "SE2", "SE3", "SE4"]
    assert list(df.columns.get_level_values(1).unique()) == list(_WEATHER_VARIABLES)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any()
    step_minutes = df.index.to_series().diff().dropna().dt.total_seconds() / 60
    assert list(step_minutes.unique()) == [60.0]


def test_fetch_weather_accepts_single_zone_string():
    start, end = date(2024, 1, 1), date(2024, 1, 2)
    payload = _weather_payload(pd.Timestamp("2024-01-01", tz="UTC"), hours=24)

    df = fetch_weather("SE3", start, end, fetch=lambda url: payload)

    assert list(df.columns.get_level_values(0).unique()) == ["SE3"]
    assert not df.isna().any().any()


def test_fetch_weather_upsamples_hourly_to_15min():
    start, end = date(2025, 10, 1), date(2025, 10, 2)
    payload = _weather_payload(pd.Timestamp("2025-10-01", tz="UTC"), hours=24)

    df = fetch_weather(["SE1"], start, end, fetch=lambda url: payload)

    assert _mtu_minutes_for(end) == 15
    step_minutes = df.index.to_series().diff().dropna().dt.total_seconds() / 60
    assert list(step_minutes.unique()) == [15.0]
    assert len(df) == 96
    temperature = df[("SE1", "temperature_2m")]
    assert (temperature.iloc[:4] == temperature.iloc[0]).all()


@pytest.mark.parametrize("zone", ["SE5", "DE", "se3", ""])
def test_fetch_weather_raises_on_unknown_zone(zone):
    with pytest.raises(ValueError, match="Unknown zone"):
        fetch_weather([zone], date(2024, 1, 1), date(2024, 1, 2), fetch=lambda url: b"{}")


def test_fetch_fx_returns_expected_shape():
    start, end = date(2024, 1, 2), date(2024, 1, 5)
    csv = _ecb_csv({"2024-01-02": 11.15, "2024-01-03": 11.19, "2024-01-04": 11.20})

    df = fetch_fx(start, end, fetch=lambda url: csv)

    assert list(df.columns) == ["fx_sek_eur"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any()


def test_fetch_fx_uses_prior_day_close():
    start, end = date(2024, 1, 3), date(2024, 1, 5)
    csv = _ecb_csv({"2024-01-02": 11.15, "2024-01-03": 11.19, "2024-01-04": 11.20})

    df = fetch_fx(start, end, fetch=lambda url: csv)

    assert len(df) == 48  # two days, hourly MTU
    # Day D's feature is the prior-day close, so 2024-01-03 uses 2024-01-02's
    # rate (11.15), never the same-day 11.19.
    assert df["fx_sek_eur"].iloc[0] == pytest.approx(11.15)
    assert df["fx_sek_eur"].iloc[24] == pytest.approx(11.19)  # 2024-01-04 -> 01-03 close
    assert df["fx_sek_eur"].iloc[-1] == pytest.approx(11.19)


def test_fetch_carbon_returns_expected_shape():
    start, end = date(2024, 1, 2), date(2024, 1, 5)
    xlsx = _eex_xlsx({"2024-01-02": 63.64, "2024-01-03": 66.07, "2024-01-04": 67.95})

    df = fetch_carbon(start, end, fetch=lambda url: xlsx)

    assert list(df.columns) == ["carbon_eua"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.isna().any().any()


def test_fetch_carbon_uses_prior_day_close():
    start, end = date(2024, 1, 3), date(2024, 1, 5)
    xlsx = _eex_xlsx({"2024-01-02": 63.64, "2024-01-03": 66.07, "2024-01-04": 67.95})

    df = fetch_carbon(start, end, fetch=lambda url: xlsx)

    assert len(df) == 48
    # Day D's feature is the prior-day close, so 2024-01-03 uses 2024-01-02's
    # auction price (63.64), never the same-day 66.07.
    assert df["carbon_eua"].iloc[0] == pytest.approx(63.64)
    assert df["carbon_eua"].iloc[24] == pytest.approx(66.07)  # 2024-01-04 -> 01-03 close
    assert df["carbon_eua"].iloc[-1] == pytest.approx(66.07)


def test_open_meteo_url_raises_when_history_exceeds_forecast_reach():
    today = datetime.now(UTC).date()
    start = today - timedelta(days=100)
    end = today + timedelta(days=2)

    with pytest.raises(ValueError, match="92-day reach"):
        _open_meteo_url(59.0, 15.0, start, end)
