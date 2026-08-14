"""Ticket 01: ENTSO-E data ingestion for SE3.

Fetches day-ahead price, load forecast, and wind forecast and normalizes them
into the `historical_data` shape forecast_pipeline/lear_forecast validate:
columns price, load_forecast, wind_forecast; MTU-start UTC DatetimeIndex; no NaNs.
"""

import os
from datetime import date

import pandas as pd
from entsoe import EntsoePandasClient

from forecast_pipeline.pipeline import _mtu_minutes_for

# ENTSO-E reports wind generation split by PSR type (onshore/offshore); sum
# whichever of those columns the zone actually reports.
_WIND_COLUMN_MARKER = "Wind"

# Domain bidding-zone names map to entsoe-py's Area enum member names.
_ENTSOE_ZONE = {"SE1": "SE_1", "SE2": "SE_2", "SE3": "SE_3", "SE4": "SE_4"}
_ENTSOE_ZONE_CODES = frozenset(_ENTSOE_ZONE.values())


def _resample_to_mtu(series: pd.Series, mtu_minutes: int) -> pd.Series:
    """Align a native-resolution series onto the target MTU grid.

    Handles the 1 Oct 2025 hourly-to-15-minute switch (ADR-0003): upsamples
    hourly data to 15-minute via forward-fill when the target grid is finer.
    """
    return series.tz_convert("UTC").resample(f"{mtu_minutes}min").ffill()


def fetch_market_data(
    zone: str,
    start: date,
    end: date,
    api_key: str | None = None,
    client: EntsoePandasClient | None = None,
) -> pd.DataFrame:
    """Fetch ENTSO-E day-ahead price, load forecast, and wind forecast for `zone`.

    `zone` is a Swedish bidding zone, one of SE1, SE2, SE3, SE4 (the
    entsoe-py codes SE_1..SE_4 are also accepted).

    Returns a DataFrame with columns price, load_forecast, wind_forecast,
    indexed by MTU-start timestamp UTC, sorted ascending, no NaNs.
    """
    zone = _ENTSOE_ZONE.get(zone, zone)
    if zone not in _ENTSOE_ZONE_CODES:
        raise ValueError(f"Unknown ENTSO-E zone {zone!r}; expected one of SE1, SE2, SE3, SE4")

    if client is None:
        api_key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise ValueError("api_key (or ENTSOE_API_KEY env var) is required when client is not provided")
        client = EntsoePandasClient(api_key=api_key)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mtu_minutes = _mtu_minutes_for(end)

    price = client.query_day_ahead_prices(zone, start=start_ts, end=end_ts)
    load_df = client.query_load_forecast(zone, start=start_ts, end=end_ts)
    if "Forecasted Load" not in load_df.columns:
        raise ValueError(
            f"Expected column 'Forecasted Load' in load forecast response, but found: {list(load_df.columns)}"
        )
    load = load_df["Forecasted Load"]
    wind_df = client.query_wind_and_solar_forecast(zone, start=start_ts, end=end_ts)
    wind_cols = [c for c in wind_df.columns if _WIND_COLUMN_MARKER in c]
    if not wind_cols:
        raise ValueError(
            f"No columns containing '{_WIND_COLUMN_MARKER}' found in wind forecast response. Found: {list(wind_df.columns)}"
        )
    wind = wind_df[wind_cols].sum(axis=1)

    df = pd.DataFrame(
        {
            "price": _resample_to_mtu(price, mtu_minutes),
            "load_forecast": _resample_to_mtu(load, mtu_minutes),
            "wind_forecast": _resample_to_mtu(wind, mtu_minutes),
        }
    )
    return df.sort_index().dropna()
