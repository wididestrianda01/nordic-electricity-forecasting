"""ENTSO-E Transparency fetchers for P14's slow and cross-border feature groups.

Hydro reservoir storage (documentType A72, weekly) and cross-border data
(net positions A25/B09, scheduled day-ahead exchanges A09, neighbour
day-ahead prices) for the Swedish bidding zones SE1-SE4.

Both fetchers return a DataFrame indexed by MTU-start UTC DatetimeIndex,
sorted ascending, with no NaNs.
"""

import os
from datetime import date

import pandas as pd
from entsoe import EntsoePandasClient

from forecast_pipeline.ingestion import (
    _ENTSOE_ZONE,
    _ENTSOE_ZONE_CODES,
    _resample_to_mtu,
)
from forecast_pipeline.pipeline import _mtu_minutes_for

# External bidding zones bordering SE1-SE4. Their day-ahead prices set the
# neighbour reference price level used downstream to compute cross-border
# price spreads (neighbour price minus the zone's own day-ahead price).
_NEIGHBOUR_ZONES = ("FI", "NO_1", "NO_3", "NO_4", "DK_1", "DK_2", "DE_LU", "PL")

# External borders of the Swedish bidding zones: (swedish zone, neighbour).
# Per-border day-ahead capacity is not freely archived across the
# 4 Nov 2024 flow-based transition, so only flows/positions/prices are
# ingested here -- capacity is out of scope.
_BORDERS = (
    ("SE1", "FI"),
    ("SE1", "NO_4"),
    ("SE2", "NO_3"),
    ("SE2", "NO_4"),
    ("SE3", "NO_1"),
    ("SE3", "DK_1"),
    ("SE3", "FI"),
    ("SE4", "DK_2"),
    ("SE4", "DE_LU"),
    ("SE4", "PL"),
)


def _get_client(api_key: str | None, client: EntsoePandasClient | None) -> EntsoePandasClient:
    if client is not None:
        return client
    api_key = api_key or os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise ValueError(
            "api_key (or ENTSOE_API_KEY env var) is required when client is not provided"
        )
    return EntsoePandasClient(api_key=api_key)


def _resolve_zones(zones: list[str]) -> list[str]:
    """Map domain bidding-zone names (SE1..SE4) to entsoe-py codes (SE_1..SE_4)."""
    resolved = [_ENTSOE_ZONE.get(zone, zone) for zone in zones]
    for zone in resolved:
        if zone not in _ENTSOE_ZONE_CODES:
            raise ValueError(
                f"Unknown ENTSO-E zone {zone!r}; expected one of SE1, SE2, SE3, SE4"
            )
    return resolved


def _ffill_to_grid(series: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill `series` onto `grid`, extending the last value to grid end."""
    series = series.tz_convert("UTC").sort_index()
    return series.reindex(series.index.union(grid)).ffill().reindex(grid)


def fetch_hydro(
    zones: list[str],
    start: date,
    end: date,
    api_key: str | None = None,
    client: EntsoePandasClient | None = None,
) -> pd.DataFrame:
    """Fetch weekly hydro reservoir storage (documentType A72) for `zones`.

    Storage is published weekly only (P7D) -- no free daily source exists --
    so treat it as a slow state variable: the most recent weekly value is
    forward-filled across the MTUs it covers, through `end`.

    Returns a DataFrame with a single column hydro_storage_mwh (MWh), indexed
    by MTU-start UTC, sorted ascending, no NaNs.
    """
    client = _get_client(api_key, client)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mtu_minutes = _mtu_minutes_for(end)
    zones = _resolve_zones(zones)

    storage = pd.concat(
        [
            client.query_aggregate_water_reservoirs_and_hydro_storage(
                zone, start=start_ts, end=end_ts
            )
            for zone in zones
        ],
        axis=1,
    ).sum(axis=1)

    grid = pd.date_range(start_ts, end_ts, freq=f"{mtu_minutes}min", inclusive="left")
    df = pd.DataFrame({"hydro_storage_mwh": _ffill_to_grid(storage, grid)})
    return df.sort_index().dropna()


def fetch_cross_border(
    zones: list[str],
    start: date,
    end: date,
    api_key: str | None = None,
    client: EntsoePandasClient | None = None,
) -> pd.DataFrame:
    """Fetch cross-border data for `zones`: net positions, scheduled day-ahead
    exchanges, and the neighbour day-ahead price level.

    Columns:
    - net_position_mwh (MW): sum of per-zone day-ahead net positions
      (documentType A25, businessType B09); positive = net export.
    - scheduled_exchange_mwh (MW): sum of scheduled day-ahead exchanges
      (documentType A09) across the zones' external borders; positive = export.
    - neighbour_price_eur_mwh (EUR/MWh): mean day-ahead price across the
      bordering zones FI, NO1, NO3, NO4, DK1, DK2, DE_LU, PL. The spread
      against a zone's own price is derived downstream (neighbour minus own).

    Per-border day-ahead capacity is not freely archived across the
    4 Nov 2024 flow-based transition, so capacity is not ingested here.
    """
    client = _get_client(api_key, client)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    mtu_minutes = _mtu_minutes_for(end)
    zones = _resolve_zones(zones)
    zone_set = set(zones)

    net_position = pd.concat(
        [
            client.query_net_position(zone, start=start_ts, end=end_ts, dayahead=True)
            for zone in zones
        ],
        axis=1,
    ).sum(axis=1)

    exchanges = []
    for swedish_zone, neighbour in _BORDERS:
        zone_code = _ENTSOE_ZONE.get(swedish_zone, swedish_zone)
        if zone_code not in zone_set:
            continue
        exchanges.append(
            client.query_scheduled_exchanges(
                zone_code, neighbour, start=start_ts, end=end_ts, dayahead=True
            )
        )
    scheduled_exchange = pd.concat(exchanges, axis=1).sum(axis=1)

    neighbour_price = pd.concat(
        [
            _resample_to_mtu(
                client.query_day_ahead_prices(zone, start=start_ts, end=end_ts),
                mtu_minutes,
            )
            for zone in _NEIGHBOUR_ZONES
        ],
        axis=1,
    ).mean(axis=1)

    df = pd.DataFrame(
        {
            "net_position_mwh": _resample_to_mtu(net_position, mtu_minutes),
            "scheduled_exchange_mwh": _resample_to_mtu(scheduled_exchange, mtu_minutes),
            "neighbour_price_eur_mwh": neighbour_price,
        }
    )
    return df.sort_index().dropna()
