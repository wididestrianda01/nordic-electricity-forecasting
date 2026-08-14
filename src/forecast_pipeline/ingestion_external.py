"""No-key external data ingestion: weather, FX, and carbon.

Each fetcher returns a DataFrame in the common `historical_data` shape used by
the pipeline: a `DatetimeIndex` of MTU-start timestamps in UTC, sorted
ascending, with no NaNs.

Daily series (FX, carbon) are forward-filled onto the MTU grid. A daily
observation is indexed at midnight UTC of its calendar date and carried forward
to every MTU of that day (and on to the next observation), matching the
"known at start of day" convention of the market-data fetcher.
"""

from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from forecast_pipeline.pipeline import _mtu_minutes_for

# Open-Meteo hourly variables, in the order the weather columns are exposed.
_WEATHER_VARIABLES = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "shortwave_radiation",
    "precipitation",
    "snowfall",
)

# Approximate bidding-zone centroids (WGS84 lat, lon) for SE1-SE4. Open-Meteo
# accepts arbitrary coordinates, so these stand in for the whole zone.
_ZONE_CENTROID = {
    "SE1": (66.0, 20.0),
    "SE2": (63.3, 16.0),
    "SE3": (59.5, 15.5),
    "SE4": (55.7, 13.3),
}

_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_ECB_EXR_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.SEK.EUR.SP00.A"

# EEX publishes one xlsx per calendar year (2020 onward); earlier years are
# legacy .xls files that this fetcher does not read.
_EEX_AUCTION_URL = (
    "https://public.eex-group.com/eex/eua-auction-report/"
    "emission-spot-primary-market-auction-report-{year}-data.xlsx"
)
_EEX_XLSX_SINCE = 2020
_EEX_SHEET = "Primary Market Auction"
_EEX_PRICE_COLUMN = "Auction Price €/tCO2"


def _http_get_bytes(url: str) -> bytes:
    """Fetch a URL and return the raw response body."""
    request = urllib.request.Request(url, headers={"User-Agent": "forecast-pipeline/0.1"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _as_date(value: date | str) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _normalize_zones(zones) -> list[str]:
    if isinstance(zones, str):
        zones = [zones]
    unknown = [z for z in zones if z not in _ZONE_CENTROID]
    if unknown:
        raise ValueError(
            f"Unknown zone(s) {unknown!r}; expected one of SE1, SE2, SE3, SE4"
        )
    return list(zones)


def _mtu_grid(start: date, end: date, mtu_minutes: int) -> pd.DatetimeIndex:
    """MTU-start timestamps covering [start, end), in UTC."""
    return pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        freq=f"{mtu_minutes}min",
        inclusive="left",
    )


def _forward_fill_to_grid(
    frame: pd.Series | pd.DataFrame, grid: pd.DatetimeIndex
) -> pd.Series | pd.DataFrame:
    """Forward-fill `frame` onto `grid`.

    Reindexing on the union of the source index and the grid then forward-
    filling handles both upsampling (hourly -> 15-minute) and the trailing
    edge: the final observation carries through to the end of the grid.
    """
    return frame.reindex(frame.index.union(grid)).ffill().reindex(grid)


def _open_meteo_url(lat: float, lon: float, start: date, end: date) -> str:
    params: dict[str, object] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_WEATHER_VARIABLES),
        "timezone": "UTC",
    }
    if end < datetime.now(UTC).date():
        base = _OPEN_METEO_ARCHIVE_URL
        params["start_date"] = start.isoformat()
        params["end_date"] = end.isoformat()
    else:
        # The forecast API serves up to 92 days of history via past_days;
        # deeper history belongs to the archive endpoint (end < today), so
        # refuse rather than silently dropping the deep tail.
        history_days = (datetime.now(UTC).date() - start).days
        if history_days > 92:
            raise ValueError(
                f"requested weather history of {history_days} days exceeds the "
                f"forecast endpoint's 92-day reach; use an end date before "
                f"today to query the archive endpoint"
            )
        base = _OPEN_METEO_FORECAST_URL
        params["past_days"] = history_days
        params["forecast_days"] = (end - datetime.now(UTC).date()).days + 1
    return f"{base}?{urllib.parse.urlencode(params)}"


def _parse_open_meteo_hourly(payload: dict) -> pd.DataFrame:
    hourly = payload["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    return pd.DataFrame(
        {variable: hourly[variable] for variable in _WEATHER_VARIABLES},
        index=index,
    )


def _parse_ecb_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise ValueError(
            f"ECB response missing TIME_PERIOD/OBS_VALUE columns; found {list(df.columns)}"
        )
    df = df.dropna(subset=["OBS_VALUE"])
    series = pd.Series(
        pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["TIME_PERIOD"], utc=True),
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def _parse_eex_auction(data: bytes) -> pd.Series:
    df = pd.read_excel(io.BytesIO(data), sheet_name=_EEX_SHEET, header=5)
    if _EEX_PRICE_COLUMN not in df.columns or "Date" not in df.columns:
        raise ValueError(
            f"EEX auction sheet missing Date/{_EEX_PRICE_COLUMN!r} columns; "
            f"found {list(df.columns)}"
        )
    df = df.dropna(subset=[_EEX_PRICE_COLUMN])
    series = pd.Series(
        pd.to_numeric(df[_EEX_PRICE_COLUMN], errors="coerce").to_numpy(),
        index=pd.to_datetime(df["Date"], utc=True),
    )
    return series[~series.index.duplicated(keep="last")].sort_index()


def fetch_weather(
    zones,
    start: date,
    end: date,
    fetch: Callable[[str], bytes] = _http_get_bytes,
) -> pd.DataFrame:
    """Fetch the six weather variables for each zone centroid from Open-Meteo.

    Returns a DataFrame indexed by MTU-start UTC timestamp with a two-level
    column index (`zone`, `variable`); the variable level holds exactly the six
    names in `_WEATHER_VARIABLES`.
    """
    zones = _normalize_zones(zones)
    start_d, end_d = _as_date(start), _as_date(end)
    grid = _mtu_grid(start_d, end_d, _mtu_minutes_for(end_d))

    frames = {}
    for zone in zones:
        lat, lon = _ZONE_CENTROID[zone]
        payload = json.loads(fetch(_open_meteo_url(lat, lon, start_d, end_d)).decode("utf-8"))
        frames[zone] = _forward_fill_to_grid(_parse_open_meteo_hourly(payload), grid)

    combined = pd.concat(frames, axis=1)
    combined.columns.names = ["zone", "variable"]
    return combined.sort_index().dropna()


def fetch_fx(
    start: date,
    end: date,
    fetch: Callable[[str], bytes] = _http_get_bytes,
) -> pd.DataFrame:
    """Fetch the daily ECB SEK/EUR reference rate, forward-filled to MTU.

    The rate published for a calendar date is that date's close, known only
    after the fact. Day D's feature is therefore the prior-day close: the
    daily series is shifted one day forward (each value carried to the next
    day) before forward-filling onto the MTU grid.
    """
    start_d, end_d = _as_date(start), _as_date(end)
    grid = _mtu_grid(start_d, end_d, _mtu_minutes_for(end_d))

    params = urllib.parse.urlencode(
        {
            "startPeriod": (start_d - timedelta(days=1)).isoformat(),
            "endPeriod": end_d.isoformat(),
            "format": "csvdata",
        }
    )
    series = _parse_ecb_csv(fetch(f"{_ECB_EXR_URL}?{params}").decode("utf-8"))
    series = series.shift(1, freq="D")
    return pd.DataFrame({"fx_sek_eur": _forward_fill_to_grid(series, grid)}).sort_index().dropna()


def fetch_carbon(
    start: date,
    end: date,
    fetch: Callable[[str], bytes] = _http_get_bytes,
) -> pd.DataFrame:
    """Fetch the daily EEX EUA primary-auction price, forward-filled to MTU.

    The auction price for a calendar date is that date's close, known only
    after the auction settles. Day D's feature is therefore the prior-day
    close: the daily series is shifted one day forward (each value carried to
    the next day) before forward-filling onto the MTU grid.
    """
    start_d, end_d = _as_date(start), _as_date(end)
    grid = _mtu_grid(start_d, end_d, _mtu_minutes_for(end_d))

    parts = [
        _parse_eex_auction(fetch(_EEX_AUCTION_URL.format(year=year)))
        for year in range(max(start_d.year, _EEX_XLSX_SINCE), end_d.year + 1)
    ]
    if not parts:
        return pd.DataFrame({"carbon_eua": pd.Series(dtype="float64", index=grid)}).dropna()
    series = pd.concat(parts)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series.shift(1, freq="D")
    return pd.DataFrame({"carbon_eua": _forward_fill_to_grid(series, grid)}).sort_index().dropna()
