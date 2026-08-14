"""Ticket 04: feature engineering -- price-only and full-features matrices.

Emits the two matrices from the locked feature spec (ticket 08):

* price-only (group 1): price lags 1/2/3/7/14/28 d, rolling 7 d mean/std,
  cyclical sin/cos hour/dow/month, a Swedish holiday flag, and the HMM regime
  label.
* full-features: price-only plus the exogenous groups (day-ahead load/wind,
  per-zone weather, hydro storage, cross-border, carbon, FX).

As-of timing: lag and rolling features are built from strictly-past prices via
`.shift` -- never the current or a future row -- and `assemble_data` joins the
ingestion fetchers' day-ahead / prior-day-close / forward-fill outputs, so no
realized exogenous value is ever used. Lag and rolling lookbacks that span a
regime boundary (4 Nov 2024, 1 Oct 2025) are masked to NaN via
`crosses_boundary`, matching the walk-forward convention in ADR-0007.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd

from forecast_pipeline import datacache
from forecast_pipeline.ingestion import fetch_market_data
from forecast_pipeline.ingestion_external import fetch_carbon, fetch_fx, fetch_weather
from forecast_pipeline.ingestion_transparency import fetch_cross_border, fetch_hydro
from forecast_pipeline.pipeline import _mtu_minutes_for, _validate
from forecast_pipeline.regime import detect_regimes
from forecast_pipeline.regime_boundaries import crosses_boundary

LAG_DAYS = (1, 2, 3, 7, 14, 28)
ROLLING_DAYS = 7

# Fixed Swedish public holidays (month, day); movable feasts derive from Easter.
_FIXED_HOLIDAYS = ((1, 1), (1, 6), (5, 1), (6, 6), (12, 25), (12, 26))


def _easter_sunday(year: int) -> date:
    """Gregorian (Anonymous) Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _swedish_holidays(year: int) -> set[date]:
    """Swedish public holidays (red days) for `year`."""
    easter = _easter_sunday(year)
    holidays = {date(year, m, d) for m, d in _FIXED_HOLIDAYS}
    holidays.update(
        {
            easter - timedelta(days=2),  # Good Friday
            easter,  # Easter Sunday
            easter + timedelta(days=1),  # Easter Monday
            easter + timedelta(days=39),  # Ascension
            easter + timedelta(days=49),  # Pentecost
        }
    )
    midsummer = next(
        date(year, 6, d) for d in range(20, 27) if date(year, 6, d).weekday() == 5
    )
    holidays.add(midsummer)
    all_saints = next(
        date(year, 10, 31) + timedelta(days=offset)
        for offset in range(7)
        if (date(year, 10, 31) + timedelta(days=offset)).weekday() == 5
    )
    holidays.add(all_saints)
    return holidays


def _is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    holiday_dates: set[date] = set()
    for year in set(index.year):
        holiday_dates.update(_swedish_holidays(int(year)))
    return np.array([ts.date() in holiday_dates for ts in index], dtype=bool)


def _cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2.0 * np.pi * values.astype(float) / period
    return np.sin(angle), np.cos(angle)


def _crossing_mask(index: pd.DatetimeIndex, lookback_days: int) -> np.ndarray:
    """True where the `lookback_days` window ending at each timestamp crosses a regime boundary."""
    return np.array(
        [
            crosses_boundary(ts.date(), ts.date() - timedelta(days=lookback_days))
            for ts in index
        ],
        dtype=bool,
    )


def _lag_features(price: pd.Series, index: pd.DatetimeIndex, mtu_minutes: int) -> pd.DataFrame:
    steps_per_day = 24 * 60 // mtu_minutes
    out = pd.DataFrame(index=index)
    for lag in LAG_DAYS:
        col = f"lag_{lag}d"
        out[col] = price.shift(lag * steps_per_day)
        out.loc[_crossing_mask(index, lag), col] = np.nan
    return out


def _rolling_features(price: pd.Series, index: pd.DatetimeIndex, mtu_minutes: int) -> pd.DataFrame:
    window = ROLLING_DAYS * (24 * 60 // mtu_minutes)
    trailing = price.shift(1)
    out = pd.DataFrame(index=index)
    out["roll_mean_7d"] = trailing.rolling(window).mean()
    out["roll_std_7d"] = trailing.rolling(window).std()
    out.loc[_crossing_mask(index, ROLLING_DAYS), ["roll_mean_7d", "roll_std_7d"]] = np.nan
    return out


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = pd.Series(index.hour + index.minute / 60.0, index=index)
    dow = pd.Series(index.dayofweek, index=index)
    month = pd.Series(index.month - 1, index=index)
    out = pd.DataFrame(index=index)
    out["hour_sin"], out["hour_cos"] = _cyclical(hour, 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(dow, 7)
    out["month_sin"], out["month_cos"] = _cyclical(month, 12)
    out["is_holiday"] = _is_holiday(index)
    return out


def _price_features(historical_data: pd.DataFrame, mtu_minutes: int) -> pd.DataFrame:
    index = historical_data.index
    price = historical_data["price"]
    features = pd.concat(
        [
            _lag_features(price, index, mtu_minutes),
            _rolling_features(price, index, mtu_minutes),
            _calendar_features(index),
        ],
        axis=1,
    )
    features["regime"] = detect_regimes(historical_data)
    return features


def _exogenous_columns(historical_data: pd.DataFrame) -> pd.DataFrame:
    return historical_data[[c for c in historical_data.columns if c != "price"]]


def build_features(
    as_of_date: date, historical_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(price_only, full_features)` from an assembled history frame.

    `price_only` holds group-1 features (lags, rolling, calendar, holiday,
    regime); `full_features` appends every non-price column of
    `historical_data` (the six exogenous groups). The raw `price` column is
    the model target and is intentionally excluded from both matrices.
    """
    mtu_minutes = _mtu_minutes_for(as_of_date)
    _validate(historical_data, mtu_minutes)

    price_only = _price_features(historical_data, mtu_minutes)
    full_features = pd.concat([price_only, _exogenous_columns(historical_data)], axis=1)
    return price_only, full_features


def _forecast_grid(historical_data: pd.DataFrame, mtu_minutes: int) -> pd.DatetimeIndex:
    """Return the D+1 forecast grid: one MTU step past the last history row."""
    periods_per_day = 24 * 60 // mtu_minutes
    start = historical_data.index[-1] + pd.Timedelta(minutes=mtu_minutes)
    return pd.date_range(start, periods=periods_per_day, freq=f"{mtu_minutes}min")


def _price_only_horizon(
    historical_data: pd.DataFrame,
    grid: pd.DatetimeIndex,
    mtu_minutes: int,
    regime_label: str,
) -> pd.DataFrame:
    """Group-1 (price-only) features on the forecast grid.

    Lags and rolling features are computed by extending the price series with
    the unknown horizon (NaN), reusing the exact shift / rolling /
    boundary-masking logic from ``build_features``. Lags resolve to
    strictly-past published prices; rolling windows that reach into the
    unknown horizon stay NaN (LightGBM treats missing covariates natively).
    """
    price = historical_data["price"]
    extended_index = cast(pd.DatetimeIndex, price.index.union(grid))
    extended = price.reindex(extended_index).sort_index()
    lags = _lag_features(extended, extended_index, mtu_minutes).loc[grid]
    rolls = _rolling_features(extended, extended_index, mtu_minutes).loc[grid]
    calendar = _calendar_features(grid)
    out = pd.concat([lags, rolls, calendar], axis=1)
    out["regime"] = regime_label
    return out


def _exogenous_horizon(historical_data: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Group-2 (exogenous) features on the forecast grid.

    Every exogenous value is forward-filled from the last published row, so the
    horizon never uses a value published after the as-of date -- the
    as-of-timing rule (day-ahead load/wind, one-day-lagged weather, prior-day
    FX/carbon close, forward-filled hydro).
    """
    exog_cols = [c for c in historical_data.columns if c != "price"]
    if not exog_cols:
        return pd.DataFrame(index=grid)
    exog = historical_data[exog_cols]
    return exog.reindex(exog.index.union(grid)).ffill().reindex(grid)


def build_horizon_features(as_of_date: date, historical_data: pd.DataFrame) -> pd.DataFrame:
    """Return the full-features matrix for the D+1 forecast grid.

    Columns are identical to ``build_features(as_of_date, historical_data)[1]``
    (the canonical full-features matrix); the index is the D+1 forecast grid
    (24 hourly or 96 15-minute rows). No value is published after ``as_of_date``:
    price lags/rolling look strictly backwards (rolling into the unknown horizon
    is NaN), calendar/regime derive from the grid / most-recent label, and every
    exogenous value is forward-filled from the last published row.
    """
    mtu_minutes = _mtu_minutes_for(as_of_date)
    _validate(historical_data, mtu_minutes)
    _, full_features = build_features(as_of_date, historical_data)

    grid = _forecast_grid(historical_data, mtu_minutes)
    horizon = _price_only_horizon(
        historical_data, grid, mtu_minutes, full_features["regime"].iloc[-1]
    )
    horizon = pd.concat([horizon, _exogenous_horizon(historical_data, grid)], axis=1)
    return horizon[list(full_features.columns)]


def assemble_data(zones, start: date, end: date, *, refresh: bool = False) -> pd.DataFrame:
    """Join all ingestion outputs into one MTU-indexed frame.

    The first zone is the forecast target (its price/load/wind); every zone
    contributes weather, hydro, and cross-border features. Weather's
    `(zone, variable)` MultiIndex columns are flattened to `ZONE_variable`.

    The joined frame is cached to disk (see `datacache`) keyed by zones and
    date range, so a re-run within the same window skips the network. Pass
    `refresh=True` to re-fetch after a fetcher or schema change.
    """

    zone_list = [zones] if isinstance(zones, str) else list(zones)
    if not zone_list:
        raise ValueError("zones must be a non-empty sequence of bidding zones")
    primary = zone_list[0]

    params = {"zones": zone_list}
    if not refresh:
        cached = datacache.load("assemble_data", start, end, params)
        if cached is not None:
            return cached

    market = fetch_market_data(primary, start, end)

    weather = fetch_weather(zone_list, start, end)
    # Weather is realized (observed) at its timestamp; per ticket 08 it may
    # only enter a forecast made at t via realized weather from t-1. Shift the
    # whole frame by one full day's MTU steps so row t holds t-1's weather.
    weather = weather.shift(24 * 60 // _mtu_minutes_for(end))
    weather.columns = [f"{zone}_{variable}" for zone, variable in weather.columns]

    hydro = fetch_hydro(zone_list, start, end)
    cross_border = fetch_cross_border(zone_list, start, end)
    fx = fetch_fx(start, end)
    carbon = fetch_carbon(start, end)

    frame = pd.concat(
        [market, weather, hydro, cross_border, fx, carbon], axis=1
    ).sort_index()
    datacache.store("assemble_data", start, end, params, frame)
    return frame
