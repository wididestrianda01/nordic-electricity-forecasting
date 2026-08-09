"""Ticket 01: pipeline skeleton and P14->P16 output contract.

Ticket 04 (regime-conditional LightGBM) replaces the placeholder quantile logic
with real quantile regression, conditioned on the HMM regime label from ticket 02.
Ticket 03 (LEAR baseline) is available as an alternative via lear_forecast.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from forecast_pipeline.lgbm import lgbm_quantile_forecast

# MTU switched from hourly to 15-minute on this date (see ADR-0003).
MTU_15MIN_SWITCH_DATE = date(2025, 10, 1)

# Empirical quantiles need enough samples to be stable against Nordic
# electricity's weekly demand seasonality -- one week is the minimum unit.
MIN_HISTORY_DAYS = 7


@dataclass(frozen=True)
class ForecastOutput:
    as_of_date: date
    mtu_minutes: int
    forecast: pd.DataFrame  # index: MTU start (UTC); columns: p10, p50, p90, regime
    baseline_forecast: pd.Series  # LEAR baseline point forecast, for comparison (ticket 03/05)


def _mtu_minutes_for(as_of_date: date) -> int:
    return 15 if as_of_date >= MTU_15MIN_SWITCH_DATE else 60


def _validate(historical_data: pd.DataFrame, mtu_minutes: int) -> None:
    if "price" not in historical_data.columns:
        raise ValueError("historical_data must have a 'price' column")
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        raise TypeError("historical_data must be indexed by timestamp")
    if not historical_data.index.is_monotonic_increasing:
        raise ValueError("historical_data index must be sorted ascending")
    if historical_data["price"].isna().any():
        raise ValueError("historical_data['price'] must not contain NaNs")

    min_rows = MIN_HISTORY_DAYS * (24 * 60 // mtu_minutes)
    if len(historical_data) < min_rows:
        raise ValueError(
            f"historical_data has {len(historical_data)} rows; "
            f"need at least {min_rows} ({MIN_HISTORY_DAYS} days at "
            f"{mtu_minutes}-minute MTU)"
        )


def forecast_pipeline(as_of_date: date, historical_data: pd.DataFrame) -> ForecastOutput:
    mtu_minutes = _mtu_minutes_for(as_of_date)
    _validate(historical_data, mtu_minutes)

    # Use regime-conditional LightGBM quantile regression (ticket 04)
    forecast = lgbm_quantile_forecast(as_of_date, historical_data, mtu_minutes)

    # Local import to avoid circular dependency: lear.py imports _mtu_minutes_for, _validate
    # from this module at module load time, so a top-level import here would create a cycle
    # (pipeline -> lear -> pipeline before pipeline finishes defining those names).
    from forecast_pipeline.lear import lear_forecast  # noqa: E402

    baseline_forecast = lear_forecast(as_of_date, historical_data)

    return ForecastOutput(
        as_of_date=as_of_date,
        mtu_minutes=mtu_minutes,
        forecast=forecast,
        baseline_forecast=baseline_forecast,
    )
