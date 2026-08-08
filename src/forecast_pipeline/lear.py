"""Ticket 03: LEAR baseline (ADR-0005).

LASSO-Estimated AutoRegressive forecaster (Lago et al.) -- a per-MTU-slot
Lasso regression on same-time-of-day price lags. This is the comparison
baseline for the regime-conditional LightGBM model (ticket 04), not the
project's headline forecaster; it returns a point forecast, not a quantile
grid.
"""

from datetime import date

import pandas as pd
from sklearn.linear_model import Lasso

from forecast_pipeline.pipeline import _mtu_minutes_for, _validate

LAG_DAYS = (1, 2, 3, 7)

# Need enough same-slot history to fit a non-degenerate Lasso per slot.
MIN_TRAINING_DAYS = max(LAG_DAYS) + 7


def _slot_of_day(index: pd.DatetimeIndex, mtu_minutes: int) -> pd.Index:
    minutes_since_midnight = index.hour * 60 + index.minute
    return minutes_since_midnight // mtu_minutes


def lear_forecast(
    as_of_date: date,
    historical_data: pd.DataFrame,
    lag_days: tuple[int, ...] = LAG_DAYS,
) -> pd.Series:
    mtu_minutes = _mtu_minutes_for(as_of_date)
    _validate(historical_data, mtu_minutes)

    days_available = len(historical_data) / (24 * 60 // mtu_minutes)
    min_days = max(lag_days) + 7
    if days_available < min_days:
        raise ValueError(
            f"historical_data has {days_available:.0f} days; "
            f"need at least {min_days} for a LEAR fit with lags {lag_days}"
        )

    day = historical_data.index.normalize()
    slot = _slot_of_day(historical_data.index, mtu_minutes)
    pivot = (
        historical_data.assign(day=day, slot=slot)
        .pivot(index="day", columns="slot", values="price")
        .sort_index()
    )

    periods_per_day = 24 * 60 // mtu_minutes
    forecast_index = pd.date_range(
        pd.Timestamp(as_of_date, tz="UTC"),
        periods=periods_per_day,
        freq=f"{mtu_minutes}min",
    )

    predictions = []
    for slot_id in range(periods_per_day):
        try:
            series = pivot[slot_id]
        except KeyError:
            raise ValueError(
                f"No training data for slot {slot_id} (missing from historical data). "
                f"Historical data must have coverage for all {periods_per_day} MTU slots."
            ) from None

        features = pd.concat({lag: series.shift(lag) for lag in lag_days}, axis=1)
        target = series
        train = features.assign(target=target).dropna()

        if len(train) == 0:
            raise ValueError(
                f"No training data for slot {slot_id} after shift and dropna. "
                f"Check historical data has no NaNs and sufficient coverage for all slots."
            )

        # ponytail: alpha=1.0 is LEAR paper baseline regularization (Lago et al., ADR-0005)
        model = Lasso(alpha=1.0)
        model.fit(train[list(lag_days)], train["target"])

        query_day = pd.Timestamp(as_of_date, tz="UTC")
        query_dict: dict[int, float] = {}
        for lag in lag_days:
            lag_date = query_day - pd.Timedelta(days=lag)
            try:
                query_dict[lag] = pivot[slot_id].loc[lag_date]
            except KeyError:
                raise ValueError(
                    f"Missing lag date {lag_date.date()} for slot {slot_id} in pivot. "
                    f"Historical data must span at least {max(lag_days)} days before forecast date."
                ) from None
        query = pd.Series(query_dict)
        predictions.append(model.predict(query.to_frame().T[list(lag_days)])[0])

    return pd.Series(predictions, index=forecast_index, name="lear_forecast")
