"""Ticket 03: LEAR baseline (ADR-0005).

LASSO-Estimated AutoRegressive forecaster (Lago et al.) -- a per-MTU-slot
Lasso regression on same-time-of-day price lags. This is the comparison
baseline for the regime-conditional LightGBM model (ticket 04), not the
project's headline forecaster; it returns a point forecast, not a quantile
grid.
"""

from datetime import date
import warnings

import pandas as pd
from sklearn.linear_model import Lasso

from forecast_pipeline.pipeline import _mtu_minutes_for, _validate
from forecast_pipeline.regime_boundaries import crosses_boundary

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

    # Build pivot for price and optional load/wind features (ticket 02)
    price_pivot = (
        historical_data.assign(day=day, slot=slot)
        .pivot(index="day", columns="slot", values="price")
        .sort_index()
    )
    load_pivot = None
    wind_pivot = None
    if "load_forecast" in historical_data.columns:
        load_pivot = (
            historical_data[["load_forecast"]].assign(day=day, slot=slot)
            .pivot(index="day", columns="slot", values="load_forecast")
            .sort_index()
        )
    if "wind_forecast" in historical_data.columns:
        wind_pivot = (
            historical_data[["wind_forecast"]].assign(day=day, slot=slot)
            .pivot(index="day", columns="slot", values="wind_forecast")
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
            price_series = price_pivot[slot_id]
        except KeyError:
            raise ValueError(
                f"No training data for slot {slot_id} (missing from historical data). "
                f"Historical data must have coverage for all {periods_per_day} MTU slots."
            ) from None

        # Build lag features for price (using string column names for consistency with load/wind)
        lag_col_dict = {f"lag_{lag}d": price_series.shift(lag) for lag in lag_days}
        features = pd.concat(lag_col_dict, axis=1)

        # Add load/wind columns if present (ticket 02)
        if load_pivot is not None:
            features["load_forecast"] = load_pivot[slot_id]
        if wind_pivot is not None:
            features["wind_forecast"] = wind_pivot[slot_id]

        # Mask lags and load/wind that cross regime boundaries (ticket 05, ADR-0007; ticket 02)
        for day in features.index:
            for lag in lag_days:
                lag_col = f"lag_{lag}d"
                lag_day = day - pd.Timedelta(days=lag)
                if crosses_boundary(day.date(), lag_day.date()):
                    features.loc[day, lag_col] = pd.NA
                    # Also mask load/wind for this row (ticket 02)
                    if load_pivot is not None:
                        features.loc[day, "load_forecast"] = pd.NA
                    if wind_pivot is not None:
                        features.loc[day, "wind_forecast"] = pd.NA

        target = price_series
        train = features.assign(target=target).dropna()

        if len(train) == 0:
            raise ValueError(
                f"No training data for slot {slot_id} after shift and dropna. "
                f"Check historical data has no NaNs and sufficient coverage for all slots."
            )

        # Collect all feature columns for training
        feature_cols = [f"lag_{lag}d" for lag in lag_days]
        if load_pivot is not None:
            feature_cols.append("load_forecast")
        if wind_pivot is not None:
            feature_cols.append("wind_forecast")

        # ponytail: alpha=1.0 is LEAR paper baseline regularization (Lago et al., ADR-0005)
        model = Lasso(alpha=1.0)
        model.fit(train[feature_cols], train["target"])

        query_day = pd.Timestamp(as_of_date, tz="UTC")
        query_dict: dict = {}
        all_features_masked = True

        # Determine if any lags cross boundary
        any_lag_crosses = False
        for lag in lag_days:
            lag_date = query_day - pd.Timedelta(days=lag)
            lag_day = lag_date.date()
            if crosses_boundary(as_of_date, lag_day):
                any_lag_crosses = True
                break

        # Build query values for lags
        for lag in lag_days:
            lag_col = f"lag_{lag}d"
            lag_date = query_day - pd.Timedelta(days=lag)
            lag_day = lag_date.date()

            if crosses_boundary(as_of_date, lag_day):
                query_dict[lag_col] = price_series.mean()
            else:
                all_features_masked = False
                try:
                    query_dict[lag_col] = price_pivot[slot_id].loc[lag_date]
                except KeyError:
                    raise ValueError(
                        f"Missing lag date {lag_date.date()} for slot {slot_id} in pivot. "
                        f"Historical data must span at least {max(lag_days)} days before forecast date."
                    ) from None

        # Build query values for load/wind (masked if lags are masked, ticket 02)
        # Note: load/wind for the forecast date may not be in historical_data
        # In practice, these would come from operational forecasts; here we use mean as fallback
        query_date = pd.Timestamp(as_of_date, tz="UTC").normalize()
        if load_pivot is not None:
            if any_lag_crosses:
                query_dict["load_forecast"] = load_pivot[slot_id].mean()
            else:
                try:
                    query_dict["load_forecast"] = load_pivot[slot_id].loc[query_date]
                except KeyError:
                    query_dict["load_forecast"] = load_pivot[slot_id].mean()
        if wind_pivot is not None:
            if any_lag_crosses:
                query_dict["wind_forecast"] = wind_pivot[slot_id].mean()
            else:
                try:
                    query_dict["wind_forecast"] = wind_pivot[slot_id].loc[query_date]
                except KeyError:
                    query_dict["wind_forecast"] = wind_pivot[slot_id].mean()

        # Warn if all features are boundary-masked
        if all_features_masked:
            warnings.warn(
                f"forecast for {as_of_date} has all lags boundary-masked; "
                "degraded to mean fallback with no discriminative signal",
                UserWarning,
                stacklevel=3,
            )

        query = pd.Series(query_dict)
        predictions.append(model.predict(query.to_frame().T[feature_cols])[0])

    return pd.Series(predictions, index=forecast_index, name="lear_forecast")
