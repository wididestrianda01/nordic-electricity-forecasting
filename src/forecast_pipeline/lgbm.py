"""Ticket 04: Regime-conditional LightGBM quantile regression model.

Replaces placeholder quantile logic with real LightGBM models trained per quantile,
conditioned on the HMM regime label from ticket 02 (ADR-0005/0006).

Three separate LightGBM regressors are trained (one per quantile: 0.1, 0.5, 0.9)
using the quantile objective. Features include:
- Lag features (price from 1/2/3/7 days ago, same MTU slot)
- Slot-of-day (which MTU slot: 0-23 for hourly, 0-95 for 15-min)
- Regime label (categorical: regime_0, regime_1, etc.)

Forecast regime uses the most recent regime label from historical data (persistence).
Quantile ordering (P10 <= P50 <= P90) is preserved post-hoc if needed.
"""

from datetime import date
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from forecast_pipeline.regime import detect_regimes
from forecast_pipeline.regime_boundaries import crosses_boundary

LAG_DAYS = (1, 2, 3, 7)


def _slot_of_day(index: pd.DatetimeIndex, mtu_minutes: int) -> np.ndarray:
    """Return the MTU slot (0-23 for hourly, 0-95 for 15-min) for each timestamp."""
    slots_per_day = 24 * 60 // mtu_minutes
    return (index.hour * (60 // mtu_minutes) + index.minute // mtu_minutes) % slots_per_day


def _build_training_features(
    historical_data: pd.DataFrame,
    regime_labels: pd.Series,
    mtu_minutes: int,
) -> pd.DataFrame:
    """Build lag and regime features for training."""
    df = historical_data[["price"]].copy()
    df["regime"] = regime_labels
    df["slot"] = _slot_of_day(df.index, mtu_minutes)

    # Create lag features
    for lag_days in LAG_DAYS:
        lag_col = f"lag_{lag_days}d"
        df[lag_col] = df["price"].shift(lag_days * 24 * 60 // mtu_minutes)

        # Mask lags that cross regime boundaries (ticket 05, ADR-0007)
        # to prevent training on inconsistent price regimes.
        for idx in df.index:
            lag_date = (idx - pd.Timedelta(days=lag_days)).date()
            if crosses_boundary(idx.date(), lag_date):
                df.loc[idx, lag_col] = np.nan

    # Drop rows with NaN lags (first 7 days + boundary-masked rows)
    df = df.dropna(subset=[f"lag_{lag_days}d" for lag_days in LAG_DAYS])

    if len(df) == 0:
        raise ValueError("After removing lag NaNs, no training data remains")

    return df


def _build_forecast_features(
    as_of_date: date,
    historical_data: pd.DataFrame,
    mtu_minutes: int,
    forecast_regime: str,
) -> pd.DataFrame:
    """Build features for forecast period (next 24 hours or 96 slots)."""
    periods_per_day = 24 * 60 // mtu_minutes
    forecast_index = pd.date_range(
        pd.Timestamp(as_of_date, tz="UTC"),
        periods=periods_per_day,
        freq=f"{mtu_minutes}min",
    )

    # For each forecast row, look back to find lag values
    forecast_rows = []
    mean_val = historical_data["price"].mean()
    for forecast_ts in forecast_index:
        slot = _slot_of_day(pd.DatetimeIndex([forecast_ts]), mtu_minutes)[0]
        row = {"slot": slot, "regime": forecast_regime}

        all_lags_masked = True
        # Collect lags from historical data
        for lag_days in LAG_DAYS:
            lag_ts = forecast_ts - pd.Timedelta(days=lag_days)
            lag_date = lag_ts.date()

            # Mask lags that cross regime boundaries (ticket 05, ADR-0007)
            # instead of looking them up in historical data.
            if crosses_boundary(as_of_date, lag_date):
                row[f"lag_{lag_days}d"] = np.nan
            else:
                all_lags_masked = False
                lag_val = historical_data[historical_data.index == lag_ts]["price"].values
                if len(lag_val) > 0:
                    row[f"lag_{lag_days}d"] = lag_val[0]
                else:
                    row[f"lag_{lag_days}d"] = np.nan

        # Warn if all lags are boundary-masked for this forecast row
        if all_lags_masked:
            warnings.warn(
                f"forecast for {as_of_date} has all lags boundary-masked; "
                "degraded to mean fallback with no discriminative signal",
                UserWarning,
                stacklevel=3,
            )

        forecast_rows.append(row)

    df = pd.DataFrame(forecast_rows, index=forecast_index)

    # Fill any missing lags with the overall mean from training
    for lag_col in [f"lag_{lag_days}d" for lag_days in LAG_DAYS]:
        df[lag_col] = df[lag_col].fillna(mean_val)

    return df


def lgbm_quantile_forecast(
    as_of_date: date,
    historical_data: pd.DataFrame,
    mtu_minutes: int,
) -> pd.DataFrame:
    """Train quantile regression models and produce forecast.

    Args:
        as_of_date: Date for which to produce the forecast.
        historical_data: DataFrame with 'price' column and datetime index.
        mtu_minutes: MTU granularity (60 for hourly, 15 for 15-minute).

    Returns:
        DataFrame with columns p10, p50, p90, regime, indexed by forecast MTU times.
    """
    # Detect regimes
    regime_labels = detect_regimes(historical_data, n_states=2)

    # Build training features
    train_df = _build_training_features(historical_data, regime_labels, mtu_minutes)
    feature_cols = [f"lag_{d}d" for d in LAG_DAYS] + ["slot", "regime"]
    X_train = train_df[feature_cols]
    y_train = train_df["price"]

    # Convert regime to categorical for LightGBM
    X_train_cat = X_train.copy()
    X_train_cat["regime"] = X_train_cat["regime"].astype("category")

    # Train one model per quantile
    models = {}
    for alpha in [0.1, 0.5, 0.9]:
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            num_leaves=8,
            min_child_samples=5,
            n_estimators=50,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train_cat, y_train, categorical_feature=["regime"])
        models[alpha] = model

    # Forecast regime: use most recent
    forecast_regime = regime_labels.iloc[-1]

    # Build forecast features
    forecast_df = _build_forecast_features(as_of_date, historical_data, mtu_minutes, forecast_regime)
    X_forecast = forecast_df[feature_cols].copy()
    X_forecast["regime"] = X_forecast["regime"].astype("category")

    # Generate predictions
    p10 = models[0.1].predict(X_forecast)
    p50 = models[0.5].predict(X_forecast)
    p90 = models[0.9].predict(X_forecast)

    # Ensure ordering: P10 <= P50 <= P90 (post-hoc correction if needed)
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)

    # Build result
    result = pd.DataFrame(
        {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "regime": forecast_regime,
        },
        index=forecast_df.index,
    )

    return result
