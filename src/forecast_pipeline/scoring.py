"""Ticket 09: forecast scoring suite.

Pure, side-effect-free metrics over quantile forecasts and aligned actuals.
Every arm in the registry is scored on identical footing: each function takes
``predictions`` (a DataFrame with ``p10``/``p50``/``p90`` columns) and
``actuals`` (a Series sharing the same index), and returns a finite float (the
mean) or a per-row Series for partitioning.

CRPS is approximated from the three forecast quantiles via the quantile-score
identity ``CRPS = 2 * mean_q pinball(q)``. With three equally weighted
quantiles centred on the median, a degenerate (point) forecast --
``p10 == p50 == p90`` -- collapses to the mean absolute error exactly, because
the weighted mean quantile level is 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Quantile levels backing the three forecast columns.
QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.50, 0.90)

#: Forecast column for each quantile level.
_QUANTILE_COLUMN: dict[float, str] = {
    0.10: "p10",
    0.50: "p50",
    0.90: "p90",
}

#: Meteorological season per calendar month (Nordic convention).
_SEASON_BY_MONTH: dict[int, str] = {
    1: "winter",
    2: "winter",
    12: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}


def _column_for(q: float) -> str:
    try:
        return _QUANTILE_COLUMN[q]
    except KeyError:
        raise ValueError(
            f"quantile level {q} unsupported; expected one of {QUANTILE_LEVELS}"
        ) from None


def _check_aligned(predictions: pd.DataFrame, actuals: pd.Series) -> None:
    missing = [c for c in ("p10", "p50", "p90") if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions missing columns {missing}")
    if not isinstance(actuals, pd.Series):
        raise TypeError("actuals must be a pandas Series")
    if not predictions.index.equals(actuals.index):
        raise ValueError("predictions and actuals must share the same index")
    if predictions[["p10", "p50", "p90"]].isna().any().any():
        raise ValueError("predictions quantiles must not contain NaNs")
    if actuals.isna().any():
        raise ValueError("actuals must not contain NaNs")


def _pinball_array(
    predictions: pd.DataFrame, actuals: pd.Series, q: float
) -> np.ndarray:
    error = actuals.to_numpy() - predictions[_column_for(q)].to_numpy()
    return np.where(error >= 0.0, q * error, (q - 1.0) * error)


def pinball_scores(
    predictions: pd.DataFrame, actuals: pd.Series, q: float
) -> pd.Series:
    """Per-row pinball (quantile) loss at level `q`, aligned to the index."""
    _check_aligned(predictions, actuals)
    loss = _pinball_array(predictions, actuals, q)
    return pd.Series(loss, index=actuals.index, name=f"pinball_{q:g}")


def pinball_loss(predictions: pd.DataFrame, actuals: pd.Series, q: float) -> float:
    """Mean pinball loss at level `q` (one of 0.10/0.50/0.90)."""
    return float(pinball_scores(predictions, actuals, q).mean())


def crps_scores(predictions: pd.DataFrame, actuals: pd.Series) -> pd.Series:
    """Per-row CRPS from the three quantiles, aligned to the index."""
    _check_aligned(predictions, actuals)
    total = sum(_pinball_array(predictions, actuals, q) for q in QUANTILE_LEVELS)
    scores = 2.0 * total / len(QUANTILE_LEVELS)
    return pd.Series(scores, index=actuals.index, name="crps")


def crps(predictions: pd.DataFrame, actuals: pd.Series) -> float:
    """Quantile-weighted CRPS (primary metric) from P10/P50/P90."""
    return float(crps_scores(predictions, actuals).mean())


def absolute_errors(predictions: pd.DataFrame, actuals: pd.Series) -> pd.Series:
    """Per-row absolute error of the median forecast, aligned to the index."""
    _check_aligned(predictions, actuals)
    return (actuals - predictions["p50"]).abs().rename("absolute_error")


def mae(predictions: pd.DataFrame, actuals: pd.Series) -> float:
    """Mean absolute error of the median (p50) forecast."""
    return float(absolute_errors(predictions, actuals).mean())


def skill_score(metric_value: float, baseline_value: float) -> float:
    """Normalized skill score ``1 - metric / baseline`` (baseline non-zero)."""
    if baseline_value == 0.0:
        raise ValueError("baseline_value must be non-zero for skill_score")
    return float(1.0 - metric_value / baseline_value)


def _step_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        raise ValueError("index needs at least two timestamps to infer the step")
    return pd.Series(index).diff().dropna().median()


def seasonal_naive_baseline(
    actuals: pd.Series, horizon: int, season_length: int | None = None
) -> pd.Series:
    """Seasonal-naive point baseline over `horizon` steps.

    Each forecast step reuses the observation one full season earlier. The
    season length defaults to one day at the index frequency (24 hourly / 96
    15-minute steps); pass `season_length` to use weekly or custom cycles.
    """
    if not isinstance(actuals, pd.Series):
        raise TypeError("actuals must be a pandas Series")
    if not isinstance(actuals.index, pd.DatetimeIndex):
        raise TypeError("actuals must be indexed by timestamp")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    step = _step_delta(actuals.index)
    if season_length is None:
        season_length = round(86400.0 / step.total_seconds())
    if season_length < 1:
        raise ValueError("season_length must be at least 1")
    if len(actuals) < season_length:
        raise ValueError(
            f"actuals has {len(actuals)} rows; need at least season_length={season_length}"
        )

    values = actuals.to_numpy()
    n = len(values)
    offsets = n - season_length + (np.arange(horizon) % season_length)

    grid = pd.date_range(start=actuals.index[-1] + step, periods=horizon, freq=step)
    return pd.Series(values[offsets], index=grid, name="seasonal_naive")

def mean_score(scores: pd.Series) -> float:
    """Overall mean of a per-row score series."""
    return float(scores.mean())


def score_by_regime(
    scores: pd.Series, regime: pd.Series | np.ndarray | list[str]
) -> pd.Series:
    """Mean `scores` grouped by regime label (aligned positionally with scores)."""
    labels = np.asarray(regime)
    if labels.shape[0] != len(scores):
        raise ValueError("regime must have the same length as scores")
    return scores.groupby(labels).mean().rename_axis("regime")


def score_by_season(scores: pd.Series) -> pd.Series:
    """Mean `scores` grouped by meteorological season derived from the index."""
    labels = [_SEASON_BY_MONTH[ts.month] for ts in scores.index]
    return scores.groupby(labels).mean().rename_axis("season")
