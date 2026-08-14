"""Ticket 15: expanding-window walk-forward backtest harness.

Wires the locked seams (ticket 09/10 contract) into the locked backtest
protocol:

* ``generate_folds`` produces yearly fold cutoffs from ``test_start`` to the
  last full day of history, dropping any cutoff whose D+1 horizon straddles a
  regime boundary or sits within the purge window around one.
* ``run_backtest`` walks every (cutoff, spec) pair, fits/predicts/score one
  expanding-window fold, and -- when ``log=True`` -- nests each fold run under
  a per-model parent run in MLflow.

The 15-minute MTU (from 2025-10-01) and the hourly regime are distinct: each
cutoff's regime is attributed by ``_mtu_minutes_for`` (60 or 15), which also
sets the forecast horizon (24 or 96 steps).
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from datetime import date, timedelta
from typing import Any

import pandas as pd

from forecast_pipeline.features import build_features, build_horizon_features
from forecast_pipeline.pipeline import _mtu_minutes_for
from forecast_pipeline.regime_boundaries import REGIME_BOUNDARIES
from forecast_pipeline.registry import ModelSpec, build_model
from forecast_pipeline.scoring import (
    crps,
    mae,
    pinball_loss,
    seasonal_naive_baseline,
    skill_score,
)
from forecast_pipeline.tracking import (
    log_fold_metrics,
    log_predictions,
    set_local_tracking_uri,
    start_fold_run,
    start_parent_run,
)

logger = logging.getLogger(__name__)

#: The only supported fold cadence (locked protocol: yearly refit).
_YEARLY = "yearly"

#: Columns of the returned walk-forward result frame (locked contract).
RESULT_COLUMNS = [
    "model",
    "family",
    "feature_set",
    "cutoff",
    "mtu_minutes",
    "crps",
    "pinball_p10",
    "pinball_p50",
    "pinball_p90",
    "mae",
    "skill_score_crps",
    "train_wall_clock",
    "inference_wall_clock",
]

#: Quantile levels backing the three forecast columns (mirrors scoring.py).
_QUANTILE_LEVELS = (0.10, 0.50, 0.90)

#: The locked 10-model roster (ticket 15); every arm pins seed 42.
DEFAULT_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("sarima", "classical", "price-only", {}, 42),
    ModelSpec("ets", "classical", "price-only", {}, 42),
    ModelSpec("lear", "ml", "price-only", {}, 42),
    ModelSpec("lgbm", "gbdt", "full-features", {}, 42),
    ModelSpec("xgboost", "gbdt", "full-features", {}, 42),
    ModelSpec("catboost", "gbdt", "full-features", {}, 42),
    ModelSpec("nbeats", "deep", "price-only", {}, 42),
    ModelSpec("tft", "deep", "full-features", {}, 42),
    ModelSpec("chronos2", "foundation", "full-features", {}, 42),
    ModelSpec("timesfm", "foundation", "price-only", {}, 42),
)


def _add_years(base: date, years: int) -> date:
    """Add ``years`` calendar years, clamping Feb 29 to Feb 28."""
    try:
        return date(base.year + years, base.month, base.day)
    except ValueError:
        return date(base.year + years, 2, 28)


def _infer_mtu_minutes(historical_data: pd.DataFrame) -> int:
    """Infer the frame's MTU (60 or 15) from its actual index step.

    The harness runs each regime (hourly vs 15-minute) on its own
    single-frequency frame, so the MTU is a property of the data, not of the
    cutoff date. Mixed or irregular frequencies are rejected explicitly
    rather than silently mislabelled.
    """
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        raise TypeError("historical_data must be indexed by timestamp")
    if len(historical_data.index) < 2:
        raise ValueError("historical_data needs at least two rows to infer MTU")
    steps = pd.Series(historical_data.index).diff().dropna()
    minutes = {int(s.total_seconds() // 60) for s in steps.unique()}
    if minutes != {60} and minutes != {15}:
        raise ValueError(
            "historical_data must be single-frequency 60- or 15-minute; "
            f"got {sorted(minutes)}-minute steps"
        )
    return minutes.pop()


def _last_full_day(historical_data: pd.DataFrame) -> date:
    """Return the latest day with a complete MTU day of rows."""
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        raise TypeError("historical_data must be indexed by timestamp")
    if historical_data.empty:
        raise ValueError("historical_data is empty")
    mtu_minutes = _infer_mtu_minutes(historical_data)
    expected = 24 * 60 // mtu_minutes
    counts = pd.Series(historical_data.index).dt.normalize().value_counts()
    for day_ts in sorted(counts.index, reverse=True):
        day = day_ts.date()
        if int(counts[day_ts]) == expected:
            return day
    raise ValueError("historical_data contains no full day")


def _cutoff_is_valid(cutoff: date, purge_days: int) -> bool:
    """True if ``cutoff``'s D+1 horizon is clear of every regime boundary."""
    horizon_end = cutoff + timedelta(days=1)
    for boundary in REGIME_BOUNDARIES:
        # (a) the horizon [cutoff, cutoff + 1 day) must contain no boundary.
        if cutoff <= boundary < horizon_end:
            return False
        # (b) cutoff must be >= purge_days away from the boundary on both sides.
        if abs((cutoff - boundary).days) < purge_days:
            return False
    return True


def generate_folds(
    historical_data: pd.DataFrame,
    *,
    test_start: date,
    refit_cadence: str = _YEARLY,
    purge_days: int = 7,
) -> list[date]:
    """Return the valid yearly fold cutoffs from ``test_start`` to the last full day.

    A cutoff is the D+1 forecast start date (the day after the last training
    day). Invalid cutoffs -- those whose horizon straddles a regime boundary or
    that fall within ``purge_days`` of one -- are dropped. Only cutoffs whose
    calendar MTU matches the frame's actual (single-frequency) MTU are kept, so
    the hourly and 15-minute regimes are generated from their own frames.
    """
    if refit_cadence != _YEARLY:
        raise ValueError(
            f"unsupported refit_cadence {refit_cadence!r}; only {_YEARLY!r} is supported"
        )
    mtu_minutes = _infer_mtu_minutes(historical_data)
    last_full_day = _last_full_day(historical_data)
    cutoffs: list[date] = []
    year_offset = 0
    while True:
        cutoff = _add_years(test_start, year_offset)
        if cutoff > last_full_day:
            break
        if _mtu_minutes_for(cutoff) == mtu_minutes and _cutoff_is_valid(cutoff, purge_days):
            cutoffs.append(cutoff)
        year_offset += 1
    return cutoffs


def _cutoff_timestamp(cutoff: date, tz: Any) -> pd.Timestamp:
    """Localize a cutoff date to midnight in the history's timezone."""
    ts = pd.Timestamp(cutoff)
    return ts.tz_localize(tz) if tz is not None else ts


def _actuals_for(
    historical_data: pd.DataFrame, cutoff_ts: pd.Timestamp
) -> pd.Series:
    """Realized price rows spanning the single forecast day at ``cutoff``."""
    horizon_end = cutoff_ts + pd.Timedelta(days=1)
    mask = (historical_data.index >= cutoff_ts) & (historical_data.index < horizon_end)
    return historical_data.loc[mask, "price"]


def _regime_tag(mtu_minutes: int) -> str:
    """Human-readable regime tag for the MLflow fold run."""
    return f"{mtu_minutes}min"


def _run_fold(
    historical_data: pd.DataFrame,
    spec: ModelSpec,
    cutoff: date,
    cutoff_index: int,
    tz: Any,
    mtu_minutes: int,
    log: bool,
    feature_columns: list[str] | None,
) -> dict[str, Any]:
    """Fit, predict, and score one (spec, cutoff) fold; return its result row."""
    cutoff_ts = _cutoff_timestamp(cutoff, tz)
    horizon = 24 * 60 // mtu_minutes

    train = historical_data.loc[historical_data.index < cutoff_ts]
    actuals = _actuals_for(historical_data, cutoff_ts)
    target = train["price"]
    if spec.feature_set == "full-features":
        _, features = build_features(cutoff, train)
        future_features = build_horizon_features(cutoff, train)
        if feature_columns is not None:
            features = features[feature_columns]
            future_features = future_features[feature_columns]
    else:
        features = None
        future_features = None

    arm = build_model(spec)

    start = time.perf_counter()
    arm.fit(target, features)
    train_wall_clock = time.perf_counter() - start

    start = time.perf_counter()
    predictions = arm.predict_quantiles(horizon, future_features)
    inference_wall_clock = time.perf_counter() - start

    crps_value = crps(predictions, actuals)
    pinball = {
        q: pinball_loss(predictions, actuals, q) for q in _QUANTILE_LEVELS
    }
    mae_value = mae(predictions, actuals)

    baseline = seasonal_naive_baseline(target, horizon)
    baseline_pred = pd.DataFrame(
        {"p10": baseline, "p50": baseline, "p90": baseline}, index=baseline.index
    )
    skill = skill_score(crps_value, crps(baseline_pred, actuals))

    fold_cm = (
        start_fold_run(fold=cutoff_index, regime=_regime_tag(mtu_minutes))
        if log
        else nullcontext()
    )
    with fold_cm:
        if log:
            log_fold_metrics(
                crps=crps_value,
                pinball_p10=pinball[0.10],
                pinball_p50=pinball[0.50],
                pinball_p90=pinball[0.90],
                mae=mae_value,
                train_wall_clock=train_wall_clock,
                inference_wall_clock=inference_wall_clock,
            )
            log_predictions(predictions)

    return {
        "model": spec.name,
        "family": spec.family,
        "feature_set": spec.feature_set,
        "cutoff": cutoff,
        "mtu_minutes": mtu_minutes,
        "crps": crps_value,
        "pinball_p10": pinball[0.10],
        "pinball_p50": pinball[0.50],
        "pinball_p90": pinball[0.90],
        "mae": mae_value,
        "skill_score_crps": skill,
        "train_wall_clock": train_wall_clock,
        "inference_wall_clock": inference_wall_clock,
    }

def run_backtest(
    historical_data: pd.DataFrame,
    specs: list[ModelSpec] | tuple[ModelSpec, ...],
    cutoffs: list[date] | tuple[date, ...],
    *,
    tracking_uri: str = "mlruns",
    log: bool = True,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Run the walk-forward backtest; return one row per (model, cutoff).

    When ``log=True`` each model config gets a parent run (via
    ``start_parent_run``) and every fold nests a child run (via
    ``start_fold_run``) beneath it, with metrics and quantile artifacts logged.

    ``feature_columns`` restricts full-features specs to the named columns
    (both the train matrix and the horizon matrix); ``None`` uses every
    canonical column. Price-only specs are unaffected. Used by the feature
    ablation and transfer check to run a model on a feature subset.
    """
    if log:
        set_local_tracking_uri(tracking_uri)
    index = historical_data.index
    tz = index.tz if isinstance(index, pd.DatetimeIndex) else None
    mtu_minutes = _infer_mtu_minutes(historical_data)
    for cutoff in cutoffs:
        if _mtu_minutes_for(cutoff) != mtu_minutes:
            raise ValueError(
                f"cutoff {cutoff} belongs to the {_mtu_minutes_for(cutoff)}-minute "
                f"regime but historical_data is {mtu_minutes}-minute; run each "
                "regime on its own single-frequency frame"
            )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        parent_cm = (
            start_parent_run(
                spec.name,
                spec.family,
                spec.feature_set,
                spec.hyperparams,
                spec.seed,
            )
            if log
            else nullcontext()
        )
        try:
            with parent_cm:
                for cutoff_index, cutoff in enumerate(cutoffs):
                    rows.append(
                        _run_fold(
                            historical_data,
                            spec,
                            cutoff,
                            cutoff_index,
                            tz,
                            mtu_minutes,
                            log,
                            feature_columns,
                        )
                    )
        except Exception:
            # A single arm failing must not abort the whole comparison: log it,
            # emit NaN rows so the model is still represented in the table, and
            # continue with the remaining specs.
            logger.warning("model %r failed; skipping its folds", spec.name, exc_info=True)
            for cutoff_index, cutoff in enumerate(cutoffs):
                rows.append(
                    {
                        "model": spec.name,
                        "family": spec.family,
                        "feature_set": spec.feature_set,
                        "cutoff": cutoff,
                        "mtu_minutes": mtu_minutes,
                        "crps": float("nan"),
                        "pinball_p10": float("nan"),
                        "pinball_p50": float("nan"),
                        "pinball_p90": float("nan"),
                        "mae": float("nan"),
                        "skill_score_crps": float("nan"),
                        "train_wall_clock": float("nan"),
                        "inference_wall_clock": float("nan"),
                    }
                )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
