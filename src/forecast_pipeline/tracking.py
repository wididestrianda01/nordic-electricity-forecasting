"""Ticket 06: MLflow logging contract for the Nordic forecasting backtest.

Local gitignored ``mlruns/`` tracking URI; nested runs -- parent = model x feature-set,
child = one run per fold. Logs the locked params/metrics/tags/artifacts contract so
every model x feature-set x fold is reproducible and comparable. The backtest runner
wires these helpers in Phase 2.

Usage (Phase 2)::

    set_local_tracking_uri()
    with start_parent_run("lightgbm", "gradient-boosted", "full", hyperparams, seed):
        for fold, (train, test) in enumerate(folds):
            with start_fold_run(fold, regime="post-oct-2025"):
                fit(...)
                log_fold_metrics(...)
                log_predictions(preds)
                log_feature_importance(importance)
    export_run_table()
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

# The filesystem tracking backend is deprecated in mlflow >= 3.15; this opt-out is
# required before the FileStore is first constructed (it is built lazily on the first
# tracking operation after set_tracking_uri).
_FILE_STORE_ALLOW_ENV = "MLFLOW_ALLOW_FILE_STORE"

DEFAULT_EXPERIMENT = "nordic-forecasting"

# Param keys (ticket 14).
PARAM_MODEL = "model"
PARAM_FAMILY = "family"
PARAM_FEATURE_SET = "feature_set"
PARAM_HYPERPARAMS = "hyperparams"
PARAM_SEED = "seed"

# Metric keys (ticket 14).
METRIC_CRPS = "CRPS"
METRIC_PINBALL_P10 = "pinball_P10"
METRIC_PINBALL_P50 = "pinball_P50"
METRIC_PINBALL_P90 = "pinball_P90"
METRIC_MAE = "MAE"
METRIC_TRAIN_WALL_CLOCK = "train_wall_clock_s"
METRIC_INFERENCE_WALL_CLOCK = "inference_wall_clock_s"

# Tag keys (ticket 14).
TAG_FOLD = "fold"
TAG_REGIME = "regime"

# MLflow's reserved parent-link tag, as surfaced by search_runs.
_PARENT_RUN_TAG = "tags.mlflow.parentRunId"


def set_local_tracking_uri(path: str | Path = "mlruns") -> str:
    """Point MLflow at a local tracking directory and return its absolute path."""
    os.environ.setdefault(_FILE_STORE_ALLOW_ENV, "true")
    uri = str(Path(path).resolve())
    mlflow.set_tracking_uri(uri)
    return uri


def start_parent_run(
    model: str,
    family: str,
    feature_set: str,
    hyperparams: Mapping[str, Any] | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> mlflow.ActiveRun:
    """Start the parent run for a model x feature-set config and log its params.

    Intended to be used as a context manager so the run ends on exit::

        with start_parent_run(...) as parent:
            ...
    """
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run(run_name=run_name or f"{model} x {feature_set}")
    mlflow.log_params(
        {
            PARAM_MODEL: model,
            PARAM_FAMILY: family,
            PARAM_FEATURE_SET: feature_set,
            PARAM_HYPERPARAMS: json.dumps(
                dict(hyperparams or {}), sort_keys=True, default=str
            ),
            PARAM_SEED: seed,
        }
    )
    return run


def start_fold_run(
    fold: int,
    regime: str | None = None,
    run_name: str | None = None,
) -> mlflow.ActiveRun:
    """Start a child run for one fold, nested under the active parent run.

    Tags the run with ``fold`` (always) and ``regime`` (when known). Must be called
    while a parent run from :func:`start_parent_run` is active::

        with start_parent_run(...):
            with start_fold_run(fold):
                ...
    """
    tags: dict[str, str] = {TAG_FOLD: str(fold)}
    if regime is not None:
        tags[TAG_REGIME] = regime
    return mlflow.start_run(run_name=run_name or f"fold-{fold}", nested=True, tags=tags)


def log_fold_metrics(
    *,
    crps: float,
    pinball_p10: float,
    pinball_p50: float,
    pinball_p90: float,
    mae: float,
    train_wall_clock: float,
    inference_wall_clock: float,
) -> None:
    """Log the locked fold-level metric set against the active (fold) run."""
    mlflow.log_metrics(
        {
            METRIC_CRPS: crps,
            METRIC_PINBALL_P10: pinball_p10,
            METRIC_PINBALL_P50: pinball_p50,
            METRIC_PINBALL_P90: pinball_p90,
            METRIC_MAE: mae,
            METRIC_TRAIN_WALL_CLOCK: train_wall_clock,
            METRIC_INFERENCE_WALL_CLOCK: inference_wall_clock,
        }
    )


def log_predictions(
    predictions: pd.DataFrame, artifact_path: str = "predictions.parquet"
) -> None:
    """Log a quantile forecast frame as a parquet artifact on the active run."""
    _log_dataframe_artifact(predictions, artifact_path)


def log_feature_importance(
    importance: Mapping[str, float] | pd.Series | pd.DataFrame | None,
    artifact_path: str = "feature_importance.parquet",
) -> None:
    """Log SHAP/feature-importance as a parquet artifact when available (no-op for None).

    Accepts a ``{feature: importance}`` mapping, a feature-indexed ``pd.Series``, or a
    pre-shaped ``pd.DataFrame`` (e.g. a SHAP value matrix).
    """
    if importance is None:
        return
    if isinstance(importance, Mapping):
        importance = (
            pd.Series(importance, name="importance").rename_axis("feature").reset_index()
        )
    elif isinstance(importance, pd.Series):
        importance = importance.rename_axis("feature").rename("importance").reset_index()
    elif not isinstance(importance, pd.DataFrame):
        raise TypeError(
            "importance must be a Mapping, pd.Series, pd.DataFrame, or None"
        )
    _log_dataframe_artifact(importance, artifact_path)


def _log_dataframe_artifact(df: pd.DataFrame, artifact_path: str) -> None:
    name = Path(artifact_path).name
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / name
        df.to_parquet(local)
        mlflow.log_artifact(str(local))


def export_run_table(
    out_dir: str | Path = "reports",
    experiment_name: str | None = None,
) -> tuple[Path, Path]:
    """Write the committed run table + Pareto table as CSVs outside ``mlruns/``.

    Returns the ``(run_table_path, pareto_table_path)`` written. The Pareto table rows
    are the model x feature-set configs (one per parent run) with mean CRPS/MAE and
    summed compute across folds; ``pareto_optimal`` flags the non-dominated front on
    (minimize CRPS, minimize total compute).
    """
    os.environ.setdefault(_FILE_STORE_ALLOW_ENV, "true")
    runs = _search_runs(experiment_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_table = _clean_run_table(runs)
    run_path = out / "run_table.csv"
    run_table.to_csv(run_path, index=False)

    pareto = _build_pareto_table(run_table)
    pareto_path = out / "pareto_table.csv"
    pareto.to_csv(pareto_path, index=False)
    return run_path, pareto_path


def _search_runs(experiment_name: str | None) -> pd.DataFrame:
    if experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()
        return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return mlflow.search_runs(search_all_experiments=True)


_PARAM_KEYS = (
    PARAM_MODEL,
    PARAM_FAMILY,
    PARAM_FEATURE_SET,
    PARAM_HYPERPARAMS,
    PARAM_SEED,
)
_METRIC_KEYS = (
    METRIC_CRPS,
    METRIC_PINBALL_P10,
    METRIC_PINBALL_P50,
    METRIC_PINBALL_P90,
    METRIC_MAE,
    METRIC_TRAIN_WALL_CLOCK,
    METRIC_INFERENCE_WALL_CLOCK,
)


def _clean_run_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Project the wide search_runs frame down to the locked contract columns."""
    src: dict[str, str] = {
        "run_id": "run_id",
        "parent_run_id": _PARENT_RUN_TAG,
        "status": "status",
    }
    src.update({key: f"params.{key}" for key in _PARAM_KEYS})
    src.update({key: f"metrics.{key}" for key in _METRIC_KEYS})
    src.update({TAG_FOLD: f"tags.{TAG_FOLD}", TAG_REGIME: f"tags.{TAG_REGIME}"})

    if runs.empty:
        return pd.DataFrame(columns=list(src))

    out = pd.DataFrame(index=runs.index)
    for name, column in src.items():
        out[name] = runs[column] if column in runs.columns else None
    return out


_PARETO_COLUMNS = [
    PARAM_MODEL,
    PARAM_FAMILY,
    PARAM_FEATURE_SET,
    TAG_REGIME,
    "n_folds",
    METRIC_CRPS,
    METRIC_MAE,
    "total_compute_s",
    "pareto_optimal",
]


def _build_pareto_table(run_table: pd.DataFrame) -> pd.DataFrame:
    if run_table.empty:
        return pd.DataFrame(columns=_PARETO_COLUMNS)

    children = run_table[run_table["parent_run_id"].notna()].copy()
    if children.empty:
        return pd.DataFrame(columns=_PARETO_COLUMNS)

    # Params (model/family/feature_set) live on the parent run; fold rows inherit them.
    parents = run_table[run_table["parent_run_id"].isna()].set_index("run_id")
    for column in (PARAM_MODEL, PARAM_FAMILY, PARAM_FEATURE_SET):
        children[column] = children["parent_run_id"].map(parents[column])
    rows = []
    for _, group in children.groupby("parent_run_id"):
        # skipna=False so a parent with any failed (NaN) fold gets a NaN mean
        # and is never flagged optimal on partial data.
        crps = pd.to_numeric(group[METRIC_CRPS], errors="coerce").mean(skipna=False)
        mae = pd.to_numeric(group[METRIC_MAE], errors="coerce").mean(skipna=False)
        total_compute = pd.to_numeric(
            group[METRIC_TRAIN_WALL_CLOCK], errors="coerce"
        ).sum() + pd.to_numeric(group[METRIC_INFERENCE_WALL_CLOCK], errors="coerce").sum()
        regime = (
            group[TAG_REGIME].iloc[0] if TAG_REGIME in group.columns else None
        )
        rows.append(
            {
                PARAM_MODEL: group[PARAM_MODEL].iloc[0],
                PARAM_FAMILY: group[PARAM_FAMILY].iloc[0],
                PARAM_FEATURE_SET: group[PARAM_FEATURE_SET].iloc[0],
                TAG_REGIME: regime,
                "n_folds": len(group),
                METRIC_CRPS: crps,
                METRIC_MAE: mae,
                "total_compute_s": total_compute,
            }
        )

    pareto = pd.DataFrame(rows)
    # The non-dominated front is computed within each regime, not pooled: a
    # model cheap and accurate in one regime must not dominate a model in the
    # other regime. NaN metrics never flag as optimal.
    pareto["pareto_optimal"] = False
    for _, group in pareto.groupby(TAG_REGIME, dropna=False):
        pareto.loc[group.index, "pareto_optimal"] = _pareto_optimal_mask(
            group[METRIC_CRPS], group["total_compute_s"]
        )
    return pareto[_PARETO_COLUMNS]


def _pareto_optimal_mask(crps: pd.Series, compute: pd.Series) -> list[bool]:
    """Non-dominated front on (minimize CRPS, minimize compute); NaN rows are not optimal."""
    crps = pd.to_numeric(crps, errors="coerce").to_numpy(dtype=float)
    compute = pd.to_numeric(compute, errors="coerce").to_numpy(dtype=float)
    mask: list[bool] = []
    for i in range(len(crps)):
        if not (np.isfinite(crps[i]) and np.isfinite(compute[i])):
            mask.append(False)
            continue
        dominated = any(
            np.isfinite(crps[j])
            and np.isfinite(compute[j])
            and crps[j] <= crps[i]
            and compute[j] <= compute[i]
            and (crps[j] < crps[i] or compute[j] < compute[i])
            for j in range(len(crps))
        )
        mask.append(not dominated)
    return mask
