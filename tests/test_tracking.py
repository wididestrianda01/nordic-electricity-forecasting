"""Tests for the MLflow tracking contract and Pareto/run-table export (ticket 17)."""

import pandas as pd
import pytest

from forecast_pipeline.tracking import (
    DEFAULT_EXPERIMENT,
    _build_pareto_table,
    _clean_run_table,
    _pareto_optimal_mask,
    export_run_table,
    log_fold_metrics,
    set_local_tracking_uri,
    start_fold_run,
    start_parent_run,
)


def _log_two_models(tracking_uri: str) -> None:
    """Log two parent runs (different models) with two fold runs each."""
    set_local_tracking_uri(tracking_uri)
    for model, family in (("lear", "ml"), ("lgbm", "gbdt")):
        with start_parent_run(model, family, "price-only" if model == "lear" else "full-features"):
            for fold in range(2):
                with start_fold_run(fold=fold, regime="60min"):
                    log_fold_metrics(
                        crps=1.0 + fold,
                        pinball_p10=0.5,
                        pinball_p50=0.5,
                        pinball_p90=0.5,
                        mae=2.0,
                        train_wall_clock=1.0,
                        inference_wall_clock=0.5,
                    )


def test_pareto_optimal_mask_known_front():
    # (crps, compute): point 0 dominates 1 and 2; a NaN row is never optimal.
    crps = pd.Series([1.0, 2.0, 3.0, float("nan")])
    compute = pd.Series([1.0, 1.5, 2.0, 0.5])
    mask = _pareto_optimal_mask(crps, compute)
    assert mask == [True, False, False, False]


def test_pareto_optimal_mask_incomparable_points_both_optimal():
    crps = pd.Series([1.0, 2.0])
    compute = pd.Series([10.0, 1.0])
    assert _pareto_optimal_mask(crps, compute) == [True, True]


def test_clean_run_table_projects_contract_columns():
    runs = pd.DataFrame(
        {
            "run_id": ["r1"],
            "status": ["FINISHED"],
            "params.model": ["lear"],
            "params.family": ["ml"],
            "params.feature_set": ["price-only"],
            "params.hyperparams": ["{}"],
            "params.seed": ["42"],
            "metrics.CRPS": [1.0],
            "metrics.pinball_P10": [0.5],
            "metrics.pinball_P50": [0.5],
            "metrics.pinball_P90": [0.5],
            "metrics.MAE": [2.0],
            "metrics.train_wall_clock_s": [1.0],
            "metrics.inference_wall_clock_s": [0.5],
            "tags.fold": ["0"],
            "tags.regime": ["60min"],
            "tags.mlflow.parentRunId": ["parent"],
            "extra.ignored": ["x"],
        }
    )
    out = _clean_run_table(runs)
    assert set(out.columns) == {
        "run_id",
        "parent_run_id",
        "status",
        "model",
        "family",
        "feature_set",
        "hyperparams",
        "seed",
        "CRPS",
        "pinball_P10",
        "pinball_P50",
        "pinball_P90",
        "MAE",
        "train_wall_clock_s",
        "inference_wall_clock_s",
        "fold",
        "regime",
    }
    assert out.iloc[0]["model"] == "lear"


def test_build_pareto_table_aggregates_parent_children():
    run_table = pd.DataFrame(
        [
            # parent (no parent_run_id) and its two fold children
            {"run_id": "p1", "parent_run_id": None, "model": "lear", "family": "ml",
             "feature_set": "price-only", "CRPS": None, "MAE": None,
             "train_wall_clock_s": None, "inference_wall_clock_s": None},
            {"run_id": "c1", "parent_run_id": "p1", "model": None, "family": None,
             "feature_set": None, "CRPS": 1.0, "MAE": 2.0,
             "train_wall_clock_s": 1.0, "inference_wall_clock_s": 0.5},
            {"run_id": "c2", "parent_run_id": "p1", "model": None, "family": None,
             "feature_set": None, "CRPS": 3.0, "MAE": 4.0,
             "train_wall_clock_s": 1.0, "inference_wall_clock_s": 0.5},
        ]
    )
    pareto = _build_pareto_table(run_table)
    assert len(pareto) == 1
    row = pareto.iloc[0]
    assert row["model"] == "lear"
    assert row["CRPS"] == pytest.approx(2.0)  # mean of 1.0, 3.0
    assert row["MAE"] == pytest.approx(3.0)
    assert row["total_compute_s"] == pytest.approx(3.0)  # (1.0+1.0)+(0.5+0.5)
    assert bool(row["pareto_optimal"])


def test_export_run_table_integration(tmp_path):
    tracking_uri = str(tmp_path / "mlruns")
    _log_two_models(tracking_uri)
    set_local_tracking_uri(tracking_uri)

    run_path, pareto_path = export_run_table(
        tmp_path / "reports", experiment_name=DEFAULT_EXPERIMENT
    )

    assert run_path.exists()
    assert pareto_path.exists()

    run_table = pd.read_csv(run_path)
    pareto = pd.read_csv(pareto_path)

    assert set(run_table["model"].dropna()) == {"lear", "lgbm"}
    # one Pareto row per model
    assert set(pareto["model"]) == {"lear", "lgbm"}
    assert {"model", "family", "feature_set", "CRPS", "MAE", "total_compute_s",
            "pareto_optimal", "n_folds"} <= set(pareto.columns)
