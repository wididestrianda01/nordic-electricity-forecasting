"""Tests for the LightGBM feature ablation (ticket 18)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.ablation import (
    _efficient_set,
    _fit_full_lgbm,
    _forward_path,
    _leave_one_out,
    evaluate_crps,
    permutation_importance,
    run_ablation,
    shap_values,
)
from forecast_pipeline.backtest import generate_folds
from forecast_pipeline.features import build_features


def _make_scorer(mapping: dict):
    return lambda cols: mapping[frozenset(cols)]


# --- pure decision logic -----------------------------------------------------


def test_forward_path_greedy_adds_best_group_then_stops():
    scorer = _make_scorer(
        {
            frozenset(["b"]): 1.0,
            frozenset(["b", "c3"]): 0.8,
            frozenset(["b", "c4"]): 0.9,
            frozenset(["b", "c3", "c4"]): 0.85,
        }
    )
    path = _forward_path(["b"], {"g3": ["c3"], "g4": ["c4"]}, scorer)
    assert [p["added"] for p in path[1:]] == ["g3"]
    assert path[0]["columns"] == ["b"]
    assert path[1]["columns"] == ["b", "c3"]
    assert path[1]["crps"] == 0.8


def test_forward_path_stops_when_no_group_improves():
    scorer = _make_scorer(
        {
            frozenset(["b"]): 1.0,
            frozenset(["b", "c3"]): 1.1,
            frozenset(["b", "c4"]): 1.2,
        }
    )
    path = _forward_path(["b"], {"g3": ["c3"], "g4": ["c4"]}, scorer)
    assert len(path) == 1  # base only


def test_efficient_set_is_smallest_within_epsilon():
    path = [
        {"columns": ["b"], "crps": 1.0},
        {"columns": ["b", "c3"], "crps": 0.995},
        {"columns": ["b", "c3", "c4"], "crps": 0.985},
    ]
    # full_crps = 0.99 -> threshold = 0.9999; first step <= threshold is ["b","c3"].
    assert _efficient_set(path, ["b", "c3", "c4"], 0.99, epsilon=0.01) == ["b", "c3"]


def test_efficient_set_falls_back_to_full_when_none_close():
    path = [{"columns": ["b"], "crps": 2.0}]
    assert _efficient_set(path, ["b", "c3"], 1.0, epsilon=0.01) == ["b", "c3"]


def test_leave_one_out_drops_each_candidate_group():
    scorer = _make_scorer(
        {
            frozenset(["b", "c3", "c4"]): 1.0,
            frozenset(["b", "c4"]): 1.2,  # drop c3
            frozenset(["b", "c3"]): 1.1,  # drop c4
        }
    )
    rows = _leave_one_out(["b", "c3", "c4"], {"g3": ["c3"], "g4": ["c4"]}, scorer)
    losses = {r["group"]: r["loss"] for r in rows}
    assert losses["g3"] == pytest.approx(0.2)
    assert losses["g4"] == pytest.approx(0.1)


# --- smoke: real LightGBM on synthetic data ----------------------------------


def _ablation_frame() -> pd.DataFrame:
    index = pd.date_range("2024-06-01", "2024-06-30", freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    n = len(index)
    frame = pd.DataFrame(
        {
            "price": rng.normal(50, 10, n),
            "load_forecast": rng.normal(1000, 100, n),
            "wind_forecast": rng.normal(200, 50, n),
            "net_position_mwh": rng.normal(0, 100, n),
            "scheduled_exchange_mwh": rng.normal(0, 100, n),
            "neighbour_price_eur_mwh": rng.normal(50, 10, n),
            "SE1_temperature_2m": rng.normal(10, 5, n),
            "hydro_storage_mwh": rng.normal(5000, 500, n),
            "carbon_eua": rng.normal(70, 10, n),
            "fx_sek_eur": rng.normal(11.3, 0.1, n),
        },
        index=index,
    )
    return frame


def test_evaluate_crps_returns_finite_float():
    frame = _ablation_frame()
    cutoffs = generate_folds(frame, test_start=date(2024, 6, 28))
    _, full = build_features(date(2024, 6, 28), frame)
    crps = evaluate_crps(frame, cutoffs, list(full.columns))
    assert np.isfinite(crps)


def test_run_ablation_produces_efficient_set(tmp_path):
    frame = _ablation_frame()
    cutoffs = generate_folds(frame, test_start=date(2024, 6, 28))

    result = run_ablation(frame, cutoffs, out_dir=tmp_path / "reports")
    assert result.efficient_set
    # group 2 (load/wind) is excluded from every arm, so never in the efficient set
    assert not set(result.efficient_set) & {"load_forecast", "wind_forecast"}
    assert result.forward_path
    assert result.leave_one_out
    assert set(result.importance.columns) == {"feature", "importance"}
    assert set(result.shap.columns) == {"feature", "mean_abs_shap"}
    assert (tmp_path / "reports" / "ablation_table.csv").exists()


def test_permutation_and_shap_rank_features():
    frame = _ablation_frame()
    _, full = build_features(date(2024, 6, 28), frame)
    columns = list(full.columns)
    arm, X, target = _fit_full_lgbm(frame, date(2024, 6, 28), columns)

    imp = permutation_importance(arm, X, target)
    shp = shap_values(arm, X)

    assert len(imp) == len(X.columns)
    assert len(shp) == len(X.columns)
    assert imp["importance"].notna().all()
    assert shp["mean_abs_shap"].notna().all()
