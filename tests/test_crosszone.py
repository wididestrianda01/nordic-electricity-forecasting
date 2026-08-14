"""Tests for the cross-zone robustness run (ticket 23)."""

import pandas as pd

from forecast_pipeline.crosszone import zone_summary


def _results() -> pd.DataFrame:
    rows = [
        # zone SE1: chronos2 best (1.0), catboost (2.0), lgbm (3.0), xgboost (4.0)
        {"zone": "SE1", "model": "chronos2", "crps": 1.0},
        {"zone": "SE1", "model": "catboost", "crps": 2.0},
        {"zone": "SE1", "model": "lgbm", "crps": 3.0},
        {"zone": "SE1", "model": "xgboost", "crps": 4.0},
        # zone SE2: catboost best (chronos2 second)
        {"zone": "SE2", "model": "catboost", "crps": 1.5},
        {"zone": "SE2", "model": "chronos2", "crps": 1.8},
        {"zone": "SE2", "model": "lgbm", "crps": 2.2},
        {"zone": "SE2", "model": "xgboost", "crps": 2.9},
    ]
    return pd.DataFrame(rows)


def test_zone_summary_ranks_within_zone():
    summary = zone_summary(_results())
    assert list(summary["zone"]) == ["SE1", "SE2"]

    se1 = summary[summary["zone"] == "SE1"].iloc[0]
    assert se1["best_model"] == "chronos2"
    assert bool(se1["chronos2_best"]) is True
    assert se1["chronos2_rank"] == 1
    assert se1["ranking"] == "chronos2, catboost, lgbm, xgboost"

    se2 = summary[summary["zone"] == "SE2"].iloc[0]
    assert se2["best_model"] == "catboost"
    assert bool(se2["chronos2_best"]) is False
    assert se2["chronos2_rank"] == 2
