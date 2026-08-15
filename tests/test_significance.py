"""Tests for the Diebold-Mariano significance runner."""

import numpy as np
import pandas as pd

from forecast_pipeline.significance import pairwise_diebold_mariano


def _scores() -> pd.DataFrame:
    """Two models, three independent folds, per-timestamp CRPS within a fold."""
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(3):
        for hour in range(24):
            ts = pd.Timestamp("2023-01-01") + pd.Timedelta(days=fold * 400, hours=hour)
            base = rng.normal(10.0, 3.0)
            rows.append({"model": "a", "cutoff": ts.date(), "timestamp": ts, "crps": base + rng.normal(0, 0.5)})
            rows.append({"model": "b", "cutoff": ts.date(), "timestamp": ts, "crps": base + 1.5 + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


def test_pairwise_diebold_mariano_aggregates_by_fold():
    """Fold-level aggregation gives a sane statistic, not the degenerate 1e7."""
    table = pairwise_diebold_mariano(_scores(), ["a", "b"])

    assert table["n_folds"].iloc[0] == 3
    # model b has uniformly higher CRPS, so model a is favoured (negative stat)
    assert table["dm_statistic"].iloc[0] < 0
    assert abs(table["dm_statistic"].iloc[0]) < 1000.0
    assert 0.0 <= table["p_value"].iloc[0] <= 1.0
