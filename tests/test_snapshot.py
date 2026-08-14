"""Tests for the curated snapshot loader (ticket 24)."""

import pandas as pd

from forecast_pipeline.snapshot import (
    MATRICES_SUBDIR,
    RESULTS_SUBDIR,
    _matrix_filename,
    headline_ranking,
    load_matrices,
    load_results,
)


def test_headline_ranking_sorts_ascending():
    results = pd.DataFrame(
        {
            "model": ["sarima", "catboost", "chronos2", "catboost", "sarima"],
            "crps": [15.0, 9.0, 9.5, 10.0, 14.0],
        }
    )
    ranking = headline_ranking(results)
    assert list(ranking["model"]) == ["catboost", "chronos2", "sarima"]
    assert ranking.loc[0, "mean_crps"] == 9.5  # mean of 9.0, 10.0
    assert ranking.loc[2, "mean_crps"] == 14.5  # mean of 15.0, 14.0


def test_load_results_offline(tmp_path):
    results = tmp_path / RESULTS_SUBDIR
    results.mkdir(parents=True)
    pd.DataFrame(
        {"model": ["catboost", "chronos2", "catboost"], "crps": [9.0, 9.5, 10.0]}
    ).to_csv(results / "results.csv", index=False)

    loaded = load_results(tmp_path)
    ranking = headline_ranking(loaded["results"])
    assert list(ranking["model"]) == ["catboost", "chronos2"]
    assert ranking.loc[0, "mean_crps"] == 9.5


def test_load_matrices_offline_roundtrip(tmp_path):
    matrices = tmp_path / MATRICES_SUBDIR
    matrices.mkdir(parents=True)
    price = pd.DataFrame({"price_lag_1": [1.0, 2.0]})
    full = pd.DataFrame({"price_lag_1": [1.0, 2.0], "carbon_eua": [80.0, 81.0]})
    price.to_parquet(matrices / _matrix_filename("SE3", "hourly", "price_only"))
    full.to_parquet(matrices / _matrix_filename("SE3", "hourly", "full_features"))

    loaded = load_matrices(tmp_path)
    assert ("SE3", "hourly") in loaded
    price_only, full_features = loaded[("SE3", "hourly")]
    assert list(price_only.columns) == ["price_lag_1"]
    assert set(full_features.columns) == {"price_lag_1", "carbon_eua"}
