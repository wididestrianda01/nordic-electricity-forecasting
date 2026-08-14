"""Tests for the tuned-vs-default comparison (ticket 22)."""

import json

import pytest

from forecast_pipeline.tuning import build_tuned_table


def test_build_tuned_table_delta_aggregation():
    models = ["sarima", "catboost", "nbeats", "chronos2"]
    families = {
        "sarima": "classical",
        "catboost": "gbdt",
        "nbeats": "deep",
        "chronos2": "foundation",
    }
    default_crps = {
        "sarima": 14.99,
        "catboost": 9.64,
        "nbeats": 14.39,
        "chronos2": 9.61,
    }
    tuned_crps = {
        "sarima": 14.50,
        "catboost": 9.40,
        "nbeats": 15.00,
        "chronos2": 9.55,
    }
    hyperparams = {
        "sarima": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1]},
        "catboost": {"depth": 5, "learning_rate": 0.08},
        "nbeats": {"num_stacks": 4, "n_epochs": 3},
        "chronos2": {"input_chunk_length": 1024},
    }

    table = build_tuned_table(
        models, families, default_crps, tuned_crps, hyperparams
    )

    assert list(table["model"]) == models
    # delta = tuned - default; negative means the tuned config is better.
    assert table.loc[0, "crps_delta"] == pytest.approx(14.50 - 14.99)
    assert table.loc[1, "crps_delta"] == pytest.approx(9.40 - 9.64)
    assert table.loc[2, "crps_delta"] == pytest.approx(15.00 - 14.39)
    assert table.loc[3, "crps_delta"] == pytest.approx(9.55 - 9.61)
    # hyperparams survive the CSV-safe serialization.
    assert json.loads(table.loc[1, "tuned_hyperparams"]) == hyperparams["catboost"]


def test_build_tuned_table_column_order():
    table = build_tuned_table(
        ["catboost"],
        {"catboost": "gbdt"},
        {"catboost": 9.64},
        {"catboost": 9.40},
        {"catboost": {"depth": 5}},
    )
    assert list(table.columns) == [
        "model",
        "family",
        "default_crps",
        "tuned_crps",
        "crps_delta",
        "tuned_hyperparams",
    ]
