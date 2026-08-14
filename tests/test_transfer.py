"""Tests for the transfer check (ticket 19)."""

from datetime import date

import numpy as np
import pandas as pd

from forecast_pipeline import transfer
from forecast_pipeline.backtest import DEFAULT_SPECS
from forecast_pipeline.transfer import check_transfer, price_only_columns


def _full_features_specs():
    return [s for s in DEFAULT_SPECS if s.feature_set == "full-features" and s.name != "lgbm"]


def test_price_only_columns_excludes_exogenous():
    columns = [
        "lag_1d",
        "regime",
        "load_forecast",
        "carbon_eua",
        "SE1_temperature_2m",
        "fx_sek_eur",
        "hydro_storage_mwh",
    ]
    assert price_only_columns(columns) == ["lag_1d", "regime"]


def test_check_transfer_computes_delta(monkeypatch):
    frame = pd.DataFrame(
        {"price": np.linspace(40, 60, 48)},
        index=pd.date_range("2024-06-01", periods=48, freq="h", tz="UTC"),
    )
    frame["load_forecast"] = 1000.0
    frame["wind_forecast"] = 200.0
    cutoffs = [date(2024, 6, 2)]

    base = ["lag_1d", "regime"]
    efficient = ["lag_1d", "regime", "carbon_eua"]

    # Map (spec_name, feature-set) -> CRPS; efficient set is always better.
    def fake_run_backtest(historical_data, specs, cutoffs, *, log=False, feature_columns=None, **kw):
        key = frozenset(feature_columns)
        base_crps = 2.0 if key == frozenset(base) else 1.0
        return pd.DataFrame(
            {
                "model": [s.name for s in specs],
                "family": [s.family for s in specs],
                "feature_set": [s.feature_set for s in specs],
                "cutoff": [cutoffs[0]] * len(specs),
                "mtu_minutes": [60] * len(specs),
                "crps": [base_crps] * len(specs),
                "pinball_p10": [0.0] * len(specs),
                "pinball_p50": [0.0] * len(specs),
                "pinball_p90": [0.0] * len(specs),
                "mae": [1.0] * len(specs),
                "skill_score_crps": [0.0] * len(specs),
                "train_wall_clock": [0.0] * len(specs),
                "inference_wall_clock": [0.0] * len(specs),
            }
        )

    monkeypatch.setattr(transfer, "run_backtest", fake_run_backtest)
    # Also feed the full-features column classification through a fixed list so
    # check_transfer's base == our `base` (frame has no exogenous columns, so
    # group 1 is everything non-price-derived; build_features would add lags).
    monkeypatch.setattr(transfer, "build_features", lambda *a, **k: (None, pd.DataFrame(index=frame.index, columns=base + ["carbon_eua"])))

    result = check_transfer(frame, cutoffs, efficient)

    assert set(result.columns) == {
        "model", "family", "crps_price_only", "crps_efficient", "crps_delta",
    }
    assert set(result["model"]) == {s.name for s in _full_features_specs()}
    assert (result["crps_price_only"] == 2.0).all()
    assert (result["crps_efficient"] == 1.0).all()
    assert (result["crps_delta"] == -1.0).all()
