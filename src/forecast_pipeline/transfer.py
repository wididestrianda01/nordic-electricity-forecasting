"""Ticket 19: transfer check -- does the efficient set generalize? (ticket 11).

The efficient feature set is discovered on LightGBM (the workhorse). This
module re-runs the other full-features arms -- XGBoost, CatBoost, TFT,
Chronos-2 -- on the price-only base (group 1) versus that efficient set, and
records the CRPS delta, confirming the set earns its cost beyond the model
that found it. Price-only arms (SARIMA/ETS/LEAR/N-BEATS/TimesFM) are out of
scope: they consume no exogenous features.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from forecast_pipeline.backtest import DEFAULT_SPECS, run_backtest
from forecast_pipeline.feature_groups import classify_columns
from forecast_pipeline.features import build_features
from forecast_pipeline.registry import ModelSpec


def price_only_columns(full_columns: list[str]) -> list[str]:
    """The group-1 (AR/calendar/regime) columns of a full-features matrix."""
    return classify_columns(full_columns)["ar_calendar_regime"]


def _evaluate(
    historical_data: pd.DataFrame,
    cutoffs: list[date],
    columns: list[str],
    spec: ModelSpec,
) -> float:
    result = run_backtest(
        historical_data, [spec], cutoffs, log=False, feature_columns=columns
    )
    return float(result["crps"].mean())


def check_transfer(
    historical_data: pd.DataFrame,
    cutoffs: list[date],
    efficient_set: list[str],
    *,
    specs: list[ModelSpec] | None = None,
) -> pd.DataFrame:
    """Run each full-features arm (except LightGBM) on base vs efficient set.

    Returns one row per arm with ``crps_price_only`` (group-1 base),
    ``crps_efficient``, and ``crps_delta`` (efficient - base; negative = the
    efficient set is better).
    """
    if specs is None:
        specs = [
            s
            for s in DEFAULT_SPECS
            if s.feature_set == "full-features" and s.name != "lgbm"
        ]

    full_columns = list(build_features(cutoffs[0], historical_data)[1].columns)
    base = price_only_columns(full_columns)

    rows: list[dict] = []
    for spec in specs:
        base_crps = _evaluate(historical_data, cutoffs, base, spec)
        efficient_crps = _evaluate(historical_data, cutoffs, efficient_set, spec)
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "crps_price_only": base_crps,
                "crps_efficient": efficient_crps,
                "crps_delta": efficient_crps - base_crps,
            }
        )
    return pd.DataFrame(rows)


def run_transfer_check(
    historical_data: pd.DataFrame,
    cutoffs: list[date],
    efficient_set: list[str],
    *,
    specs: list[ModelSpec] | None = None,
    out_dir: str | Path = "reports",
) -> pd.DataFrame:
    """Run the transfer check and write the table to ``out_dir``."""
    table = check_transfer(
        historical_data, cutoffs, efficient_set, specs=specs
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "transfer_table.csv", index=False)
    return table
