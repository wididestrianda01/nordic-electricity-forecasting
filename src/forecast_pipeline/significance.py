"""Diebold-Mariano significance tests for the bake-off (report supplement).

The aggregate results table folds per-timestamp CRPS into a per-fold mean, so
a significance test on it would have only three hourly observations. This
module re-runs the hourly walk-forward folds for a chosen set of models,
capturing the per-timestamp CRPS that the fold means discard, then computes
pairwise Diebold-Mariano tests (scoring.diebold_mariano) between them.

Runnable via ``uv run python -m forecast_pipeline.significance``; the data is
read from the disk cache, so no network or API key is required after the
initial assembly.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from forecast_pipeline.backtest import (
    DEFAULT_SPECS,
    _actuals_for,
    _cutoff_timestamp,
    _infer_mtu_minutes,
    generate_folds,
)
from forecast_pipeline.features import (
    assemble_data,
    build_features,
    build_horizon_features,
)
from forecast_pipeline.registry import build_model
from forecast_pipeline.scoring import crps_scores, diebold_mariano

#: Forecast horizon in MTU steps (hourly regime).
_HORIZON_MTU = 24


def capture_per_timestamp_crps(
    historical_data: pd.DataFrame, specs, cutoffs: list[date]
) -> pd.DataFrame:
    """Return one row per (model, timestamp) with the per-timestamp CRPS."""
    rows: list[dict] = []
    index = historical_data.index
    tz = index.tz if isinstance(index, pd.DatetimeIndex) else None
    mtu_minutes = _infer_mtu_minutes(historical_data)
    horizon = 24 * 60 // mtu_minutes
    for spec in specs:
        for cutoff in cutoffs:
            cutoff_ts = _cutoff_timestamp(cutoff, tz)
            train = historical_data.loc[historical_data.index < cutoff_ts]
            actuals = _actuals_for(historical_data, cutoff_ts)
            target = train["price"]
            if spec.feature_set == "full-features":
                _, features = build_features(cutoff, train)
                future_features = build_horizon_features(cutoff, train)
            else:
                features = None
                future_features = None
            arm = build_model(spec)
            arm.fit(target, features)
            predictions = arm.predict_quantiles(horizon, future_features)
            for timestamp, value in crps_scores(predictions, actuals).items():
                rows.append(
                    {"model": spec.name, "cutoff": cutoff, "timestamp": timestamp, "crps": value}
                )
    return pd.DataFrame(rows)


def pairwise_diebold_mariano(
    scores: pd.DataFrame, models: list[str] | None = None
) -> pd.DataFrame:
    """Pairwise two-sided DM tests; a negative statistic favours ``model_a``.

    Per-timestamp CRPS is collapsed to per-fold means first, because each
    yearly fold is an independent block (the folds are disjoint days, years
    apart). The test then compares independent fold-level observations with no
    within-block autocorrelation, which the per-timestamp MA(24) estimator
    would mis-handle.
    """
    if models is None:
        models = sorted(scores["model"].unique())
    fold_means = scores.groupby(["model", "cutoff"], as_index=False)["crps"].mean()
    rows: list[dict] = []
    for i, a in enumerate(models):
        sa = fold_means.loc[fold_means["model"] == a].set_index("cutoff")["crps"].sort_index()
        for b in models[i + 1 :]:
            sb = fold_means.loc[fold_means["model"] == b].set_index("cutoff")["crps"].sort_index()
            statistic, p_value = diebold_mariano(sa, sb, horizon=1)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "dm_statistic": statistic,
                    "p_value": p_value,
                    "n_folds": len(sa),
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pairwise DM significance tests.")
    parser.add_argument("--zones", nargs="+", default=["SE3"])
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-09-30")
    parser.add_argument(
        "--models", nargs="+", default=["chronos2", "catboost", "lgbm", "xgboost"]
    )
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args(argv)

    frame = assemble_data(args.zones, date.fromisoformat(args.start), date.fromisoformat(args.end))
    cutoffs = generate_folds(frame, test_start=date(2023, 1, 1))
    specs = [s for s in DEFAULT_SPECS if s.name in args.models]
    missing = set(args.models) - {s.name for s in specs}
    if missing:
        parser.error(f"unknown model(s): {sorted(missing)}")

    print(f"capturing per-timestamp CRPS for {[s.name for s in specs]} over {cutoffs}")
    scores = capture_per_timestamp_crps(frame, specs, cutoffs)
    table = pairwise_diebold_mariano(scores, [s.name for s in specs])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "per_timestamp_crps.csv", index=False)
    table.to_csv(out / "dm_tests.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
