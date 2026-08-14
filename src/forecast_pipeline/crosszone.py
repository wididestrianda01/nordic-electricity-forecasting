"""Ticket 23: cross-zone (SE1-SE4) robustness run.

The headline bake-off runs on SE3. This module proves the framework
generalizes by running the accuracy-vs-compute frontier (the GBDT trio:
CatBoost / LightGBM / XGBoost) plus the accuracy anchor (Chronos-2) across
SE1-SE4 at default configs, hourly regime only, over the full test window
``2023-01-01 -> 2025-09-30``. The full 10-model roster per zone is out of
scope; the question is whether the frontier and Chronos-2's accuracy edge
survive outside SE3.

Runnable via ``uv run python -m forecast_pipeline.crosszone``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from forecast_pipeline.backtest import (
    DEFAULT_SPECS,
    generate_folds,
    run_backtest,
)
from forecast_pipeline.features import assemble_data
from forecast_pipeline.pipeline import MTU_15MIN_SWITCH_DATE

#: The frontier (GBDT trio) plus the accuracy anchor (Chronos-2).
CROSS_ZONE_MODELS = ("catboost", "lgbm", "xgboost", "chronos2")

DEFAULT_TEST_START = date(2023, 1, 1)


def _cross_zone_specs() -> list:
    return [s for s in DEFAULT_SPECS if s.name in CROSS_ZONE_MODELS]


def zone_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per-zone ranking of the four models by mean CRPS (rank 1 = best).

    ``chronos2_best`` is True where Chronos-2 leads the zone, answering whether
    its accuracy edge survives outside SE3.
    """
    rows = []
    for zone, group in results.groupby("zone"):
        means = group.groupby("model")["crps"].mean().sort_values()
        rank = {model: i + 1 for i, model in enumerate(means.index)}
        rows.append(
            {
                "zone": zone,
                "best_model": means.index[0],
                "best_crps": means.iloc[0],
                "chronos2_best": means.index[0] == "chronos2",
                "chronos2_rank": rank.get("chronos2"),
                "ranking": ", ".join(means.index),
            }
        )
    return pd.DataFrame(rows)


def run_cross_zone(
    zones: list[str],
    start: date,
    end: date,
    *,
    test_start: date = DEFAULT_TEST_START,
    tracking_uri: str = "mlruns",
    out_dir: str | Path = "reports",
    refresh: bool = False,
) -> pd.DataFrame:
    """Run the 4-model frontier across each zone; return one row per (zone, model, cutoff)."""
    specs = _cross_zone_specs()
    parts: list[pd.DataFrame] = []
    for zone in zones:
        frame = assemble_data([zone], start, end, refresh=refresh)
        cutoffs = generate_folds(frame, test_start=test_start)
        if not cutoffs:
            continue
        result = run_backtest(
            frame, specs, cutoffs, tracking_uri=tracking_uri, log=True
        )
        result["zone"] = zone
        parts.append(result)

    results = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not results.empty:
        results.to_csv(out / "cross_zone_results.csv", index=False)
        zone_summary(results).to_csv(out / "cross_zone_summary.csv", index=False)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-zone robustness run.")
    parser.add_argument("--zones", nargs="+", default=["SE1", "SE2", "SE3", "SE4"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START.isoformat())
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end)
    # Cross-zone is hourly-only (ticket 23); clamp to the pre-switch boundary so
    # assemble_data resolves a 60-minute MTU instead of up-sampling to 15-minute.
    hourly_end = MTU_15MIN_SWITCH_DATE - pd.Timedelta(days=1)
    results = run_cross_zone(
        args.zones,
        date.fromisoformat(args.start),
        min(end, hourly_end),
        test_start=date.fromisoformat(args.test_start),
        tracking_uri=args.tracking_uri,
        out_dir=args.out_dir,
        refresh=args.refresh,
    )
    print(zone_summary(results).to_string(index=False))


if __name__ == "__main__":
    main()
