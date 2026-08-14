"""Ticket 21: real-data ablation + transfer execution.

The feature ablation (ticket 18) and transfer check (ticket 19) are
implemented and unit-tested on synthetic scores; this module runs them on the
real assembled hourly frame and emits the derived artifacts: the efficient
feature set, the marginal-value ranking of the seven groups, and the
per-family transfer table. It reads the cached ``assemble_data`` window, so no
network fetch is paid on re-runs.

Runnable via ``uv run python -m forecast_pipeline.ablation_transfer``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd

from forecast_pipeline.ablation import run_ablation
from forecast_pipeline.backtest import generate_folds
from forecast_pipeline.feature_groups import GROUP_ORDER, classify_columns
from forecast_pipeline.features import assemble_data
from forecast_pipeline.pipeline import MTU_15MIN_SWITCH_DATE
from forecast_pipeline.tracking import set_local_tracking_uri, start_parent_run
from forecast_pipeline.transfer import run_transfer_check

DEFAULT_TEST_START = date(2023, 1, 1)


def _efficient_set_groups(efficient_set: list[str]) -> list[str]:
    """Map the efficient-set columns back to their feature groups, in order."""
    groups = classify_columns(efficient_set)
    return [group for group in GROUP_ORDER if groups[group]]


def _marginal_value_table(forward_path: list[dict]) -> pd.DataFrame:
    """Turn the greedy forward path into a marginal-value ranking."""
    rows = []
    prev = forward_path[0]["crps"]
    for i, step in enumerate(forward_path):
        rows.append(
            {
                "step": i,
                "added_group": step.get("added", "(base)"),
                "crps": step["crps"],
                "marginal_delta": step["crps"] - prev,
            }
        )
        prev = step["crps"]
    return pd.DataFrame(rows)


def run_real_data_ablation_transfer(
    zones: list[str],
    start: date,
    end: date,
    *,
    test_start: date = DEFAULT_TEST_START,
    tracking_uri: str = "mlruns",
    out_dir: str | Path = "reports",
    refresh: bool = False,
) -> dict:
    """Run the ablation + transfer check on the real hourly frame."""
    frame = assemble_data(zones, start, end, refresh=refresh)
    cutoffs = generate_folds(frame, test_start=test_start)

    result = run_ablation(frame, cutoffs, out_dir=out_dir)
    transfer = run_transfer_check(frame, cutoffs, result.efficient_set, out_dir=out_dir)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = _efficient_set_groups(result.efficient_set)
    (out / "efficient_set.csv").write_text("\n".join(["group", *groups]) + "\n")
    marginal = _marginal_value_table(result.forward_path)
    marginal.to_csv(out / "marginal_value.csv", index=False)

    # Log the derived tables to MLflow under one parent run.
    set_local_tracking_uri(tracking_uri)
    with start_parent_run("ablation", "ablation", "full-features", {}, 42):
        for name in ("efficient_set.csv", "marginal_value.csv", "transfer_table.csv"):
            mlflow.log_artifact(str(out / name))

    return {
        "efficient_set_groups": groups,
        "efficient_set_columns": result.efficient_set,
        "marginal_value": marginal,
        "transfer": transfer,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Real-data ablation + transfer.")
    parser.add_argument("--zones", nargs="+", default=["SE3"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START.isoformat())
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end)
    hourly_end = MTU_15MIN_SWITCH_DATE - pd.Timedelta(days=1)
    summary = run_real_data_ablation_transfer(
        args.zones,
        date.fromisoformat(args.start),
        min(end, hourly_end),
        test_start=date.fromisoformat(args.test_start),
        tracking_uri=args.tracking_uri,
        out_dir=args.out_dir,
        refresh=args.refresh,
    )
    print("efficient set:", json.dumps(summary["efficient_set_groups"]))
    print(summary["marginal_value"].to_string(index=False))
    print(summary["transfer"].to_string(index=False))


if __name__ == "__main__":
    main()
