"""Ticket 16: bake-off runner -- the full 10-model comparison entrypoint.

Orchestrates ``assemble_data`` -> ``generate_folds`` -> ``run_backtest`` over
``DEFAULT_SPECS`` for the two regimes separately: the hourly (60-minute) frame
before the 1 Oct 2025 MTU switch, and the 15-minute frame from the switch
onward. Each regime is single-frequency, which ``generate_folds`` and
``run_backtest`` require (ticket 15). Results are concatenated (one row per
model x cutoff x regime) and saved outside ``mlruns/``.

Runnable via ``uv run python -m forecast_pipeline.bakeoff``.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from forecast_pipeline.backtest import (
    DEFAULT_SPECS,
    RESULT_COLUMNS,
    generate_folds,
    run_backtest,
)
from forecast_pipeline.features import assemble_data
from forecast_pipeline.pipeline import _mtu_minutes_for

DEFAULT_TEST_START = date(2023, 1, 1)


def split_regimes(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Split an assembled frame into ``{mtu_minutes: single-frequency frame}``.

    Rows are grouped by the MTU in force on their calendar date (60-minute
    before the 1 Oct 2025 switch, 15-minute from then on). Empty regimes are
    dropped; every returned frame is single-frequency.
    """
    mtu = pd.Series(
        [_mtu_minutes_for(ts.date()) for ts in frame.index], index=frame.index
    )
    return {
        minutes: frame.loc[mtu == minutes].sort_index()
        for minutes in sorted(mtu.unique())
    }


def run_bakeoff(
    zones,
    start: date,
    end: date,
    *,
    specs=DEFAULT_SPECS,
    test_start: date = DEFAULT_TEST_START,
    tracking_uri: str = "mlruns",
    out_dir: str | Path = "reports",
    refresh: bool = False,
    log: bool = True,
) -> pd.DataFrame:
    """Run the full bake-off; return one row per (model, cutoff, regime).

    Assembles data (disk-cached; ``refresh`` bypasses the cache), splits into
    the hourly and 15-min regimes, and runs the walk-forward backtest over
    ``specs`` for each regime separately. Saves the combined result to
    ``out_dir/results.parquet`` and ``out_dir/results.csv``.
    """
    frame = assemble_data(zones, start, end, refresh=refresh)

    parts: list[pd.DataFrame] = []
    for regime_frame in split_regimes(frame).values():
        cutoffs = generate_folds(regime_frame, test_start=test_start)
        if not cutoffs:
            continue
        parts.append(
            run_backtest(
                regime_frame, specs, cutoffs, tracking_uri=tracking_uri, log=log
            )
        )

    result = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=RESULT_COLUMNS)
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out / "results.parquet")
    result.to_csv(out / "results.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Nordic bake-off.")
    parser.add_argument("--zones", nargs="+", default=["SE3"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START.isoformat())
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args(argv)

    result = run_bakeoff(
        args.zones,
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        test_start=date.fromisoformat(args.test_start),
        tracking_uri=args.tracking_uri,
        out_dir=args.out_dir,
        refresh=args.refresh,
        log=not args.no_log,
    )

    summary = result.groupby("model")["crps"].mean().sort_values()
    print(f"ran {len(result)} fold(s) across {result['mtu_minutes'].nunique()} regime(s)")
    print(summary.to_string())


if __name__ == "__main__":
    main()
