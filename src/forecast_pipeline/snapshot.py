"""Ticket 24: curated data snapshot and offline loader.

A small committed snapshot so a reviewer replicates the demo with no API key.
It holds the assembled feature matrices (price-only + full-features) for SE3
in both regimes and for SE1-SE4 hourly, plus every result table. The raw
network cache (``data/cache/``) and ``mlruns/`` stay out of git; this directory
is the only committed data.

``build_snapshot`` assembles the matrices (via the cached ``assemble_data``
window) and copies the result tables from ``reports/``. ``load_matrices`` /
``load_results`` read the snapshot back offline, and ``headline_ranking``
reconstructs the committed mean-CRPS ranking from ``results.csv``.

Runnable via ``uv run python -m forecast_pipeline.snapshot --build``.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from forecast_pipeline.features import assemble_data, build_features
from forecast_pipeline.pipeline import MTU_15MIN_SWITCH_DATE

#: Snapshot root (committed; distinct from the gitignored ``data/cache/``).
#: Resolved from the module path, not the CWD, so notebooks run from any dir.
SNAPSHOT_DIR = str(Path(__file__).resolve().parents[2] / "data" / "snapshot")
MATRICES_SUBDIR = "matrices"
RESULTS_SUBDIR = "results"

#: Matrices to snapshot: (zone, regime). SE3 both regimes; SE1/SE2/SE4 hourly.
MATRIX_SPECS: tuple[tuple[str, str], ...] = (
    ("SE3", "hourly"),
    ("SE3", "15min"),
    ("SE1", "hourly"),
    ("SE2", "hourly"),
    ("SE4", "hourly"),
)

#: Result tables copied from ``reports/`` into the snapshot.
RESULT_TABLES: tuple[str, ...] = (
    "results.csv",
    "pareto_table.csv",
    "run_table.csv",
    "tuned_table.csv",
    "cross_zone_results.csv",
    "cross_zone_summary.csv",
    "transfer_table.csv",
    "ablation_table.csv",
    "marginal_value.csv",
    "efficient_set.csv",
    "feature_importance.csv",
    "feature_shap.csv",
)

DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date(2026, 8, 13)


def snapshot_root() -> Path:
    return Path(SNAPSHOT_DIR)


def _window(regime: str) -> tuple[date, date]:
    if regime == "15min":
        return MTU_15MIN_SWITCH_DATE, DEFAULT_END
    return DEFAULT_START, MTU_15MIN_SWITCH_DATE - pd.Timedelta(days=1)


def _matrix_filename(zone: str, regime: str, kind: str) -> str:
    return f"{zone.lower()}_{regime}_{kind}.parquet"


def build_snapshot(
    *,
    out_dir: str | Path = SNAPSHOT_DIR,
    reports_dir: str | Path = "reports",
    refresh: bool = False,
) -> Path:
    """Assemble the feature matrices and copy result tables into the snapshot."""
    root = Path(out_dir)
    matrices = root / MATRICES_SUBDIR
    results = root / RESULTS_SUBDIR
    matrices.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    for zone, regime in MATRIX_SPECS:
        start, end = _window(regime)
        frame = assemble_data([zone], start, end, refresh=refresh)
        as_of_date = frame.index[-1].date()
        price_only, full_features = build_features(as_of_date, frame)
        price_only.to_parquet(matrices / _matrix_filename(zone, regime, "price_only"))
        full_features.to_parquet(
            matrices / _matrix_filename(zone, regime, "full_features")
        )

    reports = Path(reports_dir)
    for name in RESULT_TABLES:
        source = reports / name
        if source.exists():
            shutil.copy(source, results / name)

    return root


def load_matrices(
    snapshot_dir: str | Path = SNAPSHOT_DIR,
) -> dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]]:
    """Load every ``(zone, regime) -> (price_only, full_features)`` pair offline."""
    matrices = Path(snapshot_dir) / MATRICES_SUBDIR
    out: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for zone, regime in MATRIX_SPECS:
        price = matrices / _matrix_filename(zone, regime, "price_only")
        full = matrices / _matrix_filename(zone, regime, "full_features")
        if price.exists() and full.exists():
            out[(zone, regime)] = (
                pd.read_parquet(price),
                pd.read_parquet(full),
            )
    return out


def load_results(snapshot_dir: str | Path = SNAPSHOT_DIR) -> dict[str, pd.DataFrame]:
    """Load every committed result table as a DataFrame (empty if absent)."""
    results = Path(snapshot_dir) / RESULTS_SUBDIR
    out: dict[str, pd.DataFrame] = {}
    for name in RESULT_TABLES:
        path = results / name
        out[name.removesuffix(".csv")] = (
            pd.read_csv(path) if path.exists() else pd.DataFrame()
        )
    return out


def headline_ranking(results: pd.DataFrame) -> pd.DataFrame:
    """Mean CRPS per model, ranked ascending (lower is better)."""
    ranking = results.groupby("model")["crps"].mean().sort_values()
    return ranking.rename("mean_crps").reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the curated snapshot.")
    parser.add_argument("--build", action="store_true", help="assemble + copy")
    parser.add_argument("--out-dir", default=SNAPSHOT_DIR)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    if args.build:
        root = build_snapshot(
            out_dir=args.out_dir,
            reports_dir=args.reports_dir,
            refresh=args.refresh,
        )
        print(f"snapshot built at {root}")
    else:
        matrices = load_matrices(args.out_dir)
        results = load_results(args.out_dir)
        print(f"matrices: {list(matrices)}")
        if not results["results"].empty:
            print(headline_ranking(results["results"]).to_string(index=False))


if __name__ == "__main__":
    main()
