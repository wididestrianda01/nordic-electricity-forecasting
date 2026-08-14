"""Ticket 22: tuned-vs-default comparison for the best model per family.

The committed bake-off compares the 10-model roster at pinned defaults; this
module adds a *secondary* comparison, not a replacement. It tunes the best
model per family on the hourly regime and records the CRPS delta against the
pinned default, leaving the default table as the headline.

Best-per-family (from the committed ranking): SARIMA, CatBoost, N-BEATS,
Chronos-2. Each family tunes by its own mechanism:

- SARIMA    -- AIC/BIC order grid (no training loop).
- CatBoost  -- bounded Optuna over learning rate / depth / iterations /
               l2_leaf_reg.
- N-BEATS   -- bounded Optuna over stacks / blocks / width / epochs.
- Chronos-2 -- config sweep over context length (pretrained weights are
               never tuned).

Search objectives evaluate a single walk-forward fold (fit before ``cutoff``,
score the D+1 horizon) through the same ``run_backtest`` harness, so a tuned
config is scored identically to the committed comparison. The tuned runs are
then re-run across all folds (and logged to MLflow) to produce the delta.

Runnable via ``uv run python -m forecast_pipeline.tuning``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_pipeline.arms_classical import _seasonal_period
from forecast_pipeline.backtest import (
    DEFAULT_SPECS,
    generate_folds,
    run_backtest,
)
from forecast_pipeline.features import assemble_data
from forecast_pipeline.pipeline import MTU_15MIN_SWITCH_DATE
from forecast_pipeline.registry import ModelSpec

#: The four tuned models (best per family).
TUNED_MODELS = ("sarima", "catboost", "nbeats", "chronos2")

#: Chronos-2 context-length sweep candidates.
CHRONOS_CONTEXT_GRID = (512, 1024, 2048)

#: SARIMA order grid: d and D are fixed at 1 (airline); p/q and P/Q vary.
_SARIMA_ORDER_GRID = tuple((p, 1, q) for p in (0, 1, 2) for q in (0, 1, 2))
_SARIMA_SEASONAL_GRID = tuple((P, 1, Q) for P in (0, 1) for Q in (0, 1))

DEFAULT_TEST_START = date(2023, 1, 1)


def _family_of(name: str) -> str:
    return next(s.family for s in DEFAULT_SPECS if s.name == name)


def _feature_set_of(name: str) -> str:
    return next(s.feature_set for s in DEFAULT_SPECS if s.name == name)


def _single_fold_crps(
    spec: ModelSpec, historical_data: pd.DataFrame, cutoff: date
) -> float:
    """CRPS of one walk-forward fold (fit before ``cutoff``, score D+1)."""
    result = run_backtest(historical_data, [spec], [cutoff], log=False)
    return float(result["crps"].mean())


# --- SARIMA: AIC/BIC order grid ---------------------------------------------


def _sarima_information_criteria(
    target: pd.Series,
    seasonal_period: int,
    order: tuple[int, int, int],
    seasonal: tuple[int, int, int],
) -> tuple[float, float]:
    """``(aic, bic)`` of a SARIMAX fit with ``simple_differencing`` (ticket 11).

    Returns ``(inf, inf)`` when the order fails to converge or raises, so the
    grid skips it instead of aborting.
    """
    try:
        model = SARIMAX(
            target.to_numpy(dtype=float),
            order=order,
            seasonal_order=(*seasonal, seasonal_period),
            simple_differencing=True,
        )
        result = model.fit(disp=False)
    # A grid order can fail for several reasons (singular matrices, LAPACK
    # errors, non-invertible MA roots); any failure means "skip this order".
    except Exception:  # noqa: BLE001
        return float("inf"), float("inf")
    return float(result.aic), float(result.bic)


def tune_sarima(
    historical_data: pd.DataFrame, cutoff: date
) -> dict[str, tuple[int, int, int]]:
    """Pick the SARIMA order minimizing AIC over the grid."""
    index = historical_data.index
    tz = index.tz if isinstance(index, pd.DatetimeIndex) else None
    cutoff_ts = pd.Timestamp(cutoff)
    if tz is not None:
        cutoff_ts = cutoff_ts.tz_localize(tz)
    train = historical_data.loc[index < cutoff_ts]
    target = train["price"]
    step = target.index[-1] - target.index[-2]
    seasonal_period = _seasonal_period(step)
    best_order: tuple[int, int, int] | None = None
    best_seasonal: tuple[int, int, int] | None = None
    best_aic = float("inf")
    for order in _SARIMA_ORDER_GRID:
        for seasonal in _SARIMA_SEASONAL_GRID:
            aic, _bic = _sarima_information_criteria(
                target, seasonal_period, order, seasonal
            )
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_seasonal = seasonal
    # Fall back to the pinned airline default if every order failed to fit.
    if best_order is None or best_seasonal is None:
        return {"order": (0, 1, 1), "seasonal_order": (0, 1, 1)}
    return {"order": best_order, "seasonal_order": best_seasonal}


# --- Optuna objectives (CatBoost, N-BEATS) ----------------------------------


def _catboost_objective(
    trial, historical_data: pd.DataFrame, cutoff: date
) -> float:
    params = {
        "iterations": trial.suggest_int("iterations", 30, 200),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }
    spec = ModelSpec("catboost", "gbdt", "full-features", params, 42)
    return _single_fold_crps(spec, historical_data, cutoff)


def tune_catboost(
    historical_data: pd.DataFrame,
    cutoff: date,
    *,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Bounded Optuna search over the CatBoost hyperparameters."""
    import optuna

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda t: _catboost_objective(t, historical_data, cutoff),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return dict(study.best_params)


def _nbeats_objective(trial, historical_data: pd.DataFrame, cutoff: date) -> float:
    params = {
        "num_stacks": trial.suggest_int("num_stacks", 2, 6),
        "num_blocks": trial.suggest_int("num_blocks", 1, 3),
        "layer_widths": trial.suggest_int("layer_widths", 16, 64),
        "n_epochs": trial.suggest_int("n_epochs", 2, 6),
    }
    spec = ModelSpec("nbeats", "deep", "price-only", params, 42)
    return _single_fold_crps(spec, historical_data, cutoff)


def tune_nbeats(
    historical_data: pd.DataFrame,
    cutoff: date,
    *,
    n_trials: int = 18,
    seed: int = 42,
) -> dict:
    """Bounded Optuna search over the N-BEATS architecture."""
    import optuna

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(
        lambda t: _nbeats_objective(t, historical_data, cutoff),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return dict(study.best_params)


# --- Chronos-2: context-length sweep ----------------------------------------


def sweep_chronos2(
    historical_data: pd.DataFrame,
    cutoff: date,
    *,
    contexts: tuple[int, ...] = CHRONOS_CONTEXT_GRID,
) -> dict:
    """Pick the Chronos-2 context length minimizing single-fold CRPS."""
    best: dict = {"input_chunk_length": contexts[0]}
    best_crps = float("inf")
    for ctx in contexts:
        spec = ModelSpec(
            "chronos2", "foundation", "full-features", {"input_chunk_length": ctx}, 42
        )
        crps = _single_fold_crps(spec, historical_data, cutoff)
        if crps < best_crps:
            best_crps = crps
            best = {"input_chunk_length": ctx}
    return best


# --- secondary table aggregation (pure, unit-tested) ------------------------


def build_tuned_table(
    models: list[str],
    families: dict[str, str],
    default_crps: dict[str, float],
    tuned_crps: dict[str, float],
    hyperparams: dict[str, dict],
) -> pd.DataFrame:
    """Merge default and tuned CRPS into the secondary tuned-vs-default table.

    ``crps_delta = tuned - default`` (negative means the tuned config is
    better). Hyperparameters are serialized so the table is CSV-safe.
    """
    rows = []
    for model in models:
        rows.append(
            {
                "model": model,
                "family": families[model],
                "default_crps": default_crps[model],
                "tuned_crps": tuned_crps[model],
                "crps_delta": tuned_crps[model] - default_crps[model],
                "tuned_hyperparams": json.dumps(hyperparams[model], sort_keys=True),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "family",
            "default_crps",
            "tuned_crps",
            "crps_delta",
            "tuned_hyperparams",
        ],
    )
def _mean_crps_by_model(result: pd.DataFrame) -> dict[str, float]:
    means = result.groupby("model")["crps"].mean()
    return {str(model): float(value) for model, value in means.items()}


def run_tuned_comparison(
    historical_data: pd.DataFrame,
    cutoffs: list[date],
    *,
    tracking_uri: str = "mlruns",
    out_dir: str | Path = "reports",
    n_trials: int = 50,
    nbeats_trials: int = 18,
) -> pd.DataFrame:
    """Tune the four models, run default + tuned across folds, write the table.

    The search scores a validation day at the end of the training window (one
    day before the first fold), so the tuning signal is disjoint from every
    backtest fold and the final delta is not optimistically biased. Both
    default and tuned specs then run across every fold (``log=False``) for a
    like-for-like delta, and the tuned specs re-run with ``log=True`` to record
    the tuned runs in MLflow.
    """
    cutoff = cutoffs[0] - timedelta(days=1)
    searches = {
        "sarima": lambda: tune_sarima(historical_data, cutoff),
        "catboost": lambda: tune_catboost(historical_data, cutoff, n_trials=n_trials),
        "nbeats": lambda: tune_nbeats(
            historical_data, cutoff, n_trials=nbeats_trials
        ),
        "chronos2": lambda: sweep_chronos2(historical_data, cutoff),
    }

    hyperparams: dict[str, dict] = {}
    tuned_specs: list[ModelSpec] = []
    for model in TUNED_MODELS:
        params = searches[model]()
        hyperparams[model] = params
        tuned_specs.append(
            ModelSpec(model, _family_of(model), _feature_set_of(model), params, 42)
        )

    default_specs = [s for s in DEFAULT_SPECS if s.name in TUNED_MODELS]
    families = {m: _family_of(m) for m in TUNED_MODELS}

    # Comparable CRPS across every fold (default and tuned run separately, no
    # MLflow logging). They share model names, so they must be scored in
    # separate passes to keep the default and tuned means distinct.
    default_result = run_backtest(historical_data, default_specs, cutoffs, log=False)
    tuned_result = run_backtest(historical_data, tuned_specs, cutoffs, log=False)
    default_crps = _mean_crps_by_model(default_result)
    tuned_crps = _mean_crps_by_model(tuned_result)

    # Record the tuned runs in MLflow.
    run_backtest(
        historical_data, tuned_specs, cutoffs, tracking_uri=tracking_uri, log=True
    )

    table = build_tuned_table(
        list(TUNED_MODELS), families, default_crps, tuned_crps, hyperparams
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tuned_table.csv", index=False)
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune the best model per family.")
    parser.add_argument("--zones", nargs="+", default=["SE3"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--test-start", default=DEFAULT_TEST_START.isoformat())
    parser.add_argument("--tracking-uri", default="mlruns")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--nbeats-trials", type=int, default=18)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end)
    hourly_end = MTU_15MIN_SWITCH_DATE - pd.Timedelta(days=1)
    historical_data = assemble_data(
        args.zones,
        date.fromisoformat(args.start),
        min(end, hourly_end),
        refresh=args.refresh,
    )
    cutoffs = generate_folds(
        historical_data, test_start=date.fromisoformat(args.test_start)
    )
    table = run_tuned_comparison(
        historical_data,
        cutoffs,
        tracking_uri=args.tracking_uri,
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        nbeats_trials=args.nbeats_trials,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
