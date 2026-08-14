"""Ticket 18: LightGBM feature-group ablation (ticket 11 methodology).

Forward greedy selection + leave-one-group-out over the exogenous feature
groups, with group 1 (AR/calendar/regime) as the always-present base and
group 2 (system fundamentals -- the day-ahead load/wind forecast) excluded
from every full-features arm by design (ticket 08 train/serve skew). The
LightGBM arm also requires the ``regime`` column (group 1), so the base can
never be dropped. The ablation therefore varies groups 3-7 (cross-border,
weather, hydro, commodities, FX); groups 1 and 2 are fixed by construction.

The efficient set is the smallest feature set whose CRPS is within
``epsilon = 1%`` of the full-features CRPS (CRPS <= full_crps * 1.01).
Within-group contribution is ranked by permutation importance (canonical) and
SHAP values (GBDT explainability) on the full set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_pipeline.backtest import DEFAULT_SPECS, run_backtest
from forecast_pipeline.feature_groups import EXCLUDED_GROUP, classify_columns
from forecast_pipeline.features import build_features
from forecast_pipeline.registry import LgbmAdapter, ModelSpec, build_model

EPSILON = 0.01


def _lgbm_spec() -> ModelSpec:
    return next(s for s in DEFAULT_SPECS if s.name == "lgbm")


def evaluate_crps(
    historical_data: pd.DataFrame, cutoffs: list[date], columns: list[str]
) -> float:
    """Mean CRPS of the LightGBM arm over `cutoffs`, restricted to `columns`."""
    result = run_backtest(
        historical_data,
        [_lgbm_spec()],
        cutoffs,
        log=False,
        feature_columns=columns,
    )
    return float(result["crps"].mean())


# --- pure decision logic (unit-testable without a model fit) -----------------


def _forward_path(
    base: list[str],
    candidates: dict[str, list[str]],
    scorer: Callable[[list[str]], float],
) -> list[dict]:
    """Greedy forward selection: add the best-improving group until none helps.

    Returns one step per accepted group, each ``{"columns", "crps", "added"}``;
    the first step is the base set with no group added.
    """
    selected = list(base)
    path: list[dict] = [{"columns": list(selected), "crps": scorer(selected)}]
    remaining = dict(candidates)
    while remaining:
        best: tuple[float, str, list[str]] | None = None
        for name, cols in remaining.items():
            trial = selected + cols
            crps = scorer(trial)
            if best is None or crps < best[0]:
                best = (crps, name, trial)
        if best is None or best[0] >= path[-1]["crps"]:
            break
        selected = best[2]
        path.append({"columns": list(selected), "crps": best[0], "added": best[1]})
        del remaining[best[1]]
    return path


def _efficient_set(
    path: list[dict],
    full_columns: list[str],
    full_crps: float,
    epsilon: float = EPSILON,
) -> list[str]:
    """Smallest set along `path` within `epsilon` of the full-features CRPS."""
    threshold = full_crps * (1.0 + epsilon)
    for step in path:
        if step["crps"] <= threshold:
            return step["columns"]
    return list(full_columns)


def _leave_one_out(
    full_columns: list[str],
    candidates: dict[str, list[str]],
    scorer: Callable[[list[str]], float],
) -> list[dict]:
    """Drop each candidate group from the full set; record the CRPS loss."""
    full_crps = scorer(full_columns)
    rows: list[dict] = []
    for name, cols in candidates.items():
        dropped = [c for c in full_columns if c not in cols]
        dropped_crps = scorer(dropped)
        rows.append(
            {"group": name, "crps": dropped_crps, "loss": dropped_crps - full_crps}
        )
    return rows


# --- importance / explainability ---------------------------------------------


def _fit_full_lgbm(
    historical_data: pd.DataFrame, as_of_date: date, columns: list[str]
) -> tuple[LgbmAdapter, pd.DataFrame, pd.Series]:
    """Fit the LightGBM arm on `columns`; return ``(arm, X, target)``."""
    _, full = build_features(as_of_date, historical_data)
    full = full[columns]
    target = historical_data["price"].reindex(full.index)
    arm = build_model(_lgbm_spec())
    arm.fit(target, full)
    X = arm._covariates(full).copy()
    if "regime" in X.columns:
        X["regime"] = pd.Categorical(X["regime"], categories=arm._regime_categories)
    return arm, X, target


def permutation_importance(
    arm: LgbmAdapter, X: pd.DataFrame, target: pd.Series, n_repeats: int = 5
) -> pd.DataFrame:
    """Permutation importance (mean MAE degradation) of the P50 regressor."""
    from sklearn.inspection import (  # type: ignore[import-untyped]
        permutation_importance,
    )

    result = permutation_importance(
        arm._models[0.5],
        X,
        target,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    frame = pd.DataFrame(
        {"feature": list(X.columns), "importance": -result.importances_mean}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def shap_values(arm: LgbmAdapter, X: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP value per feature from the P50 regressor."""
    import shap  # type: ignore[import-untyped]

    explainer = shap.TreeExplainer(arm._models[0.5])
    values = explainer.shap_values(X)
    mean_abs = np.abs(np.asarray(values)).mean(axis=0)
    frame = pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs})
    return frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# --- orchestration -----------------------------------------------------------


@dataclass
class AblationResult:
    efficient_set: list[str]
    forward_path: list[dict] = field(default_factory=list)
    leave_one_out: list[dict] = field(default_factory=list)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    shap: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_ablation(
    historical_data: pd.DataFrame,
    cutoffs: list[date],
    *,
    epsilon: float = EPSILON,
    out_dir: str | Path = "reports",
) -> AblationResult:
    """Run the full 7-group ablation on LightGBM; write tables to `out_dir`."""
    full_columns = list(build_features(cutoffs[0], historical_data)[1].columns)
    groups = classify_columns(full_columns)

    base = groups["ar_calendar_regime"]
    candidates = {
        name: cols
        for name, cols in groups.items()
        if name not in ("ar_calendar_regime", EXCLUDED_GROUP) and cols
    }

    def scorer(columns: list[str]) -> float:
        return evaluate_crps(historical_data, cutoffs, columns)

    full_crps = scorer(full_columns)
    path = _forward_path(base, candidates, scorer)
    efficient = _efficient_set(path, full_columns, full_crps, epsilon)
    leave_one_out = _leave_one_out(full_columns, candidates, scorer)

    arm, X, target = _fit_full_lgbm(historical_data, cutoffs[0], full_columns)
    importance = permutation_importance(arm, X, target)
    shap = shap_values(arm, X)

    result = AblationResult(
        efficient_set=efficient,
        forward_path=path,
        leave_one_out=leave_one_out,
        importance=importance,
        shap=shap,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(leave_one_out).to_csv(out / "ablation_table.csv", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    shap.to_csv(out / "feature_shap.csv", index=False)
    return result
