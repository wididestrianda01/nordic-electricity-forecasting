"""Ticket 27: single-page Streamlit bake-off explorer.

Reads the committed snapshot (ticket 24) and renders the full result in one
scroll: overview, Pareto frontier, the ranked headline table, a per-model
drill-down with mechanism explanation, the feature-exploration story, and the
methodology. No API key and no live network.

Run via ``uv run streamlit run app.py``.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from forecast_pipeline.snapshot import headline_ranking, load_results

st.set_page_config(page_title="Nordic electricity forecasting", layout="wide")

SNAPSHOT_DIR = os.environ.get("FORECAST_SNAPSHOT_DIR", "data/snapshot")
SNAPSHOT = load_results(SNAPSHOT_DIR)
RESULTS = SNAPSHOT.get("results", pd.DataFrame())
PARETO = SNAPSHOT.get("pareto_table", pd.DataFrame())
TUNED = SNAPSHOT.get("tuned_table", pd.DataFrame())
CROSS_ZONE = SNAPSHOT.get("cross_zone_summary", pd.DataFrame())
TRANSFER = SNAPSHOT.get("transfer_table", pd.DataFrame())
MARGINAL = SNAPSHOT.get("marginal_value", pd.DataFrame())
IMPORTANCE = SNAPSHOT.get("feature_importance", pd.DataFrame())

#: One-line mechanism per model, plus why it ranks where it does.
MODEL_EXPLANATIONS: dict[str, str] = {
    "sarima": (
        "Seasonal ARIMA with an airline specification, (0,1,1)(0,1,1,m), fit on "
        "the price series alone. It captures daily seasonality and a random-walk "
        "trend but sees no exogenous information, which caps its accuracy below "
        "the feature-driven models."
    ),
    "ets": (
        "Additive Holt-Winters exponential smoothing. The simplest model in the "
        "roster; it tracks level, trend, and daily seasonality but has no way to "
        "use weather or cross-border flows, so it trails the field."
    ),
    "lear": (
        "LASSO-estimated autoregression with exogenous terms (the LEAR model of "
        "Lago et al.). Regularized linear dynamics make it fast and stable, but "
        "the linear form misses the nonlinear price response to system stress."
    ),
    "lgbm": (
        "LightGBM with three quantile regressors on the full feature matrix. "
        "Trees capture nonlinear interactions cheaply; its near-zero training "
        "cost places it on the accuracy-per-compute frontier."
    ),
    "xgboost": (
        "XGBoost with three quantile regressors on the full feature matrix. "
        "Slightly behind LightGBM and CatBoost on this data, still firmly on the "
        "frontier because of its low compute cost."
    ),
    "catboost": (
        "CatBoost with three quantile regressors, treating the regime label as "
        "categorical. The most accurate of the tree models here, and the best "
        "accuracy-per-compute point overall."
    ),
    "nbeats": (
        "A deep neural basis expansion on the price series only. It learns "
        "trend and seasonality basis functions but, with no covariates, it "
        "cannot beat the feature-driven trees and costs much more to train."
    ),
    "tft": (
        "A temporal fusion transformer over the full feature set. The richest "
        "model in the roster, but it is dominated on both axes here: it is less "
        "accurate than the trees and far more expensive than Chronos-2."
    ),
    "chronos2": (
        "A pretrained zero-shot foundation model (Chronos-2) with past and future "
        "covariates. It edges the trees on mean CRPS by about 0.5% but costs "
        "roughly 1500x the compute, so it is accuracy-only, not on the frontier."
    ),
    "timesfm": (
        "A pretrained zero-shot foundation model (TimesFM 2.5), price-only. "
        "Strong on the hourly regime but it carries no covariates and is "
        "dominated by the feature-driven trees on the 15-minute regime."
    ),
}


def _pareto_frame() -> pd.DataFrame:
    if PARETO.empty:
        return pd.DataFrame()
    frame = PARETO.copy()
    frame["log_compute"] = pd.to_numeric(frame["total_compute_s"], errors="coerce").apply(
        lambda x: _log1p(x)
    )
    return frame


def _log1p(x: float) -> float:
    import math

    return math.log1p(x) if math.isfinite(x) else float("nan")


st.title("Nordic electricity price forecasting")
st.caption(
    "A ten-model comparison across seven feature groups, ranked on a Pareto "
    "accuracy-versus-compute basis."
)

# --- overview ----------------------------------------------------------------
st.header("Overview")
if not RESULTS.empty:
    ranking = headline_ranking(RESULTS)
    best = ranking.iloc[0]
    st.write(
        f"Across the hourly and 15-minute regimes, **{best['model']}** leads with "
        f"a mean CRPS of **{best['mean_crps']:.2f}**. The gradient-boosted trees "
        f"(CatBoost, LightGBM, XGBoost) dominate the accuracy-per-compute frontier; "
        f"Chronos-2 edges accuracy at roughly 1500x the cost; the deep and classical "
        f"models are dominated on both axes."
    )
else:
    st.warning("Snapshot not found. Run `uv run python -m forecast_pipeline.snapshot --build`.")

# --- Pareto scatter ----------------------------------------------------------
st.header("Pareto frontier")
pareto = _pareto_frame()
if not pareto.empty:
    pareto["regime_label"] = pareto["regime"].fillna("mixed")
    pareto["frontier"] = pareto["pareto_optimal"].map(
        {True: "on frontier", False: "dominated"}
    )
    chart = st.scatter_chart(
        pareto,
        x="log_compute",
        y="CRPS",
        color="regime_label",
        size=40,
    )
    st.caption(
        "Lower CRPS (more accurate) and lower log-compute are better. The frontier "
        "flag is computed within each regime (ticket 20), so a model cheap in one "
        "regime does not dominate a model in the other."
    )
    st.dataframe(
        pareto[
            [
                "model",
                "family",
                "regime",
                "n_folds",
                "CRPS",
                "MAE",
                "total_compute_s",
                "pareto_optimal",
            ]
        ].sort_values(["regime", "CRPS"]),
        use_container_width=True,
        hide_index=True,
    )

# --- ranked headline table --------------------------------------------------
st.header("Ranked headline table")
if not RESULTS.empty:
    ranking = headline_ranking(RESULTS)
    ranking = ranking.rename(columns={"mean_crps": "mean CRPS"})
    st.dataframe(ranking, use_container_width=True, hide_index=True)

# --- per-model drill-down ----------------------------------------------------
st.header("Model drill-down")
if not RESULTS.empty:
    model = st.selectbox("Model", sorted(RESULTS["model"].dropna().unique()))
    st.write(MODEL_EXPLANATIONS.get(model, ""))
    per_regime = (
        RESULTS[RESULTS["model"] == model]
        .groupby("mtu_minutes")
        .agg(crps=("crps", "mean"), mae=("mae", "mean"))
        .reset_index()
    )
    st.dataframe(per_regime, use_container_width=True, hide_index=True)

# --- feature exploration -----------------------------------------------------
st.header("Feature exploration")
if not MARGINAL.empty:
    st.subheader("Marginal value of each feature group")
    st.write(
        "Forward selection adds the group that most reduces CRPS, stopping when "
        "no group helps. A negative marginal delta means the group improved CRPS."
    )
    st.dataframe(MARGINAL, use_container_width=True, hide_index=True)
if not IMPORTANCE.empty:
    st.subheader("Permutation importance")
    st.dataframe(IMPORTANCE.head(15), use_container_width=True, hide_index=True)
if not TRANSFER.empty:
    st.subheader("Efficient-set transfer")
    st.write(
        "Each full-features arm is re-run on the price-only base versus the "
        "efficient set found on LightGBM. A negative delta means the efficient "
        "set earns its cost for that model too."
    )
    st.dataframe(TRANSFER, use_container_width=True, hide_index=True)
if not CROSS_ZONE.empty:
    st.subheader("Cross-zone robustness")
    st.dataframe(CROSS_ZONE, use_container_width=True, hide_index=True)
if not TUNED.empty:
    st.subheader("Tuned versus default")
    st.dataframe(TUNED, use_container_width=True, hide_index=True)

# --- methodology -------------------------------------------------------------
st.header("Methodology")
st.write(
    "Ten models across four families (classical, gradient-boosted, deep, "
    "foundation) are compared on the Nord Pool SE3 day-ahead price, with the "
    "cross-zone run extending the frontier to SE1-SE4. The backtest uses an "
    "expanding walk-forward window with yearly refits and a 7-day purge; the "
    "hourly regime runs 2023-01-01 to 2025-09-30 and the 15-minute regime runs "
    "from the 2025-10-01 market switch onward. The primary metric is mean CRPS, "
    "with pinball loss and MAE reported alongside. Feature groups follow the "
    "as-of timing rule so no covariate leaks future information."
)
