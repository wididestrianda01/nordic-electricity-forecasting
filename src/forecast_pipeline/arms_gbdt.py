"""Ticket 12: native gradient-boosted full-features arms (XGBoost, CatBoost).

Mirrors ``LgbmAdapter`` (ticket 08): no-arg constructors, pinned
hyperparameters, fixed seed 42. Each arm trains three quantile regressors
(0.1/0.5/0.9) on the canonical full-features matrix minus the day-ahead
``load_forecast``/``wind_forecast`` covariates; NaN lags/rolling features are
handled natively by both libraries.

The ``regime`` label is the one categorical covariate: it is re-encoded as a
deterministic ``pandas.Categorical`` (ordered by sorted training categories)
so XGBoost (``enable_categorical``) and CatBoost (``cat_features``) treat it
as categorical, not numeric. Missing labels map to a ``__missing__`` sentinel
because CatBoost rejects NaN inside a categorical feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


class _GbtQuantileArm:
    """Shared fit/predict machinery for the native gradient-boosted arms."""

    feature_set = "full-features"
    QUANTILES = (0.1, 0.5, 0.9)
    SEED = 42
    #: Day-ahead covariates excluded from every full-features arm (ticket 08).
    EXCLUDED_COVARIATES = frozenset({"load_forecast", "wind_forecast"})
    _MISSING_REGIME = "__missing__"

    def __init__(self) -> None:
        self._models: dict[float, Any] = {}
        self._covariate_columns: list[str] | None = None
        self._regime_categories: list[str] | None = None

    def _make_model(self, alpha: float) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fit_model(self, model: Any, X: pd.DataFrame, target: pd.Series) -> None:
        model.fit(X, target)

    def _covariates(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in features.columns if c not in self.EXCLUDED_COVARIATES]
        return features[cols]

    def _encode_regime(self, X: pd.DataFrame) -> pd.DataFrame:
        """Re-encode ``regime`` as a deterministic categorical (train/serve-safe)."""
        if self._regime_categories is None:
            raise ValueError("must call fit() before encoding regime covariates")
        X = X.copy()
        categories = [*self._regime_categories, self._MISSING_REGIME]
        regime = X["regime"].astype("string").fillna(self._MISSING_REGIME)
        X["regime"] = pd.Categorical(regime, categories=categories)
        return X

    def fit(
        self, target: pd.Series, features: pd.DataFrame | None = None
    ) -> _GbtQuantileArm:
        """Train one quantile regressor per quantile on the full-features matrix."""
        if features is None:
            raise ValueError(f"{type(self).__name__}.fit requires canonical full-features")
        X = self._covariates(features).copy()
        self._regime_categories = sorted(X["regime"].dropna().unique())
        X = self._encode_regime(X)
        self._covariate_columns = list(X.columns)
        for alpha in self.QUANTILES:
            model = self._make_model(alpha)
            self._fit_model(model, X, target)
            self._models[alpha] = model
        return self

    def predict_quantiles(
        self, horizon: int | None = None, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Predict P10/P50/P90 on the ``future_features`` grid.

        ``horizon`` is accepted for protocol compatibility; the grid is
        carried by ``future_features``.
        """
        if future_features is None:
            raise ValueError(
                f"{type(self).__name__}.predict_quantiles requires future_features"
            )
        if not self._models:
            raise ValueError("must call fit() before predict_quantiles()")
        X = self._encode_regime(self._covariates(future_features))
        p10 = self._models[0.1].predict(X)
        p50 = self._models[0.5].predict(X)
        p90 = self._models[0.9].predict(X)
        # Enforce P10 <= P50 <= P90 per row.
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
        return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=X.index)


class XgboostArm(_GbtQuantileArm):
    """XGBoost quantile arm: three ``XGBRegressor`` (objective quantile)."""

    def _make_model(self, alpha: float) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            enable_categorical=True,
            random_state=self.SEED,
            n_jobs=1,
        )


class CatboostArm(_GbtQuantileArm):
    """CatBoost quantile arm: three ``CatBoostRegressor`` (Quantile loss)."""

    def __init__(
        self,
        iterations: int | None = None,
        depth: int | None = None,
        learning_rate: float | None = None,
        l2_leaf_reg: float | None = None,
    ) -> None:
        super().__init__()
        self._iterations = iterations
        self._depth = depth
        self._learning_rate = learning_rate
        self._l2_leaf_reg = l2_leaf_reg

    def _make_model(self, alpha: float) -> CatBoostRegressor:
        params: dict[str, Any] = {
            "loss_function": f"Quantile:alpha={alpha}",
            "iterations": self._iterations if self._iterations is not None else 50,
            "depth": self._depth if self._depth is not None else 4,
            "learning_rate": self._learning_rate if self._learning_rate is not None else 0.1,
            "random_seed": self.SEED,
            "thread_count": 1,
            "verbose": False,
        }
        if self._l2_leaf_reg is not None:
            params["l2_leaf_reg"] = self._l2_leaf_reg
        return CatBoostRegressor(**params)

    def _fit_model(
        self, model: CatBoostRegressor, X: pd.DataFrame, target: pd.Series
    ) -> None:
        model.fit(X, target, cat_features=["regime"])
