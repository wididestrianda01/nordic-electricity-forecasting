"""Ticket 05: model registry.

A minimal, duck-typed model surface shared by every forecasting arm. Each arm
exposes ``fit(target, features=None)`` and
``predict_quantiles(horizon, future_features=None) -> pd.DataFrame`` returning
``p10``/``p50``/``p90`` columns (ticket 07 contract).

The two v1 arms are:

- ``LearAdapter``   -- degenerate quantiles from the LEAR point forecast.
  ``fit`` stores the price target; ``predict_quantiles`` derives the
  ``as_of_date`` (one MTU step past the target's last timestamp) and the
  price-only history frame, then wraps ``lear_forecast`` verbatim.
- ``LgbmAdapter``   -- regime-conditional LightGBM quantile forecast over the
  canonical feature matrices: ``fit(target, features)`` trains three quantile
  regressors on the full-features matrix and ``predict_quantiles(horizon,
  future_features)`` predicts on the ``build_horizon_features`` grid (ticket 08).

``build_model`` (ticket 10) dispatches a ``ModelSpec`` to its arm instance;
the eight darts arms are imported lazily so this module still imports before
their home modules exist (tickets 11-14).
"""

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from forecast_pipeline.lear import lear_forecast

#: Columns every arm in the registry must return from ``predict_quantiles``.
QUANTILE_COLUMNS = ("p10", "p50", "p90")


@dataclass(frozen=True)
class ModelSpec:
    """Registry metadata describing one forecasting arm."""

    name: str
    family: str
    feature_set: str
    hyperparams: dict[str, Any]
    seed: int | None


class LearAdapter:
    """Wrap ``lear_forecast`` as a degenerate quantile forecaster.

    A point forecast's CRPS collapses to MAE, so the quantile grid is
    degenerate: ``p10 == p50 == p90 ==`` the LEAR point forecast.

    Follows the uniform arm convention (ticket 10): no constructor args, data
    flows through ``fit``/``predict_quantiles``. ``predict_quantiles`` builds
    the price-only history frame from the stored target and derives the
    ``as_of_date`` as one MTU step past the target's last timestamp.
    """

    feature_set = "price-only"

    def __init__(self) -> None:
        self._target: pd.Series | None = None

    def fit(self, target: pd.Series, features: Any = None) -> "LearAdapter":
        """Store the price target; ``features`` are accepted and ignored."""
        if len(target) < 2:
            raise ValueError("LearAdapter.fit requires at least two target rows")
        self._target = target
        return self

    def predict_quantiles(
        self, horizon: int | None = None, future_features: Any = None
    ) -> pd.DataFrame:
        """Return degenerate quantiles equal to the LEAR point forecast.

        ``future_features`` is accepted for protocol compatibility but unused.
        ``horizon`` bounds the returned grid: LEAR emits one full forecast day
        (24 or 96 rows); when ``horizon`` is given and shorter, the first
        ``horizon`` rows are returned. A ``horizon`` longer than the full day
        is rejected.
        """
        if self._target is None:
            raise ValueError("must call fit() before predict_quantiles()")
        target = self._target
        step = target.index[-1] - target.index[-2]
        as_of_date = (target.index[-1] + step).date()
        point = lear_forecast(as_of_date, target.to_frame("price"))
        if horizon is not None and horizon > len(point):
            raise ValueError(
                f"LearAdapter horizon {horizon} exceeds its full-day grid of "
                f"{len(point)} rows"
            )
        if horizon is not None:
            point = point.iloc[:horizon]
        return pd.DataFrame(
            {"p10": point, "p50": point, "p90": point},
            index=point.index,
        )


class LgbmAdapter:
    """LightGBM quantile arm over the canonical full-features matrices.

    ``fit(target, features)`` trains three LightGBM quantile regressors
    (P10/P50/P90) on the full-features matrix; ``predict_quantiles(horizon,
    future_features)`` predicts on the ``build_horizon_features`` grid.

    ``load_forecast``/``wind_forecast`` are excluded from the covariates: they
    are day-ahead values that cannot be known for the D+1 horizon without a
    separate fetch, and mean-filling them (the legacy ``lgbm_quantile_forecast``
    path) creates a train/serve skew (ticket 08). NaN lags/rolling features are
    handled natively by LightGBM.
    """

    feature_set = "full-features"

    QUANTILES = (0.1, 0.5, 0.9)
    #: Day-ahead covariates excluded from the model (skew resolution, ticket 08).
    EXCLUDED_COVARIATES = frozenset({"load_forecast", "wind_forecast"})

    def __init__(self) -> None:
        self._models: dict[float, LGBMRegressor] = {}
        self._covariate_columns: list[str] | None = None
        self._regime_categories: list[str] | None = None

    def _covariates(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in features.columns if c not in self.EXCLUDED_COVARIATES]
        return features[cols]

    def fit(self, target: pd.Series, features: pd.DataFrame | None = None) -> "LgbmAdapter":
        """Train one LightGBM quantile regressor per quantile on ``features``."""
        if features is None:
            raise ValueError("LgbmAdapter.fit requires canonical full-features")
        X = self._covariates(features).copy()
        self._regime_categories = sorted(X["regime"].dropna().unique())
        X["regime"] = pd.Categorical(X["regime"], categories=self._regime_categories)
        self._covariate_columns = list(X.columns)
        for alpha in self.QUANTILES:
            model = LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                num_leaves=8,
                min_child_samples=5,
                n_estimators=50,
                random_state=42,
                verbose=-1,
            )
            # pandas categorical columns are auto-detected by LightGBM.
            model.fit(X, target)
            self._models[alpha] = model
        return self

    def predict_quantiles(
        self, horizon: int | None = None, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Predict P10/P50/P90 on the ``build_horizon_features`` grid.

        ``horizon`` is accepted for protocol compatibility; the grid is carried
        by ``future_features``.
        """
        if future_features is None:
            raise ValueError("LgbmAdapter.predict_quantiles requires future_features")
        if not self._models:
            raise ValueError("must call fit() before predict_quantiles()")
        X = self._covariates(future_features).copy()
        if "regime" in X.columns:
            X["regime"] = pd.Categorical(X["regime"], categories=self._regime_categories)
        p10 = self._models[0.1].predict(X)
        p50 = self._models[0.5].predict(X)
        p90 = self._models[0.9].predict(X)
        # Enforce P10 <= P50 <= P90 per row.
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)
        return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=X.index)


#: Dispatch table: arm name -> (required feature_set, module path, class name).
#:
#: ``module`` is ``None`` for the inline v1 adapters; otherwise it names the
#: module the arm class lives in, imported lazily inside ``build_model`` so this
#: module imports before the darts arm modules (tickets 11-14) exist. The
#: required ``feature_set`` mirrors each arm's class-level ``feature_set``.
_ARM_REGISTRY: dict[str, tuple[str, str | None, str]] = {
    "lear": ("price-only", None, "LearAdapter"),
    "lgbm": ("full-features", None, "LgbmAdapter"),
    "sarima": ("price-only", "forecast_pipeline.arms_classical", "SarimaArm"),
    "ets": ("price-only", "forecast_pipeline.arms_classical", "EtsArm"),
    "xgboost": ("full-features", "forecast_pipeline.arms_gbdt", "XgboostArm"),
    "catboost": ("full-features", "forecast_pipeline.arms_gbdt", "CatboostArm"),
    "nbeats": ("price-only", "forecast_pipeline.arms_deep", "NbeatsArm"),
    "tft": ("full-features", "forecast_pipeline.arms_deep", "TftArm"),
    "chronos2": ("full-features", "forecast_pipeline.arms_foundation", "ChronosArm"),
    "timesfm": ("price-only", "forecast_pipeline.arms_foundation", "TimesfmArm"),
}


def build_model(spec: ModelSpec) -> Any:
    """Dispatch a ``ModelSpec`` to its arm instance (ticket 10).

    ``lear``/``lgbm`` return the inline adapters. The eight darts arms are
    imported lazily (inside this function) so ``registry`` imports before
    their home modules exist. ``spec.feature_set`` must match the arm's
    class-level ``feature_set``, otherwise ``ValueError``.

    ``spec.hyperparams`` are passed as constructor keyword arguments so a
    tuned spec (ticket 22) overrides the arm's pinned defaults. Arms that do
    not accept tuning receive an empty dict (the default for every roster
    spec), so this is a no-op for the committed comparison.
    """
    entry = _ARM_REGISTRY.get(spec.name)
    if entry is None:
        raise ValueError(f"unknown model name: {spec.name!r}")
    feature_set, module, class_name = entry
    if spec.feature_set != feature_set:
        raise ValueError(
            f"feature_set mismatch for {spec.name!r}: "
            f"spec declares {spec.feature_set!r}, arm requires {feature_set!r}"
        )
    if module is None:
        arm_class = globals()[class_name]
    else:
        arm_class = getattr(importlib.import_module(module), class_name)
    return arm_class(**spec.hyperparams)
