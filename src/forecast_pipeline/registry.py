"""Ticket 05: model registry.

A minimal, duck-typed model surface shared by every forecasting arm. Each arm
exposes ``fit(target, features=None)`` and
``predict_quantiles(horizon, future_features=None) -> pd.DataFrame`` returning
``p10``/``p50``/``p90`` columns (ticket 07 contract).

The two v1 arms wrap existing forecasters verbatim (their internals are
deliberately untouched -- see tickets 03 and 04):

- ``LearAdapter``   -- degenerate quantiles from the LEAR point forecast.
- ``LgbmAdapter``   -- regime-conditional LightGBM quantile forecast.

Both wrapped functions build their own features from ``historical_data``, so the
adapters capture ``(as_of_date, historical_data)`` at construction and delegate
unchanged. The ``features``/``future_features`` arguments are accepted for
protocol compatibility with the Phase 2 darts arms but unused here.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from forecast_pipeline.lear import lear_forecast
from forecast_pipeline.lgbm import lgbm_quantile_forecast
from forecast_pipeline.pipeline import _mtu_minutes_for

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
    """Wrap ``lear_forecast`` as a quantile forecaster.

    A point forecast's CRPS collapses to MAE, so the quantile grid is
    degenerate: ``p10 == p50 == p90 ==`` the LEAR point forecast.
    """

    feature_set = "price-only"

    def __init__(self, as_of_date: date, historical_data: pd.DataFrame) -> None:
        self.as_of_date = as_of_date
        self.historical_data = historical_data

    def fit(self, target: Any = None, features: Any = None) -> "LearAdapter":
        """No-op: LEAR fits per MTU slot internally on ``predict_quantiles``.

        ``target``/``features`` are accepted for protocol compatibility but
        unused -- LEAR builds its own price-lag features from ``historical_data``.
        """
        return self

    def predict_quantiles(
        self, horizon: int | None = None, future_features: Any = None
    ) -> pd.DataFrame:
        """Return degenerate quantiles equal to the LEAR point forecast.

        ``horizon``/``future_features`` are accepted for protocol compatibility
        but unused: the wrapped forecaster emits its own one-day grid.
        """
        point = lear_forecast(self.as_of_date, self.historical_data)
        return pd.DataFrame(
            {"p10": point, "p50": point, "p90": point},
            index=point.index,
        )


class LgbmAdapter:
    """Wrap ``lgbm_quantile_forecast`` as a registry arm."""

    feature_set = "full-features"

    def __init__(self, as_of_date: date, historical_data: pd.DataFrame) -> None:
        self.as_of_date = as_of_date
        self.historical_data = historical_data

    def fit(self, target: Any = None, features: Any = None) -> "LgbmAdapter":
        """No-op: LightGBM fits its three quantile models on ``predict_quantiles``.

        ``target``/``features`` are accepted for protocol compatibility but
        unused -- the wrapped forecaster builds its own lag/load/wind/regime
        features from ``historical_data``.
        """
        return self

    def predict_quantiles(
        self, horizon: int | None = None, future_features: Any = None
    ) -> pd.DataFrame:
        """Return the wrapped forecaster's ``p10``/``p50``/``p90`` columns.

        ``horizon``/``future_features`` are accepted for protocol compatibility
        but unused: the wrapped forecaster emits its own one-day grid.
        """
        mtu_minutes = _mtu_minutes_for(self.as_of_date)
        forecast = lgbm_quantile_forecast(self.as_of_date, self.historical_data, mtu_minutes)

        # Enforce P10 <= P50 <= P90 per row (belt-and-suspenders on top of the
        # post-hoc correction already applied inside lgbm_quantile_forecast).
        p10 = forecast["p10"].clip(upper=forecast["p50"])
        p90 = forecast["p90"].clip(lower=forecast["p50"])

        return pd.DataFrame(
            {"p10": p10, "p50": forecast["p50"], "p90": p90},
            index=forecast.index,
        )
