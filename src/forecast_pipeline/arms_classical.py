"""Ticket 11: classical forecasting arms (SARIMA, ETS).

Two price-only arms built on darts. Both follow the uniform arm convention
(ticket 10): no constructor args, data flows through ``fit`` /
``predict_quantiles``, and hyperparameters plus the fixed seed are pinned
class-level.

``SarimaArm`` wraps ``darts.models.ARIMA`` with a hand-specified (not
auto-tuned) airline specification. The one-day seasonal period is inferred
from the target frequency: 24 steps for an hourly MTU, 96 steps for a
15-minute MTU. statsmodels' state-space SARIMAX is prohibitively slow at
``m = 96``, so the arm uses a thin subclass that enables statsmodels'
``simple_differencing`` -- a pure performance flag that leaves the model
specification unchanged while making the 15-minute fit tractable.

``EtsArm`` wraps ``darts.models.ExponentialSmoothing`` configured as additive
Holt-Winters (additive trend + additive seasonality).
"""
from __future__ import annotations

from typing import TypeAlias

import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA as _DartsArima
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode, SeasonalityMode
from statsmodels.tsa.statespace.sarimax import SARIMAX as _Sarimax

from forecast_pipeline.darts_seam import quantiles_to_frame, series_to_time_series

#: Seconds in one day, used to infer the one-day seasonal period from the
#: target frequency (24 steps hourly, 96 steps for a 15-minute MTU).
_DAY_SECONDS = 24 * 60 * 60

#: Any darts model used by the price-only classical arms.
_ClassicalModel: TypeAlias = _DartsArima | ExponentialSmoothing


class _SimpleDiffArima(_DartsArima):
    """``darts.models.ARIMA`` with statsmodels ``simple_differencing`` on.

    ``simple_differencing`` differences the data before the state-space
    representation. For the 15-minute MTU the one-day seasonal period is 96
    steps, and the default (non-differenced) state-space fit does not finish
    in usable time; this flag gives the same model specification an order of
    magnitude faster fit.
    """

    def _fit(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries | None = None,
        verbose: bool | None = None,
    ) -> _SimpleDiffArima:
        # Replicate darts' ARIMA._fit (grandparent setup + univariate assert +
        # statsmodels SARIMAX fit), but with ``simple_differencing=True``.
        super(_DartsArima, self)._fit(series, future_covariates, verbose=verbose)
        self._assert_univariate(series)
        self.training_historic_future_covariates = future_covariates
        model = _Sarimax(
            series.values(copy=False),
            exog=future_covariates.values(copy=False) if future_covariates else None,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            simple_differencing=True,
        )
        self.model = model.fit(disp=False)
        return self


def _seasonal_period(step: pd.Timedelta) -> int:
    """Infer the one-day seasonal period (in MTU steps) from the target step."""
    return round(_DAY_SECONDS / step.total_seconds())


class _PriceOnlyArm:
    """Shared plumbing for the price-only classical arms (ticket 11).

    Both arms derive their forecast grid from the stored target -- one MTU
    step past the target's last timestamp, then ``horizon`` steps at the
    target frequency -- and collapse a probabilistic darts forecast onto it
    with ``quantiles_to_frame``. Subclasses pin their model specification and
    build the darts model in ``_build_model``.
    """

    feature_set = "price-only"
    #: Fixed seed for reproducible probabilistic forecasts (ticket 10).
    SEED = 42
    #: Number of Monte Carlo samples drawn per probabilistic forecast.
    NUM_SAMPLES = 200

    def __init__(self) -> None:
        self._target: pd.Series | None = None
        self._model: _ClassicalModel | None = None

    def fit(self, target: pd.Series, features: pd.DataFrame | None = None) -> _PriceOnlyArm:
        """Fit the darts model on the price target; ``features`` are ignored."""
        self._target = target
        step = target.index[-1] - target.index[-2]
        seasonal_period = _seasonal_period(step)
        series = series_to_time_series(target)
        self._model = self._build_model(seasonal_period)
        self._model.fit(series)
        return self

    def predict_quantiles(
        self, horizon: int, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Forecast P10/P50/P90 onto the price-only grid (``future_features`` ignored)."""
        if self._model is None or self._target is None:
            raise ValueError("must call fit() before predict_quantiles()")
        target = self._target
        step = target.index[-1] - target.index[-2]
        grid = pd.date_range(target.index[-1] + step, periods=horizon, freq=step)
        prob = self._model.predict(
            n=horizon, num_samples=self.NUM_SAMPLES, random_state=self.SEED
        )
        return quantiles_to_frame(prob, grid)

    def _build_model(self, seasonal_period: int) -> _ClassicalModel:
        raise NotImplementedError  # pragma: no cover


class SarimaArm(_PriceOnlyArm):
    """Seasonal ARIMA (airline model) over the price series.

    Hand-specified (not auto-tuned) airline specification: ``order=(0, 1, 1)``
    with a seasonal ``(0, 1, 1, m)`` component, where ``m`` is the one-day
    seasonal period inferred from the target frequency. ``order``/
    ``seasonal_order`` overrides (ticket 22) replace the pinned defaults.
    """

    ORDER = (0, 1, 1)  # (p, d, q)
    SEASONAL_ORDER = (0, 1, 1)  # (P, D, Q); the period m is inferred.

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        if order is not None:
            self.ORDER = order
        if seasonal_order is not None:
            self.SEASONAL_ORDER = seasonal_order

    def _build_model(self, seasonal_period: int) -> _ClassicalModel:
        p, d, q = self.ORDER
        seasonal = (*self.SEASONAL_ORDER, seasonal_period)
        return _SimpleDiffArima(
            p=p, d=d, q=q, seasonal_order=seasonal, random_state=self.SEED
        )


class EtsArm(_PriceOnlyArm):
    """Additive Holt-Winters exponential smoothing over the price series."""

    TREND = ModelMode.ADDITIVE
    SEASONAL = SeasonalityMode.ADDITIVE

    def _build_model(self, seasonal_period: int) -> _ClassicalModel:
        return ExponentialSmoothing(
            trend=self.TREND,
            damped=False,
            seasonal=self.SEASONAL,
            seasonal_periods=seasonal_period,
            random_state=self.SEED,
        )
