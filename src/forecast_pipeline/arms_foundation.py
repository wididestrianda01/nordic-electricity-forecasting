"""Ticket 14: foundation-model arms (Chronos-2, TimesFM 2.5).

Zero-shot, inference-only arms over the darts foundation models. Neither arm
trains: ``fit`` only records the inputs, and the underlying darts model (with
its pretrained weights) is instantiated lazily on the first
``predict_quantiles`` call, so importing this module never touches the network.

Both arms emit native P10/P50/P90 quantiles: the spread comes from the model's
own predictive distribution, sampled ``NUM_SAMPLES`` times and reduced to the
three quantile levels by the shared ``quantiles_to_frame`` seam (ticket 10).
No post-hoc synthetic band is added.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from darts.models import Chronos2Model, TimesFM2p5Model
from darts.utils.likelihood_models import QuantileRegression

from forecast_pipeline.darts_seam import (
    encode_regime,
    frame_to_time_series,
    quantiles_to_frame,
    series_to_time_series,
)

#: Fixed RNG seed for reproducible sampling (ticket 07 contract).
SEED = 42

#: Native quantile levels every foundation arm returns.
QUANTILES = (0.1, 0.5, 0.9)

#: Monte Carlo draws reduced to ``[p10, p50, p90]`` by ``quantiles_to_frame``.
NUM_SAMPLES = 200

#: Day-ahead covariates excluded from every full-features arm (ticket 08).
EXCLUDED_COVARIATES = frozenset({"load_forecast", "wind_forecast"})


class ChronosArm:
    """Chronos-2 zero-shot arm (full-features).

    Chronos-2 natively supports past and future covariates. Past covariates
    (the canonical full-features history minus ``load_forecast``/
    ``wind_forecast``) are always fed. Future covariates are fed only when
    they span the model's ``output_chunk_length`` (96) -- i.e. the 15-minute
    regime; for the hourly regime (24 rows) they are omitted. The forecast
    grid is taken from ``future_features.index``.

    The ``regime`` label is factorized to integers (darts covariates must be
    numeric); NaN lags/rolling features in the warm-up period are passed
    through untouched.
    """

    feature_set = "full-features"

    INPUT_CHUNK_LENGTH = 512
    OUTPUT_CHUNK_LENGTH = 96
    QUANTILES = QUANTILES
    NUM_SAMPLES = NUM_SAMPLES
    EXCLUDED_COVARIATES = EXCLUDED_COVARIATES

    def __init__(self) -> None:
        self._model: Chronos2Model | None = None
        self._target: pd.Series | None = None
        self._past_covariates: pd.DataFrame | None = None

    @staticmethod
    def _covariates(features: pd.DataFrame) -> pd.DataFrame:
        """Select covariates and integer-encode the categorical ``regime``."""
        cols = [c for c in features.columns if c not in EXCLUDED_COVARIATES]
        frame = features[cols]
        if "regime" in frame.columns:
            frame["regime"] = encode_regime(frame["regime"])
        return frame

    def _ensure_model(self) -> Chronos2Model:
        if self._model is None:
            self._model = Chronos2Model(
                input_chunk_length=self.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.OUTPUT_CHUNK_LENGTH,
                likelihood=QuantileRegression(quantiles=list(self.QUANTILES)),
                pl_trainer_kwargs={"accelerator": "cpu"},
            )
        return self._model

    def fit(
        self, target: pd.Series, features: pd.DataFrame | None = None
    ) -> ChronosArm:
        """Record the target and past covariates; no training is performed."""
        if features is None:
            raise ValueError("ChronosArm.fit requires canonical full-features")
        self._target = target
        self._past_covariates = self._covariates(features)
        return self

    def predict_quantiles(
        self, horizon: int, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Forecast native P10/P50/P90 on the ``future_features`` grid."""
        if self._target is None or self._past_covariates is None:
            raise ValueError("must call fit() before predict_quantiles()")
        if future_features is None:
            raise ValueError("ChronosArm.predict_quantiles requires future_features")
        model = self._ensure_model()
        series = series_to_time_series(self._target)
        past_covariates = frame_to_time_series(self._past_covariates)
        future_covariates = None
        if len(future_features) >= self.OUTPUT_CHUNK_LENGTH:
            horizon_cov = frame_to_time_series(self._covariates(future_features))
            future_covariates = past_covariates.concatenate(horizon_cov, axis=0)
        # Zero-shot: fit(epochs=0) registers the series (and loads weights) but
        # does not train; darts still requires fit() before predict(). Future
        # covariates are passed to fit so the model records historic future
        # covariates and accepts them at predict time.
        model.fit(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        prob_ts = model.predict(
            n=horizon,
            num_samples=self.NUM_SAMPLES,
            random_state=SEED,
        )
        grid = cast(pd.DatetimeIndex, future_features.index)
        return quantiles_to_frame(prob_ts, grid)


class TimesfmArm:
    """TimesFM 2.5 zero-shot arm (price-only).

    TimesFM 2.5 supports univariate/multivariate series but no covariates, so
    the arm is strictly univariate and ignores ``features``/``future_features``.
    The forecast grid is derived as one MTU step past the target's last
    timestamp, then ``horizon`` steps at the target frequency.
    """

    feature_set = "price-only"

    INPUT_CHUNK_LENGTH = 512
    OUTPUT_CHUNK_LENGTH = 96
    QUANTILES = QUANTILES
    NUM_SAMPLES = NUM_SAMPLES

    def __init__(self) -> None:
        self._model: TimesFM2p5Model | None = None
        self._target: pd.Series | None = None

    def _ensure_model(self) -> TimesFM2p5Model:
        if self._model is None:
            self._model = TimesFM2p5Model(
                input_chunk_length=self.INPUT_CHUNK_LENGTH,
                output_chunk_length=self.OUTPUT_CHUNK_LENGTH,
                likelihood=QuantileRegression(quantiles=list(self.QUANTILES)),
            )
        return self._model

    def fit(self, target: pd.Series, features: pd.DataFrame | None = None) -> TimesfmArm:
        """Record the target; ``features`` are accepted and ignored."""
        self._target = target
        return self

    def _forecast_grid(self, horizon: int) -> pd.DatetimeIndex:
        target = self._target
        assert target is not None  # guarded by predict_quantiles
        step = target.index[-1] - target.index[-2]
        return pd.date_range(start=target.index[-1] + step, periods=horizon, freq=step)

    def predict_quantiles(
        self, horizon: int, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        if self._target is None:
            raise ValueError("must call fit() before predict_quantiles()")
        model = self._ensure_model()
        series = series_to_time_series(self._target)
        # Zero-shot: fit(epochs=0) registers the series (and loads weights) but
        # does not train; darts still requires fit() before predict().
        model.fit(series)
        prob_ts = model.predict(
            n=horizon,
            num_samples=self.NUM_SAMPLES,
            random_state=SEED,
        )
        return quantiles_to_frame(prob_ts, self._forecast_grid(horizon))
