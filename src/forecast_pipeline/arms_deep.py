"""Ticket 13: deep-learning arms (N-BEATS and TFT).

Two darts torch arms on pinned, CPU-sized budgets:

- ``NbeatsArm`` -- price-only N-BEATS. Fits on the price target alone and
  ignores ``features``/``future_features``; derives its forecast grid one MTU
  step past the target's last timestamp.
- ``TftArm`` -- full-features Temporal Fusion Transformer. Consumes the
  canonical covariate set (minus the day-ahead ``load_forecast``/
  ``wind_forecast`` columns -- ticket 08 skew resolution) as both past and
  future covariates.

Pinned training budget (both arms, so CPU smoke tests finish in seconds):
2 epochs, batch size 8, tiny input/output chunks, low-capacity Gaussian
likelihood sampled 50 times and collapsed to P10/P50/P90 by the shared seam.
The fixed seed (42) is class-level: the registry factory constructs arms with
no constructor arguments.
"""

from __future__ import annotations

import pandas as pd
import torch
from darts.models import NBEATSModel, TFTModel
from darts.utils.likelihood_models import GaussianLikelihood
from pandas.tseries.offsets import BaseOffset

from forecast_pipeline.darts_seam import (
    frame_to_time_series,
    quantiles_to_frame,
    series_to_time_series,
)

#: Pin the torch intra-op thread count to avoid thread-pool oversubscription,
#: which makes these tiny CPU models ~50x slower on many-core machines.
torch.set_num_threads(2)

#: Fixed random seed for every deep arm (factory constructs with no args).
SEED = 42

#: Samples drawn per probabilistic forecast before collapsing to quantiles.
NUM_SAMPLES = 50

#: Quiet CPU lightning trainer: no progress bar, summary, checkpointing, logger.
_PL_TRAINER_KWARGS = {
    "accelerator": "cpu",
    "enable_progress_bar": False,
    "enable_model_summary": False,
    "enable_checkpointing": False,
    "logger": False,
}


class NbeatsArm:
    """Price-only N-BEATS arm.

    Pinned CPU budget: input chunk 12, output chunk 4, 4 stacks, 1 block,
    2 layers of width 16, 2 epochs, batch size 8, Gaussian likelihood.
    """

    feature_set = "price-only"

    INPUT_CHUNK_LENGTH = 12
    OUTPUT_CHUNK_LENGTH = 4
    NUM_STACKS = 4
    NUM_BLOCKS = 1
    NUM_LAYERS = 2
    LAYER_WIDTHS = 16
    EPOCHS = 2
    BATCH_SIZE = 8

    def __init__(self) -> None:
        self._model = NBEATSModel(
            input_chunk_length=self.INPUT_CHUNK_LENGTH,
            output_chunk_length=self.OUTPUT_CHUNK_LENGTH,
            num_stacks=self.NUM_STACKS,
            num_blocks=self.NUM_BLOCKS,
            num_layers=self.NUM_LAYERS,
            layer_widths=self.LAYER_WIDTHS,
            n_epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            likelihood=GaussianLikelihood(),
            random_state=SEED,
            pl_trainer_kwargs=dict(_PL_TRAINER_KWARGS),
        )
        self._series = None
        self._last_ts: pd.Timestamp | None = None
        self._freq: str | BaseOffset | None = None

    def fit(
        self, target: pd.Series, features: pd.DataFrame | None = None
    ) -> NbeatsArm:
        """Fit N-BEATS on the price target; ``features`` is ignored."""
        index = pd.DatetimeIndex(target.index)
        freq = index.freq or pd.infer_freq(index)
        if freq is None:
            raise ValueError("target index must be regular (inferable frequency)")
        self._series = series_to_time_series(target)
        self._model.fit(self._series)
        self._last_ts = target.index[-1]
        self._freq = freq
        return self

    def _forecast_grid(self, horizon: int) -> pd.DatetimeIndex:
        """One MTU step past the last target timestamp, then ``horizon`` steps."""
        if self._last_ts is None or self._freq is None:
            raise ValueError("must call fit() before predict_quantiles()")
        return pd.date_range(self._last_ts, periods=horizon + 1, freq=self._freq)[1:]

    def predict_quantiles(
        self, horizon: int, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Forecast P10/P50/P90 onto the price-only grid."""
        if self._series is None:
            raise ValueError("must call fit() before predict_quantiles()")
        pred = self._model.predict(
            n=horizon, num_samples=NUM_SAMPLES, show_warnings=False
        )
        return quantiles_to_frame(pred, self._forecast_grid(horizon))


class TftArm:
    """Full-features Temporal Fusion Transformer arm.

    Consumes the canonical covariate set (minus day-ahead load/wind) as both
    past and future covariates. Pinned CPU budget: input chunk 8, output
    chunk 2, hidden size 8, 1 LSTM layer, 2 attention heads, 2 epochs,
    batch size 8, Gaussian likelihood.
    """

    feature_set = "full-features"

    INPUT_CHUNK_LENGTH = 8
    OUTPUT_CHUNK_LENGTH = 2
    HIDDEN_SIZE = 8
    LSTM_LAYERS = 1
    NUM_ATTENTION_HEADS = 2
    DROPOUT = 0.1
    EPOCHS = 2
    BATCH_SIZE = 8

    #: Day-ahead covariates excluded from the model (skew resolution, ticket 08).
    EXCLUDED_COVARIATES = frozenset({"load_forecast", "wind_forecast"})

    def __init__(self) -> None:
        self._model = TFTModel(
            input_chunk_length=self.INPUT_CHUNK_LENGTH,
            output_chunk_length=self.OUTPUT_CHUNK_LENGTH,
            hidden_size=self.HIDDEN_SIZE,
            lstm_layers=self.LSTM_LAYERS,
            num_attention_heads=self.NUM_ATTENTION_HEADS,
            dropout=self.DROPOUT,
            n_epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            likelihood=GaussianLikelihood(),
            random_state=SEED,
            pl_trainer_kwargs=dict(_PL_TRAINER_KWARGS),
        )
        self._hist_cov = None

    def _covariates(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in features.columns if c not in self.EXCLUDED_COVARIATES]
        return features[cols]

    def fit(
        self, target: pd.Series, features: pd.DataFrame | None = None
    ) -> TftArm:
        """Fit TFT on the price target with past+future covariates."""
        if features is None:
            raise ValueError("TftArm.fit requires canonical full-features")
        series = series_to_time_series(target)
        hist_cov = frame_to_time_series(self._covariates(features))
        # During training the "future" covariates are the known history values.
        self._model.fit(series, past_covariates=hist_cov, future_covariates=hist_cov)
        self._hist_cov = hist_cov
        return self

    def predict_quantiles(
        self, horizon: int, future_features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Forecast P10/P50/P90 onto the ``future_features`` grid."""
        if future_features is None:
            raise ValueError("TftArm.predict_quantiles requires future_features")
        if self._hist_cov is None:
            raise ValueError("must call fit() before predict_quantiles()")
        future_cov = frame_to_time_series(self._covariates(future_features))
        # Extend the history covariates into the horizon so the autoregressive
        # decoder has past+future covariates at every step.
        full_cov = self._hist_cov.concatenate(future_cov, axis=0)
        pred = self._model.predict(
            n=horizon,
            num_samples=NUM_SAMPLES,
            past_covariates=full_cov,
            future_covariates=full_cov,
            show_warnings=False,
        )
        return quantiles_to_frame(pred, future_features.index)
