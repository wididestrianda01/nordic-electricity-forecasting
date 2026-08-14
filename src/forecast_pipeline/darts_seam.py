"""Ticket 10: shared darts seam.

Thin adapters between the pandas-based arm contract (ticket 07) and the
``darts`` models used by the eight model arms (tickets 11-14). Four helpers:

- ``series_to_time_series`` -- wrap a price ``pd.Series`` as a univariate
  ``darts.TimeSeries``.
- ``frame_to_time_series`` -- wrap a feature ``pd.DataFrame`` as a
  multivariate ``darts.TimeSeries``.
- ``encode_regime`` -- integer-encode the categorical ``regime`` covariate so
  darts (float-only covariates) can consume it.
- ``quantiles_to_frame`` -- collapse a probabilistic ``darts.TimeSeries``
  into the ``[p10, p50, p90]`` frame every arm must return.

darts 0.46 does not support timezone-aware indexes (it strips ``tz`` with a
warning), so the two wrappers localize the index to naive UTC on the way in;
``quantiles_to_frame`` restores the timezone by reindexing onto the caller's
forecast grid.
"""

import numpy as np
import pandas as pd
from darts import TimeSeries

#: Quantile levels in the P14->P16 handoff order (ticket 07 contract).
_QUANTILES = (0.10, 0.50, 0.90)


def _naive_utc(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Return ``obj`` with its DatetimeIndex localized to naive UTC."""
    index = obj.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        obj = obj.copy()
        obj.index = index.tz_localize(None)
    return obj


def encode_regime(series: pd.Series) -> pd.Series:
    """Encode a categorical ``regime`` covariate as numeric floats for darts.

    ``detect_regimes`` emits string labels ``regime_N``; the test fixtures use
    plain numeric labels. Either way, return the integer state index as a float
    so darts (which requires float covariates) can consume it. String labels
    map to their trailing integer, which stays consistent across fit/predict
    subsets (unlike ``pd.factorize``, which re-indexes on the subset).
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.astype(str).str.extract(r"(\d+)$", expand=False).astype(float)


def series_to_time_series(series: pd.Series) -> TimeSeries:
    """Wrap a price ``Series`` as a univariate ``darts.TimeSeries``.

    The index must be a sorted, regular ``DatetimeIndex`` (no gaps); darts
    infers the frequency. A timezone-aware index is localized to naive UTC
    first (darts 0.46 has no timezone support); ``quantiles_to_frame``
    restores it on the way out.
    """
    return TimeSeries.from_series(_naive_utc(series))


def frame_to_time_series(frame: pd.DataFrame) -> TimeSeries:
    """Wrap a feature ``DataFrame`` as a multivariate ``darts.TimeSeries``.

    Each column becomes one component. Same index requirements as
    ``series_to_time_series``. NaN covariates (lag/rolling warm-up and
    boundary-masked cells) are zero-filled: the tree arms handle NaN natively,
    but the neural/foundation darts arms require clean float inputs.
    """
    return TimeSeries.from_dataframe(_naive_utc(frame).fillna(0.0))


def quantiles_to_frame(prob_ts: TimeSeries, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract ``[p10, p50, p90]`` from a probabilistic ``darts.TimeSeries``.

    Uses the darts 0.46 API ``TimeSeries.quantile([0.10, 0.50, 0.90])``
    (the successor to the removed ``quantile_timeseries`` method), which
    returns a deterministic series with one component per quantile. The
    single price component's quantile values are reindexed onto ``index``
    with per-row ordering enforced (``p10 <= p50 <= p90``).

    ``prob_ts`` must be a univariate stochastic series (>= 2 samples) --
    i.e. what a single-price forecasting arm emits.
    """
    q_ts = prob_ts.quantile(list(_QUANTILES))
    values = q_ts.all_values()  # shape (time, n_quantiles, 1)
    p10 = values[:, 0, 0]
    p50 = values[:, 1, 0]
    p90 = values[:, 2, 0]
    # Enforce P10 <= P50 <= P90 per row.
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    frame = pd.DataFrame(
        {"p10": p10, "p50": p50, "p90": p90},
        index=q_ts.time_index,
    )
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None and index.tz is not None:
        frame.index = frame.index.tz_localize(index.tz)
    return frame.reindex(index)
