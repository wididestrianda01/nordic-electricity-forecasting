"""Tests for the shared darts seam (ticket 10)."""

import numpy as np
import pandas as pd
from darts import TimeSeries

from forecast_pipeline.darts_seam import (
    encode_regime,
    frame_to_time_series,
    quantiles_to_frame,
    series_to_time_series,
)


def test_encode_regime_maps_string_labels_to_integers():
    idx = pd.date_range("2024-06-15", periods=4, freq="h")
    series = pd.Series(["regime_0", "regime_1", "regime_1", "regime_0"], index=idx)

    encoded = encode_regime(series)

    assert list(encoded) == [0.0, 1.0, 1.0, 0.0]
    assert encoded.dtype == float


def test_encode_regime_preserves_numeric_labels():
    idx = pd.date_range("2024-06-15", periods=3, freq="h")
    series = pd.Series([0.0, 2.0, 1.0], index=idx)

    encoded = encode_regime(series)

    assert list(encoded) == [0.0, 2.0, 1.0]

def test_series_to_time_series_wraps_hourly_price():
    idx = pd.date_range("2024-06-15", periods=48, freq="h", tz="UTC")
    price = pd.Series(np.arange(48, dtype=float), index=idx, name="price")

    ts = series_to_time_series(price)

    assert isinstance(ts, TimeSeries)
    assert ts.n_components == 1
    assert ts.components[0] == "price"
    assert len(ts) == 48
    # darts 0.46 has no timezone support; the seam localizes to naive UTC.
    assert ts.time_index.tz is None
    assert ts.time_index.equals(idx.tz_localize(None))


def test_series_to_time_series_preserves_15min_frequency():
    idx = pd.date_range("2025-10-15", periods=96, freq="15min", tz="UTC")
    price = pd.Series(np.arange(96, dtype=float), index=idx, name="price")

    ts = series_to_time_series(price)

    assert len(ts) == 96
    assert str(ts.freq) in {"15min", "15T", "<15 * Minutes>"}


def test_frame_to_time_series_wraps_multivariate_frame():
    idx = pd.date_range("2024-06-15", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame({"a": np.arange(24.0), "b": np.arange(24.0) * 2}, index=idx)

    ts = frame_to_time_series(frame)

    assert ts.n_components == 2
    assert ts.components.to_list() == ["a", "b"]
    assert len(ts) == 24
    assert ts.time_index.tz is None


def test_quantiles_to_frame_extracts_ordered_quantiles():
    idx = pd.date_range("2024-06-15", periods=5, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=40.0, scale=10.0, size=(5, 1, 200))
    prob_ts = TimeSeries.from_times_and_values(idx.tz_localize(None), samples)

    result = quantiles_to_frame(prob_ts, idx)

    assert list(result.columns) == ["p10", "p50", "p90"]
    assert result.index.equals(idx)
    assert result.index.tz is not None

    expected = np.quantile(samples[:, 0, :], [0.10, 0.50, 0.90], axis=1).T
    np.testing.assert_allclose(result["p10"].to_numpy(), expected[:, 0])
    np.testing.assert_allclose(result["p50"].to_numpy(), expected[:, 1])
    np.testing.assert_allclose(result["p90"].to_numpy(), expected[:, 2])
    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()


def test_quantiles_to_frame_reindexes_onto_supplied_index():
    """Values are placed onto the caller's grid, even a permuted one."""
    idx = pd.date_range("2024-06-15", periods=3, freq="h", tz="UTC")
    samples = np.random.default_rng(0).normal(40.0, 10.0, size=(3, 1, 50))
    prob_ts = TimeSeries.from_times_and_values(idx.tz_localize(None), samples)

    permuted = idx[[2, 0, 1]]
    result = quantiles_to_frame(prob_ts, permuted)

    assert result.index.equals(permuted)
    assert not result[["p10", "p50", "p90"]].isna().any().any()


def test_quantiles_to_frame_enforces_row_ordering(monkeypatch):
    """Crossed quantiles are clamped so p10 <= p50 <= p90 per row."""
    idx = pd.date_range("2024-06-15", periods=3, freq="h", tz="UTC")
    prob_ts = TimeSeries.from_times_and_values(
        idx.tz_localize(None), np.zeros((3, 1, 10))
    )
    # Three "quantile" components, the first row deliberately inverted.
    inverted = TimeSeries.from_times_and_values(
        idx.tz_localize(None),
        np.array(
            [
                [[50.0], [30.0], [40.0]],  # p10=50 > p50=30, p90=40 > p50
                [[1.0], [2.0], [3.0]],
                [[7.0], [8.0], [9.0]],
            ]
        ),
    )
    monkeypatch.setattr(prob_ts, "quantile", lambda q: inverted)

    result = quantiles_to_frame(prob_ts, idx)

    assert (result["p10"] <= result["p50"]).all()
    assert (result["p50"] <= result["p90"]).all()
    # Row 0 clamped: p10 = min(50, 30), p90 = max(40, 30).
    assert result.loc[idx[0], "p10"] == 30.0
    assert result.loc[idx[0], "p50"] == 30.0
    assert result.loc[idx[0], "p90"] == 40.0
