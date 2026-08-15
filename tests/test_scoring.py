"""Tests for the scoring suite (ticket 09)."""

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline.scoring import (
    crps,
    crps_scores,
    diebold_mariano,
    mae,
    mean_score,
    pinball_loss,
    score_by_regime,
    score_by_season,
    seasonal_naive_baseline,
    skill_score,
)


def _tiny() -> tuple[pd.DataFrame, pd.Series]:
    """Two aligned rows with hand-computable metrics.

    actuals [3, 4] against p10 [0, 1], p50 [1, 2], p90 [2, 3]:
      pinball(0.10) = 0.3, pinball(0.50) = 1.0, pinball(0.90) = 0.9,
      crps = 2/3 * 2.2 = 1.4667, mae = 2.0.
    """
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    preds = pd.DataFrame(
        {"p10": [0.0, 1.0], "p50": [1.0, 2.0], "p90": [2.0, 3.0]}, index=index
    )
    actuals = pd.Series([3.0, 4.0], index=index)
    return preds, actuals


def test_mae_known_value():
    preds, actuals = _tiny()
    assert mae(preds, actuals) == pytest.approx(2.0)


def test_pinball_known_values():
    preds, actuals = _tiny()
    assert pinball_loss(preds, actuals, 0.10) == pytest.approx(0.3)
    assert pinball_loss(preds, actuals, 0.50) == pytest.approx(1.0)
    assert pinball_loss(preds, actuals, 0.90) == pytest.approx(0.9)


def test_crps_known_value():
    preds, actuals = _tiny()
    assert crps(preds, actuals) == pytest.approx(2.2 * 2.0 / 3.0)


def test_pinball_median_is_half_mae():
    preds, actuals = _tiny()
    assert pinball_loss(preds, actuals, 0.50) == pytest.approx(0.5 * mae(preds, actuals))


def test_degenerate_quantiles_crps_equals_mae():
    index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    point = np.array([10.0, 20.0, 30.0, 25.0, 15.0])
    preds = pd.DataFrame({"p10": point, "p50": point, "p90": point}, index=index)
    actuals = pd.Series([11.0, 19.0, 33.0, 25.0, 12.0], index=index)
    assert crps(preds, actuals) == pytest.approx(mae(preds, actuals))


def test_skill_score_values():
    assert skill_score(0.0, 5.0) == pytest.approx(1.0)
    assert skill_score(2.0, 4.0) == pytest.approx(0.5)
    assert skill_score(4.0, 4.0) == pytest.approx(0.0)
    assert skill_score(6.0, 4.0) == pytest.approx(-0.5)


def test_seasonal_naive_baseline_hourly():
    index = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    actuals = pd.Series(np.arange(48.0), index=index)
    base = seasonal_naive_baseline(actuals, horizon=24)
    assert len(base) == 24
    assert base.index[0] == index[-1] + index.freq
    np.testing.assert_array_equal(base.to_numpy(), np.arange(24.0, 48.0))


def test_seasonal_naive_baseline_weekly_override():
    index = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
    actuals = pd.Series(np.arange(len(index), dtype=float), index=index)
    base = seasonal_naive_baseline(actuals, horizon=24, season_length=24 * 7)
    assert base.iloc[0] == pytest.approx(actuals.iloc[-(24 * 7)])


def test_seasonal_naive_baseline_15min_default():
    index = pd.date_range("2024-01-01", periods=96 * 2, freq="15min", tz="UTC")
    actuals = pd.Series(np.arange(len(index), dtype=float), index=index)
    base = seasonal_naive_baseline(actuals, horizon=96)
    assert len(base) == 96
    np.testing.assert_array_equal(base.to_numpy(), np.arange(96.0, 192.0))


def test_crps_scores_aligned():
    preds, actuals = _tiny()
    scores = crps_scores(preds, actuals)
    assert list(scores.index) == list(actuals.index)
    assert crps(preds, actuals) == pytest.approx(scores.mean())


def test_mean_score():
    scores = pd.Series([1.0, 2.0, 3.0])
    assert mean_score(scores) == pytest.approx(2.0)


def test_score_by_regime():
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    scores = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=index)
    regime = ["a", "a", "a", "b", "b", "b"]
    result = score_by_regime(scores, regime)
    assert result["a"] == pytest.approx(2.0)
    assert result["b"] == pytest.approx(20.0)
    assert result.index.name == "regime"


def test_score_by_season():
    index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-07-01",
            "2024-07-02",
            "2024-07-03",
        ],
        tz="UTC",
    )
    scores = pd.Series([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], index=index)
    result = score_by_season(scores)
    assert result["winter"] == pytest.approx(2.0)
    assert result["summer"] == pytest.approx(20.0)
    assert result.index.name == "season"


def test_unsupported_quantile_raises():
    preds, actuals = _tiny()
    with pytest.raises(ValueError):
        pinball_loss(preds, actuals, 0.25)


def test_misaligned_actuals_raise():
    preds, actuals = _tiny()
    shifted = actuals.copy()
    shifted.index = shifted.index + actuals.index.freq
    with pytest.raises(ValueError):
        mae(preds, shifted)


def test_skill_score_zero_baseline_raises():
    with pytest.raises(ValueError):
        skill_score(1.0, 0.0)


def test_diebold_mariano_identical_series():
    loss = pd.Series(np.linspace(1.0, 10.0, 200))
    stat, p = diebold_mariano(loss, loss.copy())
    assert abs(stat) < 1e-9
    assert p == pytest.approx(1.0)


def test_diebold_mariano_detects_worse_model():
    rng = np.random.default_rng(0)
    noise_a = rng.normal(0, 0.1, 500)
    noise_b = rng.normal(0, 0.1, 500)
    base = rng.normal(0, 1, 500)
    # model A has systematically higher loss than B
    loss_a = pd.Series(base + 0.5 + noise_a)
    loss_b = pd.Series(base + noise_b)
    stat, p = diebold_mariano(loss_a, loss_b)
    assert stat > 0
    assert p < 0.05


def test_diebold_mariano_rejects_mismatched_length():
    with pytest.raises(ValueError):
        diebold_mariano(pd.Series([1.0, 2.0]), pd.Series([1.0]))
