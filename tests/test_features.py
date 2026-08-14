"""Tests for feature engineering matrices (ticket 04 / build 04)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecast_pipeline import features
from forecast_pipeline.features import (
    assemble_data,
    build_features,
    build_horizon_features,
)

PRICE_ONLY_COLUMNS = [
    "lag_1d",
    "lag_2d",
    "lag_3d",
    "lag_7d",
    "lag_14d",
    "lag_28d",
    "roll_mean_7d",
    "roll_std_7d",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "regime",
]

EXOGENOUS_COLUMNS = [
    "load_forecast",
    "wind_forecast",
    "SE1_temperature_2m",
    "SE3_temperature_2m",
    "hydro_storage_mwh",
    "net_position_mwh",
    "scheduled_exchange_mwh",
    "neighbour_price_eur_mwh",
    "carbon_eua",
    "fx_sek_eur",
]


def _assembled_frame(end: date, days: int) -> pd.DataFrame:
    """Synthetic assembled frame: strictly-increasing price + every exogenous group."""
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    n = len(index)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": np.arange(n, dtype=float),
            "load_forecast": rng.normal(6000, 500, n),
            "wind_forecast": rng.normal(1500, 300, n),
            "SE1_temperature_2m": rng.normal(5, 5, n),
            "SE3_temperature_2m": rng.normal(8, 6, n),
            "hydro_storage_mwh": rng.normal(10000, 100, n),
            "net_position_mwh": rng.normal(0, 200, n),
            "scheduled_exchange_mwh": rng.normal(0, 300, n),
            "neighbour_price_eur_mwh": rng.normal(40, 10, n),
            "carbon_eua": rng.normal(70, 5, n),
            "fx_sek_eur": rng.normal(11.3, 0.1, n),
        },
        index=index,
    )


def test_build_features_returns_two_dataframes(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    price_only, full = build_features(as_of, history)
    assert isinstance(price_only, pd.DataFrame)
    assert isinstance(full, pd.DataFrame)
    assert price_only.index.equals(history.index)
    assert full.index.equals(history.index)


def test_price_only_has_expected_columns(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    price_only, _ = build_features(as_of, history)
    assert list(price_only.columns) == PRICE_ONLY_COLUMNS


def test_full_features_adds_exogenous_groups(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    _, full = build_features(as_of, history)
    # The fixture carries load/wind forecast; the remaining groups come from
    # an assembled frame in the next test.
    for col in ("load_forecast", "wind_forecast"):
        assert col in full.columns
    assert "price" not in full.columns  # target excluded from features


def test_full_features_includes_all_exogenous_columns():
    as_of = date(2024, 6, 15)
    frame = _assembled_frame(as_of, days=30)
    price_only, full = build_features(as_of, frame)
    assert set(EXOGENOUS_COLUMNS) <= set(full.columns)
    assert set(PRICE_ONLY_COLUMNS) <= set(full.columns)
    assert "price" not in full.columns
    # price-only must NOT leak exogenous columns
    assert set(price_only.columns).isdisjoint(EXOGENOUS_COLUMNS)


def test_lags_and_rolling_present(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    price_only, _ = build_features(as_of, history)
    for lag in (1, 2, 3, 7, 14, 28):
        assert f"lag_{lag}d" in price_only.columns
    assert "roll_mean_7d" in price_only.columns
    assert "roll_std_7d" in price_only.columns


def test_cyclical_and_holiday_present(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    price_only, _ = build_features(as_of, history)
    for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"):
        assert col in price_only.columns
    assert price_only["is_holiday"].dtype == bool


def test_holiday_flag_marks_known_dates():
    # 6 Jun is Sweden's National Day; 10 Jun 2024 is an ordinary Monday.
    index = pd.date_range("2024-06-05", periods=8 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"price": rng.normal(40, 10, len(index))}, index=index)
    price_only, _ = build_features(date(2024, 6, 12), frame)
    holiday = price_only.loc[pd.Timestamp("2024-06-06 00:00", tz="UTC"), "is_holiday"]
    normal = price_only.loc[pd.Timestamp("2024-06-10 00:00", tz="UTC"), "is_holiday"]
    assert bool(holiday) is True
    assert bool(normal) is False


def test_regime_label_present(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    price_only, _ = build_features(as_of, history)
    assert set(price_only["regime"].unique()) <= {"regime_0", "regime_1"}


def test_no_leakage_lags_use_strictly_past_prices():
    as_of = date(2024, 6, 15)
    frame = _assembled_frame(as_of, days=30)
    _, full = build_features(as_of, frame)
    steps = 24  # hourly MTU
    for lag in (1, 2, 3, 7, 14, 28):
        col = f"lag_{lag}d"
        # No regime boundary inside this window, so the only NaNs are the
        # leading shift rows; the value at t equals the price exactly lag days
        # earlier -- never the current or a future price.
        assert full[col].iloc[lag * steps] == frame["price"].iloc[0]
        assert np.array_equal(
            full[col].to_numpy()[lag * steps :],
            frame["price"].to_numpy()[:-lag * steps],
        )


def test_no_leakage_rolling_excludes_current_row():
    as_of = date(2024, 6, 15)
    frame = _assembled_frame(as_of, days=30)
    _, full = build_features(as_of, frame)
    steps = 24
    window = 7 * steps
    i = window + 10
    trailing = frame["price"].iloc[i - window : i]  # strictly before t
    assert full["roll_mean_7d"].iloc[i] == pytest.approx(trailing.mean())
    assert full["roll_std_7d"].iloc[i] == pytest.approx(trailing.std())
    # including the current row would shift the mean up by 0.5 (arange prices)
    assert full["roll_mean_7d"].iloc[i] != pytest.approx(
        frame["price"].iloc[i - window : i + 1].mean()
    )


def test_boundary_masking_across_nov_2024(straddle_nov_2024_scenario):
    as_of, history = straddle_nov_2024_scenario
    price_only, _ = build_features(as_of, history)
    ts = pd.Timestamp("2024-11-04 00:00", tz="UTC")
    # A 7-day lag looked up from the boundary date reaches back pre-break.
    assert np.isnan(price_only.loc[ts, "lag_7d"])
    # A 1-day lag from 3 Nov stays entirely inside the pre-break regime.
    ts_pre = pd.Timestamp("2024-11-03 00:00", tz="UTC")
    assert not np.isnan(price_only.loc[ts_pre, "lag_1d"])


def test_boundary_masking_across_oct_2025(post_oct_2025_scenario):
    as_of, history = post_oct_2025_scenario
    price_only, _ = build_features(as_of, history)
    ts = pd.Timestamp("2025-10-01 00:00", tz="UTC")
    assert np.isnan(price_only.loc[ts, "lag_1d"])
    ts_pre = pd.Timestamp("2025-09-30 00:00", tz="UTC")
    assert not np.isnan(price_only.loc[ts_pre, "lag_1d"])


def test_assemble_data_joins_and_flattens_weather(monkeypatch):
    start, end = date(2024, 10, 1), date(2024, 10, 3)
    grid = pd.date_range(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h", inclusive="left"
    )

    def market(zone, s, e):
        return pd.DataFrame(
            {"price": 1.0, "load_forecast": 2.0, "wind_forecast": 3.0}, index=grid
        )

    def weather(zones, s, e):
        cols = pd.MultiIndex.from_product(
            [["SE1", "SE3"], ["temperature_2m"]], names=["zone", "variable"]
        )
        return pd.DataFrame(np.ones((len(grid), 2)), index=grid, columns=cols)

    def hydro(zones, s, e):
        return pd.DataFrame({"hydro_storage_mwh": 4.0}, index=grid)

    def cross_border(zones, s, e):
        return pd.DataFrame(
            {
                "net_position_mwh": 5.0,
                "scheduled_exchange_mwh": 6.0,
                "neighbour_price_eur_mwh": 7.0,
            },
            index=grid,
        )

    def fx(s, e):
        return pd.DataFrame({"fx_sek_eur": 8.0}, index=grid)

    def carbon(s, e):
        return pd.DataFrame({"carbon_eua": 9.0}, index=grid)

    monkeypatch.setattr(features, "fetch_market_data", market)
    monkeypatch.setattr(features, "fetch_weather", weather)
    monkeypatch.setattr(features, "fetch_hydro", hydro)
    monkeypatch.setattr(features, "fetch_cross_border", cross_border)
    monkeypatch.setattr(features, "fetch_fx", fx)
    monkeypatch.setattr(features, "fetch_carbon", carbon)

    df = assemble_data(["SE1", "SE3"], start, end)

    assert "price" in df.columns
    assert "SE1_temperature_2m" in df.columns  # flattened from (zone, variable)
    assert "SE3_temperature_2m" in df.columns
    assert "hydro_storage_mwh" in df.columns
    assert "net_position_mwh" in df.columns
    assert "carbon_eua" in df.columns
    assert "fx_sek_eur" in df.columns


def test_assemble_data_lags_weather_by_one_day(monkeypatch):
    """Weather at t must use realized weather from t-1 (ticket 08), never t."""
    start, end = date(2024, 10, 1), date(2024, 10, 3)
    grid = pd.date_range(
        pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="h", inclusive="left"
    )

    def market(zone, s, e):
        return pd.DataFrame(
            {"price": 1.0, "load_forecast": 2.0, "wind_forecast": 3.0}, index=grid
        )

    def weather(zones, s, e):
        cols = pd.MultiIndex.from_product(
            [["SE1"], ["temperature_2m"]], names=["zone", "variable"]
        )
        # Day-of-range index (0 for day 1, 1 for day 2) so the shift is visible.
        values = (np.arange(len(grid)) // 24).astype(float)
        return pd.DataFrame(values, index=grid, columns=cols)

    def hydro(zones, s, e):
        return pd.DataFrame({"hydro_storage_mwh": 4.0}, index=grid)

    def cross_border(zones, s, e):
        return pd.DataFrame(
            {
                "net_position_mwh": 5.0,
                "scheduled_exchange_mwh": 6.0,
                "neighbour_price_eur_mwh": 7.0,
            },
            index=grid,
        )

    def fx(s, e):
        return pd.DataFrame({"fx_sek_eur": 8.0}, index=grid)

    def carbon(s, e):
        return pd.DataFrame({"carbon_eua": 9.0}, index=grid)

    monkeypatch.setattr(features, "fetch_market_data", market)
    monkeypatch.setattr(features, "fetch_weather", weather)
    monkeypatch.setattr(features, "fetch_hydro", hydro)
    monkeypatch.setattr(features, "fetch_cross_border", cross_border)
    monkeypatch.setattr(features, "fetch_fx", fx)
    monkeypatch.setattr(features, "fetch_carbon", carbon)

    df = assemble_data(["SE1"], start, end)

    day0 = pd.Timestamp("2024-10-01 00:00", tz="UTC")
    day1 = pd.Timestamp("2024-10-02 00:00", tz="UTC")
    col = "SE1_temperature_2m"
    # Leading day has no t-1 weather within range -> NaN.
    assert np.isnan(df.loc[day0, col])
    # Weather at day1 equals realized weather at day0 (t-1), not day1's own 1.0.
    assert df.loc[day1, col] == 0.0


def test_build_horizon_features_matches_full_features_columns(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    _, full = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)

    assert list(horizon.columns) == list(full.columns)
    # D+1 grid: one MTU step past the last history row, one full day.
    assert horizon.index.equals(
        pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    )


def test_build_horizon_features_15min_grid(post_oct_2025_scenario):
    as_of, history = post_oct_2025_scenario
    horizon = build_horizon_features(as_of, history)

    assert len(horizon) == 96
    assert horizon.index.equals(
        pd.date_range(history.index[-1] + pd.Timedelta(minutes=15), periods=96, freq="15min")
    )


def test_build_horizon_features_exogenous_forward_filled_no_leakage():
    """Horizon exogenous values are forward-filled from the last published row."""
    as_of = date(2024, 6, 15)
    frame = _assembled_frame(as_of, days=30)
    _, full = build_features(as_of, frame)
    horizon = build_horizon_features(as_of, frame)

    # Every exogenous column is constant across the horizon and equals its last
    # published value -- so no value published after as_of can appear.
    for col in EXOGENOUS_COLUMNS:
        assert (horizon[col] == full[col].iloc[-1]).all()


def test_build_horizon_features_lags_use_strictly_past_prices():
    as_of = date(2024, 6, 15)
    frame = _assembled_frame(as_of, days=30)
    horizon = build_horizon_features(as_of, frame)
    price = frame["price"]

    for lag in (1, 2, 3, 7, 14, 28):
        col = f"lag_{lag}d"
        expected = price.reindex(horizon.index - pd.Timedelta(days=lag)).to_numpy()
        np.testing.assert_allclose(horizon[col].to_numpy(), expected)


def test_build_horizon_features_regime_is_persisted(pre_nov_2024_scenario):
    as_of, history = pre_nov_2024_scenario
    _, full = build_features(as_of, history)
    horizon = build_horizon_features(as_of, history)

    assert (horizon["regime"] == full["regime"].iloc[-1]).all()
