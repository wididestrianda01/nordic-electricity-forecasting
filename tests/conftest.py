from datetime import date

import numpy as np
import pandas as pd
import pytest


def _hourly_history(end: date, days: int) -> pd.DataFrame:
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=42)
    prices = rng.normal(loc=40.0, scale=10.0, size=len(index))
    loads = rng.normal(loc=6000.0, scale=500.0, size=len(index))
    winds = rng.normal(loc=1500.0, scale=300.0, size=len(index))
    return pd.DataFrame(
        {"price": prices, "load_forecast": loads, "wind_forecast": winds},
        index=index,
    )


def _15min_history(end: date, days: int) -> pd.DataFrame:
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    index = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed=42)
    prices = rng.normal(loc=40.0, scale=10.0, size=len(index))
    loads = rng.normal(loc=6000.0, scale=500.0, size=len(index))
    winds = rng.normal(loc=1500.0, scale=300.0, size=len(index))
    return pd.DataFrame(
        {"price": prices, "load_forecast": loads, "wind_forecast": winds},
        index=index,
    )


@pytest.fixture
def pre_nov_2024_scenario() -> tuple[date, pd.DataFrame]:
    """as_of_date well before both regime boundaries — hourly MTU."""
    as_of = date(2024, 6, 15)
    return as_of, _hourly_history(as_of, days=30)


@pytest.fixture
def straddle_nov_2024_scenario() -> tuple[date, pd.DataFrame]:
    """historical_data spans the 4 Nov 2024 flow-based-coupling boundary; MTU still hourly."""
    as_of = date(2024, 11, 10)
    return as_of, _hourly_history(as_of, days=30)


@pytest.fixture
def post_oct_2025_scenario() -> tuple[date, pd.DataFrame]:
    """as_of_date after the 1 Oct 2025 MTU switch — 15-minute MTU."""
    as_of = date(2025, 10, 15)
    return as_of, _15min_history(as_of, days=30)


@pytest.fixture
def two_regime_scenario() -> tuple[date, pd.DataFrame]:
    """Hourly history with a genuine level/volatility shift halfway through --
    exercises HMM regime detection against a real, not merely nominal, break."""
    as_of = date(2024, 6, 15)
    days = 30
    start = pd.Timestamp(as_of) - pd.Timedelta(days=days)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=7)
    half = len(index) // 2
    low_regime = rng.normal(loc=20.0, scale=3.0, size=half)
    high_regime = rng.normal(loc=80.0, scale=15.0, size=len(index) - half)
    prices = np.concatenate([low_regime, high_regime])
    loads = rng.normal(loc=6000.0, scale=500.0, size=len(index))
    winds = rng.normal(loc=1500.0, scale=300.0, size=len(index))
    return as_of, pd.DataFrame(
        {"price": prices, "load_forecast": loads, "wind_forecast": winds},
        index=index,
    )


@pytest.fixture
def malformed_input_scenario() -> tuple[date, pd.DataFrame]:
    """historical_data too short to forecast from."""
    as_of = date(2024, 6, 15)
    return as_of, _hourly_history(as_of, days=1)


@pytest.fixture
def regime_shift_to_high_scenario() -> tuple[date, pd.DataFrame]:
    """30 days hourly data with regime shift, forecast date after transition to high regime.

    Used to verify that regime-conditional model produces higher forecasts for high regime.
    History: low regime (20±3) for first 15 days, then high regime (80±15) for last 15 days.
    Forecast: made at end of high regime, should predict high prices.
    """
    as_of = date(2024, 6, 30)  # End of the 30-day period
    start = pd.Timestamp(as_of) - pd.Timedelta(days=30)
    index = pd.date_range(start, periods=30 * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=8)
    half = len(index) // 2
    low_regime = rng.normal(loc=20.0, scale=3.0, size=half)
    high_regime = rng.normal(loc=80.0, scale=15.0, size=len(index) - half)
    prices = np.concatenate([low_regime, high_regime])
    loads = rng.normal(loc=6000.0, scale=500.0, size=len(index))
    winds = rng.normal(loc=1500.0, scale=300.0, size=len(index))
    return as_of, pd.DataFrame(
        {"price": prices, "load_forecast": loads, "wind_forecast": winds},
        index=index,
    )
