from datetime import date

import numpy as np
import pandas as pd
import pytest


def _hourly_history(end: date, days: int) -> pd.DataFrame:
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(seed=42)
    prices = rng.normal(loc=40.0, scale=10.0, size=len(index))
    return pd.DataFrame({"price": prices}, index=index)


def _15min_history(end: date, days: int) -> pd.DataFrame:
    start = pd.Timestamp(end) - pd.Timedelta(days=days)
    index = pd.date_range(start, periods=days * 96, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed=42)
    prices = rng.normal(loc=40.0, scale=10.0, size=len(index))
    return pd.DataFrame({"price": prices}, index=index)


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
    return as_of, pd.DataFrame({"price": prices}, index=index)


@pytest.fixture
def malformed_input_scenario() -> tuple[date, pd.DataFrame]:
    """historical_data too short to forecast from."""
    as_of = date(2024, 6, 15)
    return as_of, _hourly_history(as_of, days=1)
