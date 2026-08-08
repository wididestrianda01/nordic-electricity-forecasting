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
def pre_nov_2024_scenario():
    """as_of_date well before both regime boundaries — hourly MTU."""
    as_of = date(2024, 6, 15)
    return as_of, _hourly_history(as_of, days=30)


@pytest.fixture
def straddle_nov_2024_scenario():
    """historical_data spans the 4 Nov 2024 flow-based-coupling boundary; MTU still hourly."""
    as_of = date(2024, 11, 10)
    return as_of, _hourly_history(as_of, days=30)


@pytest.fixture
def post_oct_2025_scenario():
    """as_of_date after the 1 Oct 2025 MTU switch — 15-minute MTU."""
    as_of = date(2025, 10, 15)
    return as_of, _15min_history(as_of, days=30)


@pytest.fixture
def malformed_input_scenario():
    """historical_data too short to forecast from."""
    as_of = date(2024, 6, 15)
    return as_of, _hourly_history(as_of, days=1)
