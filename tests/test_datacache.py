"""Tests for the disk-backed Parquet cache (datacache)."""

from datetime import date

import pandas as pd

from forecast_pipeline import datacache


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-10-01", periods=5, freq="h", tz="UTC")
    return pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)


def test_load_returns_none_on_miss():
    assert datacache.load("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE1"]}) is None


def test_store_then_load_roundtrip():
    frame = _frame()
    datacache.store("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE1"]}, frame)

    loaded = datacache.load("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE1"]})

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)


def test_load_distinguishes_params():
    frame = _frame()
    datacache.store("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE1"]}, frame)

    assert (
        datacache.load("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE3"]}) is None
    )


def test_load_distinguishes_date_range():
    frame = _frame()
    datacache.store("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE1"]}, frame)

    assert (
        datacache.load("src", date(2024, 10, 1), date(2024, 10, 3), {"zones": ["SE1"]}) is None
    )


def test_store_leaves_no_tmp(tmp_path):
    frame = _frame()
    datacache.store("src", date(2024, 10, 1), date(2024, 10, 2), {"zones": ["SE1"]}, frame)

    leftover_tmp = list(datacache.cache_dir().glob("*.tmp"))
    assert leftover_tmp == []


def test_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_CACHE_DIR", str(tmp_path / "custom"))
    assert datacache.cache_dir() == tmp_path / "custom"
