"""Tests for feature-group classification (feature_groups)."""

from forecast_pipeline.feature_groups import (
    EXCLUDED_GROUP,
    GROUP_ORDER,
    classify_columns,
)


def test_classify_columns_partitions_all_seven_groups():
    columns = [
        # group 1 (remainder)
        "lag_1d",
        "rolling_mean_7d",
        "hour_sin",
        "holiday",
        "regime",
        # group 2 (excluded)
        "load_forecast",
        "wind_forecast",
        # group 3
        "net_position_mwh",
        "scheduled_exchange_mwh",
        "neighbour_price_eur_mwh",
        # group 4
        "SE1_temperature_2m",
        "SE3_wind_speed_100m",
        "SE4_precipitation",
        # group 5
        "hydro_storage_mwh",
        # group 6
        "carbon_eua",
        # group 7
        "fx_sek_eur",
    ]

    groups = classify_columns(columns)

    assert set(groups) == set(GROUP_ORDER)
    assert groups["system_fundamentals"] == ["load_forecast", "wind_forecast"]
    assert groups["cross_border"] == [
        "net_position_mwh",
        "scheduled_exchange_mwh",
        "neighbour_price_eur_mwh",
    ]
    assert groups["weather"] == [
        "SE1_temperature_2m",
        "SE3_wind_speed_100m",
        "SE4_precipitation",
    ]
    assert groups["hydro"] == ["hydro_storage_mwh"]
    assert groups["commodities"] == ["carbon_eua"]
    assert groups["fx"] == ["fx_sek_eur"]
    # group 1 is the remainder
    assert groups["ar_calendar_regime"] == [
        "lag_1d",
        "rolling_mean_7d",
        "hour_sin",
        "holiday",
        "regime",
    ]


def test_classify_columns_no_column_in_two_groups():
    columns = [
        "lag_1d",
        "regime",
        "load_forecast",
        "net_position_mwh",
        "SE2_temperature_2m",
        "hydro_storage_mwh",
        "carbon_eua",
        "fx_sek_eur",
    ]
    groups = classify_columns(columns)
    flattened = [c for cols in groups.values() for c in cols]
    assert len(flattened) == len(set(flattened)) == len(columns)


def test_classify_columns_empty_groups_get_empty_list():
    groups = classify_columns(["lag_1d", "regime"])
    assert groups["ar_calendar_regime"] == ["lag_1d", "regime"]
    assert groups["weather"] == []
    assert groups["fx"] == []


def test_excluded_group_is_system_fundamentals():
    assert EXCLUDED_GROUP == "system_fundamentals"
