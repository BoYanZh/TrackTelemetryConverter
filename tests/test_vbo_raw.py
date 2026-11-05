"""Opt-in native numeric views of VBO source columns."""

import numpy as np
from conftest import _read_lines

from track_telemetry_converter.channels import CH_GPS_LATITUDE, CH_GPS_LONGITUDE
from track_telemetry_converter.log import DataLog


def _small_vbo(data_rows=None):
    data_rows = data_rows or [
        "235959.50 +3736.000 -12210.000 1.25 10 11 100 200 300 400",
        "000000.50 +3736.060 -12210.060 2.50 12 13 101 201 301 401",
    ]
    return [
        "File created on 01/01/2026 at 23:00:00",
        "[header]",
        "time",
        "latitude",
        "longitude",
        "brake pressure bar",
        "velocity kmh",
        "velocity-calc kmh",
        "mystery raw",
        "mystery raw",
        "literal mystery",
        "literal mystery",
        "[column names]",
        "time lat long BrakePress velocity velocity-calc mystery mystery mystery#2 mystery#2",
        "[data]",
        *data_rows,
    ]


def test_vbo_raw_numeric_views_keep_source_values_and_elapsed_clock():
    log = DataLog()
    log.from_vbo_log(_small_vbo(), raw_channels=["time", "lat", "long", "BrakePress", "velocity"])

    assert list(log.channels["Ground Speed"].values) == [11.0, 13.0]
    assert list(log.channels["Brake Press"].values) == [125.0, 250.0]
    assert np.allclose(log.channels[CH_GPS_LATITUDE].values, [3736.0 / 60.0, 3736.06 / 60.0])
    assert np.allclose(log.channels[CH_GPS_LONGITUDE].values, [-12210.0 / 60.0, -12210.06 / 60.0])

    raw = log.raw_channels
    assert list(raw["lat"].values) == [3736.0, 3736.06]
    assert list(raw["long"].values) == [-12210.0, -12210.06]
    assert list(raw["BrakePress"].values) == [1.25, 2.5]
    assert list(raw["velocity"].values) == [10.0, 12.0]
    assert list(raw["time"].source_timestamps) == [235959.5, 0.5]
    assert raw["time"].source_timestamps is not raw["lat"].source_timestamps
    assert np.shares_memory(raw["time"].source_timestamps, raw["lat"].source_timestamps)
    assert list(raw["time"].timestamps) == [0.0, 1.0]
    assert raw["time"].sample_rate == 1.0
    assert raw["time"].timestamps_monotonic is True
    assert raw["BrakePress"].units == "bar"
    assert raw["BrakePress"].provenance["value_origin"] == "numeric_text"
    assert raw["BrakePress"].provenance["source_time_encoding"] == "HHMMSS.fraction"
    assert raw["BrakePress"].provenance["dtype"] == "<f8"
    assert raw["BrakePress"].provenance["column_index"] == 3
    assert not raw["BrakePress"].values.flags.writeable
    assert not raw["BrakePress"].timestamps.flags.writeable
    assert not raw["BrakePress"].source_timestamps.flags.writeable


def test_vbo_raw_selection_covers_losing_aliases_duplicates_and_literal_name_collisions():
    log = DataLog()
    log.from_vbo_log(_small_vbo(), raw_channels=["velocity-calc", "mystery", "mystery#3"])

    assert list(log.raw_channels) == ["velocity-calc", "mystery", "mystery#3"]
    assert list(log.raw_channels["velocity-calc"].values) == [11.0, 13.0]
    assert list(log.raw_channels["mystery"].values) == [100.0, 101.0]
    assert list(log.raw_channels["mystery#3"].values) == [200.0, 201.0]
    assert log.raw_channels["mystery#3"].provenance["raw_name"] == "mystery"
    assert log.raw_channels["mystery#3"].provenance["column_index"] == 7

    inventory = log.metadata["vbo_raw_columns"]
    assert [column["key"] for column in inventory] == [
        "time", "lat", "long", "BrakePress", "velocity", "velocity-calc",
        "mystery", "mystery#3", "mystery#2", "mystery#2#2",
    ]
    assert len({column["key"] for column in inventory}) == len(inventory)
    assert inventory[3]["source_header"] == "brake pressure bar"
    assert inventory[3]["source_units"] == "bar"

    duplicate_log = DataLog()
    duplicate_log.from_vbo_log(_small_vbo(), raw_channels="mystery")
    assert list(duplicate_log.raw_channels) == ["mystery", "mystery#3"]

    literal_log = DataLog()
    literal_log.from_vbo_log(_small_vbo(), raw_channels="mystery#2")
    assert list(literal_log.raw_channels) == ["mystery#2", "mystery#2#2"]


def test_vbo_raw_bad_numeric_cells_become_nan_without_shifting_rows():
    log = DataLog()
    log.from_vbo_log(
        _small_vbo([
            "235959.50 +3736.000 -12210.000 1.25 10 11 bad 200 300 400",
            "000000.50 +3736.060 -12210.060 2.50 12 13 101 201 301 401",
        ]),
        raw_channels="mystery",
    )

    values = log.raw_channels["mystery"].values
    assert len(values) == 2
    assert np.isnan(values[0])
    assert values[1] == 101.0
    assert list(log.raw_channels["mystery"].timestamps) == [0.0, 1.0]


def test_vbo_all_raw_columns_include_fixture_duplicates_and_default_is_unchanged():
    fixture = _read_lines("vbo_sample.vbo")
    log = DataLog()
    log.from_vbo_log(fixture, raw_channels="all")
    inventory = log.metadata["vbo_raw_columns"]

    assert len(inventory) == 47
    assert len(log.raw_channels) == 47
    assert len({column["key"] for column in inventory}) == 47
    assert sum(column["raw_name"] != column["key"] for column in inventory) == 8
    assert all(len(channel.values) == len(channel.timestamps) for channel in log.raw_channels.values())
    assert log.raw_channels["velocity"].units == "kmh"
    assert log.raw_channels["x_acc-acc"].units == "g"

    default_log = DataLog()
    default_log.from_vbo_log(fixture)
    assert default_log.raw_channels == {}
    assert "vbo_raw_columns" not in default_log.metadata


def test_vbo_raw_selection_ignores_unknown_names_and_handles_all_short_rows():
    log = DataLog()
    log.from_vbo_log(
        _small_vbo(["too short", "still short"]),
        raw_channels=["does-not-exist", "mystery"],
    )
    assert list(log.raw_channels) == ["mystery", "mystery#3"]
    assert all(len(channel.values) == 0 for channel in log.raw_channels.values())
