"""Tests for opt-in libxrk-decoded native channel access."""

from types import MappingProxyType

import numpy as np
import pytest
from conftest import EXAMPLES

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog

pa = pytest.importorskip("pyarrow")
libxrk = pytest.importorskip("libxrk")


class _FakeXrkLog:
    def __init__(self, channels, laps=None):
        self.channels = channels
        self.metadata = {}
        self.laps = laps


def _channel_table(name, times, values, units="", *, source_type=9):
    value_array = pa.array(values)
    field_metadata = {
        b"units": units.encode(),
        b"dec_pts": b"2",
        b"source_type": str(source_type).encode(),
        b"function": b"test channel",
    }
    schema = pa.schema([
        pa.field("timecodes", pa.array(times).type),
        pa.field(name, value_array.type, metadata=field_metadata),
    ])
    return pa.Table.from_arrays([pa.array(times), value_array], schema=schema)


def _patch_xrk(monkeypatch, channels, laps=None):
    log = _FakeXrkLog(channels, laps)
    monkeypatch.setattr(libxrk, "aim_xrk", lambda _path: log)
    return log


def _raw_fixture_channels():
    times = np.asarray([2000, 1000, 2000, 1500], dtype=np.int32)
    values = {
        "InlineAcc": np.asarray([1.0, 2.0, np.nan, 4.0], dtype=np.float32),
        "IMU_LongG": np.asarray([9.0, 8.0, 7.0, 6.0], dtype=np.float32),
        "BrakePress": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "mystery_sensor": np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
    }
    channels = {
        "InlineAcc": _channel_table("InlineAcc", times, values["InlineAcc"], "g"),
        "IMU_LongG": _channel_table("IMU_LongG", times, values["IMU_LongG"], "g"),
        "BrakePress": _channel_table("BrakePress", times, values["BrakePress"], "bar"),
        "mystery_sensor": _channel_table(
            "mystery_sensor", times, values["mystery_sensor"], "V",
        ),
    }
    return channels, times, values


def test_xrk_raw_all_preserves_decoded_samples_and_canonical_behavior(monkeypatch):
    from track_telemetry_converter.channels import CH_BRAKE_PRESS, CH_CG_ACCEL_LON

    channels, source_times, source_values = _raw_fixture_channels()
    _patch_xrk(monkeypatch, channels)

    log = DataLog()
    log.from_xrk_log("stub.xrk", raw_channels="all")

    assert list(log.raw_channels) == list(channels)
    raw = log.raw_channels["InlineAcc"]
    assert raw.name == "InlineAcc"
    assert raw.units == "g"
    assert raw.values.dtype == np.dtype(np.float32)
    assert raw.source_timestamps.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(raw.values, source_values["InlineAcc"])
    np.testing.assert_array_equal(raw.source_timestamps, source_times)
    np.testing.assert_array_equal(raw.timestamps, [1.0, 0.0, 1.0, 0.5])
    assert raw.timestamps_monotonic is False
    assert raw.sample_rate == 2.0
    assert raw.provenance["source"] == "XRK"
    assert raw.provenance["value_origin"] == "libxrk_decoded"
    assert raw.provenance["dtype"] == "float32"
    assert raw.provenance["source_metadata"]["source_type"] == 9
    assert raw.provenance["native_acquisition_rate_hz"] is None
    assert raw.provenance["series_sample_rate_hz"] == 2.0
    assert raw.provenance["time_origin_ms"] == 1000.0
    assert isinstance(raw.provenance, MappingProxyType)
    assert not raw.timestamps.flags.writeable
    assert not raw.values.flags.writeable
    assert not raw.source_timestamps.flags.writeable
    with pytest.raises(TypeError):
        raw.provenance["source"] = "changed"  # pyright: ignore[reportIndexIssue]

    # Raw output retains both alias candidates; the canonical path still chooses
    # the higher priority IMU source, converts brake pressure, and deduplicates.
    assert log.metadata["channel_provenance"][CH_CG_ACCEL_LON]["raw_name"] == "IMU_LongG"
    assert log.channel_metadata[CH_CG_ACCEL_LON]["raw_name"] == "IMU_LongG"
    assert log.channel_metadata[CH_CG_ACCEL_LON]["sample_rate"] == 2.0
    np.testing.assert_array_equal(log.channels[CH_CG_ACCEL_LON].values, [8.0, 6.0, 7.0])
    np.testing.assert_array_equal(log.channels[CH_BRAKE_PRESS].values, [200.0, 400.0, 300.0])
    np.testing.assert_array_equal(log.raw_channels["BrakePress"].values, source_values["BrakePress"])
    assert log.channels[CH_BRAKE_PRESS].units == "kPa"
    assert "InlineAcc" not in log.channels
    assert "IMU_LongG" not in log.channels
    assert "BrakePress" not in log.channels


def test_xrk_raw_selection_default_and_missing_names(monkeypatch):
    channels, _, _ = _raw_fixture_channels()
    _patch_xrk(monkeypatch, channels)

    default_log = DataLog()
    default_log.from_xrk_log("stub.xrk")
    assert default_log.raw_channels == {}

    selected_log = DataLog()
    selected_log.from_xrk_log("stub.xrk", raw_channels="InlineAcc")
    assert list(selected_log.raw_channels) == ["InlineAcc"]

    missing_log = DataLog()
    missing_log.from_xrk_log("stub.xrk", raw_channels=["InlineAcc", "not-recorded"])
    assert list(missing_log.raw_channels) == ["InlineAcc"]


def test_xrk_raw_target_lap_filters_unordered_samples_in_source_order(monkeypatch):
    times = np.asarray([2500, 1000, 2000, 1500], dtype=np.int64)
    values = np.asarray([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    channels = {"BrakePress": _channel_table("BrakePress", times, values, "bar")}
    laps = pa.table({"num": [1], "start_time": [1000], "end_time": [2000]})
    _patch_xrk(monkeypatch, channels, laps)

    log = DataLog()
    log.from_xrk_log("stub.xrk", target_lap=1, raw_channels="BrakePress")

    raw = log.raw_channels["BrakePress"]
    np.testing.assert_array_equal(raw.source_timestamps, [1000, 2000, 1500])
    np.testing.assert_array_equal(raw.timestamps, [0.0, 1.0, 0.5])
    np.testing.assert_array_equal(raw.values, [np.nan, 3.0, 4.0])
    assert raw.timestamps_monotonic is False


def test_xrk_raw_skips_empty_malformed_and_nonnumeric_tables(monkeypatch, capsys):
    missing_timecodes = pa.table({"orphan": pa.array([1.0, 2.0])})
    nonnumeric = pa.table({
        "timecodes": pa.array([0, 1000], type=pa.int64()),
        "text_sensor": pa.array(["low", "high"]),
    })
    nonscalar = pa.table({
        "timecodes": pa.array([0, 1000], type=pa.int64()),
        "vector_sensor": pa.array([[1.0], [2.0]]),
    })
    empty = pa.table({"timecodes": pa.array([], type=pa.int64())})
    _patch_xrk(monkeypatch, {
        "missing_timecodes": missing_timecodes,
        "text_sensor": nonnumeric,
        "vector_sensor": nonscalar,
        "empty": empty,
    })

    log = DataLog()
    log.from_xrk_log("stub.xrk", raw_channels="all")

    assert log.raw_channels == {}
    warning = capsys.readouterr().out
    assert "missing_timecodes" in warning
    assert "text_sensor" in warning
    assert "vector_sensor" in warning


def test_xrk_raw_keeps_invalid_values_and_clocks_without_poisoning_origin(monkeypatch):
    channels = {
        "BrakePress": _channel_table(
            "BrakePress", [np.nan, -np.inf, 1000.0, 2000.0],
            np.array([1.0, 2.0, np.nan, np.nan], dtype=np.float32), "bar",
        ),
    }
    _patch_xrk(monkeypatch, channels)

    log = DataLog()
    log.from_xrk_log("stub.xrk", raw_channels="all")

    raw = log.raw_channels["BrakePress"]
    assert raw.provenance["time_origin_ms"] == 1000.0
    np.testing.assert_array_equal(raw.source_timestamps, [np.nan, -np.inf, 1000.0, 2000.0])
    np.testing.assert_array_equal(raw.values, [1.0, 2.0, np.nan, np.nan])
    assert raw.timestamps_monotonic is False
    assert "Brake Press" not in log.channels
    selected = raw.between(0.0, 1.0, include_end=True)
    np.testing.assert_array_equal(selected.source_timestamps, [1000.0, 2000.0])
    assert len(selected.values) == 2 and np.all(np.isnan(selected.values))


def test_xrk_raw_channels_do_not_enter_motec_export(monkeypatch):
    channels, _, _ = _raw_fixture_channels()
    _patch_xrk(monkeypatch, channels)
    log = DataLog()
    log.from_xrk_log("stub.xrk", raw_channels="all")
    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)

    assert "mystery_sensor" in log.raw_channels
    assert {channel.name for channel in motec.ld_channels} == set(log.channels)
    assert "mystery_sensor" not in {channel.name for channel in motec.ld_channels}


def test_xrk_real_fixture_raw_values_and_source_clocks_match_decoder():
    from pathlib import Path

    source = libxrk.aim_xrk(str(Path(EXAMPLES) / "aim_sample.xrk"))
    log = DataLog()
    log.from_xrk_log(Path(EXAMPLES) / "aim_sample.xrk", raw_channels="all")

    assert set(log.raw_channels) == set(source.channels)
    assert len(log.raw_channels) > len(log.channels)
    for name, table in source.channels.items():
        raw = log.raw_channels[name]
        values = table.column(name).to_numpy(zero_copy_only=False)
        clocks = table.column("timecodes").to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(raw.values, values, err_msg=name)
        np.testing.assert_array_equal(raw.source_timestamps, clocks, err_msg=name)
        assert raw.values.dtype == values.dtype
    # The fixture includes unordered GPS data; raw access must retain it.
    assert log.raw_channels["GPS Latitude"].timestamps_monotonic is False
