"""Synthetic regression coverage for native IBT samples and lap views."""

from __future__ import annotations

import operator
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import _run_cli_in_process

from track_telemetry_converter.channels import (
    CH_BRAKE_PRESS_FL,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_ENGINE_OIL_PRESS,
    CH_GROUND_SPEED,
    CH_STEERING_ANGLE,
    CH_WHEEL_SPEED_FL,
    CH_YAW_RATE,
)
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog

_IBT_DTYPES = {
    0: np.dtype("S1"),
    1: np.dtype("?"),
    2: np.dtype("<i4"),
    3: np.dtype("<u4"),
    4: np.dtype("<f4"),
    5: np.dtype("<f8"),
}


def _variable(name, type_code, values=None, *, units="", count=None,
              offset=None, count_as_time=False, description="synthetic"):
    """Describe a variable header and optional tick values for the IBT writer."""
    dtype = _IBT_DTYPES.get(type_code)
    array = None if values is None or dtype is None else np.asarray(values, dtype=dtype)
    if count is None:
        count = 1 if array is None or array.ndim < 2 else array.shape[1]
    return {
        "name": name,
        "type_code": type_code,
        "values": array,
        "units": units,
        "count": count,
        "offset": offset,
        "count_as_time": count_as_time,
        "description": description,
    }


def _write_ibt(path, variables, *, n_ticks, tick_rate=60, buf_len=None):
    """Write a small valid IBT file with explicitly controlled variable headers."""
    variables = [dict(variable) for variable in variables]
    next_offset = 0
    for variable in variables:
        dtype = _IBT_DTYPES.get(variable["type_code"])
        count = variable["count"]
        if variable["offset"] is None:
            variable["offset"] = next_offset
        next_offset = max(
            next_offset,
            variable["offset"] + max(count, 1) * (dtype.itemsize if dtype else 1),
        )

    if buf_len is None:
        buf_len = max(1, next_offset)

    var_header_offset = 144  # 112-byte header and 32-byte disk subheader.
    session_info = (
        b"TrackDisplayName: Synthetic Track\n"
        b"UserName: Test Driver\n"
        b"CarScreenName: Test Car\n"
        b"Date: 2026-01-02\n"
    )
    session_info_offset = var_header_offset + len(variables) * 144
    buf_offset = session_info_offset + len(session_info)

    header = struct.pack(
        "<12i",
        1, 0, tick_rate, 0, len(session_info), session_info_offset,
        len(variables), var_header_offset, 1, buf_len, 0, 0,
    )
    buffer_info = struct.pack("<4i", n_ticks, buf_offset, 0, 0)
    disk_subheader = b"\x00" * 32

    variable_headers = bytearray()
    for variable in variables:
        name = variable["name"].encode("latin-1")[:31]
        description = variable["description"].encode("latin-1")[:63]
        units = variable["units"].encode("latin-1")[:31]
        variable_headers.extend(
            struct.pack(
                "<4i32s64s32s",
                variable["type_code"], variable["offset"], variable["count"],
                int(variable["count_as_time"]),
                name + b"\x00" * (32 - len(name)),
                description + b"\x00" * (64 - len(description)),
                units + b"\x00" * (32 - len(units)),
            ),
        )

    tick_data = bytearray(n_ticks * max(buf_len, 0))
    if buf_len > 0:
        for variable in variables:
            values = variable["values"]
            dtype = _IBT_DTYPES.get(variable["type_code"])
            count = variable["count"]
            offset = variable["offset"]
            if (
                values is None or dtype is None or count <= 0 or offset < 0
                or offset + dtype.itemsize * count > buf_len
            ):
                continue
            expected_shape = (n_ticks,) if count == 1 else (n_ticks, count)
            try:
                rows = np.asarray(values, dtype=dtype).reshape(expected_shape)
            except ValueError:
                continue
            row_bytes = dtype.itemsize * count
            for tick, row in enumerate(rows):
                raw = np.asarray(row, dtype=dtype).tobytes()
                start = tick * buf_len + offset
                tick_data[start:start + row_bytes] = raw

    path.write_bytes(
        header + buffer_info + b"\x00" * 48 + disk_subheader
        + variable_headers + session_info + tick_data,
    )


def _samples(log, name):
    return log.channels[name].values


def test_raw_channels_keep_native_scalar_array_types_and_metadata(tmp_path):
    path = tmp_path / "native-types.ibt"
    variables = [
        _variable("SessionTime", 5, [12.5, 12.75, 13.0], units="s", count_as_time=True),
        _variable("Signed", 2, [-7, 0, 21], units="count"),
        _variable("Unsigned", 3, [0, 2**31 + 5, 2**32 - 1], units="count"),
        _variable("Boolean", 1, [False, True, False], units=""),
        _variable(
            "SteeringWheelTorque_ST", 3,
            [[6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]],
            units="count", count=6, count_as_time=True,
        ),
        _variable(
            "Bytes", 0,
            [[b"a", b"b", b"c"], [b"x", b"y", b"z"], [b"1", b"2", b"3"]],
            units="byte", count=3,
        ),
        _variable("Vector", 4, [[1, 2], [3, 4], [5, 6]], units="m/s", count_as_time=True),
    ]
    _write_ibt(path, variables, n_ticks=3, tick_rate=10)

    log = DataLog()
    log.from_ibt_log(
        path,
        raw_channels=[
            "SessionTime", "Signed", "Unsigned", "Boolean", "SteeringWheelTorque_ST",
            "Bytes", "Vector", "Missing",
        ],
    )

    assert set(log.raw_channels) == {
        "SessionTime", "Signed", "Unsigned", "Boolean", "SteeringWheelTorque_ST", "Bytes", "Vector",
    }
    expected = {
        "SessionTime": (np.dtype("<f8"), np.array([12.5, 12.75, 13.0], dtype="<f8")),
        "Signed": (np.dtype("<i4"), np.array([-7, 0, 21], dtype="<i4")),
        "Unsigned": (np.dtype("<u4"), np.array([0, 2**31 + 5, 2**32 - 1], dtype="<u4")),
        "Boolean": (np.dtype("?"), np.array([False, True, False], dtype="?")),
        "SteeringWheelTorque_ST": (
            np.dtype("<u4"), np.array(
                [[6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]],
                dtype="<u4",
            ),
        ),
        "Bytes": (np.dtype("S1"), np.array(
            [[b"a", b"b", b"c"], [b"x", b"y", b"z"], [b"1", b"2", b"3"]], dtype="S1",
        )),
        "Vector": (np.dtype("<f4"), np.array([[1, 2], [3, 4], [5, 6]], dtype="<f4")),
    }
    for name, (dtype, values) in expected.items():
        view = log.raw_channels[name]
        assert view.values.dtype == dtype
        assert view.values.shape == values.shape
        np.testing.assert_array_equal(view.values, values)
        assert not view.values.flags.writeable
        np.testing.assert_array_equal(view.timestamps, [0.0, 0.25, 0.5])
        assert not view.timestamps.flags.writeable
        assert view.sample_rate == 10.0
        assert view.provenance["source"] == "IBT"
        assert view.provenance["variable"] == name

    assert log.source_timestamps is not None
    assert log.source_timestamps.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(log.source_timestamps, [12.5, 12.75, 13.0])
    assert not log.source_timestamps.flags.writeable
    assert log.metadata["ibt"]["variables"]["Vector"]["count"] == 2
    assert log.metadata["ibt"]["variables"]["Vector"]["count_as_time"] is True
    assert log.metadata["ibt"]["variables"]["Signed"]["type_code"] == 2
    assert log.metadata["ibt"]["variables"]["SteeringWheelTorque_ST"]["count"] == 6
    assert log.metadata["ibt"]["variables"]["SteeringWheelTorque_ST"]["count_as_time"] is True
    assert log.raw_channels["Vector"].units == "m/s"
    assert log.raw_channels["Vector"].provenance["subsample_rate"] == 20.0
    assert log.raw_channels["SteeringWheelTorque_ST"].provenance["subsample_rate"] == 60.0


def test_canonical_ibt_conversions_and_metadata(tmp_path):
    path = tmp_path / "converted.ibt"
    variables = [
        _variable("Speed", 4, [10.0, 11.5, 20.0], units="m/s"),
        _variable("SteeringWheelAngle", 4, [np.pi / 2, 0, -np.pi / 4], units="rad"),
        _variable("YawRate", 4, [np.pi / 2, 0, -np.pi / 4], units="rad/s"),
        _variable("LatAccel", 4, [9.80665, -4.903325, 0], units="m/s^2"),
        _variable("LongAccel", 4, [-9.80665, 0, 19.6133], units="m/s^2"),
        _variable("OilPress", 4, [3.1, 4.2, 0.5], units="bar"),
        _variable("LFbrakeLinePress", 4, [1, 2, 3], units="bar"),
        _variable("LFspeed", 4, [15, 0, 5], units="m/s"),
    ]
    _write_ibt(path, variables, n_ticks=3, tick_rate=10)

    log = DataLog()
    log.from_ibt_log(path)

    assert log.raw_channels == {}
    np.testing.assert_allclose(_samples(log, CH_GROUND_SPEED), [36, 41.4, 72])
    np.testing.assert_allclose(_samples(log, CH_STEERING_ANGLE), [90, 0, -45])
    np.testing.assert_allclose(_samples(log, CH_YAW_RATE), [90, 0, -45])
    np.testing.assert_allclose(_samples(log, CH_CG_ACCEL_LAT), [1, -0.5, 0])
    np.testing.assert_allclose(_samples(log, CH_CG_ACCEL_LON), [-1, 0, 2])
    np.testing.assert_allclose(_samples(log, CH_ENGINE_OIL_PRESS), [310, 420, 50])
    np.testing.assert_allclose(_samples(log, CH_BRAKE_PRESS_FL), [100, 200, 300])
    np.testing.assert_allclose(_samples(log, CH_WHEEL_SPEED_FL), [54, 0, 18])
    assert log.channels[CH_GROUND_SPEED].units == "km/h"
    assert log.channel_metadata[CH_GROUND_SPEED] == {
        "source": "IBT", "variable": "Speed", "sample_rate": 10.0,
        "native_units": "m/s", "converted": True,
    }
    assert log.channel_metadata[CH_YAW_RATE]["variable"] == "YawRate"
    assert log.channel_metadata[CH_ENGINE_OIL_PRESS]["converted"] is True


def test_malformed_and_missing_variable_headers_are_skipped(tmp_path):
    path = tmp_path / "malformed-variables.ibt"
    variables = [
        _variable("Speed", 4, [5, 6, 7], units="m/s", offset=0),
        _variable("NegativeOffset", 2, [1, 2, 3], offset=-1),
        _variable("PastBuffer", 4, [1, 2, 3], offset=30),
        _variable("UnknownType", 99, None, offset=8),
        _variable("ZeroCount", 2, None, count=0, offset=12),
        _variable("", 2, [1, 2, 3], offset=16),
    ]
    _write_ibt(path, variables, n_ticks=3, tick_rate=10, buf_len=32)

    log = DataLog()
    log.from_ibt_log(path, raw_channels="all")

    valid_names = {"Speed"}
    assert set(log.metadata["ibt"]["variables"]) == valid_names
    assert set(log.raw_channels) == valid_names
    np.testing.assert_array_equal(log.raw_channels["Speed"].values, [5, 6, 7])
    np.testing.assert_allclose(_samples(log, CH_GROUND_SPEED), [18, 21.6, 25.2])


@pytest.mark.parametrize("geometry", [{"tick_rate": 0}, {"buf_len": 0}])
def test_invalid_zero_tick_rate_or_buffer_length_returns_empty_log(tmp_path, geometry):
    path = tmp_path / f"invalid-{next(iter(geometry))}.ibt"
    _write_ibt(
        path,
        [_variable("Speed", 4, [1, 2], units="m/s")],
        n_ticks=2,
        tick_rate=geometry.get("tick_rate", 10),
        buf_len=geometry.get("buf_len", 16),
    )

    log = DataLog()
    log.from_ibt_log(path, raw_channels="all")

    assert log.channels == {}
    assert log.raw_channels == {}
    assert "ibt" not in log.metadata


def test_session_time_is_normalized_and_bad_steps_use_tick_period(tmp_path):
    path = tmp_path / "repaired-time.ibt"
    source_time = np.array([100.0, 100.25, 100.25, 100.0, 100.5, 100.75], dtype="<f8")
    _write_ibt(
        path,
        [
            _variable("SessionTime", 5, source_time, units="s", count_as_time=True),
            _variable("Speed", 4, [0, 1, 2, 3, 4, 5], units="m/s"),
        ],
        n_ticks=6,
        tick_rate=4,
    )

    log = DataLog()
    log.from_ibt_log(path, raw_channels="all")

    np.testing.assert_array_equal(log.source_timestamps, source_time)
    np.testing.assert_allclose(log.channels[CH_GROUND_SPEED].timestamps, [0, 0.25, 0.5, 0.75, 1.25, 1.5])
    assert log.metadata["ibt"]["time_source"] == "SessionTime"
    assert log.metadata["ibt"]["time_repaired"] is True
    for view in log.raw_channels.values():
        assert len(view.timestamps) == 6
        assert len(view.values) == 6
    assert len(log.raw_channels["Speed"].values) == 6


def test_complete_and_partial_lap_views_are_readonly_shared_slices(tmp_path):
    path = tmp_path / "lap-views.ibt"
    _write_ibt(
        path,
        [
            _variable("SessionTime", 5, [20.0 + i / 10 for i in range(10)], units="s"),
            _variable("Lap", 2, [1, 1, 2, 2, 2, 3, 3, 1, 1, 1]),
            _variable("LapDistPct", 4, [0.5, 0.95, 0.05, 0.4, 0.8, 0.02, 0.3, 0.1, 0.4, 0.8]),
            _variable("LapCompleted", 2, [0, 0, 1, 1, 1, 2, 2, 0, 0, 0]),
            _variable("SessionNum", 2, [1] * 10),
            _variable("IsOnTrack", 1, [True] * 10),
            _variable("OnPitRoad", 1, [False] * 10),
            _variable("Speed", 4, list(range(10)), units="m/s"),
            _variable("RawVector", 4, [[i, i + 100] for i in range(10)], units="raw", count=2),
        ],
        n_ticks=10,
        tick_rate=10,
    )

    log = DataLog()
    log.from_ibt_log(path, raw_channels=["RawVector"])

    laps = list(log.iter_laps())
    complete_info = next(info for info in log.laps_info["laps"] if info["complete"])
    partial_info = next(
        info for info in log.laps_info["laps"]
        if info["type"] == "Partial" and info["source_lap_num"] == 1
    )
    complete = log.get_lap(
        complete_info["lap_num"],
        channels=[CH_GROUND_SPEED, "MissingChannel"],
        raw_channels=["RawVector", "MissingRaw"],
    )
    partial = log.get_lap(
        partial_info["lap_num"], channels=[CH_GROUND_SPEED], raw_channels=["RawVector"],
    )

    assert [lap.info["source_lap_num"] for lap in laps] == [1, 2, 3, 1]
    assert complete.info["complete"] is True
    assert complete.info["source_lap_num"] == 2
    assert complete.info["type"] == "Timed"
    assert partial.info["complete"] is False
    assert partial.info["source_lap_num"] == 1
    assert partial.info["type"] == "Partial"
    assert set(complete.channels) == {CH_GROUND_SPEED}
    assert set(complete.raw_channels) == {"RawVector"}

    np.testing.assert_allclose(complete.channels[CH_GROUND_SPEED].timestamps, [0.2, 0.3, 0.4])
    np.testing.assert_array_equal(complete.channels[CH_GROUND_SPEED].values, [7.2, 10.8, 14.4])
    np.testing.assert_allclose(partial.channels[CH_GROUND_SPEED].timestamps, [0.7, 0.8, 0.9])
    np.testing.assert_array_equal(partial.channels[CH_GROUND_SPEED].values, [25.2, 28.8, 32.4])
    np.testing.assert_array_equal(partial.raw_channels["RawVector"].values, [[7, 107], [8, 108], [9, 109]])

    source_speed = log.channels[CH_GROUND_SPEED].values
    source_speed_times = log.channels[CH_GROUND_SPEED].timestamps
    source_raw = log.raw_channels["RawVector"].values
    assert np.shares_memory(complete.channels[CH_GROUND_SPEED].values, source_speed)
    assert np.shares_memory(complete.channels[CH_GROUND_SPEED].timestamps, source_speed_times)
    assert np.shares_memory(partial.raw_channels["RawVector"].values, source_raw)
    assert not complete.channels[CH_GROUND_SPEED].values.flags.writeable
    assert not complete.channels[CH_GROUND_SPEED].timestamps.flags.writeable
    assert not partial.raw_channels["RawVector"].values.flags.writeable
    with pytest.raises(ValueError):
        complete.channels[CH_GROUND_SPEED].values[0] = -1
    with pytest.raises(ValueError):
        partial.raw_channels["RawVector"].values[0, 0] = -1
    with pytest.raises(TypeError):
        operator.setitem(complete.info, "complete", False)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        operator.setitem(complete.channels, "missing", None)  # pyright: ignore[reportCallIssue, reportArgumentType]
    with pytest.raises(TypeError):
        operator.setitem(complete.raw_channels, "missing", None)  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_raw_all_is_not_added_to_motec_channels(tmp_path):
    path = tmp_path / "raw-not-exported.ibt"
    _write_ibt(
        path,
        [
            _variable("Speed", 4, [1, 2, 3], units="m/s"),
            _variable("RPM", 2, [1000, 2000, 3000], units="rpm"),
            _variable("OpaqueRawSignal", 3, [1, 2, 3], units="bits"),
        ],
        n_ticks=3,
        tick_rate=10,
    )

    log = DataLog()
    log.from_ibt_log(path, raw_channels="all")
    assert "OpaqueRawSignal" in log.raw_channels
    assert "OpaqueRawSignal" not in log.channels

    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)
    exported_names = {channel.name for channel in motec.ld_channels}
    assert exported_names == set(log.channels)
    assert "OpaqueRawSignal" not in exported_names


def test_synthetic_ibt_cli_writes_compatible_ld_and_ldx(tmp_path):
    source = tmp_path / "cli-smoke.ibt"
    output = tmp_path / "cli-smoke.ld"
    _write_ibt(
        source,
        [
            _variable("SessionTime", 5, [5.0, 5.1, 5.2, 5.3], units="s"),
            _variable("Speed", 4, [1, 2, 3, 4], units="m/s"),
            _variable("RPM", 2, [1000, 2000, 3000, 4000], units="rpm"),
            _variable("OpaqueRawSignal", 3, [1, 2, 3, 4], units="bits"),
        ],
        n_ticks=4,
        tick_rate=10,
    )

    result = _run_cli_in_process([str(source), "IBT", "--output", str(output)])

    assert result.returncode == 0, result.stdout + result.stderr
    ldx = output.with_suffix(".ldx")
    assert output.is_file()
    assert ldx.is_file()
    assert ET.parse(ldx).getroot().tag == "LDXFile"
    from track_telemetry_converter._vendor.ldparser import ldData

    parsed = ldData.fromfile(str(output))
    exported_names = {channel.name for channel in parsed.channs}
    assert CH_GROUND_SPEED in exported_names
    assert "OpaqueRawSignal" not in exported_names
