"""Tests for the Tesla Track Mode CSV parser."""

import numpy as np

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers import auto_detect_log_type

_TESLA_HEADER = (
    "Lap,Elapsed Time (ms),Speed (MPH),Latitude (decimal),Longitude (decimal),"
    "Lateral Acceleration (m/s^2),Longitudinal Acceleration (m/s^2),"
    "Throttle Position (%),Brake Pressure (bar),Steering Angle (deg),"
    "Yaw Rate (rad/s),State of Charge (%)\n"
)


def test_tesla_autodetect():
    assert auto_detect_log_type("tests/fixtures/tesla_sample.csv") == "TESLA"


def test_tesla_unit_conversions():
    from conftest import _read_lines

    log = DataLog()
    log.from_tesla_log(_read_lines("tesla_sample.csv"))
    assert log.channels["Ground Speed"].units == "km/h"
    assert log.channels["CG Accel Lateral"].units == "G"
    assert log.channels["Brake Press"].units == "kPa"
    assert log.channels["Chassis Yaw Rate"].units == "deg/s"

    lines = _read_lines("tesla_sample.csv")
    raw = [line.strip().split(",") for line in lines[1:] if line.strip()]
    mph = float(raw[5][2])
    assert log.channels["Ground Speed"].values[5] == mph * 1.609344
    ms2 = float(raw[5][5])
    assert log.channels["CG Accel Lateral"].values[5] == ms2 / 9.80665
    bar = float(raw[5][8])
    assert log.channels["Brake Press"].values[5] == bar * 100.0
    assert log.channels["GPS Latitude"].values[5] == float(raw[5][3])


def test_tesla_cumulative_timebase_across_lap_reset():
    rows = [_TESLA_HEADER]
    for et in (0, 1000, 2000):
        rows.append(f"0,{et},60.0,39.5,-122.3,0.0,0.0,50.0,1.0,0.0,0.0,80.0\n")
    for et in (0, 1000, 2000):
        rows.append(f"1,{et},60.0,39.5,-122.3,0.0,0.0,50.0,1.0,0.0,0.0,80.0\n")
    log = DataLog()
    log.from_tesla_log(rows)
    stamps = [m.timestamp for m in log.channels["Ground Speed"].messages]
    assert stamps == [0.0, 1.0, 2.0, 2.0, 3.0, 4.0]
    assert all(b >= a for a, b in zip(stamps, stamps[1:], strict=False))


def test_tesla_lap_zero_becomes_out_lap():
    rows = [_TESLA_HEADER]
    for et in range(0, 31000, 1000):
        rows.append(f"0,{et},60.0,39.5,-122.3,0.0,0.0,50.0,1.0,0.0,0.0,80.0\n")
    for et in range(0, 41000, 1000):
        rows.append(f"1,{et},60.0,39.5,-122.3,0.0,0.0,50.0,1.0,0.0,0.0,80.0\n")
    log = DataLog()
    log.from_tesla_log(rows)
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap", "Timed"]
    assert log.laps_info["total_laps"] == 2
    assert log.laps_info["fastest_time"] == 40.0


def test_tesla_target_lap_and_errors(capsys):
    rows = [_TESLA_HEADER]
    for et in (0, 1000):
        rows.append(f"1,{et},60.0,39.5,-122.3,0.0,0.0,50.0,1.0,0.0,0.0,80.0\n")
    for et in (0, 1000):
        rows.append(f"2,{et},80.0,39.5,-122.3,0.0,0.0,50.0,1.0,0.0,0.0,80.0\n")
    log = DataLog()
    log.from_tesla_log(rows, target_lap="2")
    assert np.array_equal(
        log.channels["Ground Speed"].values, [80.0 * 1.609344] * 2,
    )

    log = DataLog()
    log.from_tesla_log(rows, target_lap="9")
    assert "WARNING" in capsys.readouterr().out

    empty = DataLog()
    empty.from_tesla_log([])
    assert len(empty.channels) == 0
    no_header = DataLog()
    no_header.from_tesla_log(["foo,bar\n", "1,2\n"])
    assert len(no_header.channels) == 0
    assert "ERROR" in capsys.readouterr().out
