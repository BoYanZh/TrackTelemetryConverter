"""Tests for CSV, PB Buddy, Accessport, and RaceChrono CSV formats."""

import os
import tempfile

import numpy as np
from conftest import _dedup_channels, _read_lines

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.output import atomic_write_motec_pair


def test_csv_smoke():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))


def test_pbbuddy_smoke():
    log = DataLog()
    lines = _read_lines("pbbuddy_sample.csv")
    log.from_pbbuddy_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.metadata.get("venue_name") == "Thunderhill East (Cyclone)"
    assert log.duration() > 0


def test_aim_solo_smoke():
    log = DataLog()
    lines = _read_lines("aim_solo_sample.csv")
    log.from_racechrono_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "CG Accel Lateral" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.metadata.get("venue_name") == "Laguna Seca"
    assert log.datetime.strftime("%Y-%m-%d") == "2026-02-14"
    assert log.duration() > 0


def test_csv_resample_and_math():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    for ch in log.channels.values():
        assert len(ch.messages) > 0, f"Channel {ch.name} empty after resample"
    log.calculate_math_channels()


def test_accessport_smoke():
    log = DataLog()
    log.from_accessport_log(_read_lines("accessport_sample.csv"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_accessport_resample_and_math():
    log = DataLog()
    log.from_accessport_log(_read_lines("accessport_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    log.calculate_math_channels()


def test_racechrono_smoke():
    log = DataLog()
    log.from_racechrono_log(_read_lines("racechrono_sample.csv"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_racechrono_resample_and_math():
    log = DataLog()
    log.from_racechrono_log(_read_lines("racechrono_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    log.calculate_math_channels()


def test_csv_middle_non_numeric_column_keeps_right_columns():
    log = DataLog()
    lines = [
        "time,colA,colB,colC\n",
        "0.0,1.0,XXX,100.0\n",
        "0.1,2.0,YYY,200.0\n",
        "0.2,3.0,ZZZ,300.0\n",
    ]
    log.from_csv_log(lines)
    assert "colA" in log.channels
    assert "colC" in log.channels
    assert "colB" not in log.channels
    colC = log.channels["colC"].messages
    assert len(colC) == 3, f"colC messages: {len(colC)}"
    assert colC[-1].value == 300.0


def test_auto_frequency_detection():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    freq = log.detect_native_frequency()
    assert freq > 0, f"Invalid auto frequency: {freq}"
    resampled_f = log.resample("auto")
    assert resampled_f == freq


def test_racechrono_long_names_prefer_canbus_and_roundtrip():
    lines = [
        "Time (s),Longitudinal acceleration (G) *calc,"
        "Longitudinal acceleration (G) *canbus,Combined acceleration (G) *canbus,"
        "Engine oil temperature (.C) *canbus\n",
        "s,G,G,G,C\n",
        "0.0,0.10,0.20,0.30,100.0\n",
        "1.0,0.11,0.21,0.31,101.0\n",
    ]
    log = DataLog()
    log.from_racechrono_log(lines)

    assert "Longitudinal acceleration (G) *calc" not in log.channels
    assert np.array_equal(log.channels["CG Accel Longitudinal"].values, [0.20, 0.21])
    assert np.array_equal(log.channels["G Force Combined"].values, [0.30, 0.31])
    assert np.array_equal(log.channels["Engine Oil Temp"].values, [100.0, 101.0])
    assert all(len(name.encode("ascii")) <= 32 for name in log.channels)

    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ld_path = os.path.join(tmp_dir, "long_names.ld")
        ldx_path = os.path.join(tmp_dir, "long_names.ldx")
        atomic_write_motec_pair(motec, log, ld_path, ldx_path)
        parsed = ldData.fromfile(ld_path)
        assert [channel.name for channel in parsed.channs] == list(log.channels)
