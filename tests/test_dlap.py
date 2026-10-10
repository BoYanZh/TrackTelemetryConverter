"""Tests for Dragy .dlap archives (outer ZIP + AES .group + loopdata/OBD/laps)."""

import os
import tempfile

from conftest import (
    _assert_cli_roundtrip,
    _dedup_channels,
    _write_minimal_dlap,
)

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers import auto_detect_log_type

EXPECTED_GPS = (
    "Ground Speed",
    "GPS Latitude",
    "GPS Longitude",
    "GPS Altitude",
    "GPS Heading",
    "GPS Satellites",
    "CG Accel Longitudinal",
    "Brake Pos",
    "Corr Dist",
    "GPS Accuracy",
)
EXPECTED_OBD = (
    "Engine RPM",
    "Accelerator Pos",
    "Brake Press",
    "Engine Oil Temp",
    "Coolant Temp",
    "Manifold Press",
)


def _parse_minimal_dlap(**kwargs):
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "session.dlap")
        _write_minimal_dlap(path)
        log = DataLog()
        log.from_dlap_log(path, **kwargs)
        _dedup_channels(log)
        return log


def test_dlap_parse_smoke():
    log = _parse_minimal_dlap()
    for name in EXPECTED_GPS + EXPECTED_OBD:
        assert name in log.channels, f"missing channel {name}"
    assert abs(log.duration() - 240 * 0.04) < 0.5
    assert log.datetime is not None
    assert log.datetime.strftime("%Y-%m-%d") == "2026-10-03"
    assert log.metadata.get("venue_name") == "Synthetic Test Track"
    assert log.metadata.get("vehicle_id") == "TestCar"


def test_dlap_laps_info():
    log = _parse_minimal_dlap()
    laps_info = log.laps_info
    assert laps_info["total_laps"] == 2
    assert len(laps_info["laps"]) == 2
    assert laps_info["fastest_time"] > 0
    assert len(laps_info["beacons"]) == 3
    first, second = laps_info["laps"]
    assert first["lap_num"] == 1
    assert second["lap_num"] == 2
    assert second["start_time"] > first["end_time"]


def test_dlap_target_lap_filters_samples():
    full = _parse_minimal_dlap()
    lap2 = _parse_minimal_dlap(target_lap="2")
    assert lap2.duration() < full.duration()
    assert lap2.duration() > 0
    assert "Ground Speed" in lap2.channels


def test_dlap_auto_detect():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "session.dlap")
        _write_minimal_dlap(path)
        assert auto_detect_log_type(path) == "DLAP"


def test_dlap_cli_roundtrip():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "session.dlap")
        _write_minimal_dlap(path)
        _assert_cli_roundtrip(path, "DLAP", expected_channels=EXPECTED_GPS)


def test_dlap_requires_crypto(monkeypatch):
    import sys

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "session.dlap")
        _write_minimal_dlap(path)
        monkeypatch.setitem(sys.modules, "Crypto", None)
        monkeypatch.setitem(sys.modules, "Crypto.Cipher", None)
        log = DataLog()
        try:
            log.from_dlap_log(path)
        except RuntimeError as exc:
            assert "pycryptodome" in str(exc)
        else:
            raise AssertionError("expected RuntimeError without Crypto")
