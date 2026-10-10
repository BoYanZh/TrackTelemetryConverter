"""Tests for Dragy .dlap archives (outer ZIP + AES .group + loopdata/OBD/laps)."""

import os
import tempfile

from conftest import (
    _assert_cli_roundtrip,
    _dedup_channels,
    _write_minimal_dlap,
)

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.models import Message
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
    assert log.datetime_utc is not None
    # Absolute instant (timezone-independent); the synthetic epoch is
    # 2026-10-03 23:26:59 UTC, which renders as Oct 4 east of UTC.
    assert log.datetime_utc.strftime("%Y-%m-%d") == "2026-10-03"
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


def _resampled_minimal_dlap():
    from track_telemetry_converter.exporters.dlap_export import write_dlap

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "session.dlap")
        _write_minimal_dlap(path)
        log = DataLog()
        log.from_dlap_log(path)
        _dedup_channels(log)
        log.resample(25)
        out = os.path.join(tmp_dir, "roundtrip.dlap")
        assert write_dlap(log, out)
        reparsed = DataLog()
        reparsed.from_dlap_log(out)
        _dedup_channels(reparsed)
        return log, reparsed


def test_dlap_export_roundtrip():
    import numpy as np

    log, reparsed = _resampled_minimal_dlap()
    for name in ("Ground Speed", "GPS Latitude", "GPS Longitude"):
        assert name in reparsed.channels, f"missing channel {name}"
        original = np.asarray(log.channels[name].values)
        back = np.asarray(reparsed.channels[name].values)
        assert len(back) == len(original)
        assert np.allclose(back, original, atol=0.01, equal_nan=True)
    # OBD covers only part of the session: values must match where present
    # and stay NaN outside the recorded window (no forward-fill).
    original = np.asarray(log.channels["Engine RPM"].values)
    back = np.asarray(reparsed.channels["Engine RPM"].values)
    assert len(back) == len(original)
    finite = np.isfinite(back)
    assert 0 < finite.sum() < len(back)
    assert np.allclose(back[finite], original[finite], atol=0.05)
    assert reparsed.laps_info["total_laps"] == 2
    assert reparsed.datetime_utc == log.datetime_utc


def test_dlap_export_requires_gps():
    from track_telemetry_converter.exporters.dlap_export import write_dlap

    log = DataLog()
    log.add_channel("Engine RPM", "rpm", float, 0)
    log.channels["Engine RPM"].messages = [Message(0.0, 3000.0)]
    with tempfile.NamedTemporaryFile(suffix=".dlap", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        assert not write_dlap(log, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_dlap_export_without_obd():
    import zipfile

    from track_telemetry_converter.exporters.dlap_export import write_dlap

    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [Message(i * 0.04, 50.0) for i in range(50)]
    log.add_channel("GPS Latitude", "deg", float, 7)
    log.channels["GPS Latitude"].messages = [Message(i * 0.04, 39.5) for i in range(50)]
    log.add_channel("GPS Longitude", "deg", float, 7)
    log.channels["GPS Longitude"].messages = [Message(i * 0.04, -122.3) for i in range(50)]
    log.resample(25)
    with tempfile.TemporaryDirectory() as tmp_dir:
        out = os.path.join(tmp_dir, "noobd.dlap")
        assert write_dlap(log, out)
        with zipfile.ZipFile(out) as outer:
            assert len(outer.namelist()) == 1
        reparsed = DataLog()
        reparsed.from_dlap_log(out)
        assert "Ground Speed" in reparsed.channels
        assert "Engine RPM" not in reparsed.channels


def test_cli_dlap_flag_and_source_guard():
    from conftest import _run_cli_in_process

    with tempfile.TemporaryDirectory() as tmp_dir:
        src = os.path.join(tmp_dir, "in.dlap")
        _write_minimal_dlap(src)
        result = _run_cli_in_process(
            [src, "DLAP", "--output", os.path.join(tmp_dir, "in"), "--dlap"]
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert os.path.isfile(os.path.join(tmp_dir, "in_export.dlap"))
        assert "Saved .dlap track file" in result.stdout
