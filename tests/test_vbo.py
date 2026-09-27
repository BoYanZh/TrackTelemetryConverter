"""Tests for Racelogic VBOX VBO format."""

from conftest import _dedup_channels, _read_lines

from track_telemetry_converter.log import DataLog


def test_vbo_smoke():
    log = DataLog()
    lines = _read_lines("vbo_sample.vbo")
    log.from_vbo_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert "CG Accel Lateral" in log.channels
    assert "CG Accel Longitudinal" in log.channels
    assert log.datetime.strftime("%Y-%m-%d") == "2026-01-01"
    assert log.duration() > 0
