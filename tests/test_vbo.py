"""Tests for Racelogic VBOX VBO format."""

from conftest import _dedup_channels, _read_lines

from track_telemetry_converter.log import DataLog


def test_vbo_brakepress_converts_bar_to_kpa_and_records_winner():
    log = DataLog()
    log.from_vbo_log([
        "File created on 01/01/2026 at 10:00:00\n",
        "[column names]\n",
        "time velocity velocity-calc BrakePress\n",
        "[data]\n",
        "000000.00 100.0 101.0 1.5\n",
        "000001.00 110.0 111.0 2.5\n",
    ])
    assert log.channels["Brake Press"].units == "kPa"
    assert list(log.channels["Brake Press"].values) == [150.0, 250.0]
    # Later columns win; the winner is recorded instead of being implicit.
    assert list(log.channels["Ground Speed"].values) == [101.0, 111.0]
    prov = log.metadata["channel_provenance"]["Ground Speed"]
    assert prov["winner"] == "velocity-calc"
    assert prov["raw_columns"] == ["velocity", "velocity-calc"]


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
    assert log.datetime is not None
    assert log.datetime.strftime("%Y-%m-%d") == "2026-01-01"
    assert log.duration() > 0
