"""Tests for GPS course-heading derivation (derived.ensure_gps_heading)."""

import numpy as np

from track_telemetry_converter.derived import (
    course_heading_from_gps,
    ensure_gps_heading,
)
from track_telemetry_converter.log import DataLog


def _log_with_gps(n=200, freq=20.0, lat0=39.0, lon0=-122.0, dlat=0.0, dlon=0.0001):
    log = DataLog()
    t = np.arange(n) / freq
    log.add_channel("GPS Latitude", "deg", float, 7)
    log.channels["GPS Latitude"].set_samples(t, lat0 + dlat * np.arange(n))
    log.add_channel("GPS Longitude", "deg", float, 7)
    log.channels["GPS Longitude"].set_samples(t, lon0 + dlon * np.arange(n))
    return log


def test_course_heading_cardinal():
    # Due east.
    h = course_heading_from_gps([39.0, 39.0, 39.0], [-122.0, -121.999, -121.998])
    np.testing.assert_allclose(h, [90, 90, 90], atol=0.5)
    # Due south.
    h = course_heading_from_gps([39.0, 38.999, 38.998], [-122.0, -122.0, -122.0])
    np.testing.assert_allclose(h, [180, 180, 180], atol=0.5)


def test_ensure_adds_missing_heading():
    log = _log_with_gps()
    assert ensure_gps_heading(log) is True
    assert "GPS Heading" in log.channels
    np.testing.assert_allclose(
        np.asarray(log.channels["GPS Heading"].values, dtype=float),
        np.full(200, 90.0),
        atol=1.0,
    )
    assert log.metadata.get("gps_heading_synthetic") is True


def test_ensure_keeps_recorded_heading():
    log = _log_with_gps()
    t = np.arange(200) / 20.0
    log.add_channel("GPS Heading", "deg", float, 3)
    log.channels["GPS Heading"].set_samples(t, np.full(200, 42.0))
    assert ensure_gps_heading(log) is False
    np.testing.assert_allclose(
        np.asarray(log.channels["GPS Heading"].values, dtype=float),
        np.full(200, 42.0),
    )


def test_ensure_without_gps_returns_false():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples([0.0, 0.1], [10.0, 10.0])
    assert ensure_gps_heading(log) is False
    assert "GPS Heading" not in log.channels


def test_ensure_handles_mismatched_clocks():
    log = DataLog()
    t_lat = np.arange(100) / 10.0
    t_lon = np.arange(200) / 20.0
    log.add_channel("GPS Latitude", "deg", float, 7)
    log.channels["GPS Latitude"].set_samples(t_lat, np.full(100, 39.0))
    log.add_channel("GPS Longitude", "deg", float, 7)
    log.channels["GPS Longitude"].set_samples(t_lon, -122.0 + 0.0001 * np.arange(200))
    assert ensure_gps_heading(log) is True
    vals = np.asarray(log.channels["GPS Heading"].values, dtype=float)
    assert len(vals) == 100
    assert np.all(np.isfinite(vals))
