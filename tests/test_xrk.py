"""Tests for AIM XRK / XRZ format."""

import os

import numpy as np
from conftest import EXAMPLES, _dedup_channels, _import_or_skip

from track_telemetry_converter.log import DataLog


def test_xrk_smoke():
    _import_or_skip("libxrk")

    log = DataLog()
    log.from_xrk_log(os.path.join(EXAMPLES, "aim_sample.xrk"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.duration() > 0
    _dedup_channels(log)


def test_xrk_units_time_origin_and_datetime():
    _import_or_skip("libxrk")

    log = DataLog()
    log.from_xrk_log(os.path.join(EXAMPLES, "aim_sample.xrk"))

    speed = log.channels["Ground Speed"]
    rpm = log.channels["Engine RPM"]
    gps = log.channels["GPS Latitude"]
    assert speed.units == "km/h"
    assert max(speed.values) > 100.0, "XRK m/s speed was labelled km/h without conversion"
    assert rpm.start() == 0.0
    assert 0.0 < gps.start() < 0.1, "GPS must retain its shared-clock offset"
    assert abs(gps.end() - rpm.end()) < 0.2, "GPS and logger clocks must stay aligned"
    assert np.all(np.isfinite(gps.timestamps))
    assert np.all(np.diff(gps.timestamps) > 0), "XRK timestamps must be unique and monotonic"
    assert log.datetime.strftime("%Y-%m-%d %H:%M:%S") == "2016-01-23 12:09:04"
