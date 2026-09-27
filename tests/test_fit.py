"""Tests for Garmin FIT format."""

import os
import warnings

import numpy as np
from conftest import EXAMPLES, _dedup_channels, _import_or_skip

from track_telemetry_converter.log import DataLog


def test_fit_smoke():
    _import_or_skip("fitparse")

    log = DataLog()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        log.from_fit_log(os.path.join(EXAMPLES, "garmin_sample.fit"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.duration() > 0
    _dedup_channels(log)


def test_fit_unique_timestamps_finite_math_and_lap_boundaries():
    _import_or_skip("fitparse")

    log = DataLog()
    log.from_fit_log(os.path.join(EXAMPLES, "garmin_sample.fit"))
    speed = log.channels["Ground Speed"]
    assert np.all(np.diff(speed.timestamps) > 0), "FIT timestamps must be strictly increasing"

    log.calculate_math_channels()
    for name in ("CG Accel Longitudinal", "CG Accel Long Smooth"):
        assert np.all(np.isfinite(log.channels[name].values)), f"{name} contains NaN/Inf"

    laps = log.laps_info["laps"]
    assert len(log.laps_info["beacons"]) == len(laps) + 1
    for previous, current in zip(laps, laps[1:]):
        assert previous["end_time"] == current["start_time"]
