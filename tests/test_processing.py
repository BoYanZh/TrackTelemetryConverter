"""Tests for data interpolation, resampling, and math/derived channel processing."""

import numpy as np
from conftest import _read_lines

from track_telemetry_converter.derived import derive_gear_from_rpm_speed
from track_telemetry_converter.interpolation import _interp_zoh
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.models import Message


def test_g_source_modes():
    # 1. Test 'calc' mode on AiM log: forces deriving G from GPS
    log1 = DataLog()
    log1.from_racechrono_log(_read_lines("aim_solo_sample.csv"))
    log1.calculate_math_channels(g_source="calc")
    assert "CG Accel Lateral" in log1.channels
    assert "CG Accel Longitudinal" in log1.channels

    # 2. Test 'sensor' mode on PB Buddy log (no raw sensor Gs): leaves G channels derived-free
    log2 = DataLog()
    log2.from_pbbuddy_log(_read_lines("pbbuddy_sample.csv"))
    log2.calculate_math_channels(g_source="sensor")
    assert "CG Accel Lateral" not in log2.channels

    # 3. Test 'auto' mode on PB Buddy log: automatically derives Gs from GPS
    log3 = DataLog()
    log3.from_pbbuddy_log(_read_lines("pbbuddy_sample.csv"))
    log3.calculate_math_channels(g_source="auto")
    assert "CG Accel Lateral" in log3.channels


def test_discrete_channels_use_zoh():
    log = DataLog()
    log.add_channel("Gear", "", float, 0)
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Gear"].messages = [
        type("M", (), {"timestamp": 0.0, "value": 2.0})(),
        type("M", (), {"timestamp": 1.0, "value": 3.0})(),
    ]
    log.channels["Ground Speed"].messages = [
        type("M", (), {"timestamp": 0.0, "value": 0.0})(),
        type("M", (), {"timestamp": 1.0, "value": 100.0})(),
    ]
    log.resample(10)

    gear_vals = [int(m.value) for m in log.channels["Gear"].messages]
    spd_vals = [m.value for m in log.channels["Ground Speed"].messages]

    # Gear must stay integer (ZOH)
    assert all(v in (2, 3) for v in gear_vals), f"Gear values not discrete: {gear_vals}"
    # Speed gets linear interp
    assert any(0 < v < 100 for v in spd_vals), f"Speed not interpolated: {spd_vals}"


def test_mask_interp_gaps_default():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [
        Message(0.0, 10.0),
        Message(3.0, 40.0),
    ]
    log.resample(10, mask_interp_gaps=False)
    vals = [m.value for m in log.channels["Ground Speed"].messages]
    assert not any(np.isnan(v) for v in vals)
    mid_idx = len(vals) // 2
    assert 10.0 < vals[mid_idx] < 40.0


def test_mask_interp_gaps_enabled():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [
        Message(0.0, 10.0),
        Message(3.0, 40.0),
    ]
    log.resample(10, mask_interp_gaps=True)
    vals = [m.value for m in log.channels["Ground Speed"].messages]
    mid_idx = len(vals) // 2
    assert np.isnan(vals[mid_idx])


def test_empty_log_guards():
    log = DataLog()
    assert log.duration() == 0.0
    assert log.start() == 0.0
    assert log.end() == 0.0
    log.resample(20)  # must not crash


def test_interp_zoh_discrete():
    src_t = np.array([0.0, 1.0, 2.0])
    src_v = np.array([2.0, 3.0, 1.0])
    new_t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    result = _interp_zoh(new_t, src_t, src_v)
    expected = np.array([2.0, 2.0, 3.0, 3.0, 1.0, 1.0])
    assert np.array_equal(result, expected), f"Got {result}, expected {expected}"


def test_gear_preserved_after_resample():
    log = DataLog()
    log.add_channel("Gear", "", float, 0)
    n = 100
    for i in range(n):
        gear = 2 if i < n // 3 else (3 if i < 2 * n // 3 else 4)
        log.channels["Gear"].messages.append(
            type("M", (), {"timestamp": float(i) * 0.05, "value": float(gear)})()
        )
    log.resample(20)
    for m in log.channels["Gear"].messages:
        assert float(m.value).is_integer(), f"Non-integer gear value: {m.value}"


def test_sector_beacons_detection():
    log = DataLog()
    log.traps = [
        {"name": "Start/Finish", "lat": 37.0, "lon": -122.0, "type": 3},
        {"name": "Split 1", "lat": 37.0001, "lon": -122.0001, "type": 4},
    ]
    log.add_channel("GPS Latitude", "deg", float, 7)
    log.add_channel("GPS Longitude", "deg", float, 7)
    log.channels["GPS Latitude"].messages = [
        Message(0.0, 37.0002), Message(1.0, 37.0), Message(2.0, 37.0002),
        Message(10.0, 37.0003), Message(11.0, 37.0001), Message(12.0, 37.0003)
    ]
    log.channels["GPS Longitude"].messages = [
        Message(0.0, -122.0002), Message(1.0, -122.0), Message(2.0, -122.0002),
        Message(10.0, -122.0003), Message(11.0, -122.0001), Message(12.0, -122.0003)
    ]
    beacons = log.detect_beacons(min_speed_kmh=0.0, min_time_sec=0.0)
    assert len(beacons) >= 2
    assert beacons[0][1] in ("Start/Finish", "Split 1")


def test_gear_derivation_mt_at_presets():
    from track_telemetry_converter.derived import (
        DEFAULT_GEAR_RATIO_THRESHOLDS_AT,
        DEFAULT_GEAR_RATIO_THRESHOLDS_MT,
    )

    def gear_for(ratio, **kw):
        log = DataLog()
        log.add_channel("Engine RPM", "rpm", float, 0)
        log.add_channel("Ground Speed", "km/h", float, 2)
        log.channels["Engine RPM"].messages = [
            Message(0.0, ratio * 100.0), Message(1.0, ratio * 100.0)]
        log.channels["Ground Speed"].messages = [
            Message(0.0, 100.0), Message(1.0, 100.0)]
        log.calculate_math_channels(**kw)
        return list(log.channels["Gear"].values)

    # GR86/BRZ 6MT theoretical ratios classify 1-6 with the MT preset.
    for ratio, expected in zip(
        (126, 76, 53.5, 42.1, 34.7, 26.6), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        assert gear_for(ratio, gearbox="mt") == [expected, expected]
    # Previously-misclassified MT mid-gear ratios (old default gave +1 gear).
    assert gear_for(41.5, gearbox="mt") == [4.0, 4.0]
    assert gear_for(51.5, gearbox="mt") == [3.0, 3.0]
    # GR86/BRZ 6AT theoretical ratios classify 1-6 with gearbox="at".
    for ratio, expected in zip(
        (117, 68, 46.5, 33.1, 23.6, 19.2), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)):
        assert gear_for(ratio, gearbox="at") == [expected, expected]
    # Explicit thresholds still override the preset.
    assert gear_for(41.5, gear_ratio_thresholds=(110, 70, 52, 42, 33, 20)) == [5.0, 5.0]
    # Default is opt-out: no thresholds and no gearbox means no derived Gear.
    default_log = DataLog()
    default_log.add_channel("Engine RPM", "rpm", float, 0)
    default_log.channels["Engine RPM"].messages = [
        Message(0.0, 4200.0), Message(1.0, 4200.0)]
    default_log.add_channel("Ground Speed", "km/h", float, 2)
    default_log.channels["Ground Speed"].messages = [
        Message(0.0, 100.0), Message(1.0, 100.0)]
    default_log.calculate_math_channels()
    assert "Gear" not in default_log.channels
    assert (len(DEFAULT_GEAR_RATIO_THRESHOLDS_MT) == 6
            and len(DEFAULT_GEAR_RATIO_THRESHOLDS_AT) == 6)


def test_gear_derivation_aligns_channels_by_timestamp():
    from track_telemetry_converter.derived import DEFAULT_GEAR_RATIO_THRESHOLDS_MT
    log = DataLog()
    log.add_channel("Engine RPM", "rpm", float, 0)
    log.channels["Engine RPM"].messages = [
        Message(i / 10.0, 3000.0) for i in range(21)
    ]
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].messages = [
        Message(0.0, 30.0), Message(1.0, 30.0),
        Message(1.0, 30.0), Message(2.0, 30.0)
    ]

    derive_gear_from_rpm_speed(log, ratio_thresholds=DEFAULT_GEAR_RATIO_THRESHOLDS_MT)
    gear = log.channels["Gear"]
    assert np.array_equal(gear.timestamps, np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(gear.values, np.array([2.0, 2.0, 2.0]))
