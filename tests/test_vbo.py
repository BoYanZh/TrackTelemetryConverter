"""Tests for Racelogic VBOX VBO format."""

import math

import numpy as np
import pytest
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


def test_vbo_racechrono_obd_pedal_and_gear_map_to_canonical():
    log = DataLog()
    log.from_vbo_log([
        "File created on 02/11/2026 at 00:31:34\n",
        "[column names]\n",
        "sats time lat long throttle_pos-obd accelerator_pos-obd gear-obd\n",
        "[data]\n",
        "016 000000.00 +2460.060000 +5280.600000 20.0 21.0 3.0\n",
        "016 000001.00 +2460.060000 +5280.580000 80.0 79.0 4.0\n",
    ])
    assert list(log.channels["Throttle Pos"].values) == [20.0, 80.0]
    assert log.channels["Throttle Pos"].units == "%"
    assert list(log.channels["Accelerator Pos"].values) == [21.0, 79.0]
    assert log.channels["Accelerator Pos"].units == "%"
    assert list(log.channels["Gear"].values) == [3.0, 4.0]
    assert log.channels["Gear"].units == ""


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
    assert log.traps == []


def _vbo_drive_lines(gates, duration_s=60.0, freq=10.0):
    """Synthetic VBO: due-east drive at 41.001N crossing the given gates.

    Gates are ``(kind, lon_minutes, name)`` crossed at a steady ~103 km/h;
    latitude stays on the gate midpoints (2460.06 minutes).
    """
    lines = [
        "File created on 02/11/2026 at 00:31:34\n",
        "[header]\n",
        "time\n",
        "[laptiming]\n",
    ]
    for kind, lon_min, name in gates:
        lines.append(
            f"{kind} +{lon_min:.6f} +2460.000000 +{lon_min:.6f} +2460.120000 ¬ {name}\n",
        )
    lines += [
        "[column names]\n",
        "sats time lat long velocity heading\n",
        "[data]\n",
    ]
    n = int(duration_s * freq)
    for i in range(n):
        t = i / freq
        lon_min = 7347.6 - 0.02 * t
        lines.append(
            f"016 0000{t:05.2f} +2460.060000 +{lon_min:.6f} 100.0 90.0\n",
        )
    return lines


def test_vbo_laptiming_traps_parsed():
    from track_telemetry_converter.parsers import vbo_parser

    log = DataLog()
    log.from_vbo_log(_vbo_drive_lines([
        ("Split", 7347.2, "Split 1"),
        ("Start", 7346.9, "Start/Finish"),
        ("Split", 7346.6, "Split 2"),
    ]))
    assert [(t["name"], round(t["lat"], 6), round(t["lon"], 6)) for t in log.traps] == [
        ("Split 1", 41.001, -122.453333),
        ("Start/Finish", 41.001, -122.448333),
        ("Split 2", 41.001, -122.443333),
    ]
    assert all("heading_deg" not in t for t in log.traps)
    beacons = log.detect_beacons()
    assert [(n, round(t, 1)) for t, n in beacons] == [
        ("Split 1", 20.0),
        ("Start/Finish", 35.0),
        ("Split 2", 50.0),
    ]
    # A file without the section keeps traps untouched.
    plain = DataLog()
    plain.from_vbo_log(_vbo_drive_lines([]))
    assert plain.traps == []
    assert vbo_parser._load_vbo_laptiming_traps(plain, {}) is None


def test_vbo_laptiming_resolver_orders_and_rejects():
    from track_telemetry_converter.parsers import vbo_parser

    resolve = vbo_parser._resolve_gate_pair
    anchor = (41.001, -122.45)
    # long,lat order (RaceChrono) and lat,long order agree.
    assert resolve("+7347.200000", "+2460.060000", anchor) == (41.001, -122.45333333333333)
    lat_lon = resolve("+2460.060000", "+7347.200000", anchor)
    assert lat_lon is not None
    lat, lon = lat_lon
    assert abs(lat - 41.001) < 1e-9 and abs(lon + 122.45333333333333) < 1e-9
    # No anchor: magnitude settles it (|lon| > 90 * 60); ambiguous stays None.
    lat_lon = resolve("+7347.200000", "+2460.060000", None)
    assert lat_lon is not None
    lat, lon = lat_lon
    assert abs(lat - 41.001) < 1e-9
    assert resolve("+3000.000000", "+0600.000000", None) is None
    assert resolve("abc", "def", None) is None
    # Stale map far from the driven trace is skipped.
    assert resolve("+0100.000000", "+0100.000000", anchor) is None


def test_vbo_laptiming_fallback_names_and_duplicates():
    lines = _vbo_drive_lines([])
    head, sep, tail = "".join(lines).partition("[laptiming]\n")
    assert sep
    gate_lines = [
        "Start +7347.200000 +2460.000000 +7347.200000 +2460.120000\n",
        "Split +7346.900000 +2460.000000 +7346.900000 +2460.120000\n",
        "Split +7346.900000 +2460.000000 +7346.900000 +2460.120000\n",
        "Bogus line without coordinates\n",
    ]
    log = DataLog()
    log.from_vbo_log((head + sep + "".join(gate_lines) + tail).splitlines(keepends=True))
    assert [t["name"] for t in log.traps] == ["Start/Finish", "Split 1"]


def test_vbo_laptiming_parallel_gates_rejected():
    # A gate running along the direction of travel is a drive-by, not a
    # line crossing: adopting its midpoint would mistime the marker.
    log = DataLog()
    log.from_vbo_log([
        "File created on 02/11/2026 at 00:31:34\n",
        "[laptiming]\n",
        "Split +5280.200000 +2460.000000 +5280.320000 +2460.000000 ¬ Split 1\n",
        "[column names]\n",
        "sats time lat long velocity heading\n",
        "[data]\n",
        "016 000000.00 +2460.060000 +5280.600000 100.0 90.0\n",
        "016 000001.00 +2460.060000 +5280.580000 100.0 90.0\n",
        "016 000002.00 +2460.060000 +5280.100000 100.0 90.0\n",
        "016 000003.00 +2460.060000 +5280.000000 100.0 90.0\n",
    ])
    assert log.traps == []


def _vbo_circle_log(t0=3.9, duration_s=128.0, speed_kmh=162.0, freq=10.0):
    """Synthetic VBO circling back through the Sonoma S/F trap.

    Constant-speed circle (period 120 s) through the trap on a 309-degree
    course, so the first crossing sits in the head gate and (for short
    files) the last in the tail gate. ``[session data]`` names Sonoma, so
    the venue trap -- not file gates -- is exercised.
    """
    from track_telemetry_converter.channels import TRACK_BEACONS

    trap = TRACK_BEACONS["Sonoma Raceway"]
    period = 120.0
    v = speed_kmh / 3.6
    radius = v * period / (2 * math.pi)
    # Center placed so the circle passes the trap at t0 heading 309°:
    # trap-relative position at t0 is radius * (-0.629, -0.777) m (E, N),
    # velocity (-0.777, +0.629) * v m/s.
    center_lat = trap["lat"] + radius * 0.7771 / 111000.0
    center_lon = trap["lon"] + radius * 0.6293 / 85670.0
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    theta0 = math.atan2(-0.6293, -0.7771)
    theta = theta0 + 2 * math.pi * (t - t0) / period
    lat = center_lat + radius * np.cos(theta) / 111000.0
    lon = center_lon + (radius * np.sin(theta)) / 85670.0
    lines = [
        "File created on 02/11/2026 at 00:31:34\n",
        "[header]\n",
        "time\n",
        "[session data]\n",
        "name Sonoma\n",
        "[column names]\n",
        "sats time lat long velocity\n",
        "[data]\n",
    ]
    for i in range(n):
        ti = t[i]
        lines.append(
            f"016 00{int(ti // 60):02d}{ti % 60:05.2f} "
            f"+{lat[i] * 60:.6f} +{abs(lon[i]) * 60:.6f} {speed_kmh:.1f}\n",
        )
    return lines


def test_vbo_excerpt_cuts_in_timed_out_laps():
    from track_telemetry_converter.log import DataLog

    log = DataLog()
    log.from_vbo_log(_vbo_circle_log())
    assert log.metadata.get("venue_name") == "Sonoma"
    assert [round(t, 1) for t, _ in log.laps_info["beacons"]] == [3.9, 123.9]
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap", "Timed", "Partial"]
    assert [round(lap["duration"], 1) for lap in log.laps_info["laps"]][:2] == [3.9, 120.0]
    assert log.laps_info["laps"][2]["duration"] == pytest.approx(4.1, abs=0.2)
    assert log.laps_info["fastest_lap"] == 2
    assert log.laps_info["fastest_time"] == pytest.approx(120.0, abs=0.2)
    assert log.laps_info["sector_beacons"] == []


def test_vbo_mid_file_crossings_need_no_rescue():
    from track_telemetry_converter.log import DataLog

    log = DataLog()
    log.from_vbo_log(_vbo_circle_log(t0=30.0, duration_s=200.0))
    assert [round(t, 1) for t, _ in log.laps_info["beacons"]] == [30.0, 150.0]
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap", "Timed", "In Lap"]


def test_vbo_rescue_rejects_slow_pass():
    from track_telemetry_converter.log import DataLog

    log = DataLog()
    log.from_vbo_log(_vbo_circle_log(speed_kmh=50.0))
    assert log.laps_info["beacons"] == []
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap"]
    assert log.laps_info["fastest_time"] == 0.0
