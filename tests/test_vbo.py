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
    lat, lon = resolve("+2460.060000", "+7347.200000", anchor)
    assert abs(lat - 41.001) < 1e-9 and abs(lon + 122.45333333333333) < 1e-9
    # No anchor: magnitude settles it (|lon| > 90 * 60); ambiguous stays None.
    lat, lon = resolve("+7347.200000", "+2460.060000", None)
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
