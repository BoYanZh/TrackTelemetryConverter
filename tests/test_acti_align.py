"""Tests for the ACTI input dialect (parsers/acti_parser.py + CLI wiring)."""

import math

import numpy as np
import pytest

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.parsers import acti_parser, auto_detect_log_type


def _loop(n=400, radius=300.0):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _write_synthetic_ld(path, with_car_coord=True, laps=2, freq=20.0, with_lap_time=True):
    """Minimal ACTI-shaped .ld: circular trajectory + speed + Lap Time sawtooth."""
    n_lap = 400
    n = n_lap * laps
    t = np.arange(n) / freq
    xy = np.vstack([_loop(n_lap) for _ in range(laps)])
    log = DataLog()
    if with_car_coord:
        log.add_channel("Car Coord X", "m", float, 3)
        log.channels["Car Coord X"].set_samples(t, xy[:, 0])
        log.add_channel("Car Coord Y", "m", float, 3)
        log.channels["Car Coord Y"].set_samples(t, xy[:, 1])
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples(t, np.full(n, 100.0))
    if with_lap_time:
        log.add_channel("Lap Time", "s", float, 3)
        log.channels["Lap Time"].set_samples(t, np.tile(np.linspace(0, n_lap / freq, n_lap), laps))
    log.add_channel("Throttle Pos", "%", float, 1)
    log.channels["Throttle Pos"].set_samples(t, np.full(n, 100.0))
    ml = MotecLog()
    ml.driver = "Test Driver"
    ml.vehicle_id = "test_car"
    ml.venue_name = "Test Venue"
    ml.initialize()
    ml.add_all_channels(log)
    ml.write(path)
    return path


def _anchors():
    return {
        "test_loop": {
            "name": "Test Loop",
            "ref_lat": 39.0,
            "ref_lon": -122.0,
            "theta_deg": 0.0,
            "dx_m": 0.0,
            "dy_m": 0.0,
        },
    }


def test_transform_identity():
    cfg = {"ref_lat": 39.0, "ref_lon": -122.0, "theta_deg": 0.0, "dx_m": 0.0, "dy_m": 0.0}
    x = np.array([0.0, 100.0, -50.0])
    y = np.array([0.0, 0.0, 25.0])
    lat, lon, xm, ym = acti_parser.transform_ac_to_gps(x, y, cfg)
    np.testing.assert_allclose(xm, x)
    np.testing.assert_allclose(ym, y)
    assert lat[0] == 39.0 and lon[0] == -122.0
    assert lat[2] > 39.0 and lon[1] > -122.0


def test_transform_rotation_preserves_distances():
    cfg = {
        "ref_lat": 39.5,
        "ref_lon": -122.3,
        "theta_deg": 178.7,
        "dx_m": 613.0,
        "dy_m": 398.0,
    }
    xy = _loop()
    _, _, xm, ym = acti_parser.transform_ac_to_gps(xy[:, 0], xy[:, 1], cfg)
    mapped = np.column_stack([xm, ym])
    d_in = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T).sum()
    d_out = np.hypot(*np.diff(np.vstack([mapped, mapped[:1]]), axis=0).T).sum()
    np.testing.assert_allclose(d_out, d_in, rtol=1e-9)


def test_transform_legacy_config_without_scale():
    cfg = {
        "ref_lat": 39.540546,
        "ref_lon": -122.331482,
        "theta_deg": 180.018,
        "dx_m": -49.94,
        "dy_m": -0.838,
    }
    lat, lon, _, _ = acti_parser.transform_ac_to_gps([0.0], [0.0], cfg)
    assert math.isfinite(lat[0]) and math.isfinite(lon[0])


def test_heading_cardinal_lines():
    north = acti_parser.compute_gps_heading([0, 0, 0], [0, 10, 20])
    np.testing.assert_allclose(north, [0, 0, 0], atol=1e-6)
    east = acti_parser.compute_gps_heading([0, 10, 20], [0, 0, 0])
    np.testing.assert_allclose(east, [90, 90, 90], atol=1e-6)


def test_heading_holds_at_standstill():
    hdg = acti_parser.compute_gps_heading([0, 10, 10, 10], [0, 0, 0, 0])
    np.testing.assert_allclose(hdg, [90, 90, 90, 90], atol=1e-6)


def test_fit_recovers_known_transform():
    src = _loop()
    ang = np.radians(178.7)
    r = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (r @ src.T).T + np.array([613.0, 398.0])
    sc, _, _, rms, _, rev, _ = acti_parser.fit_similarity(src, dst)
    assert rev is False
    assert abs(sc - 1.0) < 1e-6
    assert rms < 1e-6


def test_fit_rejects_wrong_layout():
    src = _loop(radius=300.0)
    dst = _loop(radius=900.0)
    sc, _, _, _, _, _, _ = acti_parser.fit_similarity(src, dst)
    assert abs(sc - 1.0) > 0.05


def test_parse_acti_log_end_to_end(tmp_path):
    src = str(tmp_path / "acti.ld")
    _write_synthetic_ld(src)
    log = DataLog()
    log.from_acti_log(src, track="test_loop", anchors=_anchors())
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert "GPS Heading" in log.channels
    assert "Throttle Pos" in log.channels
    assert "Accelerator Pos" not in log.channels
    laps = log.laps_info["laps"]
    assert [lap["lap_num"] for lap in laps] == [1, 2]
    assert log.laps_info["total_laps"] == 2
    assert "Lap Number" in log.channels
    lats = log.channels["GPS Latitude"].values
    assert np.ptp(lats) > 0
    np.testing.assert_allclose(np.median(lats), 39.0, atol=0.01)


def test_parse_acti_log_short_first_group_is_out_lap(tmp_path):
    """A mid-lap recording start still opens with an Out Lap, never Partial."""
    src = str(tmp_path / "acti.ld")
    freq = 20.0
    n_short, n_lap = 100, 400  # 5 s partial + one 20 s lap
    n = n_short + n_lap
    t = np.arange(n) / freq
    xy = _loop(n)
    log = DataLog()
    log.add_channel("Car Coord X", "m", float, 3)
    log.channels["Car Coord X"].set_samples(t, xy[:, 0])
    log.add_channel("Car Coord Y", "m", float, 3)
    log.channels["Car Coord Y"].set_samples(t, xy[:, 1])
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples(t, np.full(n, 100.0))
    log.add_channel("Lap Time", "s", float, 3)
    log.channels["Lap Time"].set_samples(
        t, np.concatenate([np.linspace(0, 5.0, n_short), np.linspace(0, 20.0, n_lap)]),
    )
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(log)
    ml.write(src)

    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    laps = out.laps_info["laps"]
    assert [lap["type"] for lap in laps] == ["Out Lap", "In Lap"]
    assert laps[0]["lap_num"] == 1


def test_parse_acti_log_without_lap_time_is_single_out_lap(tmp_path):
    """No Lap Time structure: the whole log is one Out Lap, never empty."""
    src = str(tmp_path / "acti.ld")
    _write_synthetic_ld(src, with_lap_time=False)
    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    assert out.laps_info["total_laps"] == 1
    lap = out.laps_info["laps"][0]
    assert lap["type"] == "Out Lap"
    assert lap["start_time"] == 0.0
    assert lap["duration"] > 0.0
    assert "Lap Number" in out.channels
    np.testing.assert_allclose(out.channels["Lap Number"].values, np.ones(800))


def test_parse_acti_log_rejects_unknown_track(tmp_path):
    src = str(tmp_path / "acti.ld")
    _write_synthetic_ld(src)
    log = DataLog()
    try:
        log.from_acti_log(src, anchors=_anchors())
    except ValueError as exc:
        assert "test_loop" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown track")


def test_parse_acti_log_rejects_non_acti_ld(tmp_path):
    src = str(tmp_path / "plain.ld")
    _write_synthetic_ld(src, with_car_coord=False)
    log = DataLog()
    try:
        log.from_acti_log(src, track="test_loop", anchors=_anchors())
    except ValueError as exc:
        assert "Car Coord" in str(exc)
    else:
        raise AssertionError("expected ValueError without Car Coord channels")


def test_parse_acti_log_selects_single_lap(tmp_path):
    src = str(tmp_path / "acti.ld")
    _write_synthetic_ld(src)
    log = DataLog()
    log.from_acti_log(src, target_lap=2, track="test_loop", anchors=_anchors())
    assert log.laps_info["total_laps"] == 1
    lap = log.laps_info["laps"][0]
    assert lap["lap_num"] == 2
    assert lap["start_time"] == 0.0
    assert abs(lap["duration"] - 20.0) < 0.01
    for ch in log.channels.values():
        assert len(ch.timestamps) == 400
        assert ch.timestamps[0] == 0.0


def test_auto_detect_acti_and_plain_ld(tmp_path):
    acti = str(tmp_path / "acti.ld")
    _write_synthetic_ld(acti)
    assert auto_detect_log_type(acti) == "ACTI"
    plain = str(tmp_path / "plain.ld")
    _write_synthetic_ld(plain, with_car_coord=False)
    assert auto_detect_log_type(plain) != "ACTI"


def test_best_lap_bounds_skips_short_in_lap():
    class _Ch:
        def __init__(self, name, data, freq=20.0):
            self.name = name
            self.data = np.asarray(data, dtype=float)
            self.freq = freq

    class _Ld:
        channs: list

    n1, n2, n3 = 2000, 1855, 1678
    lt = np.concatenate(
        [np.linspace(0, 100, n1), np.linspace(0, 92.75, n2), np.linspace(0, 83.9, n3)],
    )
    spd = np.concatenate([np.full(n1, 90.0), np.full(n2, 101.0), np.full(n3, 81.0)])
    ld = _Ld()
    ld.channs = [_Ch("Lap Time", lt), _Ch("Ground Speed", spd)]
    s, e, dur = acti_parser._best_lap_bounds(ld)
    assert abs(dur - 92.75) < 0.1
    assert (s, e) == (n1, n1 + n2)


def test_all_lap_bounds_interpolates_reset_crossing():
    """Lap durations use the ms-precise timer, not the 20 Hz sample grid."""
    freq = 20.0
    # True crossings at 19.97 / 39.94 / 59.91: each reset falls between two
    # samples, so the grid durations (20.00) would be off by tens of ms.
    crossings = [19.97, 39.94, 59.91]
    n = 1250
    t = np.arange(n) / freq

    def _prev(ti):
        past = [c for c in crossings if c <= ti]
        return past[-1] if past else crossings[0] - 20.0

    lt = np.array([ti - _prev(ti) for ti in t])

    class _Ch:
        def __init__(self, name, data, freq=freq):
            self.name = name
            self.data = np.asarray(data, dtype=float)
            self.freq = freq

    class _Ld:
        channs: list

    ld = _Ld()
    ld.channs = [_Ch("Lap Time", lt)]
    groups, resets = acti_parser._all_lap_bounds(ld)
    assert len(groups) == 4  # out + 2 timed + in
    durs = [g[0] for g in groups]
    assert abs(durs[0] - 19.97) < 0.005
    assert abs(durs[1] - 19.97) < 0.005
    assert abs(durs[2] - 19.97) < 0.005
    assert abs(durs[3] - (62.5 - 59.91)) < 0.005
    # Only true reset crossings become beacons: no marker at log start
    # (mid-lap) or file end (truncated lap), so i2 shows Out/Lap/In rows
    # with no phantom zero-length lap shifting the numbers by one.
    assert list(resets) == pytest.approx([19.97, 39.94, 59.91], abs=0.005)
    info = acti_parser._build_laps_info(groups, freq, resets)
    assert abs(info["laps"][1]["duration"] - 19.97) < 0.005
    beacons = [b[0] for b in info["beacons"]]
    assert beacons == pytest.approx([19.97, 39.94, 59.91], abs=0.005)


def test_all_lap_bounds_prefers_official_last_lap_time():
    """Completed laps use the game's official Last Lap Time (0 ms), not interp."""
    freq = 20.0
    crossings = [19.97, 39.94, 59.91]
    n = 1250
    t = np.arange(n) / freq

    def _prev(ti):
        past = [c for c in crossings if c <= ti]
        return past[-1] if past else crossings[0] - 20.0

    lt = np.array([ti - _prev(ti) for ti in t])
    # Official times latched at each reset, observed during the next group.
    # Deliberately off the interpolated 19.97 to prove the snap.
    last = np.zeros(n)
    last[799:1199] = 19.9
    last[1199:] = 19.85

    class _Ch:
        def __init__(self, name, data, freq=freq):
            self.name = name
            self.data = np.asarray(data, dtype=float)
            self.freq = freq

    class _Ld:
        channs: list

    ld = _Ld()
    ld.channs = [_Ch("Lap Time", lt), _Ch("Last Lap Time", last)]
    groups, resets = acti_parser._all_lap_bounds(ld)
    durs = [g[0] for g in groups]
    assert durs[0] == pytest.approx(19.97, abs=0.005)  # out lap: interp
    assert durs[1] == pytest.approx(19.9, abs=1e-9)  # official
    assert durs[2] == pytest.approx(19.85, abs=1e-9)  # official
    assert list(resets) == pytest.approx([19.97, 39.87, 59.72], abs=0.005)
    info = acti_parser._build_laps_info(groups, freq, resets)
    assert info["fastest_time"] == pytest.approx(19.85, abs=1e-9)
    assert [b[0] for b in info["beacons"]] == pytest.approx(
        [19.97, 39.87, 59.72], abs=0.005)


def test_all_lap_bounds_ignores_wild_official_time():
    """A garbage Last Lap Time falls back to interpolation, never corrupts."""

    class _Ch:
        def __init__(self, name, data, freq=20.0):
            self.name = name
            self.data = np.asarray(data, dtype=float)
            self.freq = freq

    class _Ld:
        channs: list

    n = 800
    lt = np.tile(np.linspace(0, 20, 400), 2)
    last = np.zeros(n)
    last[400:] = 999.0  # stale garbage, far from the ~20 s interpolated lap
    ld = _Ld()
    ld.channs = [_Ch("Lap Time", lt), _Ch("Last Lap Time", last)]
    groups, resets = acti_parser._all_lap_bounds(ld)
    assert [g[0] for g in groups] == pytest.approx([20.0, 20.0], abs=1e-9)
    assert list(resets) == pytest.approx([20.0], abs=1e-9)


def test_all_lap_bounds_falls_back_on_non_timer_values():
    """An implausible post-reset value keeps the grid duration, never NaN."""

    class _Ch:
        def __init__(self, name, data, freq=20.0):
            self.name = name
            self.data = np.asarray(data, dtype=float)
            self.freq = freq

    class _Ld:
        channs: list

    # Drop from 100 to 50 looks like a reset edge, but 50 s is not a
    # plausible "time since reset" at 20 Hz -> fall back to sample count.
    lt = np.concatenate([np.linspace(0, 100, 2000), np.linspace(50, 130, 1600)])
    ld = _Ld()
    ld.channs = [_Ch("Lap Time", lt)]
    groups, resets = acti_parser._all_lap_bounds(ld)
    assert len(groups) == 2
    assert abs(groups[0][0] - 100.0) < 1e-9
    assert abs(groups[1][0] - 80.0) < 1e-9
    assert list(resets) == pytest.approx([100.0], abs=1e-9)


def test_shipped_bidirectional_markers():
    """Only tracks run in both directions take _cw/_ccw (thill west, buttonwillow)."""
    assert acti_parser.ACTI_TRACK_PROFILES["thunderhill_west"].get("bidirectional") is True
    assert acti_parser.ACTI_TRACK_PROFILES["buttonwillow_1a_ccw"].get("bidirectional") is True
    assert not acti_parser.ACTI_TRACK_PROFILES["sonoma"].get("bidirectional", False)


def test_cli_registers_acti():
    from track_telemetry_converter.cli import _BATCH_TYPE_EXTENSIONS, build_parser

    assert _BATCH_TYPE_EXTENSIONS["ACTI"] == (".ld",)
    parser = build_parser()
    for action in parser._actions:
        if action.dest == "log_type":
            assert action.choices is not None and "ACTI" in action.choices
            break
    else:
        raise AssertionError("ACTI missing from log_type choices")


def test_shipped_profiles_have_scale_near_unity():
    for key, cfg in acti_parser.ACTI_TRACK_PROFILES.items():
        assert abs(float(cfg.get("scale", 1.0)) - 1.0) <= 0.005, key
        for field in ("ref_lat", "ref_lon", "theta_deg", "dx_m", "dy_m"):
            assert field in cfg, (key, field)


def test_detect_track_key():
    profiles = acti_parser.ACTI_TRACK_PROFILES
    assert acti_parser.resolve_track_profile("thunderhill_west", "anything.ld", profiles)
    assert (
        acti_parser.resolve_track_profile(None, "laguna_seca_run.ld", profiles) == "laguna_seca"
    )
    assert acti_parser.resolve_track_profile(None, "unknown.ld", profiles) is None
    assert acti_parser.resolve_track_profile("nope", "x.ld", profiles) is None


def test_cli_help_exits_zero():
    from track_telemetry_converter import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
