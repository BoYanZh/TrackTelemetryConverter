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
    assert "Accelerator Pos" in log.channels
    np.testing.assert_array_equal(
        log.channels["Accelerator Pos"].values, log.channels["Throttle Pos"].values,
    )
    laps = log.laps_info["laps"]
    assert [lap["lap_num"] for lap in laps] == [1, 2]
    assert log.laps_info["total_laps"] == 2
    assert "Lap Number" in log.channels
    lats = log.channels["GPS Latitude"].values
    assert np.ptp(lats) > 0
    np.testing.assert_allclose(np.median(lats), 39.0, atol=0.01)


def test_parse_acti_log_flying_start_first_lap_is_timed(tmp_path):
    """Recording starts on the line at speed: the opening full-length
    segment counts as a timed lap (the i2 out-lap row comes from the
    leading 0.0 beacon, not from this segment)."""
    src = str(tmp_path / "acti.ld")
    _write_synthetic_ld(src)  # closed 2x20 s loops at 100 km/h from the origin
    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    laps = out.laps_info["laps"]
    assert [lap["type"] for lap in laps] == ["Timed", "In Lap"]
    assert abs(out.laps_info["fastest_time"] - 20.0) < 0.5
    assert out.laps_info["fastest_lap"] == 1


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
    # Beacons lead with 0.0 so i2 renders the opening "Out Lap
    # 0:00.000" row; the real S/F crossings follow. The file-end
    # truncation is still not beaconed.
    assert list(resets) == pytest.approx([19.97, 39.94, 59.91], abs=0.005)
    info = acti_parser._build_laps_info(groups, freq, resets)
    assert abs(info["laps"][1]["duration"] - 19.97) < 0.005
    beacons = [b[0] for b in info["beacons"]]
    assert beacons == pytest.approx([0.0, 19.97, 39.94, 59.91], abs=0.005)


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
        [0.0, 19.97, 39.87, 59.72], abs=0.005)


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
def test_parse_acti_log_prefers_recorded_accelerator_over_mirror(tmp_path):
    """A genuinely logged Accelerator Pos is kept verbatim, never mirrored."""
    n = 400
    freq = 20.0
    t = np.arange(n) / freq
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    xy = np.column_stack([300 * np.cos(th), 300 * np.sin(th)])
    log = DataLog()
    log.add_channel("Car Coord X", "m", float, 3)
    log.channels["Car Coord X"].set_samples(t, xy[:, 0])
    log.add_channel("Car Coord Y", "m", float, 3)
    log.channels["Car Coord Y"].set_samples(t, xy[:, 1])
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples(t, np.full(n, 100.0))
    log.add_channel("Lap Time", "s", float, 3)
    log.channels["Lap Time"].set_samples(t, np.linspace(0, n / freq, n))
    log.add_channel("Throttle Pos", "%", float, 1)
    log.channels["Throttle Pos"].set_samples(t, np.full(n, 100.0))
    log.add_channel("Accelerator Pos", "%", float, 1)
    log.channels["Accelerator Pos"].set_samples(t, np.full(n, 42.0))
    src = str(tmp_path / "acti.ld")
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(log)
    ml.write(src)

    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    np.testing.assert_allclose(out.channels["Accelerator Pos"].values, np.full(n, 42.0))


def _write_realign_synthetic_ld(path, venue_name="Thunderhill East Bypass test", freq=20.0,
                                resets=(10.0, 30.0), cross_t=20.0, duration_s=40.0,
                                with_sectors=False):
    """ACTI-shaped .ld whose sim resets miss the real S/F (crossed at cross_t).

    The Car Coord trace drives due south through the real Thunderhill East
    Start/Finish trap at 50 m/s, crossing it at t=cross_t, while the Lap Time
    channel resets at the given sim times like a mod timing line ~63 m off.
    With ``with_sectors``, each sim lap also latches three Last Sector Time
    values (one per third), so game sector splits exist to realign.
    """
    from track_telemetry_converter.channels import TRACK_BEACONS

    sf = TRACK_BEACONS["Thunderhill East Bypass"]
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    speed_ms = 50.0
    # y runs so y_real crosses 0 (the beacon) at t=cross_t.
    y = speed_ms * (cross_t - t)
    log = DataLog()
    log.add_channel("Car Coord X", "m", float, 3)
    log.channels["Car Coord X"].set_samples(t, np.zeros(n))
    log.add_channel("Car Coord Y", "m", float, 3)
    log.channels["Car Coord Y"].set_samples(t, y)
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples(t, np.full(n, speed_ms * 3.6))
    bounds = [0.0, *resets, duration_s]
    lt = np.empty(n)
    for i, ti in enumerate(t):
        lo = max(b for b in bounds if b <= ti + 1e-9)
        lt[i] = ti - lo
    log.add_channel("Lap Time", "s", float, 3)
    log.channels["Lap Time"].set_samples(t, lt)
    if with_sectors:
        # One latch value per third of each sim lap; consecutive laps use
        # disjoint values so every sim lap boundary latches too.
        vals = np.empty(n)
        base = 40.0
        for li in range(len(bounds) - 1):
            lo, hi = bounds[li], bounds[li + 1]
            idx = np.nonzero((t >= lo - 1e-9) & (t < hi - 1e-9))[0]
            for s, part in enumerate(np.array_split(idx, 3)):
                vals[part] = base + s
            base += 3.0
        log.add_channel("Last Sector Time", "s", float, 1)
        log.channels["Last Sector Time"].set_samples(t, vals)
    log.add_channel("Throttle Pos", "%", float, 1)
    log.channels["Throttle Pos"].set_samples(t, np.full(n, 100.0))
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(log)
    ml.write(path)
    return {
        "test_east": {
            "name": venue_name,
            "ref_lat": sf["lat"],
            "ref_lon": sf["lon"],
            "theta_deg": 0.0,
            "dx_m": 0.0,
            "dy_m": 0.0,
        },
    }


def test_parse_acti_log_realign_sf_cuts_at_real_line(tmp_path):
    src = str(tmp_path / "acti.ld")
    anchors = _write_realign_synthetic_ld(src)
    out = DataLog()
    out.from_acti_log(src, track="test_east", anchors=anchors, realign_sf=True)
    beacons = [b[0] for b in out.laps_info["beacons"]]
    assert beacons[0] == 0.0
    assert len(beacons) == 2
    assert abs(beacons[1] - 20.0) < 0.1
    laps = out.laps_info["laps"]
    assert len(laps) == 2
    assert laps[0]["type"] == "Out Lap"
    assert abs(laps[0]["end_time"] - beacons[1]) < 1e-9
    assert abs(laps[1]["start_time"] - beacons[1]) < 1e-9


def test_parse_acti_log_realign_sf_off_keeps_sim_cuts(tmp_path):
    src = str(tmp_path / "acti.ld")
    anchors = _write_realign_synthetic_ld(src)
    out = DataLog()
    out.from_acti_log(src, track="test_east", anchors=anchors)
    beacons = [b[0] for b in out.laps_info["beacons"]]
    assert beacons == pytest.approx([0.0, 10.0, 30.0], abs=0.1)


def test_parse_acti_log_realign_sf_falls_back_without_real_line(tmp_path):
    src = str(tmp_path / "acti.ld")
    anchors = _write_realign_synthetic_ld(src, venue_name="Test Loop")
    out = DataLog()
    out.from_acti_log(src, track="test_east", anchors=anchors, realign_sf=True)
    beacons = [b[0] for b in out.laps_info["beacons"]]
    assert beacons == pytest.approx([0.0, 10.0, 30.0], abs=0.1)


def test_parse_acti_log_realign_sf_recovers_tail_crossing(tmp_path):
    """AC reset 1 s before file end: the real crossing sits inside the 5 s
    tail gate, so normal detection misses it; the relaxed pass must still
    move the end boundary instead of leaving it on the sim line."""
    src = str(tmp_path / "acti.ld")
    anchors = _write_realign_synthetic_ld(src, resets=(19.0,), cross_t=20.0, duration_s=21.0)
    out = DataLog()
    out.from_acti_log(src, track="test_east", anchors=anchors, realign_sf=True)
    beacons = [b[0] for b in out.laps_info["beacons"]]
    assert beacons[0] == 0.0
    assert len(beacons) == 2
    assert abs(beacons[1] - 20.0) < 0.1


def test_parse_acti_log_realign_sf_recovers_head_crossing(tmp_path):
    """AC reset 8 s after recording start: the real crossing sits inside the
    15 s head gate; the relaxed pass must still move it."""
    src = str(tmp_path / "acti.ld")
    anchors = _write_realign_synthetic_ld(src, resets=(8.0,), cross_t=9.2, duration_s=40.0)
    out = DataLog()
    out.from_acti_log(src, track="test_east", anchors=anchors, realign_sf=True)
    beacons = [b[0] for b in out.laps_info["beacons"]]
    assert beacons[0] == 0.0
    assert len(beacons) == 2
    assert abs(beacons[1] - 9.2) < 0.1


def test_acti_local_header_mtime_applies_to_ld_and_ldx(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from track_telemetry_converter.conversion import _apply_mtime_from_log

    local_zone = timezone(timedelta(hours=-8))

    class LocalHeader(datetime):
        # Deterministic Windows/local-clock behavior even on UTC CI hosts.
        def astimezone(self, tz=None):
            aware = self.replace(tzinfo=local_zone) if self.tzinfo is None else self
            return datetime.astimezone(aware, tz)

    src = _write_synthetic_ld(str(tmp_path / "acti_local.ld"))
    parsed = acti_parser.ldData.fromfile(src)
    parsed.head.datetime = LocalHeader(2026, 11, 7, 22, 3)
    monkeypatch.setattr(acti_parser.ldData, "fromfile", lambda _: parsed)
    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    expected_start = datetime(2026, 11, 7, 22, 3, tzinfo=local_zone)
    assert out.datetime_utc == expected_start.astimezone(timezone.utc)
    paths = [tmp_path / "out.ld", tmp_path / "out.ldx"]
    for path in paths:
        path.write_bytes(b"test")
    _apply_mtime_from_log(paths, out, source_path=src)
    expected_end = expected_start.timestamp() + out.duration()
    for path in paths:
        assert abs(path.stat().st_mtime - expected_end) < 0.001


def _write_sector_synthetic_ld(path, laps=2, freq=20.0):
    """ACTI-shaped .ld with a 3-sector Last Sector Time latch per lap."""
    n_lap = 400
    n = n_lap * laps
    t = np.arange(n) / freq
    xy = np.vstack([_loop(n_lap) for _ in range(laps)])
    log = DataLog()
    log.add_channel("Car Coord X", "m", float, 3)
    log.channels["Car Coord X"].set_samples(t, xy[:, 0])
    log.add_channel("Car Coord Y", "m", float, 3)
    log.channels["Car Coord Y"].set_samples(t, xy[:, 1])
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples(t, np.full(n, 100.0))
    log.add_channel("Lap Time", "s", float, 3)
    log.channels["Lap Time"].set_samples(
        t, np.tile(np.linspace(0, n_lap / freq, n_lap), laps),
    )
    third = n_lap // 3
    one_lap = np.concatenate(
        [np.full(third, 10.0), np.full(third, 7.0), np.full(n_lap - 2 * third, 6.0)],
    )
    log.add_channel("Last Sector Time", "s", float, 1)
    log.channels["Last Sector Time"].set_samples(t, np.tile(one_lap, laps))
    log.add_channel("Throttle Pos", "%", float, 1)
    log.channels["Throttle Pos"].set_samples(t, np.full(n, 100.0))
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(log)
    ml.write(path)
    return path


def _fake_sector_ld(channs):
    class _Ch:
        def __init__(self, name, data, freq=20.0):
            self.name = name
            self.data = np.asarray(data, dtype=float)
            self.freq = freq

    class _Ld:
        pass

    ld = _Ld()
    ld.channs = [_Ch(name, data) for name, data in channs]
    return ld


def test_extract_sector_beacons_two_splits_per_lap():
    freq = 20.0
    n_lap = 400
    lt = np.tile(np.linspace(0, 20.0, n_lap), 2)
    third = n_lap // 3
    one_lap = np.concatenate(
        [np.full(third, 10.0), np.full(third, 7.0), np.full(n_lap - 2 * third, 6.0)],
    )
    ld = _fake_sector_ld([("Lap Time", lt), ("Last Sector Time", np.tile(one_lap, 2))])
    groups, resets = acti_parser._all_lap_bounds(ld)
    sectors = acti_parser._extract_sector_beacons(ld, groups, freq, resets)
    assert [name for _, name in sectors] == ["S1", "S2", "S1", "S2"]
    assert [t for t, _ in sectors] == pytest.approx([6.65, 13.3, 26.65, 33.3], abs=0.06)
    # Lap-end latches never leak into Sections (already covered by Beacons).
    for t, _ in sectors:
        assert all(abs(t - b) > 0.5 for b in resets)


def test_extract_sector_beacons_skips_garbage_flicker_and_mismatch():
    freq = 20.0
    lt = np.linspace(0, 20.0, 400)
    groups = [(20.0, 0, 400, 0.0, 20.0)]
    # Pit-standing latch (569 s) and single-sample flicker are not sectors.
    lst = np.zeros(400)
    lst[100:] = 45.0
    lst[200:] = 569.0
    lst[300] = 1.0
    lst[301] = 45.0
    ld = _fake_sector_ld([("Lap Time", lt), ("Last Sector Time", lst)])
    sectors = acti_parser._extract_sector_beacons(ld, groups, freq, [20.0])
    assert [t for t, _ in sectors] == pytest.approx([5.0, 15.05], abs=0.06)
    # Missing channel, constant channel, grid mismatch, realigned groups.
    assert acti_parser._extract_sector_beacons(
        _fake_sector_ld([("Lap Time", lt)]), groups, freq, [20.0],
    ) == []
    assert acti_parser._extract_sector_beacons(
        _fake_sector_ld([("Lap Time", lt), ("Last Sector Time", np.zeros(400))]),
        groups, freq, [20.0],
    ) == []
    assert acti_parser._extract_sector_beacons(
        _fake_sector_ld([("Lap Time", lt), ("Last Sector Time", np.zeros(399))]),
        groups, freq, [20.0],
    ) == []
    assert acti_parser._extract_sector_beacons(
        _fake_sector_ld([("Lap Time", lt), ("Last Sector Time", np.zeros(400))]),
        [(20.0, 0, 0, 0.0, 20.0)], freq, [20.0],
    ) == []


def test_acti_end_to_end_sector_beacons_and_ldx_sections(tmp_path):
    import xml.etree.ElementTree as ET

    src = _write_sector_synthetic_ld(str(tmp_path / "acti.ld"))
    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    assert [name for _, name in out.laps_info["sector_beacons"]] == ["S1", "S2", "S1", "S2"]
    assert [t for t, _ in out.laps_info["sector_beacons"]] == pytest.approx(
        [6.65, 13.3, 26.65, 33.3], abs=0.06,
    )
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(out)
    ldx_path = str(tmp_path / "out.ldx")
    ml.write_ldx(ldx_path, out.laps_info, beacons=[])
    groups = {
        group.attrib.get("Name"): [m.attrib.get("Name") for m in group.findall("Marker")]
        for group in ET.parse(ldx_path).getroot().findall(".//MarkerGroup")
    }
    assert groups["Sections"] == ["S1", "S2", "S1", "S2"]
    assert len(groups["Beacons"]) == 2


def test_acti_ldx_merges_gps_splits_with_game_sectors(tmp_path):
    import xml.etree.ElementTree as ET

    src = _write_sector_synthetic_ld(str(tmp_path / "acti.ld"))
    out = DataLog()
    out.from_acti_log(src, track="test_loop", anchors=_anchors())
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(out)
    ldx_path = str(tmp_path / "out.ldx")
    ml.write_ldx(ldx_path, out.laps_info, beacons=[(10.0, "T1")])
    groups = {
        group.attrib.get("Name"): [float(m.attrib.get("Time")) / 1e6 for m in group.findall("Marker")]
        for group in ET.parse(ldx_path).getroot().findall(".//MarkerGroup")
    }
    assert groups["Sections"] == pytest.approx([6.65, 10.0, 13.3, 26.65, 33.3], abs=0.06)


def test_acti_single_lap_filter_rebases_sectors(tmp_path):
    src = _write_sector_synthetic_ld(str(tmp_path / "acti.ld"))
    out = DataLog()
    out.from_acti_log(src, target_lap=2, track="test_loop", anchors=_anchors())
    assert [name for _, name in out.laps_info["sector_beacons"]] == ["S1", "S2"]
    assert [t for t, _ in out.laps_info["sector_beacons"]] == pytest.approx(
        [6.65, 13.3], abs=0.06,
    )


def test_extract_realigned_sector_beacons_two_splits_per_lap():
    n_lap = 400
    one_lap = np.concatenate(
        [np.full(133, 10.0), np.full(133, 7.0), np.full(n_lap - 266, 6.0)],
    )
    ld = _fake_sector_ld([("Last Sector Time", np.tile(one_lap, 2))])
    segs = [(20.0, 0, 0, 0.0, 20.0), (20.0, 0, 0, 20.0, 40.0)]
    sectors = acti_parser._extract_realigned_sector_beacons(ld, segs, [0.0, 20.0, 40.0], 40.0)
    assert [name for _, name in sectors] == ["S1", "S2", "S1", "S2"]
    assert [t for t, _ in sectors] == pytest.approx([6.65, 13.3, 26.65, 33.3], abs=0.06)


def test_extract_realigned_sector_beacons_straddling_lap():
    # Sim resets at 10/30, real line crossed once at 20: the mod-line
    # lap-end latches at 10.0/30.0 are interior splits of the rebuilt laps.
    one0 = np.concatenate([np.full(67, 10.0), np.full(66, 11.0), np.full(67, 12.0)])
    one1 = np.concatenate([np.full(133, 13.0), np.full(134, 14.0), np.full(133, 15.0)])
    one2 = np.concatenate([np.full(67, 16.0), np.full(66, 17.0), np.full(67, 18.0)])
    ld = _fake_sector_ld([("Last Sector Time", np.concatenate([one0, one1, one2]))])
    segs = [(20.0, 0, 0, 0.0, 20.0), (20.0, 0, 0, 20.0, 40.0)]
    sectors = acti_parser._extract_realigned_sector_beacons(ld, segs, [0.0, 20.0], 40.0)
    assert [name for _, name in sectors] == ["S1", "S2", "S3", "S4"] * 2
    assert [t for t, _ in sectors] == pytest.approx(
        [3.35, 6.65, 10.0, 16.65, 23.35, 30.0, 33.35, 36.65], abs=0.06,
    )


def test_extract_realigned_sector_beacons_rejects_garbage():
    lst = np.zeros(800)
    lst[100:] = 45.0
    lst[200:] = 569.0  # pit standing: not a sector
    lst[300] = 1.0  # single-sample flicker
    lst[301] = 45.0
    lst[400:] = 46.0  # lap-end latch exactly on the new beacon at 20.0
    ld = _fake_sector_ld([("Last Sector Time", lst)])
    segs = [(20.0, 0, 0, 0.0, 20.0), (20.0, 0, 0, 20.0, 40.0)]
    sectors = acti_parser._extract_realigned_sector_beacons(ld, segs, [0.0, 20.0], 40.0)
    assert [t for t, _ in sectors] == pytest.approx([5.0, 15.05], abs=0.06)
    assert acti_parser._extract_realigned_sector_beacons(
        _fake_sector_ld([]), segs, [0.0, 20.0], 40.0,
    ) == []
    assert acti_parser._extract_realigned_sector_beacons(ld, [], [0.0, 20.0], 40.0) == []


def test_acti_realign_sf_keeps_game_sectors(tmp_path):
    import xml.etree.ElementTree as ET

    src = str(tmp_path / "acti.ld")
    anchors = _write_realign_synthetic_ld(src, with_sectors=True)
    out = DataLog()
    out.from_acti_log(src, track="test_east", anchors=anchors, realign_sf=True)
    assert [b for b, _ in out.laps_info["beacons"]] == pytest.approx([0.0, 20.0], abs=0.1)
    sectors = out.laps_info["sector_beacons"]
    assert [name for _, name in sectors] == ["S1", "S2", "S3", "S4"] * 2
    assert [t for t, _ in sectors] == pytest.approx(
        [3.35, 6.65, 10.0, 16.65, 23.35, 30.0, 33.35, 36.65], abs=0.15,
    )
    ml = MotecLog()
    ml.initialize()
    ml.add_all_channels(out)
    ldx_path = str(tmp_path / "out.ldx")
    ml.write_ldx(ldx_path, out.laps_info, beacons=[])
    groups = {
        group.attrib.get("Name"): [m.attrib.get("Name") for m in group.findall("Marker")]
        for group in ET.parse(ldx_path).getroot().findall(".//MarkerGroup")
    }
    assert groups["Sections"] == ["S1", "S2", "S3", "S4"] * 2
