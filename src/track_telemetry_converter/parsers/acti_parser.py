"""Parser for Assetto Corsa ACTI MoTeC logs.

ACTI records simulator-frame coordinates (``Car Coord X/Y``, metres) instead
of GPS. This parser maps the useful channels to canonical names and places
the trajectory on Earth with a per-track similarity profile (rotation about
the origin plus translation, then projection around a reference point).
Profiles ship below; custom ones merge in through the ``anchors`` argument.
Unknown tracks fail closed: pass an explicit ``track`` key or calibrate one
with :func:`calibrate_track`.
"""

from __future__ import annotations

import datetime

import numpy as np

from .._vendor.ldparser import ldData
from ..channels import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_POS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_CG_ACCEL_VERT,
    CH_ENGINE_RPM,
    CH_GEAR,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GROUND_SPEED,
    CH_LAP_NUMBER,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_YAW_RATE,
)

EARTH_RADIUS_M = 6371000.0

#: Shipped track profiles. ``scale`` is near unity (within 0.5%, enforced by
#: ``test_shipped_profiles_have_scale_near_unity``) for every shipped track;
#: calibration refuses to save anything that deviates more than 5%.
#: ``bidirectional`` marks tracks that are actually run in both directions
#: (thunderhill_west, buttonwillow): only those ever take the opt-in
#: --direction-suffix ``_cw``/``_ccw`` names. Single-direction tracks such as
#: sonoma would only gain noise, so the suffix is skipped for them even when
#: the flag is passed. Custom anchors may set ``bidirectional`` the same way.
ACTI_TRACK_PROFILES = {
    "thunderhill_east_bypass": {
        "name": "Thunderhill Raceway Park (East Bypass)",
        "ref_lat": 39.540546,
        "ref_lon": -122.331482,
        "theta_deg": 180.018,
        "dx_m": -49.94,
        "dy_m": -0.838,
        "rmse_error_m": 1.12,
    },
    "ca9_south": {
        "name": "CA-9 S",
        "ref_lat": 37.25430017,
        "ref_lon": -122.03870167,
        "theta_deg": 180.8691,
        "dx_m": -5079.164,
        "dy_m": -1236.784,
        "rmse_error_m": 2.08,
    },
    "thunderhill_west": {
        "name": "Thunderhill West",
        "ref_lat": 39.536232,
        "ref_lon": -122.339561,
        "theta_deg": 180.259,
        "scale": 1.001,
        "dx_m": 641.107,
        "dy_m": 481.77,
        "rmse_error_m": 4.41,
        "bidirectional": True,
    },
    "laguna_seca": {
        "name": "Laguna Seca",
        "ref_lat": 36.584494,
        "ref_lon": -121.753899,
        "theta_deg": 179.127,
        "scale": 0.9954,
        "dx_m": 53.744,
        "dy_m": -16.414,
        "rmse_error_m": 4.32,
    },
    "sonoma": {
        "name": "Sonoma Raceway",
        "ref_lat": 38.162827,
        "ref_lon": -122.458832,
        "theta_deg": 141.506,
        "scale": 1.0036,
        "dx_m": -533.463,
        "dy_m": -1358.614,
        "rmse_error_m": 4.19,
    },
    "buttonwillow_1a_ccw": {
        "name": "Buttonwillow 1A CCW",
        "ref_lat": 35.489531,
        "ref_lon": -119.546513,
        "theta_deg": 180.17,
        "scale": 1.0003,
        "dx_m": -372.26,
        "dy_m": 415.941,
        "rmse_error_m": 4.1,
        "bidirectional": True,
    },
}

#: Pseudo-track key for the mixed Thunderhill Raceway Park source directory.
#: East-cluster Car Coord X spans [-514, +368] m while the West-cluster
#: (ig stint_21~32) spans [+502, +940] m; the gap midpoint is ~435 m, so
#: ``min(X) > 435`` selects West, otherwise East bypass. Uses the minimum
#: (not the mean) because the West cluster is entirely positive -- a single
#: threshold on the minimum separates the two clusters with ~67 m margin
#: on each side.
THUNDERHILL_AUTO_KEY = "thunderhill_auto"
THUNDERHILL_AUTO_THRESHOLD_M = 435.0
THUNDERHILL_AUTO_WEST = "thunderhill_west"
THUNDERHILL_AUTO_EAST = "thunderhill_east_bypass"

#: ACTI source channel -> (canonical channel, units, decimals).
_ACTI_CHANNEL_MAP = {
    "Ground Speed": (CH_GROUND_SPEED, "km/h", 2),
    "CG Accel Lateral": (CH_CG_ACCEL_LAT, "G", 2),
    "CG Accel Longitudinal": (CH_CG_ACCEL_LON, "G", 2),
    "CG Accel Vertical": (CH_CG_ACCEL_VERT, "G", 2),
    "Chassis Yaw Rate": (CH_YAW_RATE, "deg/s", 2),
    "Steering Angle": (CH_STEERING_ANGLE, "deg", 2),
    # Assetto Corsa exposes the pedal as "Throttle Pos". It maps to the
    # canonical throttle channel following the RaceChrono PPS convention
    # (pedal values under the throttle name); there is no throttle-body
    # channel to confuse it with, so no pedal/throttle ambiguity arises.
    # The mirror below carries the same pedal series on the canonical
    # accelerator channel, so sim runs overlay 1:1 with real-car logs that
    # keep pedal (Accelerator Pos) and throttle body (Throttle Pos) apart.
    "Throttle Pos": (CH_THROTTLE_POS, "%", 1),
    "Brake Pos": (CH_BRAKE_POS, "%", 1),
    "Engine RPM": (CH_ENGINE_RPM, "rpm", 0),
    "Gear": (CH_GEAR, "", 0),
}

_DROPPED_SIM_FRAME_CHANNELS = frozenset({"Car Coord X", "Car Coord Y", "Car Coord Z"})


def transform_ac_to_gps(ac_x, ac_y, track_cfg):
    """Map ACTI metres to WGS84 degrees. Pure function of arrays + config."""
    ref_lat = float(track_cfg.get("ref_lat", 39.540017))
    ref_lon = float(track_cfg.get("ref_lon", -122.331175))
    theta_rad = np.radians(float(track_cfg.get("theta_deg", 180.0)))
    scale = float(track_cfg.get("scale", 1.0))
    dx_m = float(track_cfg.get("dx_m", 0.0))
    dy_m = float(track_cfg.get("dy_m", 0.0))

    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    ac_x = np.asarray(ac_x, dtype=float)
    ac_y = np.asarray(ac_y, dtype=float)

    x_real_m = scale * (cos_t * ac_x - sin_t * ac_y) + dx_m
    y_real_m = scale * (sin_t * ac_x + cos_t * ac_y) + dy_m

    lat = ref_lat + np.degrees(y_real_m / EARTH_RADIUS_M)
    lon = ref_lon + np.degrees(x_real_m / (EARTH_RADIUS_M * np.cos(np.radians(ref_lat))))
    return lat, lon, x_real_m, y_real_m


def compute_gps_heading(x_real_m, y_real_m):
    """Course heading in degrees (0..360, 0=North, 90=East). Holds last value at standstill."""
    x_real_m = np.asarray(x_real_m, dtype=float)
    y_real_m = np.asarray(y_real_m, dtype=float)
    dx = np.gradient(x_real_m)
    dy = np.gradient(y_real_m)
    dist_step = np.hypot(dx, dy)

    raw_heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    moving = dist_step > 0.01
    if not np.any(moving):
        return np.zeros_like(raw_heading)

    hdg = np.copy(raw_heading)
    last_hdg = 0.0
    for i in range(len(hdg)):
        if moving[i]:
            last_hdg = hdg[i]
        else:
            hdg[i] = last_hdg

    first_mov = int(np.argmax(moving))
    if moving[first_mov]:
        hdg[:first_mov] = hdg[first_mov]
    return np.round(hdg, 3)


def heading_direction(headings, min_total_deg=180.0):
    """Classify loop direction from a GPS-heading series.

    Sums wrapped per-sample heading deltas over the whole run: clockwise
    driving rotates the course heading clockwise (total > 0, ``cw``),
    counter-clockwise driving the opposite (``ccw``). Wrap-aware, so a
    start/finish heading near 0/360 cannot flip the result.

    Returns ``(direction, total_deg)`` with direction ``"cw"`` / ``"ccw"``
    / ``None`` (too few samples, NaN-only, or ``|total|`` below
    ``min_total_deg`` -- less than about half a lap of evidence).
    """
    try:
        arr = np.asarray(list(headings), dtype=float)
    except (TypeError, ValueError):
        return None, 0.0
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return None, 0.0
    diffs = (np.diff(arr) + 540.0) % 360.0 - 180.0
    total = float(np.sum(diffs))
    if abs(total) < min_total_deg:
        return None, total
    return ("cw" if total > 0 else "ccw"), total


def _resample_xy(xy, n=800):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    t = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])])


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    h = a.T @ b / len(src)
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    scale = s.sum() / ((a**2).sum() / len(src))
    t = mu_d - scale * r @ mu_s
    fit = scale * (r @ src.T).T + t
    rms = float(np.sqrt(((fit - dst) ** 2).sum(1).mean()))
    return scale, r, t, rms


def fit_similarity(sim_xy, ref_xy, samples=800, shift_step=4):
    """Best rigid fit of sim (metres) onto a reference trajectory.

    Searches both travel directions and cyclic start shifts. Returns
    ``(scale, R, t, rms, lateral_p95, reversed, shift)``.
    """
    src = _resample_xy(np.asarray(sim_xy, dtype=float), samples)
    dst = _resample_xy(np.asarray(ref_xy, dtype=float), samples)
    best = None
    for rev in (False, True):
        s0 = src[::-1] if rev else src
        cur = None
        for shift in range(0, samples, shift_step):
            sc, r, t, rms = _umeyama(np.roll(s0, shift, axis=0), dst)
            if cur is None or rms < cur[0]:
                cur = (rms, shift, sc, r, t)
        if cur is None:
            raise ValueError("Similarity search found no candidate shift")
        if best is None or cur[0] < best[0]:
            best = (cur[0], rev, cur[1], cur[2], cur[3], cur[4])
    if best is None:
        raise ValueError("Similarity search found no candidate direction")
    rms, rev, shift, sc, r, t = best
    fit = sc * (r @ np.roll(src[::-1] if rev else src, shift, axis=0).T).T + t
    lat_dev = np.hypot(*(fit - dst).T)
    return sc, r, t, rms, float(np.percentile(lat_dev, 95)), rev, shift


def resolve_track_profile(track, filename, profiles):
    """Pick a profile by explicit key, else by filename match. None when ambiguous."""
    if track:
        return track if track in profiles else None
    lower_path = filename.lower()
    for key, cfg in profiles.items():
        if key in lower_path or cfg.get("name", "").lower() in lower_path:
            return key
    return None


def probe_thunderhill_layout(log_path):
    """Decide east/west from the Car Coord X channel only.

    Reads just the ``Car Coord X`` channel (other channels' data stays
    untouched thanks to lazy loading) and returns ``(key, xmin)`` where
    ``key`` is ``thunderhill_west`` when ``min(X)`` exceeds
    :data:`THUNDERHILL_AUTO_THRESHOLD_M`, else ``thunderhill_east_bypass``.
    """
    ld = ldData.fromfile(log_path)
    ch = _get_ld_channel(ld, "Car Coord X")
    if ch is None:
        raise ValueError(
            f"'{log_path}' has no Car Coord X channel; "
            "cannot auto-select the Thunderhill layout",
        )
    x = np.asarray(ch.data, dtype=float)
    if x.size == 0:
        raise ValueError(
            f"'{log_path}' has an empty Car Coord X channel; "
            "cannot auto-select the Thunderhill layout",
        )
    xmin = float(np.min(x))
    key = THUNDERHILL_AUTO_WEST if xmin > THUNDERHILL_AUTO_THRESHOLD_M else THUNDERHILL_AUTO_EAST
    return key, xmin


def _get_ld_channel(ld, name):
    for c in ld.channs:
        if c.name == name:
            return c
    return None


def _channel_data(ld, name):
    for c in ld.channs:
        if c.name == name:
            freq = float(c.freq) if c.freq and c.freq > 0 else 20.0
            return np.asarray(c.data, dtype=float), freq
    return None, 20.0


def _best_lap_bounds(ld):
    """Bounds (start, end) of the fastest full-length lap.

    Lap Time resets first, Lap Number transitions second. Groups cluster by
    travelled distance so shortened in/out laps and pit driving never win.
    """
    n = max((len(c.data) for c in ld.channs if len(c.data)), default=0)
    bounds = []
    lt, freq = _channel_data(ld, "Lap Time")
    if lt is not None and len(lt) > 100:
        edges = [0]
        for i in range(1, len(lt)):
            if lt[i] < lt[i - 1] - 1.0:
                edges.append(i)
        edges.append(len(lt))
        if len(edges) > 2:
            bounds = [(edges[k], edges[k + 1], freq) for k in range(len(edges) - 1)]
    if not bounds:
        ln, freq = _channel_data(ld, "Lap Number")
        if ln is not None and len(ln) > 100:
            edges = [0]
            for i in range(1, len(ln)):
                if ln[i] != ln[i - 1]:
                    edges.append(i)
            edges.append(len(ln))
            if len(edges) > 2:
                bounds = [(edges[k], edges[k + 1], freq) for k in range(len(edges) - 1)]
    timed = [((e - s) / f, s, e) for s, e, f in bounds if 30.0 < (e - s) / f < 900.0]
    if not timed:
        raise ValueError("No complete laps found")
    spd, _ = _channel_data(ld, "Ground Speed")
    if spd is not None and len(spd) == n:
        dists = []
        for _dur, s, e in timed:
            dists.append(float(np.trapezoid(spd[s:e] / 3.6, dx=1.0 / 20.0)))
        members = [sum(1 for d in dists if abs(d - di) <= 0.05 * di) for di in dists]
        target = dists[max(range(len(dists)), key=lambda i: (members[i], dists[i]))]
        timed = [t for t, d in zip(timed, dists, strict=True) if abs(d - target) <= 0.05 * target]
        if not timed:
            raise ValueError("No full-length laps found")
    dur, s, e = min(timed)
    return s, e, dur


def _official_lap_durations(last, n, edges, freq):
    """Game-official durations for complete laps, keyed by group index.

    The game latches each completed lap's time into Last Lap Time at the
    reset, so the value observed during group k+1 belongs to lap k. Exact
    to the ms (e.g. 116.867); sample-grid interpolation instead jitters
    within the 1/300 s timer quantum, and downstream renderers (i2 1.0)
    jitter the same way, so no interpolation can reproduce them at 0 ms.
    Use the official value when sane (finite, positive, within 2 s of the
    interpolated duration); otherwise leave the group on interpolation.
    """
    official = {}
    if last is None or len(last) != n:
        return official
    for k in range(1, len(edges) - 2):
        seg = np.asarray(last[edges[k + 1]:edges[k + 2]], dtype=float)
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        cand = float(np.median(seg))
        interp = (edges[k + 1] - edges[k]) / freq
        if cand > 0.0 and abs(cand - interp) <= 2.0:
            official[k] = cand
    return official


def _all_lap_bounds(ld, min_lap_sec=15.0):
    """Every Lap Time group as (groups, resets).

    ``groups`` is a list of (duration, start, end, t_start, t_end).
    Completed laps take the game's official duration from Last Lap Time
    (see ``_official_lap_durations``); partial edge laps (out/in) and files
    without that channel use reset-crossing interpolation: the Lap Time
    channel is the game's running timer, but the reset falls between two
    20 Hz samples, so ``(e - s) / freq`` is only accurate to the 0.05 s
    sample grid. The true crossing sits ``lt[e]`` seconds before sample
    ``e`` (time elapsed since the reset) and ``lt[s]`` seconds after sample
    ``s``, hence ``duration = (e - s) / freq - post + pre``. The first group
    started before recording (pre = 0: the Out Lap counts from log start)
    and a trailing group has no post sample (post = 0: the In Lap counts to
    file end). Non-timer-looking edge values fall back to the grid duration.

    ``resets`` holds the start/finish crossing times (one per observed Lap
    Time reset), accumulated from the final durations so beacon diffs always
    equal the displayed lap durations. Only true crossings become Beacons:
    log start (mid-lap) and file end (truncated lap) are not crossings, so
    emitting markers there makes i2 render a phantom zero-length lap and
    shifts every lap number by one.

    Completed laps preferably use the game's own official time latched in
    Last Lap Time (exact to the ms); sample interpolation only jitters
    within the 1/300 s timer quantum and can never reproduce a downstream
    renderer's own jitter at 0 ms.
    """
    lt, freq = _channel_data(ld, "Lap Time")
    if lt is None or len(lt) < 100:
        return [], []
    last, _ = _channel_data(ld, "Last Lap Time")
    edges = [0]
    for i in range(1, len(lt)):
        if lt[i] < lt[i - 1] - 1.0:
            edges.append(i)
    edges.append(len(lt))
    if len(edges) <= 2:
        return [], []
    n = len(lt)
    official = _official_lap_durations(last, n, edges, freq)
    bounds = []
    resets = []
    cum = 0.0
    for k in range(len(edges) - 1):
        s, e = edges[k], edges[k + 1]
        grid = (e - s) / freq
        post = float(lt[e]) if e < n and np.isfinite(lt[e]) else 0.0
        pre = float(lt[s]) if s > 0 and np.isfinite(lt[s]) else 0.0
        if not (0.0 <= post <= 5.0 and 0.0 <= pre <= 5.0):
            dur = grid
        else:
            t_start_meas = (s / freq - pre) if s > 0 else 0.0
            t_end_meas = (e / freq - post) if e < n else e / freq
            dur = t_end_meas - t_start_meas
            if k in official:
                dur = official[k]
        t_start, t_end = cum, cum + dur
        bounds.append((dur, s, e, t_start, t_end))
        cum = t_end
        if e < n:
            resets.append(cum)
    return bounds, resets


def _gps_to_local_m(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0, lon0 = float(np.median(lat)), float(np.median(lon))
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y]), lat0, lon0


def calibrate_track(rw_ld_path, acti_ld_path, track_key, track_name=None):
    """Fit a track profile from a real-world .ld and an ACTI .ld.

    Uses each file's fastest full-length lap. Refuses to return a profile
    when the fitted scale deviates more than 5% from 1.0 or lateral p95
    exceeds 25 m: that means a wrong layout, not a bad fit.
    """
    rw_ld = ldData.fromfile(rw_ld_path)
    acti_ld = ldData.fromfile(acti_ld_path)

    rs, re, rdur = _best_lap_bounds(rw_ld)
    rw_lat_ch = _get_ld_channel(rw_ld, "GPS Latitude")
    rw_lon_ch = _get_ld_channel(rw_ld, "GPS Longitude")
    if rw_lat_ch is None or rw_lon_ch is None:
        raise ValueError(f"{rw_ld_path}: missing GPS channels for calibration")
    rw_xy, lat0, lon0 = _gps_to_local_m(
        np.asarray(rw_lat_ch.data, dtype=float)[rs:re],
        np.asarray(rw_lon_ch.data, dtype=float)[rs:re],
    )

    as_, ae, adur = _best_lap_bounds(acti_ld)
    ac_x_ch = _get_ld_channel(acti_ld, "Car Coord X")
    ac_y_ch = _get_ld_channel(acti_ld, "Car Coord Y")
    if ac_x_ch is None or ac_y_ch is None:
        raise ValueError(f"{acti_ld_path}: missing Car Coord channels for calibration")
    print(f"  reference lap {rdur:.2f}s, sim lap {adur:.2f}s")

    sc, r, t, rms, p95, rev, shift = fit_similarity(
        np.column_stack(
            [
                np.asarray(ac_x_ch.data, dtype=float)[as_:ae],
                np.asarray(ac_y_ch.data, dtype=float)[as_:ae],
            ],
        ),
        rw_xy,
        samples=1200,
        shift_step=2,
    )
    print(f"  scale={sc:.4f} rms={rms:.2f}m lateral_p95={p95:.1f}m reversed={rev} shift={shift}")
    if abs(sc - 1.0) > 0.05:
        raise ValueError(
            f"Refusing profile '{track_key}': scale {sc:.3f} deviates >5% from 1.0 "
            "(wrong layout, not a bad fit)",
        )
    if p95 > 25.0:
        raise ValueError(
            f"Refusing profile '{track_key}': lateral p95 {p95:.1f}m > 25m (layout mismatch)",
        )

    theta_deg = round(float(np.degrees(np.arctan2(r[1, 0], r[0, 0])) % 360), 3)
    return {
        "name": track_name if track_name else track_key.replace("_", " ").title(),
        "ref_lat": round(lat0, 6),
        "ref_lon": round(lon0, 6),
        "theta_deg": theta_deg,
        "scale": round(float(sc), 4),
        "dx_m": round(float(t[0]), 3),
        "dy_m": round(float(t[1]), 3),
        "rmse_error_m": round(rms, 2),
    }


def _opening_group_is_complete_lap(data_log, t_end_first, dur_first):
    """Whether the opening ACTI segment is a complete lap, not an out-lap.

    The sim often starts recording on (or rolling toward) the timing line --
    e.g. an AC spawn -- so the first logged segment already runs
    start-line to start-line at full pace and deserves to count as a timed
    lap (the i2 Out Lap 0:00.000 row comes from the leading 0.0 beacon, not
    from this segment). A genuine pit-out or mid-lap join stays an Out Lap:
    the segment must be full-length (>= 15 s), start within 150 m of where
    it ends (the line), and start rolling (>= 80 km/h over the first 3 s,
    ruling out pit stalls and standing starts).
    """
    if dur_first < 15.0:
        return False
    try:
        lat_v = np.asarray(data_log.channels[CH_GPS_LATITUDE].values, dtype=float)
        lon_v = np.asarray(data_log.channels[CH_GPS_LONGITUDE].values, dtype=float)
        t_v = np.asarray(data_log.channels[CH_GPS_LATITUDE].timestamps, dtype=float)
        spd_v = np.asarray(data_log.channels[CH_GROUND_SPEED].values, dtype=float)
        spd_t = np.asarray(data_log.channels[CH_GROUND_SPEED].timestamps, dtype=float)
    except (KeyError, AttributeError, TypeError, ValueError):
        return False
    if lat_v.size == 0 or t_v.size == 0 or spd_v.size == 0:
        return False
    i1 = int(np.argmin(np.abs(t_v - t_end_first)))
    dist_m = np.sqrt(
        ((lat_v[0] - lat_v[i1]) * 111000.0) ** 2 + ((lon_v[0] - lon_v[i1]) * 85670.0) ** 2
    )
    if not np.isfinite(dist_m) or dist_m > 150.0:
        return False
    early = spd_v[spd_t <= (spd_t[0] + 3.0)]
    early = early[np.isfinite(early)]
    if early.size == 0 or float(np.mean(early)) < 80.0:
        return False
    return True


def _build_laps_info(groups, freq, resets=None, first_timed=False):
    """laps_info from Lap Time groups: first row is the Out Lap, unless it is
    a complete lap in disguise.

    The opening group is usually the run from recording start (pits or
    mid-lap) to the first start/finish crossing -- never a timed lap -- so it
    is typed "Out Lap" unconditionally. The exception (``first_timed``) is a
    flying recording start right at the timing line -- e.g. an AC spawn --
    whose opening segment already runs line-to-line at full pace; that one
    counts as timed, while the leading 0.0 beacon still renders the i2 Out
    Lap 0:00.000 row ahead of it. Interior full-length groups are timed
    (>=15 s), the trailing group is the In Lap; short fragments of the
    trailing edge stay "Partial". The leading 0.0 beacon always renders the
    i2 Out Lap 0:00.000 row, so a timed opening segment still sits behind an
    out-lap row in the lap table.

    Group tuples carry interpolated (duration, start, end, t_start, t_end).
    ``resets`` holds the interpolated start/finish crossing times, which
    become Beacons (falling back to group boundaries when resets is None,
    e.g. legacy callers). A (0.0, "Start/Finish") beacon is always prepended
    (unless a true crossing already sits at zero) so that i2 renders a
    leading "Out Lap 0:00.000" row ahead of the timed laps.
    """
    laps = []
    for k, item in enumerate(groups):
        dur, s, e = item[0], item[1], item[2]
        if len(item) >= 5:
            start_s, end_s = float(item[3]), float(item[4])
        else:
            start_s = s / freq
            end_s = e / freq
        if k == 0:
            kind = "Timed" if (first_timed and dur >= 15.0) else "Out Lap"
        elif k == len(groups) - 1:
            kind = "In Lap" if dur > 15.0 else "Partial"
        else:
            kind = "Timed" if dur >= 15.0 else "Partial"
        laps.append(
            {
                "lap_num": k + 1,
                "lap_label": str(k + 1),
                "start_time": start_s,
                "end_time": end_s,
                "duration": dur,
                "type": kind,
                "stint": 0,
            },
        )
    if resets is None:
        beacons = []
        for item in groups:
            dur, s, e = item[0], item[1], item[2]
            if len(item) >= 5:
                start_s, end_s = float(item[3]), float(item[4])
            else:
                start_s = s / freq
                end_s = e / freq
            for boundary in (start_s, end_s):
                if not any(abs(boundary - b[0]) <= 1e-6 for b in beacons):
                    beacons.append((boundary, "Start/Finish"))
    else:
        # A leading 0.0 beacon makes i2 render the opening "Out Lap
        # 0:00.000" row ahead of the timed laps; the real S/F crossings
        # follow from `resets`. Skip it when a true crossing already sits
        # at (or within rounding of) zero.
        beacon_times = [float(t) for t in resets]
        if not beacon_times or beacon_times[0] > 1e-9:
            beacon_times = [0.0] + beacon_times
        beacons = [(t, "Start/Finish") for t in beacon_times]
    timed = [lap for lap in laps if lap["type"] == "Timed"]
    return {
        "laps": laps,
        "total_laps": len(laps),
        "fastest_lap": min(timed, key=lambda lap: lap["duration"])["lap_num"] if timed else 1,
        "fastest_time": min((lap["duration"] for lap in timed), default=0.0),
        "beacons": sorted(beacons, key=lambda item: item[0]),
    }


def _filter_acti_lap(data_log, laps_info, target_lap):
    """Keep one lap, rebased to zero. Returns the laps_info to store."""
    if target_lap is None or str(target_lap).lower() == "all":
        return laps_info
    try:
        lap_no = int(target_lap)
        recs = [lap for lap in laps_info["laps"] if lap["lap_num"] == lap_no]
    except (ValueError, IndexError):
        recs = []
    if not recs:
        print(f"WARNING: Lap '{target_lap}' not found in ACTI file; keeping all laps")
        return laps_info
    t_lo, t_hi = recs[0]["start_time"], recs[0]["end_time"]
    for ch in data_log.channels.values():
        keep_t, keep_v = [], []
        for t, v in zip(ch.timestamps, ch.values, strict=False):
            if t_lo <= t <= t_hi:
                keep_t.append(t - t_lo)
                keep_v.append(v)
        ch.set_samples(np.asarray(keep_t), np.asarray(keep_v))
    duration = t_hi - t_lo
    return {
        "laps": [
            {
                "lap_num": recs[0]["lap_num"],
                "lap_label": recs[0].get("lap_label", str(recs[0]["lap_num"])),
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration,
                "type": recs[0].get("type", "Timed"),
                "stint": recs[0].get("stint", 0),
            },
        ],
        "total_laps": 1,
        "fastest_lap": recs[0]["lap_num"],
        "fastest_time": duration,
        "beacons": [(0.0, "Start/Finish"), (duration, "Start/Finish")],
    }


def _merge_profiles(anchors):
    profiles = dict(ACTI_TRACK_PROFILES)
    if anchors:
        for key, cfg in anchors.items():
            merged = dict(profiles.get(key, {}))
            merged.update(cfg)
            profiles[key] = merged
    return profiles


# Merge window for --realign-sf: an AC reset this far (or nearer) from a GPS
# Start/Finish crossing is the same lap boundary seen from two lines, so the
# GPS time wins; a reset farther away is a genuine crossing the GPS detector
# gated out (first 15 s / last 5 s of file) and is kept. Half of
# detect_beacons' min_lap_sec, far below any real lap duration.
_REALIGN_MERGE_SEC = 20.0


def _realign_groups_to_real_sf(data_log, groups, resets, duration):
    """Re-cut ACTI segments at the real-world Start/Finish line.

    The sim's Lap Time resets fire at the mod's own timing line, which can
    sit metres away from the physical start lights (Thunderhill East is
    ~63 m / ~1.25 s off). When the transformed GPS trace crosses a known
    ``Start/Finish`` trap, rebuild the segments from those crossings so
    sim laps align corner-for-corner with real logs.

    Returns ``(groups, resets)`` rebuilt from the merged boundaries, or the
    inputs unchanged when no usable real line exists: unknown venue,
    sector-style trap lists (e.g. CA-9, whose traps are not named
    Start/Finish), or no crossing detected (wrong direction, recording
    never reaches the line). An AC reset with no GPS crossing within
    ``_REALIGN_MERGE_SEC`` keeps its sim value: the GPS detector gates out
    crossings in the first 15 s and last 5 s of the file, and a reset there
    is still a genuine line crossing, not a phantom. Rebuilt segments keep
    the ``(duration, 0, 0, t_start, t_end)`` shape that
    :func:`_build_laps_info` accepts.
    """
    try:
        found = [t for t, name in data_log.detect_beacons() if name == "Start/Finish"]
    except (ValueError, TypeError, AttributeError):
        return groups, resets
    found = sorted(t for t in found if 0.0 < t < duration)
    if found:
        keep = [
            float(r)
            for r in resets
            if not any(abs(float(r) - c) <= _REALIGN_MERGE_SEC for c in found)
        ]
    else:
        # Nothing passed the full gates (short file, line inside the head /
        # tail zones): every sim reset is a replace candidate.
        keep = [float(r) for r in resets]
    if keep:
        # Second pass with the head/tail gates lifted: a kept reset means the
        # car was at the mod line but GPS saw no valid crossing. When the
        # trace actually reached the real line just inside the gated zone
        # (recording starts/ends right at the line), adopt that crossing.
        # Relaxed crossings only ever replace kept resets -- never add
        # boundaries -- so pit-lane crawls near the beacon cannot inject
        # phantom laps.
        try:
            relaxed = [
                t
                for t, name in data_log.detect_beacons(min_time_sec=0.0, end_gate_sec=0.0)
                if name == "Start/Finish"
            ]
        except (ValueError, TypeError, AttributeError):
            relaxed = []
        relaxed = sorted(t for t in relaxed if 0.0 < t < duration)
        used = set()
        patched = []
        for k in keep:
            cands = [c for c in relaxed if c not in used and abs(c - k) <= _REALIGN_MERGE_SEC]
            if cands:
                c = min(cands, key=lambda c: abs(c - k))
                used.add(c)
                patched.append(c)
            else:
                patched.append(k)
        keep = patched
    if not found and keep == [float(r) for r in resets]:
        # Relaxed pass changed nothing: keep the original Lap Time cuts verbatim.
        return groups, resets
    crossings = sorted(found + [k for k in keep if 0.0 < k < duration])
    if not crossings:
        return groups, resets
    bounds = [0.0] + crossings + [duration]
    segs = [(b - a, 0, 0, a, b) for a, b in zip(bounds, bounds[1:], strict=False) if b - a > 0]
    if not segs:
        return groups, resets
    return segs, crossings


def parse_acti_log(data_log, log_path, target_lap=None, track=None, anchors=None, realign_sf=False):
    """Populate channels from an Assetto Corsa ACTI .ld file.

    Car Coord X/Y place the trajectory through the track profile; GPS
    Latitude/Longitude/Heading are synthesised. ``track`` selects the profile
    explicitly, otherwise the filename must match one unambiguously.
    ``realign_sf`` re-cuts laps at the real-world Start/Finish line (from
    the transformed GPS trace) instead of the sim's own timing line,
    falling back to the sim cuts when no real crossing is detected.
    """
    data_log.clear()
    data_log.laps_info = {}

    profiles = _merge_profiles(anchors)
    if track == THUNDERHILL_AUTO_KEY:
        track, xmin = probe_thunderhill_layout(log_path)
        print(f"thunderhill_auto: Car Coord X min {xmin:.1f} m -> track profile '{track}'")
    key = resolve_track_profile(track, log_path, profiles)
    if key is None or key not in profiles:
        raise ValueError(
            "Cannot determine ACTI track profile for "
            f"'{log_path}'. Pass track with one of: "
            f"{sorted([*profiles.keys(), THUNDERHILL_AUTO_KEY])}",
        )
    cfg = profiles[key]

    ld = ldData.fromfile(log_path)

    def channel_length(name):
        c = _get_ld_channel(ld, name)
        return len(c.data) if c is not None else 0

    n = max((channel_length(c.name) for c in ld.channs), default=0)
    if n == 0:
        return

    ch_x = _get_ld_channel(ld, "Car Coord X")
    ch_y = _get_ld_channel(ld, "Car Coord Y")
    if ch_x is None or ch_y is None:
        raise ValueError(
            f"'{log_path}' has no Car Coord X/Y channels; not an ACTI log",
        )
    fx = float(ch_x.freq) if ch_x.freq and ch_x.freq > 0 else 20.0
    nx = len(ch_x.data)
    t_gps = np.arange(nx) / fx

    lat, lon, x_real_m, y_real_m = transform_ac_to_gps(ch_x.data, ch_y.data, cfg)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    heading = compute_gps_heading(x_real_m, y_real_m)

    for raw_name, (out_name, unit, dec) in _ACTI_CHANNEL_MAP.items():
        src = _get_ld_channel(ld, raw_name)
        if src is None or len(src.data) == 0:
            continue
        data_log.add_channel(out_name, unit, float, dec)
        fch = float(src.freq) if src.freq and src.freq > 0 else 20.0
        vals = np.asarray(src.data, dtype=float)
        data_log.channels[out_name].set_samples(np.arange(len(vals)) / fch, vals)

    # Mirror the sim pedal onto the canonical accelerator channel. The ACTI
    # app logs the driver's pedal input as "Throttle Pos" and never a
    # throttle-body angle; real-car logs keep the two apart (Accelerator
    # Pos = pedal, Throttle Pos = body). Without this mirror a sim run
    # could never be overlaid on a real car's Accelerator Pos trace.
    # A genuinely logged "Accelerator Pos" source channel always wins and
    # is preserved verbatim by the loop below instead.
    pedal = _get_ld_channel(ld, "Throttle Pos")
    if (
        pedal is not None
        and len(pedal.data) > 0
        and CH_ACCELERATOR_POS not in data_log.channels
        and _get_ld_channel(ld, "Accelerator Pos") is None
    ):
        fch = float(pedal.freq) if pedal.freq and pedal.freq > 0 else 20.0
        vals = np.asarray(pedal.data, dtype=float)
        data_log.add_channel(CH_ACCELERATOR_POS, "%", float, 1)
        data_log.channels[CH_ACCELERATOR_POS].set_samples(np.arange(len(vals)) / fch, vals)

    # Preserve every other source channel under its original name, unless
    # that name is already taken by a mapped / synthesised channel above.
    # Mapped raw names are skipped even when the canonical rename differs,
    # so a renamed channel never appears twice.
    for src in ld.channs:
        name = getattr(src, "name", "")
        if not name:
            continue
        if name in data_log.channels:
            continue
        if name in _ACTI_CHANNEL_MAP:
            continue
        try:
            vals = np.asarray(src.data, dtype=float)
        except (TypeError, ValueError):
            continue
        if vals.size == 0:
            continue
        fch = float(src.freq) if getattr(src, "freq", 0) and src.freq > 0 else 20.0
        try:
            kind = np.dtype(src.dtype).kind if getattr(src, "dtype", None) is not None else "f"
        except TypeError:
            kind = "f"
        data_type = float if kind in ("f", "c") else int
        try:
            dec = int(getattr(src, "dec", 0) or 0)
        except (TypeError, ValueError):
            dec = 0
        unit = getattr(src, "unit", "") or ""
        data_log.add_channel(name, unit, data_type, dec)
        data_log.channels[name].set_samples(np.arange(len(vals)) / fch, vals)

    data_log.add_channel(CH_GPS_LATITUDE, "deg", float, 7)
    data_log.channels[CH_GPS_LATITUDE].set_samples(t_gps, lat)
    data_log.add_channel(CH_GPS_LONGITUDE, "deg", float, 7)
    data_log.channels[CH_GPS_LONGITUDE].set_samples(t_gps, lon)
    data_log.add_channel(CH_GPS_HEADING, "deg", float, 3)
    data_log.channels[CH_GPS_HEADING].set_samples(t_gps, np.asarray(heading, dtype=float))

    groups, resets = _all_lap_bounds(ld)
    data_log.metadata["venue_name"] = cfg.get("name", "")
    if realign_sf and groups:
        groups, resets = _realign_groups_to_real_sf(data_log, groups, resets, data_log.duration())
    if groups:
        lap_ids = np.zeros(len(t_gps))
        for k, item in enumerate(groups):
            # Group edges index the Lap Time grid; map the interpolated
            # reset crossings onto the GPS clock.
            t0 = float(item[3]) if len(item) >= 5 else item[1] / fx
            t1 = float(item[4]) if len(item) >= 5 else item[2] / fx
            lap_ids[(t_gps >= t0) & (t_gps < t1)] = k + 1
        last_t1 = float(groups[-1][4]) if len(groups[-1]) >= 5 else groups[-1][1] / fx
        lap_ids[t_gps >= last_t1] = len(groups)
        data_log.add_channel(CH_LAP_NUMBER, "", int, 0)
        data_log.channels[CH_LAP_NUMBER].set_samples(t_gps, lap_ids)
    elif len(t_gps):
        # No Lap Time structure at all (missing channel or no recorded
        # reset): the whole log is the out lap, so the export still
        # carries exactly one Out Lap instead of an empty lap list.
        data_log.add_channel(CH_LAP_NUMBER, "", int, 0)
        data_log.channels[CH_LAP_NUMBER].set_samples(t_gps, np.ones(len(t_gps)))

    data_log.metadata["driver"] = getattr(ld.head, "driver", "")
    data_log.metadata["vehicle_id"] = getattr(ld.head, "vehicleid", "")
    # Resolved profile key + direction relevance for the CLI: only
    # bidirectional tracks take --direction-suffix _cw/_ccw names.
    data_log.metadata["acti_track"] = key
    data_log.metadata["acti_bidirectional"] = bool(cfg.get("bidirectional", False))
    # The ACTI app stores tyre info in the header short comment
    # (e.g. "Tires: Street (ST)"); carry it over so i2 still shows it.
    # An explicit --short_comment always wins in _build_motec_log.
    head_short = getattr(ld.head, "short_comment", "")
    if head_short:
        data_log.metadata["short_comment"] = head_short
    head_dt = getattr(ld.head, "datetime", None)
    if isinstance(head_dt, datetime.datetime):
        data_log.datetime = head_dt
        if head_dt.tzinfo is None:
            data_log.datetime_utc = head_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            data_log.datetime_utc = head_dt.astimezone(datetime.timezone.utc)

    if groups:
        first_item = groups[0]
        if len(first_item) >= 5:
            first_end = float(first_item[4])
        else:
            first_end = first_item[2] / fx
        first_timed = _opening_group_is_complete_lap(data_log, first_end, float(first_item[0]))
        laps_info = _build_laps_info(groups, fx, resets, first_timed=first_timed)
    else:
        duration = data_log.duration()
        laps_info = {
            "laps": [
                {
                    "lap_num": 1,
                    "lap_label": "1",
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration,
                    "type": "Out Lap",
                    "stint": 0,
                },
            ],
            "total_laps": 1,
            "fastest_lap": 1,
            "fastest_time": 0.0,
            "beacons": [(0.0, "Start/Finish")] if duration > 0 else [],
        }
    data_log.laps_info = _filter_acti_lap(data_log, laps_info, target_lap)
