"""Dragy `.dlap` telemetry archive export for TrackTelemetryConverter.

Writes the resampled log as a `.dlap` file (outer ZIP with one AES-128-CBC
encrypted `<id>.group` inner ZIP), mirroring the layout Dragy's app
produces, so converted sessions round-trip through this project's own
DLAP importer and stay structurally compatible with Dragy's frontend:

- `loopdata_<ts>.txt`: 25 Hz-style GPS/IMU CSV, 27 columns. Columns the
  frontend never reads (16, 18-24 diagnostics) are written as neutral
  placeholders; per-lap time, brake flag, and row index are derived from
  the log's laps and channels.
- `<ts>.obd`: OBD rows (`t,hz,rpm,acc,brake,oil,coolant,map`) when OBD
  channels exist; omitted otherwise.
- `<id>.json`: session metadata with per-lap `startDataIndex` /
  `endDataIndex` row ranges and the OBD pid order.

Column semantics and crypto parameters are the inverse of
`parsers/dlap_parser.py` (see that module for provenance notes). No
track-map PNG is written on export (the importer does not require one).
"""

from __future__ import annotations

import datetime
import io
import json
import zipfile

import numpy as np

from ..channels import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LON,
    CH_COOLANT_TEMP,
    CH_CORR_DIST,
    CH_ENGINE_OIL_TEMP,
    CH_ENGINE_RPM,
    CH_GPS_ACCURACY,
    CH_GPS_ALTITUDE,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GPS_SATS,
    CH_GROUND_SPEED,
    CH_MANIFOLD_PRESS,
    DISCRETE_CHANNELS,
)
from ..interpolation import _interp_zoh, _mask_interp_outside_range

# OBD pid order written to the session JSON (must match the *.obd columns).
_OBD_PID_ORDER = (
    "PID_RPM",
    "PID_ACC_POS",
    "PID_BRAKE_PRESS",
    "PID_OIL_TEMP",
    "PID_COOLANT_TEMP",
    "PID_MAP",
)
_OBD_PID_CHANNELS = {
    "PID_RPM": CH_ENGINE_RPM,
    "PID_ACC_POS": CH_ACCELERATOR_POS,
    "PID_BRAKE_PRESS": CH_BRAKE_PRESS,
    "PID_OIL_TEMP": CH_ENGINE_OIL_TEMP,
    "PID_COOLANT_TEMP": CH_COOLANT_TEMP,
    "PID_MAP": CH_MANIFOLD_PRESS,
}


def _new_aes_cipher():
    try:
        from Crypto.Cipher import AES
    except ImportError as exc:
        raise RuntimeError(
            "Dragy .dlap export requires the 'pycryptodome' package "
            '(pip install ".[dlap]").'
        ) from exc
    return AES.new(
        bytes.fromhex("6932466c655a6e640000000000000000"),
        AES.MODE_CBC,
        bytes.fromhex("69336576386c656e0000000000000000"),
    )


def _sample_on_grid(channel, grid):
    """Sample a channel onto the time grid (ZOH for discrete channels).

    NaN outside the channel's own sample span stays NaN (partial-coverage
    channels such as OBD must not be forward-filled into gaps).
    """
    ts = np.asarray(channel.timestamps, dtype=np.float64)
    vs = np.asarray(channel.values, dtype=np.float64)
    mask = np.isfinite(ts) & np.isfinite(vs)
    if mask.sum() == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    if mask.sum() == 1:
        return np.full_like(grid, vs[mask][0], dtype=np.float64)
    if channel.name in DISCRETE_CHANNELS:
        return _mask_interp_outside_range(
            _interp_zoh(grid, ts[mask], vs[mask]), grid, ts[mask]
        )
    return _mask_interp_outside_range(
        np.interp(grid, ts[mask], vs[mask]), grid, ts[mask]
    )


def _session_stamp(data_log):
    """Compact timestamp for member names, from wall clock or zeros."""
    origin_ms = getattr(data_log, "time_origin_epoch_ms", None)
    if origin_ms:
        moment = datetime.datetime.fromtimestamp(
            origin_ms / 1000.0, datetime.timezone.utc
        )
    elif getattr(data_log, "datetime_utc", None) is not None:
        moment = data_log.datetime_utc
    elif getattr(data_log, "datetime", None) is not None:
        moment = data_log.datetime
    else:
        moment = datetime.datetime(2000, 1, 1)
    return moment.strftime("%Y%m%d%H%M%S")


def _lap_times(data_log, grid):
    """Per-row lap-relative times from laps_info (else session time)."""
    laps = []
    info = getattr(data_log, "laps_info", None) or {}
    for lap in info.get("laps", []):
        try:
            start = float(lap["start_time"])
            end = float(lap["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            laps.append((start, end))
    laps.sort()
    if not laps:
        return grid - grid[0]
    user_time = np.full_like(grid, np.nan, dtype=np.float64)
    for start, end in laps:
        inside = (grid >= start) & (grid <= end)
        user_time[inside] = grid[inside] - start
    return user_time


def _cumulative_distance(speed_kmh, grid):
    """Integrate speed into metres travelled from session start."""
    speed_ms = np.where(np.isfinite(speed_kmh), speed_kmh, 0.0) / 3.6
    dt = np.diff(grid, prepend=grid[0])
    dt = np.where(dt > 0, dt, 0.0)
    return np.cumsum(speed_ms * dt)


def _write_loopdata(data_log, grid, usertime, distance, get):
    """Render the 27-column loopdata CSV text."""
    lat = get(CH_GPS_LATITUDE)
    lon = get(CH_GPS_LONGITUDE)
    speed = get(CH_GROUND_SPEED)
    acc = get(CH_CG_ACCEL_LON)
    alt = get(CH_GPS_ALTITUDE)
    accuracy = get(CH_GPS_ACCURACY)
    sats = get(CH_GPS_SATS)
    heading = get(CH_GPS_HEADING)
    brake = get(CH_BRAKE_POS)
    origin_ms = getattr(data_log, "time_origin_epoch_ms", None)
    epoch0 = origin_ms / 1000.0 if origin_ms else 0.0
    lines = []
    for i, t in enumerate(grid):
        def num(value, decimals):
            if not np.isfinite(value):
                return ""
            return f"{value:.{decimals}f}"

        row = [
            num(usertime[i], 3),
            f"{t:.2f}",
            num(speed[i], 3),
            num(acc[i], 3),
            num(alt[i], 1),
            num(lat[i], 6),
            num(lon[i], 6),
            num(distance[i], 2),
            num(accuracy[i], 1),
            num(sats[i], 0),
            num(heading[i], 2),
            "1" if np.isfinite(brake[i]) and brake[i] > 50.0 else "0",
            "0.000",
            "0.000",
            "0.000",
            str(i),
            "",
            "0.000",
            "3",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            "0.000",
            f"{epoch0 + t:.2f}" if epoch0 else "0.00",
            f"{t:.3f}",
        ]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def _write_obd(grid, get):
    """Render the OBD CSV text, or None when no OBD channels exist."""
    series = {}
    for pid in _OBD_PID_ORDER:
        chan_name = _OBD_PID_CHANNELS[pid]
        vals = get(chan_name)
        if np.isfinite(vals).any():
            series[pid] = vals
    if not series:
        return None, []
    span = grid[-1] - grid[0]
    hz = len(grid) / span if span > 0 else 0.0
    lines = []
    for i, t in enumerate(grid):
        if not any(np.isfinite(series[pid][i]) for pid in series):
            continue
        values = [
            f"{series[pid][i]:.4f}" if np.isfinite(series[pid][i]) else ""
            for pid in _OBD_PID_ORDER
        ]
        lines.append(",".join([f"{t:.3f}", f"{hz:.2f}", *values]))
    if not lines:
        return None, []
    return "\n".join(lines) + "\n", list(_OBD_PID_ORDER)


def _write_session_json(data_log, grid, stamp, obd_name, pids):
    """Render the session metadata JSON text."""
    info = getattr(data_log, "laps_info", None) or {}
    freq = 0.0
    if len(grid) > 1 and grid[-1] > grid[0]:
        freq = (len(grid) - 1) / (grid[-1] - grid[0])
    laps = []
    for lap in info.get("laps", []):
        try:
            start = float(lap["start_time"])
            end = float(lap["end_time"])
            num = int(lap.get("lap_num", len(laps) + 1))
        except (KeyError, TypeError, ValueError):
            continue
        if not end > start:
            continue
        lo = int(np.searchsorted(grid, start, side="left"))
        hi = int(np.searchsorted(grid, end, side="right")) - 1
        lo = max(0, min(lo, len(grid) - 1))
        hi = max(lo, min(hi, len(grid) - 1))
        duration = end - start
        laps.append({
            "idx": num,
            "startDataIndex": lo,
            "endDataIndex": hi,
            "frequence": round(freq, 3),
            "useTime": round(duration, 3),
            "distance": 0.0,
            "testTime": 0.0,
        })
    meta = getattr(data_log, "metadata", None) or {}
    session = {
        "trackTitle": meta.get("venue_name", ""),
        "brandName": meta.get("vehicle_id", ""),
        "manufacture": "TrackTelemetryConverter",
        "totalTime": round(float(grid[-1] - grid[0]), 3),
        "lapNum": len(laps),
        "laps": laps,
    }
    if obd_name is not None:
        session["obd"] = {
            "version": 5,
            "fileName": obd_name,
            "pidEnums": json.dumps(list(pids)),
        }
    return json.dumps(session)


def write_dlap(data_log, dlap_path):
    """Write resampled DataLog channels to a Dragy `.dlap` archive.

    Returns True on success, False if no GPS data is available.
    """
    lat_chan = data_log.channels.get(CH_GPS_LATITUDE)
    lon_chan = data_log.channels.get(CH_GPS_LONGITUDE)
    if lat_chan is None or lon_chan is None:
        return False
    grid = np.asarray(lat_chan.timestamps, dtype=np.float64)
    if len(grid) == 0:
        return False

    def get(name):
        chan = data_log.channels.get(name)
        if chan is None:
            return np.full_like(grid, np.nan, dtype=np.float64)
        return _sample_on_grid(chan, grid)

    lat = get(CH_GPS_LATITUDE)
    lon = get(CH_GPS_LONGITUDE)
    if not np.isfinite(lat).any() or not np.isfinite(lon).any():
        return False

    speed = get(CH_GROUND_SPEED)
    distance = get(CH_CORR_DIST)
    if not np.isfinite(distance).any():
        distance = _cumulative_distance(speed, grid)
    usertime = _lap_times(data_log, grid)
    stamp = _session_stamp(data_log)
    group_id = f"ttc_{stamp}"
    loop_name = f"loopdata_{stamp[:8]}_{stamp[8:]}.txt"
    obd_text, pids = _write_obd(grid, get)
    obd_name = f"{stamp}.obd" if obd_text is not None else None

    inner_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_buffer, "w", compression=zipfile.ZIP_DEFLATED) as inner:
        inner.writestr(loop_name, _write_loopdata(data_log, grid, usertime, distance, get))
        inner.writestr(f"{group_id}.json", _write_session_json(data_log, grid, stamp, obd_name, pids))
        if obd_text is not None and obd_name is not None:
            inner.writestr(obd_name, obd_text)
    plain = inner_buffer.getvalue()
    pad = 16 - len(plain) % 16
    encrypted = _new_aes_cipher().encrypt(plain + bytes([pad]) * pad)

    with zipfile.ZipFile(dlap_path, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr(f"{group_id}.group", encrypted)
    return True
