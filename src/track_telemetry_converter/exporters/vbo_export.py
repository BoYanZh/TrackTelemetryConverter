"""Racelogic VBO telemetry export for TrackTelemetryConverter.

Writes the resampled log as a space-delimited `.vbo` file (the format
Racelogic VBOX and Dragy export), so converted sessions can be opened in
Circuit Tools, LapTimer, and similar GPS analysis tools.

The seven core GPS columns are always written. Extra columns are derived
dynamically from whatever channels exist via `_EXTRA_COLUMN_MAP`
(canonical channel -> VBO column name), so newly recognized inputs
round-trip without touching this file; channels whose units don't match
the column's expected units are skipped rather than mislabeled. Column
names are chosen to round-trip through this project's own VBO importer.
Times are UTC wall-clock (`HHMMSS.sss`) derived from the log's
absolute origin when available.

VBO has no NaN representation, so gaps are forward-filled (last-good value),
matching Dragy's own VBO writer behavior.
"""

from __future__ import annotations

import datetime

import numpy as np

from ..channels import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_COOLANT_TEMP,
    CH_ENGINE_OIL_TEMP,
    CH_ENGINE_RPM,
    CH_GEAR,
    CH_GEARBOX_TEMP,
    CH_GPS_ALTITUDE,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GPS_SATS,
    CH_GROUND_SPEED,
    CH_MANIFOLD_PRESS,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_VEHICLE_SPEED,
    CH_WHEEL_SPEED_FL,
    CH_WHEEL_SPEED_FR,
    CH_WHEEL_SPEED_RL,
    CH_WHEEL_SPEED_RR,
    CH_YAW_RATE,
    DISCRETE_CHANNELS,
)

# (header name, column name, format function) for the core GPS columns.
_CORE_COLUMNS = ("sats", "time", "lat", "long", "velocity", "heading", "height")
_HEADER_NAMES = {
    "sats": "satellites",
    "time": "time",
    "lat": "latitude",
    "long": "longitude",
    "velocity": "velocity kmh",
    "heading": "heading",
    "height": "height",
}

# Extra canonical channel -> (column name, header name, expected units, format).
# Exported columns are derived from whatever channels exist, so newly
# recognized inputs (e.g. RaceChrono `-canbus` columns) round-trip without
# touching this file. Column names must be single whitespace-free tokens:
# the space-delimited [data] section is mapped positionally. Header names
# mirror Dragy/RaceChrono layouts so this project's own importer reads them
# back. The `-obd` names cover Dragy's OBD family, `-canbus` the rest.
_EXTRA_COLUMN_MAP = {
    CH_CG_ACCEL_LON: ("longacc", "long accel g", "G", lambda v: f"{v:.3f}"),
    CH_CG_ACCEL_LAT: ("latacc", "lat accel g", "G", lambda v: f"{v:.3f}"),
    CH_ENGINE_RPM: ("rpm-obd", "rpm-obd", "rpm", lambda v: f"{v:.1f}"),
    CH_ACCELERATOR_POS: ("acceleration-obd", "acceleration-obd", "%", lambda v: f"{v:.3f}"),
    CH_THROTTLE_POS: ("throttle_pos-canbus", "throttle_pos-canbus", "%", lambda v: f"{v:.3f}"),
    CH_BRAKE_PRESS: ("brake_pressure-obd", "brake_pressure-obd", "kPa", lambda v: f"{v:.3f}"),
    CH_ENGINE_OIL_TEMP: ("engine_oil_temp-obd", "engine_oil_temp-obd", "C", lambda v: f"{v:.3f}"),
    CH_COOLANT_TEMP: ("coolant_temp-obd", "coolant_temp-obd", "C", lambda v: f"{v:.3f}"),
    CH_MANIFOLD_PRESS: ("manifold_pressure-obd", "manifold_pressure-obd", "kPa", lambda v: f"{v:.3f}"),
    CH_GEAR: ("gear-canbus", "gear-canbus", "", lambda v: f"{v:.0f}"),
    CH_GEARBOX_TEMP: ("gearbox_temp-canbus", "gearbox_temp-canbus", "C", lambda v: f"{v:.3f}"),
    CH_YAW_RATE: ("yaw_rate-canbus", "yaw_rate-canbus", "deg/s", lambda v: f"{v:.3f}"),
    CH_STEERING_ANGLE: ("steering_angle-canbus", "steering_angle-canbus", "deg", lambda v: f"{v:.3f}"),
    CH_VEHICLE_SPEED: ("velocity-canbus", "velocity-canbus kmh", "km/h", lambda v: f"{v:.3f}"),
    CH_WHEEL_SPEED_FL: ("wheel_speed_fl-canbus", "wheel_speed_fl-canbus", "km/h", lambda v: f"{v:.3f}"),
    CH_WHEEL_SPEED_FR: ("wheel_speed_fr-canbus", "wheel_speed_fr-canbus", "km/h", lambda v: f"{v:.3f}"),
    CH_WHEEL_SPEED_RL: ("wheel_speed_rl-canbus", "wheel_speed_rl-canbus", "km/h", lambda v: f"{v:.3f}"),
    CH_WHEEL_SPEED_RR: ("wheel_speed_rr-canbus", "wheel_speed_rr-canbus", "km/h", lambda v: f"{v:.3f}"),
}


def _fill_gaps(values):
    """Forward-fill then back-fill NaN runs (VBO cannot represent NaN)."""
    filled = np.array(values, dtype=np.float64, copy=True)
    if np.all(~np.isfinite(filled)):
        return filled
    last = np.nan
    for i in range(len(filled)):
        if np.isfinite(filled[i]):
            last = filled[i]
        elif np.isfinite(last):
            filled[i] = last
    if not np.isfinite(filled[0]):
        first_valid = filled[np.isfinite(filled)][0]
        filled[~np.isfinite(filled)] = first_valid
    return filled


def _sample_on_grid(channel, grid):
    """Sample a channel onto the time grid (ZOH for discrete channels)."""
    from ..interpolation import _interp_zoh

    ts = np.asarray(channel.timestamps, dtype=np.float64)
    vs = np.asarray(channel.values, dtype=np.float64)
    mask = np.isfinite(ts) & np.isfinite(vs)
    if mask.sum() == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    if mask.sum() == 1:
        return np.full_like(grid, vs[mask][0], dtype=np.float64)
    if channel.name in DISCRETE_CHANNELS:
        return _interp_zoh(grid, ts[mask], vs[mask])
    return np.interp(grid, ts[mask], vs[mask])


def _utc_origin(data_log):
    """Absolute UTC origin as a datetime, or None if unknown."""
    origin_ms = getattr(data_log, "time_origin_epoch_ms", None)
    if origin_ms:
        return datetime.datetime.fromtimestamp(
            origin_ms / 1000.0, datetime.timezone.utc
        ).replace(tzinfo=None)
    dt_utc = getattr(data_log, "datetime_utc", None)
    if dt_utc is not None:
        return dt_utc
    return None


def write_vbo(data_log, vbo_filename):
    """Write resampled DataLog channels to a Racelogic `.vbo` file.

    Returns True on success, False if no GPS data is available.
    """
    lat_chan = data_log.channels.get(CH_GPS_LATITUDE)
    lon_chan = data_log.channels.get(CH_GPS_LONGITUDE)
    if lat_chan is None or lon_chan is None:
        return False
    grid = np.asarray(lat_chan.timestamps, dtype=np.float64)
    if len(grid) == 0:
        return False

    lat = _fill_gaps(_sample_on_grid(lat_chan, grid))
    lon = _fill_gaps(_sample_on_grid(lon_chan, grid))
    if not np.isfinite(lat).any() or not np.isfinite(lon).any():
        return False

    def get(name):
        chan = data_log.channels.get(name)
        if chan is None:
            return np.full_like(grid, np.nan, dtype=np.float64)
        return _fill_gaps(_sample_on_grid(chan, grid))

    sats = get(CH_GPS_SATS)
    vel = get(CH_GROUND_SPEED)
    heading = get(CH_GPS_HEADING) % 360.0
    height = get(CH_GPS_ALTITUDE)

    extras = []
    export_names = dict(_EXTRA_COLUMN_MAP)
    # Accelerator and throttle pedals carry the same signal (the pipeline
    # mirrors one into the other); export a single pedal column to avoid a
    # redundant duplicate.
    if CH_ACCELERATOR_POS in data_log.channels and CH_THROTTLE_POS in data_log.channels:
        del export_names[CH_THROTTLE_POS]
    for ch_name, (col_name, header_name, unit, fmt) in export_names.items():
        chan = data_log.channels.get(ch_name)
        if chan is None:
            continue
        if unit and chan.units != unit:
            continue
        vals = _fill_gaps(_sample_on_grid(chan, grid))
        if not np.isfinite(vals).any():
            continue
        extras.append((col_name, header_name, fmt, vals))

    origin = _utc_origin(data_log)
    if origin is not None:
        date_line = "File created on %s @ %s" % (
            origin.strftime("%d/%m/%Y"), origin.strftime("%H:%M")
        )
        day_start = origin.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        date_line = "File created by TrackTelemetryConverter"
        day_start = None

    with open(vbo_filename, "w", encoding="utf-8", newline="") as f:
        f.write(date_line + "\n\n[header]\n")
        for col in _CORE_COLUMNS:
            f.write(_HEADER_NAMES[col] + "\n")
        for _, header_name, _, _ in extras:
            f.write(header_name + "\n")
        f.write("\n[channel units]\n\n[comments]\n")
        f.write("Generated by TrackTelemetryConverter\n")
        f.write("\n[column names]\n")
        f.write(" ".join([*_CORE_COLUMNS, *(c for c, _, _, _ in extras)]) + "\n\n[data]\n")
        for i, t in enumerate(grid):
            if day_start is not None:
                tod = (day_start + datetime.timedelta(seconds=float(t))).time()
                tod_s = tod.hour * 3600.0 + tod.minute * 60.0 + tod.second + tod.microsecond / 1e6
            else:
                tod_s = float(t) % 86400.0
            hh = int(tod_s // 3600)
            mm = int((tod_s % 3600) // 60)
            ss = tod_s % 60
            row = [
                "%03d" % (int(round(sats[i])) if np.isfinite(sats[i]) else 0),
                "%02d%02d%06.3f" % (hh, mm, ss),
                "%+011.6f" % (lat[i] * 60.0),
                "%+011.6f" % (lon[i] * 60.0),
                "%07.3f" % vel[i],
                "%06.2f" % heading[i],
                "%+09.2f" % height[i],
            ]
            for _, _, fmt, vals in extras:
                row.append(fmt(vals[i]))
            f.write(" ".join(row) + "\n")
    return True
