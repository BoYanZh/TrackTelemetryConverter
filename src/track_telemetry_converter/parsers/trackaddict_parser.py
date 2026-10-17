"""Parser for TrackAddict / RaceRender-Data CSV telemetry log files.

TrackAddict ("Interpolated CSV" share) writes a `#` comment metadata block
(RaceRender Data signature, vehicle, GPS/OBD devices, End/Split points with
coordinates, precise `# Lap` times) followed by one quoted header row and
unquoted data rows. Column units ride in the header itself: MPH vs Km/h,
PSI vs kPa, ft vs m, F vs C, plus `*OBD` source suffixes.
"""

from __future__ import annotations

import datetime
import math

from ..channels import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_POS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_COOLANT_TEMP,
    CH_ENGINE_RPM,
    CH_GEAR,
    CH_GEARBOX_TEMP,
    CH_GPS_ACCURACY,
    CH_GPS_ALTITUDE,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GROUND_SPEED,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_VEHICLE_SPEED,
)
from ..provenance import record_channel_provenance, record_dropped_duplicate
from .racechrono_csv_parser import _build_timing_laps

_MPH_TO_KMH = 1.60934
_PSI_TO_KPA = 6.89476
_FT_TO_M = 1.0 / 3.28084
_F_TO_C_FACTOR = 5.0 / 9.0
_F_TO_C_OFFSET = -32.0

# Header (lowercased) -> (canonical channel, unit, decimals, factor, offset).
# value = (raw + offset) * factor.
_EXACT_COLUMNS = {
    "latitude": (CH_GPS_LATITUDE, "deg", 7, 1.0, 0.0),
    "longitude": (CH_GPS_LONGITUDE, "deg", 7, 1.0, 0.0),
    "altitude (m)": (CH_GPS_ALTITUDE, "m", 1, 1.0, 0.0),
    "altitude (ft)": (CH_GPS_ALTITUDE, "m", 1, _FT_TO_M, 0.0),
    "speed (mph)": (CH_GROUND_SPEED, "km/h", 2, _MPH_TO_KMH, 0.0),
    "speed (km/h)": (CH_GROUND_SPEED, "km/h", 2, 1.0, 0.0),
    "heading": (CH_GPS_HEADING, "deg", 1, 1.0, 0.0),
    "accuracy (m)": (CH_GPS_ACCURACY, "m", 1, 1.0, 0.0),
    "accel x": (CH_CG_ACCEL_LON, "G", 2, 1.0, 0.0),
    "accel y": (CH_CG_ACCEL_LAT, "G", 2, 1.0, 0.0),
    "accel z": ("Accel Z", "G", 2, 1.0, 0.0),
    "x": (CH_CG_ACCEL_LON, "G", 2, 1.0, 0.0),
    "y": (CH_CG_ACCEL_LAT, "G", 2, 1.0, 0.0),
    "z": ("Accel Z", "G", 2, 1.0, 0.0),
    "throttle": (CH_THROTTLE_POS, "%", 2, 1.0, 0.0),
    "accelerator pedal (%) *obd": (CH_ACCELERATOR_POS, "%", 2, 1.0, 0.0),
    "rpm": (CH_ENGINE_RPM, "rpm", 2, 1.0, 0.0),
    "mph": (CH_GROUND_SPEED, "km/h", 2, _MPH_TO_KMH, 0.0),
    "kph": (CH_GROUND_SPEED, "km/h", 2, 1.0, 0.0),
    "speed (m/s)": (CH_GROUND_SPEED, "km/h", 2, 3.6, 0.0),
    "gear": (CH_GEAR, "", 0, 1.0, 0.0),
    "brake (calculated)": (CH_BRAKE_POS, "%", 0, 100.0, 0.0),
    "barometric pressure (psi)": ("Barometric Pressure", "kPa", 2, _PSI_TO_KPA, 0.0),
    "barometric pressure (kpa)": ("Barometric Pressure", "kPa", 2, 1.0, 0.0),
    "pressure altitude (ft)": ("Pressure Altitude", "m", 1, _FT_TO_M, 0.0),
    "pressure altitude (m)": ("Pressure Altitude", "m", 1, 1.0, 0.0),
    "engine speed (rpm) *obd": (CH_ENGINE_RPM, "rpm", 2, 1.0, 0.0),
    "vehicle speed (mph) *obd": (CH_VEHICLE_SPEED, "km/h", 2, _MPH_TO_KMH, 0.0),
    "vehicle speed (km/h) *obd": (CH_VEHICLE_SPEED, "km/h", 2, 1.0, 0.0),
    "throttle position (%) *obd": (CH_THROTTLE_POS, "%", 2, 1.0, 0.0),
    "engine coolant temp (f) *obd": (CH_COOLANT_TEMP, "C", 2, _F_TO_C_FACTOR, _F_TO_C_OFFSET),
    "engine coolant temp (c) *obd": (CH_COOLANT_TEMP, "C", 2, 1.0, 0.0),
    "transmission fluid temp (f) *obd": (CH_GEARBOX_TEMP, "C", 2, _F_TO_C_FACTOR, _F_TO_C_OFFSET),
    "transmission fluid temp (c) *obd": (CH_GEARBOX_TEMP, "C", 2, 1.0, 0.0),
    "steering angle (deg) *obd": (CH_STEERING_ANGLE, "deg", 1, 1.0, 0.0),
}

# Headers that carry no measurement (timing/flags/predictions/metadata).
_SKIPPED_COLUMNS = {
    "time", "utc time", "lap", "sector",
    "predicted lap time", "predicted vs best lap",
    "gps_update", "gps_delay", "obd_update",
}

# Metric twin wins when both unit variants of one quantity are present.
_METRIC_WINS = {
    "altitude (ft)": "altitude (m)",
    "barometric pressure (psi)": "barometric pressure (kpa)",
    "pressure altitude (ft)": "pressure altitude (m)",
}


def _split_csv_line(line):
    import csv

    try:
        return [p.strip().strip('"').strip("'") for p in next(csv.reader([line]))]
    except Exception:  # pragma: no cover - csv.reader rarely raises; split fallback
        return [p.strip().strip('"').strip("'") for p in line.split(",")]


def _parse_trap_coordinate(text):
    """Parse "lat, lon @ heading[deg]" into (lat, lon, heading) or None."""
    try:
        coords, _, heading = text.partition("@")
        lat_s, _, lon_s = coords.partition(",")
        hdg = heading.strip().split()[0]
        return float(lat_s), float(lon_s), float(hdg)
    except (ValueError, IndexError):
        return None


def _parse_comment_metadata(data_log, log_lines, data_header_idx):
    """Read `#` metadata: vehicle, End/Split trap points (laps come from data)."""
    for line in log_lines[:data_header_idx]:
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        content = stripped[1:].strip()
        key, _, value = content.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "vehicle" and value:
            data_log.metadata["vehicle_id"] = value
        elif key == "end point" and value:
            parsed = _parse_trap_coordinate(value)
            if parsed:
                lat, lon, hdg = parsed
                data_log.traps.append({
                    "lat": lat, "lon": lon, "heading_deg": hdg, "name": "Start/Finish",
                })
        elif key.startswith("split point") and value:
            parsed = _parse_trap_coordinate(value)
            if parsed:
                lat, lon, hdg = parsed
                label = key.replace("split point", "").strip() or "?"
                data_log.traps.append({
                    "lat": lat, "lon": lon, "heading_deg": hdg, "name": f"Split {label}",
                })


def _find_data_header(log_lines):
    """Return the index of the quoted Time/... header row, or -1."""
    for i, line in enumerate(log_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = _split_csv_line(stripped)
        if len(parts) > 2 and parts[0].lower() == "time":
            return i
    print("ERROR: Could not find 'Time' column in TrackAddict CSV log.")
    return -1


def _generic_obd_column(raw_name):
    """Map an unknown `Name (unit) [*OBD]` column via its unit suffix."""
    name = raw_name[:-len(" *OBD")].strip() if raw_name.lower().endswith(" *obd") else raw_name
    base, _, unit = name.rpartition("(")
    unit = unit.rstrip(")").strip().lower()
    base = base.strip() or raw_name
    mapping = {
        "mph": ("km/h", _MPH_TO_KMH, 0.0),
        "km/h": ("km/h", 1.0, 0.0),
        "kmh": ("km/h", 1.0, 0.0),
        "m/s": ("km/h", 3.6, 0.0),
        "f": ("C", _F_TO_C_FACTOR, _F_TO_C_OFFSET),
        "c": ("C", 1.0, 0.0),
        "psi": ("kPa", _PSI_TO_KPA, 0.0),
        "kpa": ("kPa", 1.0, 0.0),
        "bar": ("bar", 1.0, 0.0),
        "ft": ("m", _FT_TO_M, 0.0),
        "m": ("m", 1.0, 0.0),
        "rpm": ("rpm", 1.0, 0.0),
        "%": ("%", 1.0, 0.0),
        "deg": ("deg", 1.0, 0.0),
        "g": ("G", 1.0, 0.0),
    }
    if unit in mapping:
        out_unit, factor, offset = mapping[unit]
        return (base, out_unit, 2, factor, offset)
    return (raw_name, "", 2, 1.0, 0.0)


def _build_columns(data_log, headers):
    """Create channels; return (col_specs, lap_idx, time_idx, utc_idx).

    col_specs: (values index, Channel, factor, offset).
    """
    lowered = [h.strip().lower() for h in headers]
    present = set(lowered)
    skip = set(_SKIPPED_COLUMNS)
    for imperial, metric in _METRIC_WINS.items():
        if metric in present:
            skip.add(imperial)

    col_specs = []
    lap_idx = time_idx = utc_idx = -1
    winners = {}
    for i, raw in enumerate(headers):
        key = raw.strip().lower()
        if key == "time":
            time_idx = i
            continue
        if key == "utc time":
            utc_idx = i
            continue
        if key == "lap":
            lap_idx = i
            continue
        if key in skip:
            continue
        spec = _EXACT_COLUMNS.get(key)
        if spec is None:
            out_name, out_unit, dec, factor, offset = _generic_obd_column(raw.strip())
        else:
            out_name, out_unit, dec, factor, offset = spec
        if out_name not in data_log.channels:
            data_log.add_channel(out_name, out_unit, float, dec)
            winners[out_name] = raw.strip()
        else:
            record_dropped_duplicate(data_log, out_name, winners[out_name], raw.strip())
            continue
        col_specs.append((i, data_log.channels[out_name], factor, offset))
    pedal_raw = winners.get(CH_ACCELERATOR_POS)
    if pedal_raw is not None and pedal_raw.strip().lower() == "accelerator pedal (%) *obd":
        # This format exports accelerator-pedal and throttle-position as
        # separate OBD columns, so the pedal column is the accelerator.
        record_channel_provenance(
            data_log, CH_ACCELERATOR_POS, pedal_raw,
            power_semantics="accelerator_pedal",
            semantics_basis=(
                "TrackAddict exports accelerator-pedal and throttle-position "
                "as separate OBD columns; the pedal column is the accelerator"
            ),
        )
    return col_specs, lap_idx, time_idx, utc_idx


def _parse_data_rows(data_log, log_lines, header_idx, col_specs, lap_idx, time_idx, utc_idx):
    """Single pass over data rows; return per-lap timing metadata."""
    chan_buffers = [([], []) for _ in col_specs]
    laps_timing = {}
    first_utc = None
    for line in log_lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        values = stripped.split(",")
        if time_idx >= len(values):
            continue
        try:
            t = float(values[time_idx].strip().strip('"').strip("'"))
        except ValueError:
            continue
        lap_val = ""
        if 0 <= lap_idx < len(values):
            lap_val = values[lap_idx].strip().strip('"').strip("'")
        if utc_idx >= 0 and first_utc is None and utc_idx < len(values):
            try:
                first_utc = float(values[utc_idx].strip().strip('"').strip("'"))
            except ValueError:
                first_utc = False
        if lap_val:
            info = laps_timing.get(lap_val)
            if info is None:
                laps_timing[lap_val] = {"start_time": t, "end_time": t, "messages_count": 1}
            else:
                info["end_time"] = t
                info["messages_count"] += 1
        for k, (val_idx, _, factor, offset) in enumerate(col_specs):
            if val_idx >= len(values):
                continue
            s = values[val_idx].strip().strip('"').strip("'")
            if not s:
                continue
            try:
                val = float(s)
            except ValueError:
                continue
            if factor != 1.0 or offset:
                val = (val + offset) * factor
            ts_buf, val_buf = chan_buffers[k]
            ts_buf.append(t)
            val_buf.append(val)
    for k, (_, channel, _, _) in enumerate(col_specs):
        ts_buf, val_buf = chan_buffers[k]
        if ts_buf:
            channel.set_samples(ts_buf, val_buf)
    if first_utc:
        data_log.datetime = datetime.datetime.fromtimestamp(
            first_utc, tz=datetime.timezone.utc,
        ).replace(tzinfo=None)
    return laps_timing


def parse_trackaddict_log(data_log, log_lines, target_lap=None):
    """Creates channels populated with messages from a TrackAddict CSV log file.

    Lap timing comes from the Lap column; precise `# Lap` tags are informational
    (data-derived spans at 10-20 Hz are exact to a sample). End/Split points
    from the `#` metadata become GPS traps for beacon detection.
    """
    data_log.clear()
    data_log.laps_info = {}
    data_log.datetime = None

    if not log_lines:
        return
    header_idx = _find_data_header(log_lines)
    if header_idx == -1:
        return

    _parse_comment_metadata(data_log, log_lines, header_idx)

    headers = _split_csv_line(log_lines[header_idx].strip())
    col_specs, lap_idx, time_idx, utc_idx = _build_columns(data_log, headers)
    if time_idx == -1:
        return

    target_lap_str = None
    if target_lap is not None and str(target_lap).lower() != "all":
        target_lap_str = str(target_lap).strip()

    laps_timing = _parse_data_rows(data_log, log_lines, header_idx, col_specs,
                                   lap_idx, time_idx, utc_idx)
    if target_lap_str is not None:
        if target_lap_str in laps_timing:
            t_lo = laps_timing[target_lap_str]["start_time"]
            t_hi = laps_timing[target_lap_str]["end_time"]
            for ch in data_log.channels.values():
                ch.messages = [m for m in ch.messages if t_lo <= m.timestamp <= t_hi]
        else:
            print(f"WARNING: Lap '{target_lap}' not found in TrackAddict file; keeping all laps")

    from ..derived import derive_yaw_rate_from_gps_heading
    derive_yaw_rate_from_gps_heading(data_log)

    if laps_timing:
        lap_items, fastest_lap, fastest_dur = _build_timing_laps(data_log, laps_timing)
        for lap in lap_items:
            if lap["lap_label"] == "0":
                lap["type"] = "Out Lap"
                lap["lap_label"] = "Out Lap"
        timed = [lap for lap in lap_items if lap["type"] == "Timed"]
        if timed:
            best = min(timed, key=lambda lap: lap["duration"])
            fastest_lap, fastest_dur = best["lap_num"], best["duration"]
        data_log.laps_info = {
            "laps": lap_items,
            "total_laps": len(lap_items),
            "fastest_lap": fastest_lap if fastest_lap is not None else 1,
            "fastest_time": fastest_dur if fastest_dur != math.inf else 0.0,
        }
    empty_channels = [name for name, ch in data_log.channels.items() if not ch.messages]
    for name in empty_channels:
        del data_log.channels[name]
