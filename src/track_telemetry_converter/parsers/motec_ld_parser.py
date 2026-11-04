"""Parser for real-world MoTeC .ld logger files.

Assetto Corsa ACTI simulator logs are also ``.ld`` but carry simulator-frame
``Car Coord X/Y`` channels without GPS; those stay on the ACTI path
(:mod:`acti_parser`, selected by :func:`parsers._sniff_ld_for_acti`). This
parser handles every other ``.ld``: genuine logger files and the converter's
own ``.ld`` outputs, both of which record GPS (``GPS Latitude`` /
``Latitude``) and canonical MoTeC channel names.

Each LD channel keeps its own clock (``arange(n) / freq``), matching the
ACTI parser. No cross-channel rebasing is performed here; resampling stays
in the conversion flow. Discrete lap channels that were linearly
interpolated on export (e.g. ``Lap #`` drifting ``4.00006...``) are rounded
back before transition detection, and near-integer conformance is enforced.
"""

from __future__ import annotations

import datetime
import re

import numpy as np

from .._vendor.ldparser import ldData
from ..channels import (
    CH_ACCELERATOR_POS,
    CH_AMBIENT_TEMP,
    CH_BRAKE_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_CG_ACCEL_VERT,
    CH_CLUTCH_POS,
    CH_COOLANT_TEMP,
    CH_DEVICE_BATTERY,
    CH_ENGINE_OIL_PRESS,
    CH_ENGINE_OIL_TEMP,
    CH_ENGINE_RPM,
    CH_FUEL_LEVEL,
    CH_G_COMBINED,
    CH_GEAR,
    CH_GPS_ACCURACY,
    CH_GPS_ALTITUDE,
    CH_GPS_FIX,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GPS_SATS,
    CH_GROUND_SPEED,
    CH_INTAKE_TEMP,
    CH_LAP_NUMBER,
    CH_LEAN_ANGLE,
    CH_MANIFOLD_PRESS,
    CH_RUNNING_TIME,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_TIRE_PRESS_FL,
    CH_TIRE_PRESS_FR,
    CH_TIRE_PRESS_RL,
    CH_TIRE_PRESS_RR,
    CH_VEHICLE_SPEED,
    CH_WHEEL_SPEED_FL,
    CH_WHEEL_SPEED_FR,
    CH_WHEEL_SPEED_RL,
    CH_WHEEL_SPEED_RR,
    CH_YAW_RATE,
    CHANNEL_ALIASES,
)

#: Canonical names indexed case-insensitively (identity matches).
_CANONICAL_BY_LOWER = {}
for _name in (
    CH_GROUND_SPEED, CH_VEHICLE_SPEED, CH_CG_ACCEL_LAT, CH_CG_ACCEL_LON,
    CH_CG_ACCEL_VERT, CH_GPS_LATITUDE, CH_GPS_LONGITUDE, CH_GPS_HEADING,
    CH_GPS_ALTITUDE, CH_GPS_SATS, CH_GPS_ACCURACY, CH_GPS_FIX,
    CH_LAP_NUMBER, CH_RUNNING_TIME, CH_DEVICE_BATTERY, CH_THROTTLE_POS,
    CH_ACCELERATOR_POS, CH_BRAKE_PRESS, CH_BRAKE_POS, CH_CLUTCH_POS,
    CH_ENGINE_RPM, CH_STEERING_ANGLE, CH_COOLANT_TEMP, CH_ENGINE_OIL_TEMP,
    CH_ENGINE_OIL_PRESS, CH_GEAR, CH_YAW_RATE, CH_LEAN_ANGLE,
    CH_INTAKE_TEMP, CH_MANIFOLD_PRESS, CH_AMBIENT_TEMP, CH_G_COMBINED,
    CH_WHEEL_SPEED_FL, CH_WHEEL_SPEED_FR, CH_WHEEL_SPEED_RL,
    CH_WHEEL_SPEED_RR, CH_FUEL_LEVEL,
):
    _CANONICAL_BY_LOWER[_name.casefold()] = _name

_ALIAS_BY_LOWER = {key.casefold(): value for key, value in CHANNEL_ALIASES.items()}

#: Trailing " *source" tags on RaceChrono-export LD channels.
_SUFFIX_RE = re.compile(r"\s+\*(canbus|calc|acc|gyro|hrm|obd)\s*$", re.IGNORECASE)
#: Trailing " (unit)" hints, e.g. "Speed (m/s) *canbus".
_PAREN_UNIT_RE = re.compile(r"\s*\(([^()]*)\)\s*$")

#: Stripped base name (casefolded) -> (canonical, unit, decimals).
_STRIPPED_ALIASES = {
    "speed": (CH_GROUND_SPEED, "km/h", 2),
    "gps speed": (CH_GROUND_SPEED, "km/h", 2),
    "vehicle speed": (CH_VEHICLE_SPEED, "km/h", 2),
    "rpm": (CH_ENGINE_RPM, "rpm", 0),
    "throttle position": (CH_THROTTLE_POS, "%", 2),
    "accelerator position": (CH_ACCELERATOR_POS, "%", 2),
    "brake position": (CH_BRAKE_POS, "%", 2),
    "brake pressure": (CH_BRAKE_PRESS, "kPa", 2),
    "steering angle": (CH_STEERING_ANGLE, "deg", 2),
    "coolant temperature": (CH_COOLANT_TEMP, "C", 2),
    "air temperature": (CH_AMBIENT_TEMP, "C", 2),
    "ambient temperature": (CH_AMBIENT_TEMP, "C", 2),
    "engine oil temp": (CH_ENGINE_OIL_TEMP, "C", 2),
    "engine oil temperature": (CH_ENGINE_OIL_TEMP, "C", 2),
    "engine oil press": (CH_ENGINE_OIL_PRESS, "kPa", 2),
    "engine oil pressure": (CH_ENGINE_OIL_PRESS, "kPa", 2),
    "gear": (CH_GEAR, "", 0),
    "yaw rate": (CH_YAW_RATE, "deg/s", 3),
    "lateral acceleration": (CH_CG_ACCEL_LAT, "G", 4),
    "longitudinal acceleration": (CH_CG_ACCEL_LON, "G", 4),
    "vertical acceleration": (CH_CG_ACCEL_VERT, "G", 4),
    "wheel speed fl": (CH_WHEEL_SPEED_FL, "km/h", 2),
    "wheel speed fr": (CH_WHEEL_SPEED_FR, "km/h", 2),
    "wheel speed rl": (CH_WHEEL_SPEED_RL, "km/h", 2),
    "wheel speed rr": (CH_WHEEL_SPEED_RR, "km/h", 2),
    "tyre pressure fl": (None, "kPa", 2),
    "tyre pressure fr": (None, "kPa", 2),
    "tyre pressure rl": (None, "kPa", 2),
    "tyre pressure rr": (None, "kPa", 2),
    "tire pressure fl": (None, "kPa", 2),
    "tire pressure fr": (None, "kPa", 2),
    "tire pressure rl": (None, "kPa", 2),
    "tire pressure rr": (None, "kPa", 2),
    "latitude": (CH_GPS_LATITUDE, "deg", 7),
    "longitude": (CH_GPS_LONGITUDE, "deg", 7),
    "bearing": (CH_GPS_HEADING, "deg", 2),
    "altitude": (CH_GPS_ALTITUDE, "m", 2),
    "lean angle": (CH_LEAN_ANGLE, "deg", 2),
    "combined acceleration": (CH_G_COMBINED, "G", 4),
    "accuracy": (CH_GPS_ACCURACY, "m", 2),
    "device battery level": (CH_DEVICE_BATTERY, "%", 2),
    "satellites": (CH_GPS_SATS, "", 0),
    "gps fix": (CH_GPS_FIX, "", 0),
    "lap #": (CH_LAP_NUMBER, "", 0),
    "lap number": (CH_LAP_NUMBER, "", 0),
    "fuel level": (CH_FUEL_LEVEL, "l", 2),
    "manifold press": (CH_MANIFOLD_PRESS, "kPa", 2),
    "manifold pressure": (CH_MANIFOLD_PRESS, "kPa", 2),
    "intake temp": (CH_INTAKE_TEMP, "C", 2),
    "intake temperature": (CH_INTAKE_TEMP, "C", 2),
    "clutch position": (CH_CLUTCH_POS, "%", 2),
}

_TYRE_TO_CANONICAL = {
    "tyre pressure fl": CH_TIRE_PRESS_FL,
    "tyre pressure fr": CH_TIRE_PRESS_FR,
    "tyre pressure rl": CH_TIRE_PRESS_RL,
    "tyre pressure rr": CH_TIRE_PRESS_RR,
    "tire pressure fl": CH_TIRE_PRESS_FL,
    "tire pressure fr": CH_TIRE_PRESS_FR,
    "tire pressure rl": CH_TIRE_PRESS_RL,
    "tire pressure rr": CH_TIRE_PRESS_RR,
}

#: Segments longer than this are pit/red-flag stints, not laps. Ten minutes
#: clears the longest plausible circuit lap (Nurburgring Nordschleife) while
#: keeping multi-hour logger files from poisoning the timed-lap set.
_MAX_LAP_SEC = 600.0

#: Channels consumed as clocks/structure, never exported as telemetry.
_CLOCK_CHANNELS = {"running time", "elapsed time"}
_SKIP_CHANNELS = {"session fragment #"}


def _split_ld_name(raw_name: str) -> tuple[str, str]:
    """Return (base name, unit hint) with source suffixes removed."""
    base = _SUFFIX_RE.sub("", raw_name).strip()
    unit_hint = ""
    match = _PAREN_UNIT_RE.search(base)
    if match:
        unit_hint = match.group(1).strip()
        base = _PAREN_UNIT_RE.sub("", base).strip()
    return base, unit_hint


def _resolve_alias(raw_name: str, unit: str) -> tuple[str, str, int] | None:
    """Map an LD channel name to (canonical, unit, decimals)."""
    if raw_name in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[raw_name]
    if raw_name.casefold() in _CANONICAL_BY_LOWER:
        canonical = _CANONICAL_BY_LOWER[raw_name.casefold()]
        alias = _ALIAS_BY_LOWER.get(raw_name.casefold())
        if alias is not None and alias[0] == canonical:
            return alias
        decimals = 7 if canonical in (CH_GPS_LATITUDE, CH_GPS_LONGITUDE) else 2
        if canonical in (CH_LAP_NUMBER, CH_GEAR, CH_GPS_SATS, CH_GPS_FIX):
            decimals = 0
        if canonical == CH_YAW_RATE:
            decimals = 3
        return (canonical, unit or alias[1] if alias else unit, decimals)
    base, _ = _split_ld_name(raw_name)
    lowered = base.casefold()
    if lowered in _ALIAS_BY_LOWER:
        return _ALIAS_BY_LOWER[lowered]
    if lowered in _CANONICAL_BY_LOWER:
        return _resolve_alias(_CANONICAL_BY_LOWER[lowered], unit)
    if lowered in _STRIPPED_ALIASES:
        canonical, target_unit, decimals = _STRIPPED_ALIASES[lowered]
        if canonical is None:
            canonical = _TYRE_TO_CANONICAL[lowered]
        return (canonical, target_unit, decimals)
    return None


def _convert_values(values: np.ndarray, source_unit: str, target_unit: str,
                    raw_name: str = "channel") -> np.ndarray:
    """Convert an LD channel array into the alias unit."""
    result = np.asarray(values, dtype=np.float64)
    source = (source_unit or "").strip().lower().replace(".c", "c").replace("°c", "c")
    target = (target_unit or "").strip().lower()
    if not source or source == target:
        return result
    if source in {"m/s", "mps"} and target == "km/h":
        return result * 3.6
    if source == "mph" and target == "km/h":
        return result * 1.609344
    if source in {"bar", "kpa", "pa"} and target in {"bar", "kpa", "pa"}:
        scales = {"bar": 100000.0, "kpa": 1000.0, "pa": 1.0}
        return result * (scales[source] / scales[target])
    if source == "f" and target == "c":
        return (result - 32.0) * (5.0 / 9.0)
    print(f"WARNING: No unit conversion for LD channel '{raw_name}' "
          f"from '{source_unit}' to '{target_unit}'; keeping raw values")
    return result


def _channel_samples(ld, name: str) -> tuple[np.ndarray, float] | tuple[None, float]:
    for channel in ld.channs:
        if channel.name == name:
            try:
                values = np.asarray(channel.data, dtype=float)
            except (TypeError, ValueError):
                return None, 20.0
            if values.size == 0:
                return None, 20.0
            freq = float(channel.freq) if channel.freq and channel.freq > 0 else 20.0
            return values, freq
    return None, 20.0


def _courtesy_heading(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Course heading (deg, 0=N) from GPS when the file records no bearing."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(np.median(lat[np.isfinite(lat)])) if np.any(np.isfinite(lat)) else 0.0
    x = np.radians(lon - np.median(lon[np.isfinite(lon)])) * 6371000.0 * np.cos(np.radians(lat0))
    y = np.radians(lat - np.median(lat[np.isfinite(lat)])) * 6371000.0
    dx = np.gradient(x)
    dy = np.gradient(y)
    step = np.hypot(dx, dy)
    raw = np.degrees(np.arctan2(dx, dy)) % 360.0
    moving = np.isfinite(step) & (step > 0.01)
    if not np.any(moving):
        return np.zeros_like(raw)
    heading = np.copy(raw)
    last = 0.0
    for i in range(len(heading)):
        if moving[i] and np.isfinite(heading[i]):
            last = heading[i]
        else:
            heading[i] = last
    first = int(np.argmax(moving))
    heading[:first] = heading[first]
    return np.round(heading, 3)


def _lap_groups_from_channel(values: np.ndarray) -> list[tuple[int, int, int]] | None:
    """Group sample indices by recorded lap number (rounded, conformance-gated)."""
    rounded = np.rint(values)
    if values.size < 100:
        return None
    if np.count_nonzero(np.isfinite(values)) < values.size:
        return None
    if float(np.mean(np.abs(values - rounded) <= 0.01)) < 0.99:
        return None
    edges = [0]
    for i in range(1, len(rounded)):
        if rounded[i] != rounded[i - 1]:
            edges.append(i)
    edges.append(len(rounded))
    if len(edges) <= 2:
        return None
    return [(int(rounded[s]), s, e) for s, e in zip(edges[:-1], edges[1:], strict=False)]


def _build_laps_from_groups(groups: list[tuple[int, int, int]], freq: float,
                            min_lap_sec: float) -> dict:
    """laps_info from lap-number groups; first group is always the Out Lap."""
    laps = []
    beacons = []
    for k, (lap_num, s, e) in enumerate(groups):
        dur = (e - s) / freq
        start_s = s / freq
        end_s = e / freq
        if k == 0:
            kind = "Out Lap"
        elif k == len(groups) - 1:
            kind = "In Lap" if min_lap_sec < dur <= _MAX_LAP_SEC else "Partial"
        else:
            kind = "Timed" if min_lap_sec <= dur <= _MAX_LAP_SEC else "Partial"
        laps.append({
            "lap_num": lap_num,
            "lap_label": str(lap_num),
            "start_time": start_s,
            "end_time": end_s,
            "duration": dur,
            "type": kind,
            "stint": 0,
        })
        if k < len(groups) - 1:
            beacons.append((end_s, "Start/Finish"))
    timed = [lap for lap in laps if lap["type"] == "Timed"]
    return {
        "laps": laps,
        "total_laps": len(laps),
        "fastest_lap": min(timed, key=lambda lap: lap["duration"])["lap_num"] if timed else laps[0]["lap_num"],
        "fastest_time": min((lap["duration"] for lap in timed), default=0.0),
        "beacons": sorted(beacons, key=lambda item: item[0]),
    }


def _build_laps_from_beacons(beacons: list[tuple[float, str]], duration: float,
                             min_lap_sec: float) -> dict:
    """laps_info from GPS start/finish crossings; edge segments stay untimed."""
    bounds = [0.0, *[t for t, _ in beacons], duration]
    laps = []
    for k in range(len(bounds) - 1):
        start_s, end_s = bounds[k], bounds[k + 1]
        dur = end_s - start_s
        if k == 0:
            kind = "Out Lap"
        elif k == len(bounds) - 2:
            kind = "In Lap" if min_lap_sec < dur <= _MAX_LAP_SEC else "Partial"
        else:
            kind = "Timed" if min_lap_sec <= dur <= _MAX_LAP_SEC else "Partial"
        laps.append({
            "lap_num": k + 1,
            "lap_label": str(k + 1),
            "start_time": start_s,
            "end_time": end_s,
            "duration": dur,
            "type": kind,
            "stint": 0,
        })
    timed = [lap for lap in laps if lap["type"] == "Timed"]
    return {
        "laps": laps,
        "total_laps": len(laps),
        "fastest_lap": min(timed, key=lambda lap: lap["duration"])["lap_num"] if timed else 1,
        "fastest_time": min((lap["duration"] for lap in timed), default=0.0),
        "beacons": sorted(beacons, key=lambda item: item[0]),
    }


def _filter_target_lap(data_log, laps_info: dict, target_lap) -> dict:
    """Keep one lap, rebased to zero. Returns the laps_info to store."""
    if target_lap is None or str(target_lap).lower() == "all":
        return laps_info
    try:
        lap_no = int(target_lap)
        recs = [lap for lap in laps_info["laps"] if lap["lap_num"] == lap_no]
    except (ValueError, IndexError):
        recs = []
    if not recs:
        print(f"WARNING: Lap '{target_lap}' not found in MoTeC LD file; keeping all laps")
        return laps_info
    t_lo, t_hi = recs[0]["start_time"], recs[0]["end_time"]
    for channel in data_log.channels.values():
        keep_t, keep_v = [], []
        for t, v in zip(channel.timestamps, channel.values, strict=False):
            if t_lo <= t <= t_hi:
                keep_t.append(t - t_lo)
                keep_v.append(v)
        channel.set_samples(np.asarray(keep_t), np.asarray(keep_v))
    duration = t_hi - t_lo
    return {
        "laps": [{
            "lap_num": recs[0]["lap_num"],
            "lap_label": recs[0].get("lap_label", str(recs[0]["lap_num"])),
            "start_time": 0.0,
            "end_time": duration,
            "duration": duration,
            "type": recs[0].get("type", "Timed"),
            "stint": recs[0].get("stint", 0),
        }],
        "total_laps": 1,
        "fastest_lap": recs[0]["lap_num"],
        "fastest_time": duration,
        "beacons": [(0.0, "Start/Finish"), (duration, "Start/Finish")],
    }


def parse_motec_ld_log(data_log, ld_file_path, target_lap=None, min_lap_sec=15.0):
    """Populate channels from a real-world MoTeC .ld logger file.

    ACTI simulator logs (``Car Coord X/Y`` without GPS) are rejected: they
    need a track profile via :func:`acti_parser.parse_acti_log`.
    """
    data_log.clear()
    data_log.laps_info = {}

    ld = ldData.fromfile(ld_file_path)
    names = [getattr(channel, "name", "") for channel in ld.channs]
    if "Car Coord X" in names and "GPS Latitude" not in names and "Latitude" not in names:
        raise ValueError(
            f"'{ld_file_path}' looks like an ACTI simulator log "
            "(Car Coord X/Y without GPS); parse it with the ACTI parser "
            "and a --track profile instead",
        )

    _apply_header_metadata(data_log, ld)
    provenance, _, native_rates, mapped = _populate_channels(data_log, ld)
    lat_t, _ = _series(data_log, CH_GPS_LATITUDE)
    lon_t, _ = _series(data_log, CH_GPS_LONGITUDE)
    has_gps = lat_t is not None and lon_t is not None
    if has_gps:
        _synthesize_heading(data_log, provenance, native_rates)
    _record_source_labels(data_log, provenance, has_gps)
    _record_session_fragments(data_log, ld)

    duration = data_log.duration()
    laps_info = _laps_from_recorded_numbers(data_log, ld, provenance, mapped, min_lap_sec)
    if laps_info is not None:
        lap_timing = "Recorded MoTeC LD lap-number transitions on the lap channel clock"
    else:
        lap_timing = ""
        if has_gps:
            try:
                beacons = data_log.detect_beacons()
            except (ValueError, TypeError):
                beacons = []
            if beacons:
                laps_info = _build_laps_from_beacons(beacons, duration, min_lap_sec)
                lap_timing = "GPS start/finish beacons on the preserved LD channel clock"
    if laps_info is None:
        laps_info = {
            "laps": [{"lap_num": 1, "lap_label": "1", "start_time": 0.0,
                      "end_time": duration, "duration": duration,
                      "type": "Out Lap", "stint": 0}],
            "total_laps": 1,
            "fastest_lap": 1,
            "fastest_time": 0.0,
            "beacons": [],
        }
        lap_timing = "No recorded lap numbers or GPS beacons; single Out Lap over the file"
        _ensure_lap_number_channel(data_log, laps_info, fill=1.0)
    else:
        _ensure_lap_number_channel(data_log, laps_info, fill=None)

    data_log.metadata["lap_timing_semantics"] = lap_timing
    data_log.laps_info = _filter_target_lap(data_log, laps_info, target_lap)

def _apply_header_metadata(data_log, ld) -> None:
    """Copy driver/vehicle/venue/event/datetime from the LD header."""
    head = ld.head
    data_log.metadata["driver"] = getattr(head, "driver", "")
    data_log.metadata["vehicle_id"] = getattr(head, "vehicleid", "")
    data_log.metadata["venue_name"] = getattr(head, "venue", "")
    event = getattr(head, "event", None)
    if event is not None:
        if getattr(event, "name", ""):
            data_log.metadata["event_name"] = event.name
        if getattr(event, "session", ""):
            data_log.metadata["event_session"] = event.session
    if getattr(head, "short_comment", ""):
        data_log.metadata["short_comment"] = head.short_comment
    head_dt = getattr(head, "datetime", None)
    if isinstance(head_dt, datetime.datetime):
        data_log.datetime = head_dt
        if head_dt.tzinfo is None:
            data_log.datetime_utc = head_dt.replace(tzinfo=datetime.timezone.utc)
        else:
            data_log.datetime_utc = head_dt.astimezone(datetime.timezone.utc)


def _populate_channels(data_log, ld) -> tuple[dict, dict, dict, set]:
    """Map LD channels to canonical names; preserve the rest verbatim."""
    provenance: dict[str, dict] = {}
    inventory: dict[str, dict] = {}
    native_rates: dict[str, float] = {}
    mapped: set[str] = set()

    for channel in ld.channs:
        raw_name = getattr(channel, "name", "")
        if not raw_name:
            continue
        lowered = raw_name.casefold()
        base_lower, _ = _split_ld_name(raw_name)
        base_lower = base_lower.casefold()
        if lowered in _SKIP_CHANNELS:
            continue
        if lowered in _CLOCK_CHANNELS or base_lower in _CLOCK_CHANNELS:
            continue
        try:
            values = np.asarray(channel.data, dtype=float)
        except (TypeError, ValueError):
            print(f"WARNING: Skipping unreadable LD channel '{raw_name}'")
            continue
        if values.size == 0:
            continue
        freq = float(channel.freq) if channel.freq and channel.freq > 0 else 20.0
        try:
            kind = np.dtype(channel.dtype).kind if getattr(channel, "dtype", None) is not None else "f"
        except TypeError:
            kind = "f"
        data_type = float if kind in ("f", "c") else int
        try:
            decimals = int(getattr(channel, "dec", 0) or 0)
        except (TypeError, ValueError):
            decimals = 0
        raw_unit = getattr(channel, "unit", "") or ""
        _, hint = _split_ld_name(raw_name)
        if (not raw_unit or raw_unit.strip().lower() == "m") and hint:
            # Generic "m" stamps (e.g. ldData.frompd exports) carry no rate
            # information; an explicit "(m/s)"-style hint in the name wins.
            # Channels without a hint (altitude, distance) keep "m".
            if hint.strip().lower() != raw_unit.strip().lower():
                raw_unit = hint

        alias = _resolve_alias(raw_name, raw_unit)
        timestamps = np.arange(len(values)) / freq
        out_name: str | None = None
        if alias is not None:
            out_name, target_unit, alias_dec = alias
            if out_name in mapped:
                out_name = None
            else:
                mapped.add(out_name)
                values = _convert_values(values, raw_unit, target_unit, raw_name)
                data_log.add_channel(out_name, target_unit, data_type, alias_dec)
                data_log.channels[out_name].set_samples(timestamps, values)
                provenance[out_name] = {"raw_name": raw_name, "source_units": raw_unit,
                                        "output_units": target_unit,
                                        "series_sample_rate_hz": freq}
                native_rates[out_name] = freq
        if alias is None or out_name is None:
            if raw_name in data_log.channels:
                continue
            data_log.add_channel(raw_name, raw_unit, data_type, decimals)
            data_log.channels[raw_name].set_samples(timestamps, values)
        inventory[raw_name] = {"samples": int(values.size), "rate_hz": freq,
                               "units": raw_unit}

    data_log.metadata["motec_ld_channel_inventory"] = inventory
    data_log.metadata["channel_provenance"] = provenance
    data_log.metadata["native_sample_rate_hz"] = native_rates
    data_log.metadata["time_axis"] = "LD per-channel clocks (arange/freq); no cross-channel rebasing"
    return provenance, inventory, native_rates, mapped


def _series(data_log, canonical: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    channel = data_log.channels.get(canonical)
    if channel is None or len(channel.timestamps) == 0:
        return None, None
    return channel.timestamps, channel.values


def _synthesize_heading(data_log, provenance: dict, native_rates: dict) -> bool:
    """Derive course heading from GPS when the file records no bearing."""
    if CH_GPS_HEADING in data_log.channels:
        return False
    lat_t, lat_v = _series(data_log, CH_GPS_LATITUDE)
    lon_t, lon_v = _series(data_log, CH_GPS_LONGITUDE)
    if lat_t is None or lon_t is None:
        return False
    try:
        heading = _courtesy_heading(np.asarray(lat_v), np.asarray(lon_v))
    except (TypeError, ValueError):
        print("WARNING: Could not derive GPS heading from LD coordinates")
        return False
    data_log.add_channel(CH_GPS_HEADING, "deg", float, 3)
    data_log.channels[CH_GPS_HEADING].set_samples(np.asarray(lat_t), heading)
    provenance[CH_GPS_HEADING] = {"raw_name": "computed_from_gps_course",
                                  "source_units": "deg", "output_units": "deg",
                                  "series_sample_rate_hz": native_rates.get(CH_GPS_LATITUDE)}
    data_log.metadata["gps_heading_synthetic"] = True
    return True


def _record_source_labels(data_log, provenance: dict, has_gps: bool) -> None:
    # Recorded channels are real logger sensors, but wiring is unverified,
    # so sources stay descriptive, never "can"/"imu".
    yaw_prov = provenance.get(CH_YAW_RATE)
    if yaw_prov and yaw_prov["raw_name"] != "computed_from_gps_course":
        data_log.metadata["yaw_rate_source"] = f"motec_ld_recorded_{yaw_prov['raw_name']}"
    elif has_gps:
        data_log.metadata["yaw_rate_source"] = "gps_heading_derivative"
    for canonical, key in ((CH_CG_ACCEL_LAT, "lateral_accel_source"),
                           (CH_CG_ACCEL_LON, "longitudinal_accel_source")):
        prov = provenance.get(canonical)
        if prov:
            data_log.metadata[key] = f"motec_ld_recorded_{prov['raw_name']}"
    if has_gps:
        data_log.metadata["gps_source"] = "motec_ld_recorded"


def _drop_drifting_lap_channel(data_log, provenance: dict, mapped: set, raw_name: str) -> None:
    # The lap channel failed near-integer conformance (typically a
    # resample-interpolated export): its drifting values must not survive as
    # a Lap Number channel. The beacon path below rebuilds clean per-lap ids.
    data_log.channels.pop(CH_LAP_NUMBER, None)
    provenance.pop(CH_LAP_NUMBER, None)
    data_log.metadata["native_sample_rate_hz"].pop(CH_LAP_NUMBER, None)
    mapped.discard(CH_LAP_NUMBER)
    print(f"WARNING: LD lap channel '{raw_name}' is not near-integer; "
          "lap numbers will come from GPS beacons instead")


def _record_session_fragments(data_log, ld) -> None:
    frag_values, _ = _channel_samples(ld, "Session fragment #")
    if frag_values is None:
        return
    fragments = sorted({int(v) for v in np.rint(frag_values) if np.isfinite(v)})
    data_log.metadata["motec_ld_session_fragments"] = fragments
    if len(fragments) > 1:
        print(f"WARNING: LD file spans {len(fragments)} session fragments; "
              "analyzed as one session, over-long segments stay Partial")


def _laps_from_recorded_numbers(data_log, ld, provenance: dict, mapped: set,
                                min_lap_sec: float) -> dict | None:
    """laps_info from a conforming lap-number channel; None when unusable."""
    for candidate in ("Lap Number", "Lap #"):
        values, freq = _channel_samples(ld, candidate)
        if values is None:
            continue
        groups = _lap_groups_from_channel(np.asarray(values, dtype=float))
        if groups is not None:
            return _build_laps_from_groups(groups, freq, min_lap_sec)
        _drop_drifting_lap_channel(data_log, provenance, mapped, candidate)
        return None
    return None


def _ensure_lap_number_channel(data_log, laps_info: dict, fill: float | None) -> None:
    """(Re)build a clean Lap Number channel on the speed/GPS clock."""
    if CH_LAP_NUMBER in data_log.channels and fill is None:
        return
    clock_t, _ = _series(data_log, CH_GROUND_SPEED)
    if clock_t is None:
        clock_t, _ = _series(data_log, CH_GPS_LATITUDE)
    if clock_t is None:
        return
    data_log.add_channel(CH_LAP_NUMBER, "", int, 0)
    if fill is not None:
        data_log.channels[CH_LAP_NUMBER].set_samples(
            np.asarray(clock_t), np.full(len(clock_t), fill))
        return
    lap_ids = np.zeros(len(clock_t))
    for lap in laps_info["laps"]:
        lap_ids[(clock_t >= lap["start_time"]) & (clock_t < lap["end_time"])] = lap["lap_num"]
    lap_ids[clock_t >= laps_info["laps"][-1]["end_time"]] = laps_info["laps"][-1]["lap_num"]
    data_log.channels[CH_LAP_NUMBER].set_samples(np.asarray(clock_t), lap_ids)
