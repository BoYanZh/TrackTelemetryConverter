"""Parser for RCZ telemetry log files.

Extracted from the original DataLog.rcz_log methods with identical behavior."""

from __future__ import annotations

import datetime
import logging
import os
import re
from typing import Any

import numpy as np

from ..channels import (
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_CG_ACCEL_VERT,
    CH_COOLANT_TEMP,
    CH_DEVICE_BATTERY,
    CH_ENGINE_OIL_TEMP,
    CH_G_COMBINED,
    CH_GPS_ACCURACY,
    CH_GPS_ALTITUDE,
    CH_GPS_AUX_E,
    CH_GPS_AUX_F,
    CH_GPS_FIX,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GPS_SATS,
    CH_GROUND_SPEED,
    CH_LAP_NUMBER,
    CH_MAG_X,
    CH_MAG_Y,
    CH_MAG_Z,
    CH_PITCH_RATE,
    CH_ROLL_RATE,
    CH_RUNNING_TIME,
    CH_YAW_RATE,
    RCZ_PID_MAP,
)
from ..derived import derive_yaw_rate_from_gps_heading
from ..errors import TelemetryParseError
from ..interpolation import _interp_zoh, _mask_interp_gaps, _mask_interp_outside_range

_PARTIAL_OUT_LAP_SPEED_KMH = 5.0
_MIN_EPOCH_MS = 100_000_000_000
_EPOCH_REFERENCE_WINDOW_MS = 7 * 24 * 60 * 60 * 1000

# RCZ binary channel naming conventions (RaceChrono internal format).
# GPS + accel + gyro families always travel together per recording
# device (e.g. 1/100 + 2/101 + 3/102, or 1/200 + 2/201 + 3/202).
# The numbers are NOT hardware-fixed: the same physical receiver can
# show up under different numbers on different phones/pairings (one
# 2026-09 weekend logged 200 on one phone and 100 on another at the
# same track), and one session (2026-03-07) logged GPS from 300 while
# phone IMU rode along as 2_100/3_101 at its native 66/50 Hz.
# Identify the source by its clock signature instead: an external
# 25 Hz receiver (e.g. RaceBox) streams GPS+IMU on a single clock
# (rateRatio 1.0000000, 0.0 ms drift over a full session), while
# phone IMU keeps its own native rate.
#   4/101 = phone IMU / secondary OBD
#   12/* = primary OBD (car ECU via CAN)
_GPS_DEV_PFX = tuple(
    f"channel_1_{device_type}_0_"
    for device_type in ("300", "200", "100")
)
_GPS_TS_KEYS = tuple(
    f"channel_1_{device_type}_0_1_1"
    for device_type in ("300", "200", "100")
)
_OBD_DEV4_TS_KEY = "channel_4_101_0_1_1"

# Known value/timestamp suffixes under the channel_1_/2_/3_ families,
# derived from the existing probes in this module (not a second table):
# clock+uptime (_resolve_rcz_clock/_validate_uptime_clocks: 1_1/2_1),
# GPS (_populate_rcz_gps: 4_0/3_1/5_0/6_0),
# GPS aux (_populate_rcz_gps_aux: 30002_0/30003_0/30007_0/46_0/30004_0/30005_0),
# accel+gyro (_populate_rcz_imu/_populate_rcz_secondary_imu:
# 9_0/10_0/11_0/12_0/13_0/14_0/1_1).
# Device 8 (magnetometer 28/29/30), device 12/4 OBD and channel2_ value
# files keep their own fallback/debug handling and are excluded here.
_RCZ_KNOWN_123_SUFFIXES = frozenset({
    "1_1", "2_1",
    "3_1", "4_0", "5_0", "6_0",
    "9_0", "10_0", "11_0", "12_0", "13_0", "14_0",
    "46_0", "30002_0", "30003_0", "30004_0", "30005_0", "30007_0",
})
_RCZ_123_RE = re.compile(r"^channel_([123])_(\d+)_0_(.+)$")


def _warn_rcz_unknown_suffixes(namelist):
    """Print one WARNING for channel_1_/2_/3_ suffixes no probe consumes.

    Metadata- and channel-neutral: the unknown files are only listed
    (key + device) so future suffixes (e.g. a new 30008) never vanish
    silently. Device 8/12/4 families are excluded (own handling).
    """
    unknowns = []
    for name in sorted(namelist):
        base = os.path.basename(name)
        match = _RCZ_123_RE.match(base)
        if match is None:
            continue
        suffix = match.group(3)
        if suffix not in _RCZ_KNOWN_123_SUFFIXES:
            unknowns.append((name, f"{match.group(1)}_{match.group(2)}"))
    if unknowns:
        details = ", ".join(
            f"{key} (device {device})" for key, device in unknowns
        )
        print(
            "WARNING: RCZ stint carries unmapped channel suffixes: "
            f"{details}",
        )


def _decode_epoch_ms(data, reference_ms=None):
    """Return a monotonic int64 Unix-ms stream, or None if implausible."""
    if len(data) < 8 or len(data) % 8:
        return None
    values = np.frombuffer(data, dtype="<i8")
    if len(values) == 0 or int(values[0]) < _MIN_EPOCH_MS:
        return None
    if len(values) > 1 and np.any(np.diff(values) < 0):
        return None
    if reference_ms and abs(int(values[0]) - int(reference_ms)) > _EPOCH_REFERENCE_WINDOW_MS:
        return None
    return values


def _relative_epoch_seconds(epoch_ms, origin_epoch_ms):
    return (epoch_ms.astype(np.float64) - float(origin_epoch_ms)) / 1000.0


def _set_absolute_origin(data_log, epoch_ms):
    origin_ms = int(round(float(epoch_ms)))
    data_log.time_origin_epoch_ms = origin_ms
    data_log.datetime_utc = datetime.datetime.fromtimestamp(
        origin_ms / 1000.0, tz=datetime.timezone.utc,
    )
    # Keep the existing local-naive DataLog.datetime contract for exporters.
    data_log.datetime = datetime.datetime.fromtimestamp(origin_ms / 1000.0)




def _resolve_rcz_prefix(z, target_session, target_stint):
    """Resolve archive prefix and prefix-stripped namelist for session/stint."""
    all_names = z.namelist()
    session_prefix = ""
    if target_session is not None:
        session_prefix = f"sessions/{target_session}/"
        if session_prefix + "session.json" not in all_names:
            raise ValueError(f"RCZ session not found: {target_session}")
    prefix = session_prefix
    if target_stint is not None and str(target_stint).lower() != "all":
        stint_value = int(target_stint)
        if stint_value > 0:
            prefix += f"resume_{stint_value}/"
    namelist = [n[len(prefix):] for n in all_names if n.startswith(prefix)]
    if prefix == session_prefix:
        # The archive root also contains resume_N/ files. Parsing stint 0
        # must not ingest later-stint CAN channels merely because their names
        # share the root prefix. Keep unrelated metadata directories intact.
        namelist = [
            name for name in namelist
            if not (
                name.startswith("resume_")
                and "/" in name
                and name.split("/", 1)[0][7:].isdigit()
            )
        ]
    return prefix, session_prefix, namelist

def _read_rcz_entry(z, prefix, name):
    """Read one prefix-relative entry from the RCZ archive."""
    return z.read(prefix + name)

def _load_rcz_session(z, all_names, session_prefix):
    """Load session.json; return (session_json, first_t) or None when missing."""
    import json
    session_key = session_prefix + "session.json"
    if session_key not in all_names:
        raise TelemetryParseError(
            "Invalid RCZ file, missing session.json",
            code="RCZ_SESSION_MISSING",
            stage="parse.session",
            source=z.filename,
            details={"expected_entry": session_key},
        )

    session_json = json.loads(z.read(session_key).decode("utf-8"))
    first_t = session_json.get("firstTimestamp", 0)
    return session_json, first_t




def _load_rcz_fragment(z, all_names, prefix):
    """Load the stint's sessionfragment.json; None when absent/unreadable."""
    import json
    key = prefix + "sessionfragment.json"
    if key not in all_names:
        return None
    try:
        return json.loads(z.read(key).decode("utf-8"))
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        return None


def _fragment_device_id(value):
    """Normalize a fragment device index to the channel-name component."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return str(int(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def _parse_rcz_fragment_decl(fragment):
    """Extract device declarations from sessionfragment.json.

    Returns {"gps": ..., "accel": ..., "gyro": ..., "gyro_bias": ...,
    "videos_count": ...} with None for anything undeclared. The gyro bias
    is recorded for consumers, not applied: the parser stays a faithful
    reader (cf. docs/research/rcz-device-model.md).
    """
    decl: dict[str, Any] = {"gps": None, "accel": None, "gyro": None,
            "gyro_bias": None, "videos_count": None}
    if not isinstance(fragment, dict):
        return decl
    decl["gps"] = _fragment_device_id(
        fragment.get("primaryGpsDeviceIndex",
                     fragment.get("primaryGpsDevice")),
    )
    decl["accel"] = _fragment_device_id(
        fragment.get("imuAccDevice", fragment.get("imuAccDeviceIndex")),
    )
    decl["gyro"] = _fragment_device_id(
        fragment.get("imuGyroDevice", fragment.get("imuGyroDeviceIndex")),
    )
    gyro = decl["gyro"]
    bias = None
    devices2 = fragment.get("devices2")
    if isinstance(devices2, dict) and gyro is not None:
        entry = devices2.get(gyro)
        if not isinstance(entry, dict):
            try:
                entry = devices2.get(int(gyro))
            except (TypeError, ValueError):
                entry = None
        if isinstance(entry, dict) and isinstance(entry.get("gyroBias"), list):
            bias = list(entry["gyroBias"])
    if bias is None and gyro is not None:
        devices = fragment.get("devices")
        items = devices.get("items") if isinstance(devices, dict) else devices
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = _fragment_device_id(
                    item.get("index", item.get("id", item.get("deviceIndex"))),
                )
                if item_id == gyro and isinstance(item.get("gyroBias"), list):
                    bias = list(item["gyroBias"])
                    break
    if bias is not None:
        try:
            decl["gyro_bias"] = [float(v) for v in bias]
        except (TypeError, ValueError):
            decl["gyro_bias"] = None
    videos = fragment.get("videos")
    if isinstance(videos, list):
        decl["videos_count"] = len(videos)
    return decl


def _apply_rcz_fragment_meta(data_log, decl):
    """Record fragment declarations in metadata (no channel changes)."""
    roles = {k: decl[k] for k in ("gps", "accel", "gyro") if decl.get(k) is not None}
    if roles:
        data_log.metadata["rcz_device_roles"] = roles
    if decl.get("gyro_bias") is not None:
        data_log.metadata["rcz_gyro_bias_declared"] = decl["gyro_bias"]
    if decl.get("videos_count") is not None:
        data_log.metadata["rcz_videos_count"] = decl["videos_count"]
        logging.getLogger(__name__).debug(
            "RCZ fragment declares %d video(s); counted in metadata only",
            decl["videos_count"],
        )


def _declared_first(order, declared):
    """Move the fragment-declared id to the head, keeping fallback order."""
    if declared in order:
        return (declared,) + tuple(x for x in order if x != declared)
    return tuple(order)


def _load_rcz_traps(data_log, z, all_names, session_prefix, session_json):
    """Load track trap markers into data_log.traps."""
    import json
    traps_list = []
    track_key = session_prefix + "trackId.json"
    try:
        if track_key in all_names:
            track_json = json.loads(z.read(track_key).decode("utf-8"))
            traps_list = track_json.get("track", {}).get("traps", [])
        elif session_prefix and "json/track_storage.json" in all_names:
            track_storage = json.loads(
                z.read("json/track_storage.json").decode("utf-8"),
            )
            tracks = track_storage.get("tracks", [])
            track = next(
                (
                    item for item in tracks
                    if session_json.get("trackLocalUuid")
                    and item.get("localUuid") == session_json.get("trackLocalUuid")
                ),
                None,
            )
            if track is None:
                track = next(
                    (
                        item for item in tracks
                        if item.get("id") == session_json.get("trackId")
                    ),
                    None,
                )
            if track:
                traps_list = track.get("traps", [])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        traps_list = []
    for trap in traps_list:
        if "centerLatitude" in trap and "centerLongitude" in trap:
            entry = {
                "name": trap.get("name", "Split"),
                "lat": trap["centerLatitude"] / 6000000.0,
                "lon": trap["centerLongitude"] / 6000000.0,
                "type": trap.get("type", 4),
            }
            # Full trap fields: bearing/width are stored x1000 in the
            # archive (180000 -> 180.0 deg; width scale mm -> m is assumed
            # and unverified). Anything else stays in the trap dict only --
            # never widened into channels. Unknown future keys go to
            # debug log, not to silent loss.
            if trap.get("bearing") is not None:
                entry["bearing_deg"] = trap["bearing"] / 1000.0
            if trap.get("width") is not None:
                entry["width_m"] = trap["width"] / 1000.0
            if trap.get("orderValue") is not None:
                entry["order"] = trap["orderValue"]
            if trap.get("uniDirectional") is not None:
                entry["uni_directional"] = bool(trap["uniDirectional"])
            known_trap_keys = {
                "name", "centerLatitude", "centerLongitude", "bearing",
                "width", "type", "orderValue", "uniDirectional",
            }
            unknown = sorted(set(trap) - known_trap_keys)
            if unknown:
                logging.getLogger(__name__).debug(
                    "RCZ trap %r carries unmapped keys: %s",
                    trap.get("name", "Split"), unknown,
                )
            data_log.traps.append(entry)

def _apply_rcz_session_meta(data_log, session_json, rcz_file_path):
    """Apply session datetime and metadata fields."""
    created_ms = session_json.get("timeCreated")
    if created_ms and created_ms > 1e8:
        data_log.session_created_epoch_ms = int(created_ms)
    ts_ms = created_ms or session_json.get("firstTimestamp")
    if ts_ms and ts_ms > 1e8:
        data_log.datetime = datetime.datetime.fromtimestamp(ts_ms / 1000.0)
        data_log.datetime_utc = datetime.datetime.fromtimestamp(
            ts_ms / 1000.0, tz=datetime.timezone.utc,
        )
    else:
        data_log.datetime = data_log._extract_datetime_from_text([], rcz_file_path)

    # Store metadata
    data_log.rcz_metadata = {
        "title": session_json.get("title", ""),
        "trackName": session_json.get("trackName", ""),
        "timeCreated": session_json.get("timeCreated", 0),
    }
    data_log._extract_metadata([], rcz_file_path)
    if session_json.get("trackName"):
        data_log.metadata["venue_name"] = session_json.get("trackName")
    if session_json.get("title"):
        data_log.metadata["event_session"] = session_json.get("title")
    notes = session_json.get("notes") or session_json.get("description")
    if notes:
        data_log.metadata["short_comment"] = notes

def _select_rcz_stint(target_stint):
    """Resolve the active stint number (0 when unspecified)."""
    if target_stint is None or str(target_stint).lower() == "all":
        return 0
    try:
        return int(target_stint)
    except ValueError:
        return 0




def _resolve_rcz_clock(read_channel, namelist, epoch_reference, declared_gps=None):
    """Resolve the GPS timestamp channel; return (time_file, timestamps_ms).

    A fragment-declared GPS device is tried first; the legacy
    (300, 200, 100) order stays as fallback so packs without a fragment --
    and packs whose declared family is absent -- parse exactly as before.
    """
    ordered = _declared_first(("300", "200", "100"), declared_gps)
    for device_type in ordered:
        candidate = f"channel_1_{device_type}_0_1_1"
        if candidate in namelist:
            decoded = _decode_epoch_ms(read_channel(candidate), epoch_reference)
            if decoded is not None:
                return candidate, decoded

    for name in namelist:
        if name.startswith("channel_") and name.endswith("_1_1"):
            decoded = _decode_epoch_ms(read_channel(name), epoch_reference)
            if decoded is not None:
                return name, decoded

    return None, None

def _init_rcz_timebase(data_log, timestamps_ms):
    """Set the absolute origin; return (times_sec, n_samples, time_origin_epoch_ms)."""
    time_origin_epoch_ms = int(timestamps_ms[0])
    _set_absolute_origin(data_log, time_origin_epoch_ms)
    times_sec = _relative_epoch_seconds(timestamps_ms, time_origin_epoch_ms)
    n_samples = len(times_sec)
    return times_sec, n_samples, time_origin_epoch_ms




def _resolve_rcz_gps(data_log, read_channel, namelist, n_samples, declared_gps=None):
    """Identify the GPS sensor family and read speed values.

    The fragment-declared GPS family is preferred; the legacy
    (300, 200, 100) priority stays as fallback so single-family packs
    resolve identically with or without a fragment.
    """
    gps_prefix = None
    ordered_types = _declared_first(("300", "200", "100"), declared_gps)
    for device_type in ordered_types:
        candidate_prefix = f"channel_1_{device_type}_0_"
        if any(name.startswith(candidate_prefix) for name in namelist):
            gps_prefix = candidate_prefix
            break
    gps_speed_values = None
    if gps_prefix:
        # Record which sensor family supplied GPS (e.g. rcz_gps_1_100),
        # so a later session can tell at a glance which recording
        # device was the source. NOTE: the number is not hardware-fixed
        # (see comment above) -- identify the hardware by clock
        # signature, not by this string.
        _gps_parts = gps_prefix.split("_")
        data_log.metadata["gps_source"] = f"rcz_gps_{_gps_parts[1]}_{_gps_parts[2]}"
        speed_key = gps_prefix + "4_0"
        if speed_key in namelist:
            raw_speed = np.frombuffer(read_channel(speed_key), dtype="<i4")
            if len(raw_speed) >= n_samples:
                gps_speed_values = (raw_speed[:n_samples] / 1000.0) * 3.6
    return gps_prefix, gps_speed_values

def _reconstruct_rcz_laps(session_json, active_stint, times_sec, timestamps_ms, first_t, min_lap_sec, gps_speed_values, n_samples):
    """Reconstruct Out/Timed/In laps overlapping the binary recording."""
    # Map laps from session.json
    laps_raw = session_json.get("laps", [])

    # Group raw laps by stint
    stints_map = {}
    for lap in laps_raw:
        st_id = int(lap.get("sessionResume", 0))
        if st_id not in stints_map:
            stints_map[st_id] = []
        stints_map[st_id].append(lap)

    stint_laps = stints_map.get(active_stint, [])

    # Reconstruct clean lap structure for this stint by filtering laps that overlap with binary recording range
    stint_duration_s = times_sec[-1] if len(times_sec) else 0.0
    stint_start_ms = timestamps_ms[0] if len(timestamps_ms) else first_t
    stint_end_ms = timestamps_ms[-1] if len(timestamps_ms) else (first_t + int(stint_duration_s * 1000))

    # Filter raw laps that actually overlap with binary recording range.
    # Laps flagged isInvalid never seed reconstruction (all 305 corpus
    # laps are False, so this is a zero-impact guard with a test).
    overlapping_laps = []
    for lap in stint_laps:
        if lap.get("isInvalid"):
            continue
        s_ms = lap.get("startTimestamp")
        f_ms = lap.get("finishTimestamp")
        if not s_ms:
            continue
        cmp_end = f_ms if f_ms is not None else stint_end_ms
        if s_ms < stint_end_ms and cmp_end > stint_start_ms:
            overlapping_laps.append(lap)

    reconstructed_laps = []

    # 1. Out Lap (if leading non-timed padding exceeds the configured minimum)
    first_timed_s = (overlapping_laps[0].get("startTimestamp") - stint_start_ms) / 1000.0 if overlapping_laps else 0.0
    if first_timed_s > min_lap_sec:
        out_lap_type = "Out Lap"
        if (
            gps_speed_values is not None
            and len(gps_speed_values) > 0
            and np.isfinite(gps_speed_values[0])
            and gps_speed_values[0] >= _PARTIAL_OUT_LAP_SPEED_KMH
        ):
            out_lap_type = "Partial Out Lap"
            print(
                "WARNING: RCZ recording begins mid-lap at "
                f"{gps_speed_values[0]:.1f} km/h; marking the leading segment "
                "as Partial Out Lap",
            )
        reconstructed_laps.append({
            "type": out_lap_type,
            "lap_label": out_lap_type,
            "lap_num": 1,
            "start_time": 0.0,
            "end_time": first_timed_s,
            "duration": first_timed_s,
            "stint": active_stint,
        })

    # 2. Timed Laps & In Lap inside recording window
    for lap in overlapping_laps:
        s_ms = lap.get("startTimestamp")
        f_ms = lap.get("finishTimestamp")
        orig_num = lap.get("number")

        start_s = max(0.0, (s_ms - stint_start_ms) / 1000.0)

        if f_ms is not None:
            end_s = min(stint_duration_s, (f_ms - stint_start_ms) / 1000.0)
            dur_s = end_s - start_s
            if dur_s >= min_lap_sec:
                reconstructed_laps.append({
                    "type": "Timed",
                    "lap_label": str(len(reconstructed_laps) + 1),
                    "lap_num": len(reconstructed_laps) + 1,
                    "start_time": start_s,
                    "end_time": end_s,
                    "duration": dur_s,
                    "stint": active_stint,
                    "orig_num": orig_num,
                })
        else:
            end_s = stint_duration_s
            dur_s = end_s - start_s
            if dur_s >= min_lap_sec:
                reconstructed_laps.append({
                    "type": "In Lap",
                    "lap_label": "In Lap",
                    "lap_num": len(reconstructed_laps) + 1,
                    "start_time": start_s,
                    "end_time": end_s,
                    "duration": dur_s,
                    "stint": active_stint,
                    "orig_num": orig_num,
                })

    # 3. Trailing In Lap check
    if overlapping_laps and overlapping_laps[-1].get("finishTimestamp") is not None:
        last_finish_ms = overlapping_laps[-1].get("finishTimestamp")
        in_dur_s = (stint_end_ms - last_finish_ms) / 1000.0
        if in_dur_s > min_lap_sec:
            reconstructed_laps.append({
                "type": "In Lap",
                "lap_label": "In Lap",
                "lap_num": len(reconstructed_laps) + 1,
                "start_time": (last_finish_ms - stint_start_ms) / 1000.0,
                "end_time": stint_duration_s,
                "duration": in_dur_s,
                "stint": active_stint,
            })

    # Build lap_numbers array for data logging
    lap_numbers = np.ones(n_samples, dtype=int)
    for r in reconstructed_laps:
        m = (times_sec >= r["start_time"]) & (times_sec <= r["end_time"])
        lap_numbers[m] = r["lap_num"]
    return reconstructed_laps, lap_numbers, stint_duration_s

def _select_rcz_target_lap(data_log, reconstructed_laps, lap_numbers, times_sec, timestamps_ms, n_samples, target_lap):
    """Filter to one lap window; refresh the absolute origin for it."""
    # Filter target_lap if specified
    target_lap_int = None
    if target_lap is not None and str(target_lap).lower() != "all":
        try:
            target_lap_int = int(target_lap)
        except ValueError:
            pass

    selected_laps = reconstructed_laps
    selected_time_origin = 0.0
    if target_lap_int is not None:
        matching_laps = [
            lap for lap in reconstructed_laps if lap["lap_num"] == target_lap_int
        ]
        if matching_laps:
            mask = (lap_numbers == target_lap_int)
            selected_samples = times_sec[mask]
            if len(selected_samples) > 0:
                selected_time_origin = float(selected_samples[0])
            selected_lap = dict(matching_laps[0])
            selected_lap["start_time"] = 0.0
            selected_lap["end_time"] = float(selected_lap["duration"])
            selected_laps = [selected_lap]
        else:
            print(f"WARNING: Lap '{target_lap}' not found in RCZ stint; keeping all laps")
            target_lap_int = None
            mask = np.ones(n_samples, dtype=bool)
    else:
        mask = np.ones(n_samples, dtype=bool)

    export_times_sec = times_sec - selected_time_origin
    sample_times = export_times_sec[mask]
    original_sample_times = times_sec[mask]
    if len(original_sample_times) > 0 and len(timestamps_ms) > 0 and timestamps_ms[0] > 1e8:
        actual_start_ms = timestamps_ms[0] + (original_sample_times[0] * 1000.0)
        _set_absolute_origin(data_log, actual_start_ms)
    data_log.rcz_metadata["timeOriginEpochMs"] = data_log.time_origin_epoch_ms
    return selected_laps, mask, export_times_sec, sample_times, target_lap_int

def _build_rcz_laps_info(data_log, selected_laps, target_lap_int, stint_duration_s):
    """Store the lap summary for LDX export."""
    # Store laps_info for ldx export
    fastest_lap_num = 1
    fastest_dur = float("inf")
    for r in selected_laps:
        if r["type"] == "Timed" and r["duration"] < fastest_dur:
            fastest_dur = r["duration"]
            fastest_lap_num = r["lap_num"]

    data_log.laps_info = {
        "laps": selected_laps,
        "total_laps": len(selected_laps),
        "fastest_lap": fastest_lap_num,
        "fastest_time": fastest_dur if fastest_dur != float("inf") else 0.0,
        "session_duration": (
            float(selected_laps[0]["duration"])
            if target_lap_int is not None and selected_laps
            else stint_duration_s
        ),
    }

def _populate_rcz_channel(data_log, n_samples, mask, sample_times, name, units, values_array, decimals=2):
    """Append one resampled channel unless its values are too short."""
    if len(values_array) < n_samples:
        return
    values_array = values_array[:n_samples]
    filtered_vals = values_array[mask]
    data_log.add_channel(name, units, float, decimals)
    data_log.channels[name].decimals = decimals
    data_log.channels[name].set_samples(sample_times, filtered_vals)




def _populate_rcz_gps(data_log, namelist, gps_prefix, gps_speed_values, n_samples, read_channel, populate_channel):
    """Populate GPS speed/position/altitude/heading channels."""
    if gps_prefix:
        # 2. Parse Speed
        if gps_speed_values is not None:
            populate_channel(CH_GROUND_SPEED, "km/h", gps_speed_values, 2)

        # 3. Parse Latitude & Longitude
        _ll_key = gps_prefix + "3_1"
        if _ll_key in namelist:
            raw_ll = np.frombuffer(read_channel(_ll_key), dtype="<i4")
            if len(raw_ll) >= n_samples * 2:
                raw_ll = raw_ll[:n_samples * 2].reshape(-1, 2)
                populate_channel(CH_GPS_LATITUDE, "deg", raw_ll[:, 0] / 6000000.0, 7)
                populate_channel(CH_GPS_LONGITUDE, "deg", raw_ll[:, 1] / 6000000.0, 7)

        # 4. Parse Altitude
        _alt_key = gps_prefix + "5_0"
        if _alt_key in namelist:
            raw_alt = np.frombuffer(read_channel(_alt_key), dtype="<i4")
            if len(raw_alt) >= n_samples:
                populate_channel(CH_GPS_ALTITUDE, "m", raw_alt / 1000.0)

        # 5. Parse GPS Heading
        _hdg_key = gps_prefix + "6_0"
        if _hdg_key in namelist:
            raw_hdg = np.frombuffer(read_channel(_hdg_key), dtype="<i4")
            if len(raw_hdg) >= n_samples:
                populate_channel(CH_GPS_HEADING, "deg", raw_hdg / 1000.0)

def _populate_rcz_gps_aux(data_log, namelist, gps_prefix, n_samples, read_channel, populate_channel):
    """Collect GPS-family value files of verified + uncertain semantics.

    30002/30003/30007/46 ride the GPS clock 1:1 (4 bytes/sample) and are
    VERIFIED against a paired RaceChrono CSV export of the same session
    (2026-10-04 Thunder Hill East Bypass: all 5 stints, 126,746 GPS
    samples -- A↔satellites and B↔fix match exactly, C↔accuracy and
    D↔battery within CSV text rounding), so they land under their
    canonical names. 30004/30005 stay neutral (GPS Aux E/F): INT32_MAX
    marks unknown precision -- kept missing (NaN), not a
    two-million-metre error -- and no paired CSV column exists to promote
    them. Absent or short files are skipped, so older packs are unaffected.
    """
    if not gps_prefix:
        return

    def _read_i4(key):
        if key not in namelist:
            return None
        values = np.frombuffer(read_channel(key), dtype="<i4")
        if len(values) < n_samples:
            return None
        return values[:n_samples].astype(np.float64)

    # 30002: modern packs store raw counts (12 ~ 16); the 2026-02-19
    # pack stores count x 1000 (3000 ~ 16000). Counts above 100 cannot be
    # raw satellites, so divide.
    aux_a = _read_i4(gps_prefix + "30002_0")
    if aux_a is not None:
        if np.nanmax(aux_a) > 100:
            aux_a = aux_a / 1000.0
        populate_channel(CH_GPS_SATS, "", aux_a, 0)

    aux_b = _read_i4(gps_prefix + "30003_0")
    if aux_b is not None:
        populate_channel(CH_GPS_FIX, "", aux_b, 0)

    aux_c = _read_i4(gps_prefix + "30007_0")
    if aux_c is not None:
        populate_channel(CH_GPS_ACCURACY, "m", aux_c / 1000.0)

    aux_d = _read_i4(gps_prefix + "46_0")
    if aux_d is not None:
        populate_channel(CH_DEVICE_BATTERY, "%", aux_d / 1000.0, 1)

    # E/F (30004/30005): INT32_MAX marks unknown precision -- keep it
    # missing (NaN), not a 2-million-metre error.
    for key, out_name in (("30004_0", CH_GPS_AUX_E), ("30005_0", CH_GPS_AUX_F)):
        aux = _read_i4(gps_prefix + key)
        if aux is None:
            continue
        aux = np.where(aux >= 2147483647, np.nan, aux) / 1000.0
        populate_channel(out_name, "m", aux)

def _populate_rcz_magnetometer(data_log, namelist, times_sec, n_samples, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel):
    """Archive device-8 magnetometer files (28/29/30) as raw channels.

    GPS-Aux套路: value semantics are UNCERTAIN (app-table names X/Y/Z
    magnetic field + int32 ranges only, no paired cross-check), so raw
    counts are stored x1.0 under neutral Magnetic Field names. A valid
    paired `channel_8_<type>_0_1_1` epoch stream is required (observed
    valid in the 2026-03-07 pack); values resample onto the GPS grid via
    their own clock, falling back to 1:1 truncation. The `*_2_1` uptime
    companion is validation-only (see _validate_uptime_clocks) and never
    stored -- an all-zero degraded stream changes nothing here.
    """
    import os
    import re
    _mag_re = re.compile(r"^channel_8_(\d+)_0_(28|29|30)_0$")
    families = sorted({
        m.group(1)
        for m in (_mag_re.match(os.path.basename(n)) for n in namelist)
        if m is not None
    })
    if not families:
        return
    if len(families) > 1:
        logging.getLogger(__name__).debug(
            "RCZ stint carries %d magnetometer families %s; keeping primary only",
            len(families), families,
        )
    mag_type = families[0]
    ts_candidates = [
        n for n in namelist
        if os.path.basename(n) == f"channel_8_{mag_type}_0_1_1"
    ]
    if not ts_candidates:
        return
    mag_t = None
    for ts_key in ts_candidates:
        raw_ts = _decode_epoch_ms(read_channel(ts_key), time_origin_epoch_ms)
        if raw_ts is not None:
            mag_t = _relative_epoch_seconds(raw_ts, time_origin_epoch_ms)
            break
    if mag_t is None:
        return
    for suffix, out_name in (
        ("28_0", CH_MAG_X), ("29_0", CH_MAG_Y), ("30_0", CH_MAG_Z),
    ):
        val_candidates = [
            n for n in namelist
            if os.path.basename(n) == f"channel_8_{mag_type}_0_{suffix}"
        ]
        if not val_candidates:
            continue
        raw = np.frombuffer(
            read_channel(val_candidates[0]), dtype="<i4",
        ).astype(np.float64)
        if len(raw) == len(mag_t):
            resampled = np.interp(times_sec, mag_t, raw)
            if mask_interp_gaps:
                resampled = _mask_interp_gaps(resampled, times_sec, mag_t)
            resampled = _mask_interp_outside_range(resampled, times_sec, mag_t)
            populate_channel(out_name, "", resampled, 0)
        elif len(raw) >= n_samples:
            populate_channel(out_name, "", raw, 0)

def _populate_rcz_imu(data_log, namelist, times_sec, n_samples, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel, declared_accel=None, declared_gyro=None):
    """Populate accelerometer/gyro channels with drift correction.

    Fragment-declared IMU devices are preferred; the legacy box-first
    order stays as fallback so single-set packs resolve identically with
    or without a fragment.
    """
    # 6. Parse Accelerations (device 2, type 201)
    # If a per-device timestamp file exists, resample onto GPS time grid using
    # np.interp.  This corrects for slight rate drift (e.g. IMU at 25.008 Hz vs
    # GPS at 25.000 Hz) which accumulates to 48-sample / 1.9 s error over a
    # 1115 s session, causing a measurable lag in the exported data.
    def _imu_times(ts_key):
        """Return relative seconds for a device's int64 epoch stream."""
        if ts_key not in namelist:
            return None
        raw = _decode_epoch_ms(read_channel(ts_key), time_origin_epoch_ms)
        if raw is None:
            return None
        return _relative_epoch_seconds(raw, time_origin_epoch_ms)

    def _parse_imu_channel(
        ch_key, out_name, units, scale, decimals=2, ts_key=None,
    ):
        if ch_key not in namelist:
            return
        raw = np.frombuffer(read_channel(ch_key), dtype="<i4").astype(np.float64) * scale
        imu_t = _imu_times(ts_key)
        if imu_t is not None and len(imu_t) == len(raw):
            # Resample via timestamps - handles rate drift and small offsets
            resampled = np.interp(times_sec, imu_t, raw)
            if mask_interp_gaps:
                resampled = _mask_interp_gaps(resampled, times_sec, imu_t)
            resampled = _mask_interp_outside_range(resampled, times_sec, imu_t)
            populate_channel(out_name, units, resampled, decimals)
        elif len(raw) >= n_samples:
            # Fallback: naive truncation (off by  1 sample)
            populate_channel(out_name, units, raw, decimals)

    accel_type = next(
        (
            device_type
            # 301/302 = external box, 201/202 = newer phone, 101/102 =
            # older phone, 100/101 = 2026-03-07 phone scheme (accel rides
            # as 2_100, gyro as 3_101 alongside the 300-series box).
            # Box first: order is priority, append-only to keep older
            # packs byte-identical. A fragment declaration moves to the
            # head without changing the fallback order.
            for device_type in _declared_first(
                ("301", "201", "101", "100"), declared_accel)
            if f"channel_2_{device_type}_0_10_0" in namelist
        ),
        None,
    )
    if accel_type is not None:
        accel_prefix = f"channel_2_{accel_type}_0_"
        accel_ts_file = accel_prefix + "1_1"
        _parse_imu_channel(
            accel_prefix + "10_0",
            CH_CG_ACCEL_LAT,
            "G",
            1.0 / 10000.0,
            ts_key=accel_ts_file,
        )
        _parse_imu_channel(
            accel_prefix + "9_0",
            CH_CG_ACCEL_LON,
            "G",
            1.0 / 10000.0,
            ts_key=accel_ts_file,
        )
        _parse_imu_channel(
            accel_prefix + "11_0",
            CH_CG_ACCEL_VERT,
            "G",
            1.0 / 10000.0,
            ts_key=accel_ts_file,
        )
        # Accelerometer channel 11 measures vertical acceleration
        # including gravity (~1 G while stationary), not lean angle.
        acceleration_source = f"rcz_accelerometer_2_{accel_type}"
        data_log.metadata["lateral_accel_source"] = acceleration_source
        data_log.metadata["longitudinal_accel_source"] = acceleration_source
        data_log.metadata["vertical_accel_source"] = acceleration_source

    # 7. Parse Gyroscope / Yaw Rate (device 3, type 302 / 202 / 102)
    gyro_z_file = None
    gyro_ts_file = None
    primary_gyro_type = None
    for g_type in _declared_first(("302", "202", "102", "101"), declared_gyro):
        gz_cand = f"channel_3_{g_type}_0_14_0"
        ts_cand = f"channel_3_{g_type}_0_1_1"
        if gz_cand in namelist:
            gyro_z_file = gz_cand
            gyro_ts_file = ts_cand
            primary_gyro_type = g_type
            break

    if gyro_z_file:
        # Device-3 gyroscope Z has the opposite sign from the
        # positive vehicle/CAN yaw convention used for this channel.
        _parse_imu_channel(gyro_z_file, CH_YAW_RATE, "deg/s",
                           -1.0 / 1000.0, ts_key=gyro_ts_file)
        data_log.metadata["yaw_rate_source"] = f"rcz_gyro_3_{primary_gyro_type}"
        data_log.metadata["yaw_axis_sign_verified"] = True
        # Gyroscope X/Y ride the same device clock (12 = X/roll,
        # 13 = Y/pitch per the RaceChrono rotation-rate table). Same
        # sensor-frame scale as Z; X/Y sign is unverified (no CAN
        # reference) and left as-recorded. Missing files are skipped
        # inside _parse_imu_channel, so older packs are unaffected.
        gyro_prefix = f"channel_3_{primary_gyro_type}_0_"
        _parse_imu_channel(gyro_prefix + "12_0", CH_ROLL_RATE, "deg/s",
                           -1.0 / 1000.0, ts_key=gyro_ts_file)
        _parse_imu_channel(gyro_prefix + "13_0", CH_PITCH_RATE, "deg/s",
                           -1.0 / 1000.0, ts_key=gyro_ts_file)

    _populate_rcz_secondary_imu(
        data_log, namelist, times_sec, time_origin_epoch_ms,
        mask_interp_gaps, read_channel, populate_channel,
        primary_accel_type=accel_type, primary_gyro_type=primary_gyro_type,
    )


def _populate_rcz_secondary_imu(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel, primary_accel_type, primary_gyro_type):
    """Parse a second IMU set (if present) into *-suffixed channels.

    Dual-device packs (e.g. 2026-03-07: phone IMU 2_100/3_101 riding
    alongside the primary 300-series box) carry a full extra
    accelerometer/gyro set. The primary selection above is untouched, so
    single-set packs -- the common case -- produce byte-identical output;
    the extra set only surfaces here under Secondary names, with existing
    channel values unchanged. Secondary gyro sign follows the primary
    convention but is unverified (no CAN reference for the second device).
    """

    def _imu_times(ts_key):
        if ts_key not in namelist:
            return None
        raw = _decode_epoch_ms(read_channel(ts_key), time_origin_epoch_ms)
        if raw is None:
            return None
        return _relative_epoch_seconds(raw, time_origin_epoch_ms)

    def _parse_secondary(ch_key, out_name, units, scale, ts_key=None):
        if ch_key in namelist and out_name not in data_log.channels:
            raw = np.frombuffer(read_channel(ch_key), dtype="<i4").astype(np.float64) * scale
            imu_t = _imu_times(ts_key) if ts_key else None
            if imu_t is not None and len(imu_t) == len(raw):
                resampled = np.interp(times_sec, imu_t, raw)
                if mask_interp_gaps:
                    resampled = _mask_interp_gaps(resampled, times_sec, imu_t)
                resampled = _mask_interp_outside_range(resampled, times_sec, imu_t)
            else:
                resampled = raw
            # populate_channel truncates to the GPS grid when long enough;
            # a shorter secondary set is skipped, never zero-filled.
            populate_channel(out_name, units, resampled)

    secondary_accel = next(
        (
            device_type
            for device_type in ("301", "201", "101", "100")
            if device_type != primary_accel_type
            and f"channel_2_{device_type}_0_10_0" in namelist
        ),
        None,
    )
    secondary_gyro = next(
        (
            device_type
            for device_type in ("302", "202", "102", "101")
            if device_type != primary_gyro_type
            and f"channel_3_{device_type}_0_14_0" in namelist
        ),
        None,
    )
    if secondary_accel is None and secondary_gyro is None:
        # Also note a second GPS family sharing one stint (not observed in
        # the corpus -- multi-GPS packs switch families per stint instead).
        gps_families = {
            name.split("_")[2]
            for name in namelist
            if name.startswith("channel_1_") and name.endswith("_1_1")
        }
        if len(gps_families) > 1:
            logging.getLogger(__name__).debug(
                "RCZ stint carries %d GPS families %s; keeping primary only",
                len(gps_families), sorted(gps_families),
            )
        return
    if secondary_accel is not None:
        prefix = f"channel_2_{secondary_accel}_0_"
        ts_key = prefix + "1_1"
        _parse_secondary(prefix + "10_0", CH_CG_ACCEL_LAT + " Secondary", "G", 1.0 / 10000.0, ts_key=ts_key)
        _parse_secondary(prefix + "9_0", CH_CG_ACCEL_LON + " Secondary", "G", 1.0 / 10000.0, ts_key=ts_key)
        _parse_secondary(prefix + "11_0", CH_CG_ACCEL_VERT + " Secondary", "G", 1.0 / 10000.0, ts_key=ts_key)
    if secondary_gyro is not None:
        prefix = f"channel_3_{secondary_gyro}_0_"
        ts_key = prefix + "1_1"
        _parse_secondary(prefix + "14_0", CH_YAW_RATE + " Secondary", "deg/s", -1.0 / 1000.0, ts_key=ts_key)
        _parse_secondary(prefix + "12_0", CH_ROLL_RATE + " Secondary", "deg/s", -1.0 / 1000.0, ts_key=ts_key)
        _parse_secondary(prefix + "13_0", CH_PITCH_RATE + " Secondary", "deg/s", -1.0 / 1000.0, ts_key=ts_key)
    data_log.metadata["secondary_imu_source"] = (
        f"rcz_secondary_2_{secondary_accel}/3_{secondary_gyro}"
    )


def _check_uptime_stream(raw_bytes, clock_len):
    """Structural check on one device `*_2_1` uptime stream.

    Encoding (int32 pairs, odd words zero) is not a wall clock -- its tick
    rate varies by device -- so no timestamps are derived from it. What is
    checked: pair structure, monotonic non-decreasing ticks, and 1:1 count
    accompaniment with the sibling `*_1_1` epoch stream. Returns a small
    summary dict for metadata; violations are for the caller to warn on.
    """
    summary = {
        "samples": 0, "monotonic": False, "clock_match": False,
        "median_ticks_per_sample": None,
    }
    if len(raw_bytes) < 8 or len(raw_bytes) % 8:
        return summary
    words = np.frombuffer(raw_bytes, dtype="<i4")
    if len(words) % 2:
        return summary
    ticks = words[0::2].astype(np.float64)
    summary["samples"] = len(ticks)
    summary["monotonic"] = bool(np.all(np.diff(ticks) >= 0))
    summary["clock_match"] = len(ticks) == clock_len
    if len(ticks) > 1:
        summary["median_ticks_per_sample"] = float(np.median(np.diff(ticks)))
    return summary


def _validate_uptime_clocks(data_log, namelist, read_channel):
    """Cross-clock drift sentinel over each device's `*_2_1` uptime file.

    Deliberately metadata-only (no diagnostic channels): the tick unit is
    device-specific and unverified, so the check asserts structure, not
    time. Any violation prints a warning; the per-device summary lands in
    metadata for post-hoc inspection.
    """
    import os
    import re
    # Device-family uptime shape only: channel_<dev>_<type>_0_2_1.
    # channel_12_<sub>_<pid>_2_1 files are per-PID OBD timestamp streams,
    # not uptime -- matching them warns falsely on every CAN pack (624
    # such files in the corpus), so device 12 is excluded explicitly.
    _uptime_re = re.compile(r"^channel_(?!12_)\d+_\d+_0_2_1$")
    report = {}
    for name in sorted(namelist):
        base = os.path.basename(name)
        if not _uptime_re.match(base):
            continue
        head, _, _ = base.rpartition("_2_1")
        clock_key = next(
            (n for n in namelist if os.path.basename(n) == head + "_1_1"),
            None,
        )
        clock_len = -1
        if clock_key is not None:
            try:
                clock_len = len(np.frombuffer(read_channel(clock_key), dtype="<i8"))
            except (ValueError, TypeError):
                clock_len = -1
        try:
            summary = _check_uptime_stream(read_channel(name), clock_len)
        except (ValueError, TypeError):
            summary = {"samples": 0, "monotonic": False, "clock_match": False,
                       "median_ticks_per_sample": None}
        report[name] = summary
        if not summary["monotonic"] or not summary["clock_match"]:
            print(
                f"WARNING: RCZ uptime stream {name} looks corrupt "
                f"(monotonic={summary['monotonic']}, "
                f"clock_match={summary['clock_match']})",
            )
        else:
            logging.getLogger(__name__).debug(
                "RCZ uptime %s: %d samples, median %.1f ticks/sample",
                name, summary["samples"],
                summary["median_ticks_per_sample"] or 0.0,
            )
    if report:
        data_log.metadata["rcz_uptime_clocks"] = report

def _populate_rcz_obd12(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel):
    """Populate primary OBD (device 12) CAN channels."""
    import os
    # 8. Parse OBD-II / CAN Channels
    # RCZ stores binary channel values as contiguous IEEE 754 float64 (double precision) values.
    # Values are ALREADY in engineering units (1:1 scale).
    # Internal RCZ channel PID to MoTeC channel name & unit mapping:
    rcz_pid_map = RCZ_PID_MAP
    yaw_rate_from_can = False
    # PID 43 timestamp files without a value companion (common: 18 corpus
    # packs) are collected below and emitted after the loop as an all-NaN
    # placeholder -- only if no real G Force Combined arrived meanwhile.
    pid43_degraded_times = None

    for name in sorted(namelist):
        base_fname = os.path.basename(name)
        if not (base_fname.startswith("channel_12_") or base_fname.startswith("channel2_12_")) or not base_fname.endswith("_1_1"):
            continue
        parts = base_fname.split("_")
        if len(parts) < 4:
            continue
        dev_sub = parts[2]
        pid = parts[3]
        raw_times = _decode_epoch_ms(read_channel(name), time_origin_epoch_ms)
        dir_prefix = os.path.dirname(name)
        companion_fname = f"channel2_12_{dev_sub}_{pid}_{pid}_3"
        companion = (dir_prefix + "/" + companion_fname) if dir_prefix else companion_fname
        if companion not in namelist or raw_times is None:
            # Degraded branch: the sensor declared timestamps but stored no
            # values. Emit NaN (masked, not zero) so the gap is visible and
            # calculate_g_sum can still recompute the channel at math time.
            # Normal path below is untouched.
            if (
                pid == "43"
                and companion not in namelist
                and raw_times is not None
                and CH_G_COMBINED not in data_log.channels
                and pid43_degraded_times is None
            ):
                pid43_degraded_times = raw_times
            continue
        # Correct binary format: float64 (double precision, 8 bytes per value)
        value_data = np.frombuffer(read_channel(companion), dtype="<f8")
        count = min(len(raw_times), len(value_data))
        if count < 2:
            continue
        raw_times = raw_times[:count]
        raw_values = value_data[:count]
        rel_times = _relative_epoch_seconds(raw_times, time_origin_epoch_ms)
        if rel_times[-1] <= 0:
            continue
        if pid == "1004":
            values = _interp_zoh(times_sec, rel_times, raw_values)
        else:
            values = np.interp(times_sec, rel_times, raw_values)
            if mask_interp_gaps:
                values = _mask_interp_gaps(values, times_sec, rel_times)
        values = _mask_interp_outside_range(values, times_sec, rel_times)
        if pid in rcz_pid_map:
            ch_name, ch_unit, ch_scale, ch_offset = rcz_pid_map[pid]
            if pid == "51" and not yaw_rate_from_can:
                vals_processed = values * ch_scale + ch_offset
                populate_channel(ch_name, ch_unit, vals_processed)
                yaw_rate_from_can = True
                data_log.metadata["yaw_rate_source"] = (
                    f"rcz_can_12_{dev_sub}_pid_51"
                )
                # CAN yaw is the vehicle reference itself.
                data_log.metadata["yaw_axis_sign_verified"] = True
            elif ch_name not in data_log.channels:
                vals_processed = values * ch_scale + ch_offset
                if pid == "1004":
                    # Do not turn unavailable gear samples into neutral.
                    vals_processed = np.round(vals_processed).clip(-1, 6)
                populate_channel(ch_name, ch_unit, vals_processed)
        else:
            populate_channel(f"OBD_{pid}", "", values)

    if pid43_degraded_times is not None and CH_G_COMBINED not in data_log.channels:
        # Timestamp-only PID 43: no samples to interpolate, so mask the
        # whole GPS grid as missing rather than synthesising zeros.
        populate_channel(
            CH_G_COMBINED, "G", np.full(len(times_sec), np.nan),
        )
        data_log.metadata["g_combined_source"] = "rcz_pid43_timestamps_only"

def _populate_rcz_obd4(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel):
    """Populate secondary OBD (device 4/101, phone IMU flavored PIDs)."""
    import os
    # 8b. Parse OBD-II / CAN Channels - Device 4, Type 101 (shared timestamp format)
    # Format: channel_4_101_0_1_1 = shared int64 epoch-ms timestamps for all PIDs
    #         channel2_4_101_0_{pid}_3 = per-PID float64 values (same length as timestamps)
    # Device 4/type 101 uses phone-IMU channels for PIDs 7/8/49/50 - different from
    # the GR86-ECU meaning of the same PIDs on device 12/type 100.
    _g = 9.80665
    _f_to_c = 5.0 / 9.0
    _f_to_c_offset = -32.0 * _f_to_c
    rcz_dev4_pid_overrides = {
        # PID 7 on device 4 = phone lateral accelerometer (m/s^2), NOT ECU Roll Angle
        "7":  (CH_CG_ACCEL_LAT,     "G",    1.0 / _g, 0.0),
        # PID 8 on device 4 = phone vertical accelerometer (m/s^2, includes gravity) - skip
        "8":  None,
        # PIDs 49/50 on device 4 = phone roll/pitch rate (rotation-rate
        # class, same as pid 51/14): +/-30 / +/-19 deg/s in the 2026-02-19
        # pack matches gyro ranges, so deg/s x1.0 like device-12 pid 51.
        "49": (CH_ROLL_RATE,        "deg/s", 1.0, 0.0),
        "50": (CH_PITCH_RATE,       "deg/s", 1.0, 0.0),
        # Device-4 coolant/oil temps are Fahrenheit, not Celsius: the
        # 2026-02-19 pack reads 154 ~ 201 (10026) and 144 ~ 246 (10066),
        # i.e. exactly the device-12 Celsius ranges (67 ~ 95 C, 57 ~ 119 C)
        # through x9/5+32. Mapping them as C would emit 200 C coolant.
        "10026": (CH_COOLANT_TEMP,    "C", _f_to_c, _f_to_c_offset),
        "10066": (CH_ENGINE_OIL_TEMP, "C", _f_to_c, _f_to_c_offset),
    }
    dev4_ts_key = _OBD_DEV4_TS_KEY
    dir_prefix = os.path.dirname(dev4_ts_key)
    if any(os.path.basename(n) == os.path.basename(dev4_ts_key) and
           os.path.dirname(n) == dir_prefix for n in namelist):
        actual_key = next(
            n for n in namelist
            if os.path.basename(n) == os.path.basename(_OBD_DEV4_TS_KEY)
        )
        obd4_epoch_ms = _decode_epoch_ms(read_channel(actual_key), time_origin_epoch_ms)
        if obd4_epoch_ms is not None:
            obd4_rel_times = _relative_epoch_seconds(obd4_epoch_ms, time_origin_epoch_ms)
            # Merge standard map with device-4-specific overrides
            dev4_map = {**RCZ_PID_MAP, **{
                k: v for k, v in rcz_dev4_pid_overrides.items() if v is not None
            }}
            for pid in sorted(set(list(RCZ_PID_MAP.keys()) + list(rcz_dev4_pid_overrides.keys())), key=str):
                override = rcz_dev4_pid_overrides.get(pid, "NOT_OVERRIDDEN")
                if override is None:
                    continue  # explicitly skipped for device 4
                mapping = dev4_map.get(pid)
                if mapping is None:
                    continue
                ch_name, ch_unit, ch_scale, ch_offset = mapping
                # Skip PIDs that use the GPS speed channel (already parsed above)
                if pid == "4":
                    continue
                val_fname = f"channel2_4_101_0_{pid}_3"
                val_key = next(
                    (n for n in namelist if os.path.basename(n) == val_fname),
                    None,
                )
                if val_key is None:
                    continue
                value_data = np.frombuffer(read_channel(val_key), dtype="<f8")
                count = min(len(obd4_epoch_ms), len(value_data))
                if count < 2:
                    continue
                rel_t = obd4_rel_times[:count]
                vals = value_data[:count]
                if pid == "1004":
                    interpolated = _interp_zoh(times_sec, rel_t, vals)
                else:
                    interpolated = np.interp(times_sec, rel_t, vals)
                    if mask_interp_gaps:
                        interpolated = _mask_interp_gaps(interpolated, times_sec, rel_t)
                interpolated = _mask_interp_outside_range(interpolated, times_sec, rel_t)
                processed = interpolated * ch_scale + ch_offset
                if pid == "1004":
                    # Keep missing gear samples missing, not neutral.
                    processed = np.round(processed).clip(-1, 6)
                if ch_name not in data_log.channels:
                    populate_channel(ch_name, ch_unit, processed)


def parse_rcz_log(data_log, rcz_file_path, target_lap=None, target_stint=None,
                  min_lap_sec=15.0, mask_interp_gaps=False,
                  target_session=None):
    """ Creates channels populated with messages directly from a RaceChrono .rcz archive.

    rcz_file_path: Path to the .rcz file
    target_lap: String, int, or None. If None or 'all', processes all laps in the session.
    target_session: Nested session directory ID in a RaceChrono backup, or None.
    """
    import functools
    import os
    import zipfile

    data_log.clear()
    data_log.laps_info = {}

    min_lap_sec = float(min_lap_sec)
    if not np.isfinite(min_lap_sec) or min_lap_sec <= 0:
        raise ValueError("min_lap_sec must be a positive finite number")

    if not os.path.isfile(rcz_file_path):
        print(f"ERROR: RCZ file {rcz_file_path} does not exist")
        return

    with zipfile.ZipFile(rcz_file_path, "r") as z:
        all_names = z.namelist()
        prefix, session_prefix, namelist = _resolve_rcz_prefix(z, target_session, target_stint)
        read_channel = functools.partial(_read_rcz_entry, z, prefix)

        session_data = _load_rcz_session(z, all_names, session_prefix)
        if session_data is None:
            return
        session_json, first_t = session_data

        _load_rcz_traps(data_log, z, all_names, session_prefix, session_json)
        _apply_rcz_session_meta(data_log, session_json, rcz_file_path)

        active_stint = _select_rcz_stint(target_stint)
        fragment = _load_rcz_fragment(z, all_names, prefix)
        decl = _parse_rcz_fragment_decl(fragment)
        if fragment is not None:
            _apply_rcz_fragment_meta(data_log, decl)
        epoch_reference = session_json.get("firstTimestamp") or session_json.get("timeCreated")
        time_file, timestamps_ms = _resolve_rcz_clock(
            read_channel, namelist, epoch_reference,
            declared_gps=decl.get("gps"),
        )
        if not time_file or timestamps_ms is None:
            raise TelemetryParseError(
                "Could not find timestamp channel in RCZ file",
                code="RCZ_TIMESTAMP_MISSING",
                stage="parse.timebase",
                source=rcz_file_path,
            )

        times_sec, n_samples, time_origin_epoch_ms = _init_rcz_timebase(data_log, timestamps_ms)

        # Resolve the GPS channels before reconstructing laps so a recording
        # that starts at speed can be identified as a partial out lap.
        gps_prefix, gps_speed_values = _resolve_rcz_gps(
            data_log, read_channel, namelist, n_samples,
            declared_gps=decl.get("gps"),
        )

        reconstructed_laps, lap_numbers, stint_duration_s = _reconstruct_rcz_laps(
            session_json, active_stint, times_sec, timestamps_ms, first_t, min_lap_sec, gps_speed_values, n_samples,
        )

        selected_laps, mask, export_times_sec, sample_times, target_lap_int = _select_rcz_target_lap(
            data_log, reconstructed_laps, lap_numbers, times_sec, timestamps_ms, n_samples, target_lap,
        )

        _build_rcz_laps_info(data_log, selected_laps, target_lap_int, stint_duration_s)
        populate_channel = functools.partial(_populate_rcz_channel, data_log, n_samples, mask, sample_times)
        # Lap Number Channel
        populate_channel(CH_LAP_NUMBER, "", lap_numbers, 0)
        # Running Time Channel
        populate_channel(CH_RUNNING_TIME, "s", export_times_sec, 2)
        _populate_rcz_gps(data_log, namelist, gps_prefix, gps_speed_values, n_samples, read_channel, populate_channel)
        _populate_rcz_gps_aux(data_log, namelist, gps_prefix, n_samples, read_channel, populate_channel)

        _populate_rcz_magnetometer(data_log, namelist, times_sec, n_samples, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel)

        _populate_rcz_imu(data_log, namelist, times_sec, n_samples, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel,
                          declared_accel=decl.get("accel"), declared_gyro=decl.get("gyro"))

        _populate_rcz_obd12(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel)

        _populate_rcz_obd4(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel)

        _validate_uptime_clocks(data_log, namelist, read_channel)

        _warn_rcz_unknown_suffixes(namelist)

    # Fallback for Chassis Yaw Rate if missing
    derive_yaw_rate_from_gps_heading(data_log)
