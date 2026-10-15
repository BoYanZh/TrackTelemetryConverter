"""Parser for RCZ telemetry log files.

Extracted from the original DataLog.rcz_log methods with identical behavior."""

from __future__ import annotations

import datetime

import numpy as np

from ..channels import (
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_CG_ACCEL_VERT,
    CH_GPS_ALTITUDE,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GROUND_SPEED,
    CH_LAP_NUMBER,
    CH_RUNNING_TIME,
    CH_YAW_RATE,
    RCZ_PID_MAP,
)
from ..derived import derive_yaw_rate_from_gps_heading
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
        print("ERROR: Invalid RCZ file, missing session.json")
        return

    session_json = json.loads(z.read(session_key).decode("utf-8"))
    first_t = session_json.get("firstTimestamp", 0)
    return session_json, first_t




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
            data_log.traps.append({
                "name": trap.get("name", "Split"),
                "lat": trap["centerLatitude"] / 6000000.0,
                "lon": trap["centerLongitude"] / 6000000.0,
                "type": trap.get("type", 4),
            })

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




def _resolve_rcz_clock(read_channel, namelist, epoch_reference):
    """Resolve the GPS timestamp channel; return (time_file, timestamps_ms)."""
    time_file = None
    timestamps_ms = None
    for candidate in _GPS_TS_KEYS:
        if candidate in namelist:
            decoded = _decode_epoch_ms(read_channel(candidate), epoch_reference)
            if decoded is not None:
                time_file = candidate
                timestamps_ms = decoded
                break

    if not time_file:
        for name in namelist:
            if name.startswith("channel_") and name.endswith("_1_1"):
                decoded = _decode_epoch_ms(read_channel(name), epoch_reference)
                if decoded is not None:
                    time_file = name
                    timestamps_ms = decoded
                    break

    return time_file, timestamps_ms

def _init_rcz_timebase(data_log, timestamps_ms):
    """Set the absolute origin; return (times_sec, n_samples, time_origin_epoch_ms)."""
    time_origin_epoch_ms = int(timestamps_ms[0])
    _set_absolute_origin(data_log, time_origin_epoch_ms)
    times_sec = _relative_epoch_seconds(timestamps_ms, time_origin_epoch_ms)
    n_samples = len(times_sec)
    return times_sec, n_samples, time_origin_epoch_ms




def _resolve_rcz_gps(data_log, read_channel, namelist, n_samples):
    """Identify the GPS sensor family and read speed values."""
    gps_prefix = None
    for candidate_prefix in _GPS_DEV_PFX:
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

    # Filter raw laps that actually overlap with binary recording range
    overlapping_laps = []
    for lap in stint_laps:
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

def _populate_rcz_imu(data_log, namelist, times_sec, n_samples, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel):
    """Populate accelerometer/gyro channels with drift correction."""
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
            for device_type in ("301", "201", "101")
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
    for g_type in ("302", "202", "102"):
        gz_cand = f"channel_3_{g_type}_0_14_0"
        ts_cand = f"channel_3_{g_type}_0_1_1"
        if gz_cand in namelist:
            gyro_z_file = gz_cand
            gyro_ts_file = ts_cand
            break

    if gyro_z_file:
        # Device-3 gyroscope Z has the opposite sign from the
        # positive vehicle/CAN yaw convention used for this channel.
        _parse_imu_channel(gyro_z_file, CH_YAW_RATE, "deg/s",
                           -1.0 / 1000.0, ts_key=gyro_ts_file)
        data_log.metadata["yaw_rate_source"] = f"rcz_gyro_3_{g_type}"

def _populate_rcz_obd12(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel):
    """Populate primary OBD (device 12) CAN channels."""
    import os
    # 8. Parse OBD-II / CAN Channels
    # RCZ stores binary channel values as contiguous IEEE 754 float64 (double precision) values.
    # Values are ALREADY in engineering units (1:1 scale).
    # Internal RCZ channel PID to MoTeC channel name & unit mapping:
    rcz_pid_map = RCZ_PID_MAP
    yaw_rate_from_can = False

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
            elif ch_name not in data_log.channels:
                vals_processed = values * ch_scale + ch_offset
                if pid == "1004":
                    # Do not turn unavailable gear samples into neutral.
                    vals_processed = np.round(vals_processed).clip(-1, 6)
                populate_channel(ch_name, ch_unit, vals_processed)
        else:
            populate_channel(f"OBD_{pid}", "", values)

def _populate_rcz_obd4(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel):
    """Populate secondary OBD (device 4/101, phone IMU flavored PIDs)."""
    import os
    # 8b. Parse OBD-II / CAN Channels - Device 4, Type 101 (shared timestamp format)
    # Format: channel_4_101_0_1_1 = shared i32[::2] uptime timestamps for all PIDs
    #         channel2_4_101_0_{pid}_3 = per-PID float64 values (same length as timestamps)
    # Device 4/type 101 uses phone-IMU channels for PIDs 7/8/49/50 - different from
    # the GR86-ECU meaning of the same PIDs on device 12/type 100.
    _g = 9.80665
    rcz_dev4_pid_overrides = {
        # PID 7 on device 4 = phone lateral accelerometer (m/s^2), NOT ECU Roll Angle
        "7":  (CH_CG_ACCEL_LAT,     "G",    1.0 / _g, 0.0),
        # PID 8 on device 4 = phone vertical accelerometer (m/s^2, includes gravity) - skip
        "8":  None,
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
            for pid in set(list(RCZ_PID_MAP.keys()) + list(rcz_dev4_pid_overrides.keys())):
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
        epoch_reference = session_json.get("firstTimestamp") or session_json.get("timeCreated")
        time_file, timestamps_ms = _resolve_rcz_clock(read_channel, namelist, epoch_reference)
        if not time_file or timestamps_ms is None:
            print("ERROR: Could not find timestamp channel in RCZ file")
            return

        times_sec, n_samples, time_origin_epoch_ms = _init_rcz_timebase(data_log, timestamps_ms)

        # Resolve the GPS channels before reconstructing laps so a recording
        # that starts at speed can be identified as a partial out lap.
        gps_prefix, gps_speed_values = _resolve_rcz_gps(data_log, read_channel, namelist, n_samples)

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

        _populate_rcz_imu(data_log, namelist, times_sec, n_samples, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel)

        _populate_rcz_obd12(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel)

        _populate_rcz_obd4(data_log, namelist, times_sec, time_origin_epoch_ms, mask_interp_gaps, read_channel, populate_channel)

    # Fallback for Chassis Yaw Rate if missing
    derive_yaw_rate_from_gps_heading(data_log)
