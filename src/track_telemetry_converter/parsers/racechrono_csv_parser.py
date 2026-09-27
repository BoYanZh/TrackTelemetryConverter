"""Parser for RACECHRONO telemetry log files.

Extracted from the original DataLog.racechrono_log methods with identical behavior."""
from __future__ import annotations

import math

from ..channels import (
    CH_BRAKE_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_COOLANT_TEMP,
    CH_CORR_DIST,
    CH_DEVICE_BATTERY,
    CH_ENGINE_OIL_PRESS,
    CH_ENGINE_OIL_TEMP,
    CH_ENGINE_RPM,
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
    CH_RUNNING_TIME,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_VEHICLE_SPEED,
    CH_WHEEL_SPEED_FL,
    CH_WHEEL_SPEED_FR,
    CH_WHEEL_SPEED_RL,
    CH_WHEEL_SPEED_RR,
    CH_YAW_RATE,
)
from ..derived import derive_yaw_rate_from_gps_heading


def parse_racechrono_log(data_log, log_lines, target_lap=None):
    """ Creates channels populated with messages from a RaceChrono CSV log file.

    This maps standard RaceChrono columns to MoTeC-compatible names and units.

    log_lines: List, containing CSV log lines
    target_lap: String, int, or None. If None or 'all', processes all laps in the session.
    """
    data_log.clear()
    data_log.laps_info = {}
    file_p = getattr(data_log, "log_file_path", "")
    data_log.datetime = data_log._extract_datetime_from_text(log_lines, file_p)
    data_log._extract_metadata(log_lines, file_p)

    if not log_lines:
        return

    # Find the table header row (typically starts with "Time (s)", "timestamp", or "Time" and has > 2 columns)
    header_idx = -1
    import csv
    for i, line in enumerate(log_lines):
        line_clean = line.strip().strip('"').strip("'")
        if not line_clean:
            continue
        try:
            parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
        except Exception:
            parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
        if parts and len(parts) > 2:
            first_col = parts[0].lower()
            if first_col in ["time (s)", "timestamp", "time"]:
                header_idx = i
                break

    if header_idx == -1:
        print("ERROR: Could not find 'Time (s)', 'timestamp', or 'Time' column in CSV log.")
        return

    header = log_lines[header_idx].strip("\n")
    # Split and strip any quotes from header names
    raw_headers = [h.strip().strip('"').strip("'") for h in header.split(",")]

    # Check for unit row (row right after header row if it contains units like "s", "mph", "km/h")
    unit_row = []
    if header_idx + 1 < len(log_lines):
        next_line = log_lines[header_idx + 1].strip("\n")
        splits = [s.strip().strip('"').strip("'") for s in next_line.split(",")]
        if splits and splits[0].lower() in ["s", "sec", "seconds", ""]:
            unit_row = splits

    channel_names = raw_headers[1:]

    # Check if yaw_rate exists, otherwise check for y_rate_of_rotation / z_rate_of_rotation / GPS Gyro
    has_yaw = any(h.lower() in ["yaw_rate", "gps gyro", "chassis yaw rate"] for h in channel_names)

    # Base mapping from typical RaceChrono / AiM / MoTeC columns to MoTeC terminology and units.
    rc_to_motec_map = {
        "lap_number": {"name": CH_LAP_NUMBER, "units": ""},
        "elapsed_time": {"name": CH_RUNNING_TIME, "units": "s"},
        "distance_traveled": {"name": CH_CORR_DIST, "units": "m"},
        "accuracy": {"name": CH_GPS_ACCURACY, "units": "m"},
        "altitude": {"name": CH_GPS_ALTITUDE, "units": "m"},
        "bearing": {"name": CH_GPS_HEADING, "units": "deg"},
        "device_battery_level": {"name": CH_DEVICE_BATTERY, "units": "%"},
        "fix_type": {"name": CH_GPS_FIX, "units": ""},
        "latitude": {"name": CH_GPS_LATITUDE, "units": "deg"},
        "longitude": {"name": CH_GPS_LONGITUDE, "units": "deg"},
        "satellites": {"name": CH_GPS_SATS, "units": ""},
        "speed": {"name": CH_GROUND_SPEED, "units": "km/h"},
        "combined_acc": {"name": CH_G_COMBINED, "units": "G"},
        "lateral_acc": {"name": CH_CG_ACCEL_LAT, "units": "G"},
        "lean_angle": {"name": CH_LEAN_ANGLE, "units": "deg"},
        "longitudinal_acc": {"name": CH_CG_ACCEL_LON, "units": "G"},
        "accelerator_pos": {"name": CH_THROTTLE_POS, "units": "%"},
        "brake_pos": {"name": CH_BRAKE_POS, "units": "%"},
        "brake_pressure": {"name": CH_BRAKE_PRESS, "units": "kPa"},
        "coolant_temp": {"name": CH_COOLANT_TEMP, "units": "C"},
        "engine_oil_temp": {"name": CH_ENGINE_OIL_TEMP, "units": "C"},
        "rpm": {"name": CH_ENGINE_RPM, "units": "rpm"},
        "steering_angle": {"name": CH_STEERING_ANGLE, "units": "deg"},
        "yaw_rate": {"name": CH_YAW_RATE, "units": "deg/s"},
        "gear": {"name": CH_GEAR, "units": ""},
        "gear_position": {"name": CH_GEAR, "units": ""},
        # AiM Solo / RaceStudio CSV Mappings
        "gps speed": {"name": CH_GROUND_SPEED, "units": "km/h"},
        "gps latacc": {"name": CH_CG_ACCEL_LAT, "units": "G"},
        "gps lonacc": {"name": CH_CG_ACCEL_LON, "units": "G"},
        "gps gyro": {"name": CH_YAW_RATE, "units": "deg/s"},
        "gps lat": {"name": CH_GPS_LATITUDE, "units": "deg"},
        "gps lon": {"name": CH_GPS_LONGITUDE, "units": "deg"},
        "gps altitude": {"name": CH_GPS_ALTITUDE, "units": "m"},
        "gps heading": {"name": CH_GPS_HEADING, "units": "deg"},
        "gps posaccuracy": {"name": CH_GPS_ACCURACY, "units": "m"},
        "pps": {"name": CH_THROTTLE_POS, "units": "%"},
        "steerangle": {"name": CH_STEERING_ANGLE, "units": "deg"},
        "brakepress": {"name": CH_BRAKE_PRESS, "units": "kPa"},
        "oiltemp": {"name": CH_ENGINE_OIL_TEMP, "units": "C"},
        "ect": {"name": CH_COOLANT_TEMP, "units": "C"},
        "intake air temp": {"name": CH_INTAKE_TEMP, "units": "C"},
        "oilpressure0": {"name": CH_ENGINE_OIL_PRESS, "units": "kPa"},
        "yawrate": {"name": CH_YAW_RATE, "units": "deg/s"},
        "wheelspeedfl": {"name": CH_WHEEL_SPEED_FL, "units": "km/h"},
        "wheelspeedfr": {"name": CH_WHEEL_SPEED_FR, "units": "km/h"},
        "wheelspeedrl": {"name": CH_WHEEL_SPEED_RL, "units": "km/h"},
        "wheelspeedrr": {"name": CH_WHEEL_SPEED_RR, "units": "km/h"},
        "speedv": {"name": CH_VEHICLE_SPEED, "units": "km/h"},
        # MoTeC CSV Mappings
        "corr speed": {"name": CH_GROUND_SPEED, "units": "km/h"},
        "gps latitude": {"name": CH_GPS_LATITUDE, "units": "deg"},
        "gps longitude": {"name": CH_GPS_LONGITUDE, "units": "deg"},
        "corr dist": {"name": CH_CORR_DIST, "units": "m"},
        # RaceChrono dynamic-column exports. These source-qualified names exceed
        # MoTeC LD's 32-byte channel-name field, so map the measured CAN values
        # to their standard names before writing.
        "longitudinal acceleration (g) *canbus": {"name": CH_CG_ACCEL_LON, "units": "G"},
        "combined acceleration (g) *canbus": {"name": CH_G_COMBINED, "units": "G"},
        "engine oil temperature (.c) *canbus": {"name": CH_ENGINE_OIL_TEMP, "units": "C"},
    }

    # Fallback for yaw rate if not named "yaw_rate" or "gps gyro"
    if not has_yaw:
        if "z_rate_of_rotation" in channel_names:
            rc_to_motec_map["z_rate_of_rotation"] = {"name": CH_YAW_RATE, "units": "deg/s"}
        elif "y_rate_of_rotation" in channel_names:
            rc_to_motec_map["y_rate_of_rotation"] = {"name": CH_YAW_RATE, "units": "deg/s"}

    # Explicitly ignore uncalibrated raw IMU channels
    ignored_columns = {
        "x acc",
        "y acc",
        "z acc",
        # Prefer the measured CAN value below over RaceChrono's calculated
        # longitudinal acceleration. Their LD-truncated names collide.
        "longitudinal acceleration (g) *calc",
    }

    active_columns = []
    lap_number_idx = -1
    col_unit_map = {}

    for i, raw_name in enumerate(channel_names):
        raw_lower = raw_name.lower().strip().replace("_", " ").replace("-", " ")
        if not raw_name or raw_lower in ignored_columns:
            continue

        if raw_lower == "lap_number" or raw_lower == "lap":
            lap_number_idx = i + 1

        unit_str = ""
        if unit_row and i + 1 < len(unit_row):
            unit_str = unit_row[i + 1]

        if raw_lower in rc_to_motec_map:
            motec_name = rc_to_motec_map[raw_lower]["name"]
            motec_units = rc_to_motec_map[raw_lower]["units"]
        else:
            # Fallback to general parsing if no exact map is found
            if " (" in raw_name and raw_name.endswith(")"):
                motec_name, motec_units = raw_name.rsplit(" (", 1)
                motec_units = motec_units[:-1]
            else:
                motec_name = raw_name
                motec_units = unit_str

        col_unit_map[raw_name] = unit_str.lower()

        if motec_name not in data_log.channels:
            data_log.add_channel(motec_name, motec_units, float, 0)
            active_columns.append((i, motec_name, raw_name))

    # First pass to parse all valid data rows
    valid_lines = []
    for line in log_lines[header_idx + 1:]:
        line_clean = line.strip("\n")
        if not line_clean:
            continue

        # Skip unit rows (typically right after headers and start with empty timestamp or strings)
        values = line_clean.split(",")
        if not values[0]:
            continue

        try:
            float(values[0].strip().strip('"').strip("'"))
        except ValueError:
            continue

        valid_lines.append(values)

    # Parse target_lap filter if specified
    target_lap_str = None
    if target_lap is not None and str(target_lap).lower() != "all":
        target_lap_str = str(target_lap).strip()

    # Track lap timing metadata & channel sample buffers
    laps_timing = {}
    chan_buffers = {name: ([], []) for _, name, _ in active_columns}

    for values in valid_lines:
        if lap_number_idx != -1 and lap_number_idx < len(values):
            lap_val = values[lap_number_idx].strip().strip('"').strip("'")
            if target_lap_str and lap_val != target_lap_str:
                continue
        else:
            lap_val = "1"

        try:
            t = float(values[0].strip().strip('"').strip("'"))
        except ValueError:
            continue

        # Record lap start/end times
        if lap_val not in laps_timing:
            laps_timing[lap_val] = {"start_time": t, "end_time": t, "messages_count": 0}
        laps_timing[lap_val]["end_time"] = t
        laps_timing[lap_val]["messages_count"] += 1

        for col_idx, name, raw_name in active_columns:
            if col_idx + 1 >= len(values):
                continue

            val_str = values[col_idx + 1].strip().strip('"').strip("'")
            if not val_str:
                continue

            try:
                val = float(val_str)

                # Convert speed based on unit or column name
                raw_lower = raw_name.lower().strip()
                unit_lower = col_unit_map.get(raw_name, "")
                if raw_lower == "speed":
                    val *= 3.6
                elif unit_lower == "mph":
                    val *= 1.60934
                elif unit_lower == "psi" and name == CH_BRAKE_PRESS:
                    val *= 6.89476
                elif unit_lower == "f":
                    val = (val - 32.0) * (5.0 / 9.0)
                elif unit_lower == "bar" and "press" in name.lower():
                    val *= 100.0
                elif raw_lower in ("yaw_rate", "z_rate_of_rotation", "y_rate_of_rotation") and name == CH_YAW_RATE:
                    val *= -1.0

                ts_buf, val_buf = chan_buffers[name]
                ts_buf.append(t)
                val_buf.append(val)

                val_text_split = val_str.split(".")
                decimals_present = 0 if len(val_text_split) == 1 else len(val_text_split[1])
                data_log.channels[name].decimals = max(decimals_present, data_log.channels[name].decimals)
            except ValueError:
                pass

    for name, (ts_buf, val_buf) in chan_buffers.items():
        if ts_buf:
            data_log.channels[name].set_samples(ts_buf, val_buf)

    # Fallback: derive Chassis Yaw Rate from GPS Heading derivative if missing
    derive_yaw_rate_from_gps_heading(data_log)

    # Construct laps_info summary (with Out Lap and In Lap)
    header_beacons = []
    import csv
    for line in log_lines[:header_idx]:
        line_clean = line.strip().strip('"').strip("'")
        try:
            parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
        except Exception:
            parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
        if parts and parts[0].lower() == "beacon markers":
            for val in parts[1:]:
                try:
                    header_beacons.append(float(val))
                except ValueError:
                    pass

    if header_beacons:
        lap_items = []
        fastest_lap = None
        fastest_dur = math.inf
        start_t = 0.0
        total_beacons = len(header_beacons)
        for idx, btime in enumerate(header_beacons):
            dur = btime - start_t
            if dur < 1.0:
                start_t = btime
                continue

            if idx == 0:
                if dur > 15.0:
                    lap_items.append({
                        "type": "Out Lap",
                        "lap_label": "Out Lap",
                        "lap_num": len(lap_items) + 1,
                        "start_time": start_t,
                        "end_time": btime,
                        "duration": dur,
                        "stint": 0
                    })
            elif idx == total_beacons - 1:
                if dur > 15.0:
                    lap_items.append({
                        "type": "In Lap",
                        "lap_label": "In Lap",
                        "lap_num": len(lap_items) + 1,
                        "start_time": start_t,
                        "end_time": btime,
                        "duration": dur,
                        "stint": 0
                    })
            else:
                if dur >= 15.0:
                    lap_num = len(lap_items) + 1
                    lap_items.append({
                        "type": "Timed",
                        "lap_label": str(lap_num),
                        "lap_num": lap_num,
                        "start_time": start_t,
                        "end_time": btime,
                        "duration": dur,
                        "stint": 0
                    })
                    if dur < fastest_dur:
                        fastest_dur = dur
                        fastest_lap = lap_num
            start_t = btime

        data_log.laps_info = {
            "laps": lap_items,
            "total_laps": len(lap_items),
            "fastest_lap": fastest_lap if fastest_lap is not None else 1,
            "fastest_time": fastest_dur if fastest_dur != math.inf else 0.0
        }
    elif laps_timing:
        lap_items = []
        fastest_lap = None
        fastest_dur = math.inf

        def lap_key(k):
            try:
                return (0, int(k), "")
            except ValueError:
                return (1, 0, str(k))

        sorted_lap_keys = sorted(laps_timing.keys(), key=lap_key)

        # Determine base start time and total duration using relative timestamps
        first_key = sorted_lap_keys[0]
        base_time = laps_timing[first_key]["start_time"]

        total_duration = 0.0
        if data_log.channels:
            any_ch = next(iter(data_log.channels.values()))
            if any_ch.messages:
                total_duration = any_ch.messages[-1].timestamp - any_ch.messages[0].timestamp

        # Convert all lap start/end times to relative time from session start (0.0s)
        rel_laps = []
        for k in sorted_lap_keys:
            info = laps_timing[k]
            s_rel = info["start_time"] - base_time
            e_rel = info["end_time"] - base_time
            dur = e_rel - s_rel
            rel_laps.append((k, s_rel, e_rel, dur))

        # Filter valid timed laps (duration >= 20.0s)
        valid_timed = [item for item in rel_laps if item[3] >= 20.0]

        if not valid_timed:
            # Fallback: treat entire session as 1 timed lap
            lap_items.append({
                "type": "Timed",
                "lap_label": "1",
                "lap_num": 1,
                "start_time": 0.0,
                "end_time": total_duration,
                "duration": total_duration,
                "stint": 0
            })
        else:
            # 1. Out Lap (if leading non-timed padding > 15.0s)
            first_start = valid_timed[0][1]
            if first_start > 15.0:
                lap_items.append({
                    "type": "Out Lap",
                    "lap_label": "Out Lap",
                    "lap_num": 1,
                    "start_time": 0.0,
                    "end_time": first_start,
                    "duration": first_start,
                    "stint": 0
                })

            # 2. Timed Laps
            for k, s, e, dur in valid_timed:
                lap_num = len(lap_items) + 1
                lap_items.append({
                    "type": "Timed",
                    "lap_label": str(k),
                    "lap_num": lap_num,
                    "start_time": s,
                    "end_time": e,
                    "duration": dur,
                    "stint": 0
                })
                if dur < fastest_dur:
                    fastest_dur = dur
                    fastest_lap = lap_num

            # 3. In Lap (if trailing non-timed padding > 15.0s)
            last_end = valid_timed[-1][2]
            if total_duration - last_end > 15.0:
                in_dur = total_duration - last_end
                lap_items.append({
                    "type": "In Lap",
                    "lap_label": "In Lap",
                    "lap_num": len(lap_items) + 1,
                    "start_time": last_end,
                    "end_time": total_duration,
                    "duration": in_dur,
                    "stint": 0
                })

        data_log.laps_info = {
            "laps": lap_items,
            "total_laps": len(lap_items),
            "fastest_lap": fastest_lap if fastest_lap is not None else 1,
            "fastest_time": fastest_dur if fastest_dur != math.inf else 0.0
        }

    # Cleanup channels without any data
    empty_channels = [name for name, ch in data_log.channels.items() if not ch.messages]
    for name in empty_channels:
        del data_log.channels[name]
