"""Parser for Tesla Track Mode CSV telemetry log files (telemetry-v1-*.csv).

The car's USB export writes one header row and unquoted data rows at ~42 Hz.
Two format facts drive this parser:

- `Elapsed Time (ms)` resets to ~0 at every lap start, so the cumulative
  session timeline is rebuilt by accumulating each lap's max elapsed time.
- Columns are fixed Tesla vocabulary: MPH speeds, m/s^2 accelerations,
  rad/s yaw rate, bar brake pressure, percent-estimate thermals/slip.
  (A newer 33-column "render" variant with extra PSI columns is passed
  through unconverted rather than merged into wrong units.)

Limitations: the file carries no wall-clock date (datetime stays None) and
no venue string (venue resolves through filename/GPS fallback as usual).
"""

from __future__ import annotations

from ..channels import (
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LON,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GROUND_SPEED,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_TIRE_PRESS_FL,
    CH_TIRE_PRESS_FR,
    CH_TIRE_PRESS_RL,
    CH_TIRE_PRESS_RR,
    CH_YAW_RATE,
)
from .racechrono_csv_parser import _build_timing_laps

_MPH_TO_KMH = 1.609344
_G = 9.80665
_RAD_TO_DEG = 57.29578
_BAR_TO_KPA = 100.0

# Header -> (canonical channel, unit, decimals, factor, offset).
# value = (raw + offset) * factor.
_TESLA_COLUMNS = {
    "speed (mph)": (CH_GROUND_SPEED, "km/h", 2, _MPH_TO_KMH, 0.0),
    "latitude (decimal)": (CH_GPS_LATITUDE, "deg", 7, 1.0, 0.0),
    "longitude (decimal)": (CH_GPS_LONGITUDE, "deg", 7, 1.0, 0.0),
    "lateral acceleration (m/s^2)": (CH_CG_ACCEL_LAT, "G", 3, 1.0 / _G, 0.0),
    "longitudinal acceleration (m/s^2)": (CH_CG_ACCEL_LON, "G", 3, 1.0 / _G, 0.0),
    "throttle position (%)": (CH_THROTTLE_POS, "%", 2, 1.0, 0.0),
    "brake pressure (bar)": (CH_BRAKE_PRESS, "kPa", 2, _BAR_TO_KPA, 0.0),
    "steering angle (deg)": (CH_STEERING_ANGLE, "deg", 2, 1.0, 0.0),
    "steering angle rate (deg/s)": ("Steering Angle Rate", "deg/s", 2, 1.0, 0.0),
    "yaw rate (rad/s)": (CH_YAW_RATE, "deg/s", 3, _RAD_TO_DEG, 0.0),
    "power level (kw)": ("Power Level", "kW", 2, 1.0, 0.0),
    "state of charge (%)": ("State of Charge", "%", 2, 1.0, 0.0),
    "tire pressure front left (bar)": (CH_TIRE_PRESS_FL, "kPa", 2, _BAR_TO_KPA, 0.0),
    "tire pressure front right (bar)": (CH_TIRE_PRESS_FR, "kPa", 2, _BAR_TO_KPA, 0.0),
    "tire pressure rear left (bar)": (CH_TIRE_PRESS_RL, "kPa", 2, _BAR_TO_KPA, 0.0),
    "tire pressure rear right (bar)": (CH_TIRE_PRESS_RR, "kPa", 2, _BAR_TO_KPA, 0.0),
    "brake temperature front left (% est.)": ("Brake Temp FL", "%", 2, 100.0, 0.0),
    "brake temperature front right (% est.)": ("Brake Temp FR", "%", 2, 100.0, 0.0),
    "brake temperature rear left (% est.)": ("Brake Temp RL", "%", 2, 100.0, 0.0),
    "brake temperature rear right (% est.)": ("Brake Temp RR", "%", 2, 100.0, 0.0),
    "front inverter temp (%)": ("Front Inverter Temp", "%", 2, 100.0, 0.0),
    "rear inverter temp (%)": ("Rear Inverter Temp", "%", 2, 100.0, 0.0),
    "battery temp (%)": ("Battery Temp", "%", 2, 100.0, 0.0),
    "tire slip front left (% est.)": ("Tire Slip FL", "%", 2, 100.0, 0.0),
    "tire slip front right (% est.)": ("Tire Slip FR", "%", 2, 100.0, 0.0),
    "tire slip rear left (% est.)": ("Tire Slip RL", "%", 2, 100.0, 0.0),
    "tire slip rear right (% est.)": ("Tire Slip RR", "%", 2, 100.0, 0.0),
}


def _find_tesla_header(log_lines):
    """Return the index of the `Lap,Elapsed Time (ms),...` header row, or -1."""
    for i, line in enumerate(log_lines):
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) > 2 and parts[0] == "Lap" and parts[1] == "Elapsed Time (ms)":
            return i
    print("ERROR: Could not find Tesla 'Lap,Elapsed Time (ms),...' header.")
    return -1


def _build_columns(data_log, headers):
    """Create channels; return (col_specs, lap_idx, et_idx)."""
    try:
        lap_idx = next(i for i, h in enumerate(headers) if h.strip() == "Lap")
        et_idx = next(i for i, h in enumerate(headers) if h.strip() == "Elapsed Time (ms)")
    except StopIteration:
        return [], -1, -1
    col_specs = []
    for i, raw in enumerate(headers):
        if i in (lap_idx, et_idx):
            continue
        spec = _TESLA_COLUMNS.get(raw.strip().lower())
        if spec is None:
            out_name, out_unit, dec, factor, offset = (raw.strip(), "", 2, 1.0, 0.0)
        else:
            out_name, out_unit, dec, factor, offset = spec
        if out_name and out_name not in data_log.channels:
            data_log.add_channel(out_name, out_unit, float, dec)
        if out_name:
            col_specs.append((i, data_log.channels[out_name], factor, offset))
    return col_specs, lap_idx, et_idx


def _parse_data_rows(data_log, log_lines, header_idx, col_specs, lap_idx, et_idx, target_lap_str):
    """Rebuild the cumulative timeline across per-lap elapsed-time resets."""
    chan_buffers = [([], []) for _ in col_specs]
    laps_timing = {}
    lap_base = {}
    last_t = 0.0
    for line in log_lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        values = stripped.split(",")
        if max(lap_idx, et_idx) >= len(values):
            continue
        lap_val = values[lap_idx].strip()
        try:
            et_s = float(values[et_idx].strip()) / 1000.0
        except ValueError:
            continue
        if lap_val not in lap_base:
            lap_base[lap_val] = last_t
        t = lap_base[lap_val] + et_s
        # Firmware glitches (notably Intel-Atom double-logging) can emit
        # non-monotone timestamps; clamp instead of going backwards.
        if t < last_t:
            t = last_t
        last_t = max(last_t, t)
        if lap_val:
            info = laps_timing.get(lap_val)
            if info is None:
                laps_timing[lap_val] = {"start_time": t, "end_time": t, "messages_count": 1}
            else:
                info["end_time"] = t
                info["messages_count"] += 1
        if target_lap_str is not None and lap_val != target_lap_str:
            continue
        for k, (val_idx, _, factor, offset) in enumerate(col_specs):
            if val_idx >= len(values):
                continue
            s = values[val_idx].strip()
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
    return laps_timing


def parse_tesla_log(data_log, log_lines, target_lap=None):
    """Creates channels populated with messages from a Tesla Track Mode CSV log.

    Lap 0 (out/pit) is relabeled from Timed to Out Lap; timed laps follow
    their Lap numbers.
    """
    import math

    data_log.clear()
    data_log.laps_info = {}
    data_log.datetime = None

    if not log_lines:
        return
    header_idx = _find_tesla_header(log_lines)
    if header_idx == -1:
        return

    headers = [h.strip() for h in log_lines[header_idx].strip().split(",")]
    col_specs, lap_idx, et_idx = _build_columns(data_log, headers)
    if lap_idx == -1 or et_idx == -1:
        return

    target_lap_str = None
    if target_lap is not None and str(target_lap).lower() != "all":
        target_lap_str = str(target_lap).strip()

    laps_timing = _parse_data_rows(data_log, log_lines, header_idx, col_specs,
                                   lap_idx, et_idx, target_lap_str)
    if target_lap_str is not None and target_lap_str not in laps_timing:
        print(f"WARNING: Lap '{target_lap}' not found in Tesla file; keeping all laps")

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
