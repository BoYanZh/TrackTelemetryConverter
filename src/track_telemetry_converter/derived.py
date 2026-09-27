"""Vehicle dynamics & math channel derivation (extracted from DataLog methods)."""

from __future__ import annotations

import numpy as np

from .channels import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LAT,
    CH_CG_ACCEL_LAT_SMOOTH,
    CH_CG_ACCEL_LON,
    CH_CG_ACCEL_LON_SMOOTH,
    CH_G_COMBINED,
    CH_GPS_HEADING,
    CH_GROUND_SPEED,
    CH_STEERING_ANGLE,
    CH_THROTTLE_POS,
    CH_YAW_RATE,
)
from .models import Message

DEFAULT_GEAR_RATIO_THRESHOLDS = (110.0, 70.0, 52.0, 42.0, 33.0, 20.0)
def calculate_math_channels(data_log, g_source="auto", gear_ratio_thresholds=None):
    """
    g_source: "auto" (default), "sensor", or "calc"
      - "auto": Use raw IMU sensor G channels if present; otherwise derive from GPS.
      - "sensor": Only use raw IMU sensor G channels. Do not derive G from GPS.
      - "calc": Force deriving G channels from GPS Speed + Yaw Rate, overriding sensor channels.
    """
    derive_yaw_rate_from_gps_heading(data_log)

    if g_source == "calc":
        if CH_CG_ACCEL_LAT in data_log.channels:
            del data_log.channels[CH_CG_ACCEL_LAT]
        if CH_CG_ACCEL_LON in data_log.channels:
            del data_log.channels[CH_CG_ACCEL_LON]
        derive_cg_accel_lateral(data_log, force=True)
        derive_cg_accel_longitudinal(data_log, force=True)
    elif g_source == "sensor":
        pass
    else:  # "auto"
        derive_cg_accel_lateral(data_log, force=False)
        derive_cg_accel_longitudinal(data_log, force=False)

    derive_smoothed_accel(data_log)
    calculate_g_sum(data_log)
    derive_brake_pos(data_log)
    calculate_input_rates(data_log)
    mirror_throttle_accel(data_log)
    derive_gear_from_rpm_speed(data_log, ratio_thresholds=gear_ratio_thresholds)


def derive_gear_from_rpm_speed(data_log, ratio_thresholds=None):
    """Derive integer Gear (1-6) from timestamp-aligned RPM/speed ratios."""
    from .channels import CH_ENGINE_RPM, CH_GEAR, CH_GROUND_SPEED
    if CH_GEAR in data_log.channels:
        return
    if CH_ENGINE_RPM not in data_log.channels or CH_GROUND_SPEED not in data_log.channels:
        return
    rpm_chan = data_log.channels[CH_ENGINE_RPM]
    spd_chan = data_log.channels[CH_GROUND_SPEED]
    if len(rpm_chan.messages) < 2 or len(spd_chan.messages) < 2:
        return

    rpm_times = np.asarray(rpm_chan.timestamps, dtype=np.float64)
    rpm_values = np.asarray(rpm_chan.values, dtype=np.float64)
    order = np.argsort(rpm_times, kind="stable")
    rpm_times = rpm_times[order]
    rpm_values = rpm_values[order]
    keep = np.concatenate([rpm_times[1:] != rpm_times[:-1], [True]])
    rpm_times = rpm_times[keep]
    rpm_values = rpm_values[keep]
    if len(rpm_times) < 2:
        return

    speed_times = np.asarray(spd_chan.timestamps, dtype=np.float64)
    speed_values = np.asarray(spd_chan.values, dtype=np.float64)
    order = np.argsort(speed_times, kind="stable")
    speed_times = speed_times[order]
    speed_values = speed_values[order]
    keep = np.concatenate([speed_times[1:] != speed_times[:-1], [True]])
    speed_times = speed_times[keep]
    speed_values = speed_values[keep]
    overlap = (speed_times >= rpm_times[0]) & (speed_times <= rpm_times[-1])
    if np.count_nonzero(overlap) < 2:
        return

    t_arr = speed_times[overlap]
    spd_vals = speed_values[overlap].copy()
    rpm_vals = np.interp(t_arr, rpm_times, rpm_values)
    if spd_chan.units == "mph":
        spd_vals *= 1.60934
    elif spd_chan.units == "m/s":
        spd_vals *= 3.6

    thresholds = ratio_thresholds
    if thresholds is None:
        thresholds = data_log.metadata.get("gear_ratio_thresholds", DEFAULT_GEAR_RATIO_THRESHOLDS)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (6,) or not np.all(np.diff(thresholds) < 0):
        raise ValueError("gear ratio thresholds must contain six strictly descending values")

    ratios = np.zeros(len(t_arr), dtype=np.float64)
    mask = (spd_vals > 5.0) & (rpm_vals > 500.0)
    ratios[mask] = rpm_vals[mask] / spd_vals[mask]

    gear_vals = np.zeros(len(t_arr), dtype=np.float64)
    unassigned = mask.copy()
    for gear, threshold in enumerate(thresholds, start=1):
        selected = unassigned & (ratios > threshold)
        gear_vals[selected] = float(gear)
        unassigned[selected] = False

    data_log.add_channel(CH_GEAR, "", float, 0)
    data_log.channels[CH_GEAR].set_samples(t_arr, gear_vals)


def derive_smoothed_accel(data_log, window_sec=0.5):
    """ Derive 0.5s moving average smoothed G channels for clean G-G diagrams in MoTeC. """
    for raw_name, smooth_name in [(CH_CG_ACCEL_LAT, CH_CG_ACCEL_LAT_SMOOTH),
                                  (CH_CG_ACCEL_LON, CH_CG_ACCEL_LON_SMOOTH)]:
        if raw_name in data_log.channels and smooth_name not in data_log.channels:
            ch = data_log.channels[raw_name]
            if len(ch.messages) < 5:
                continue
            freq = ch.avg_frequency()
            w = max(1, int(window_sec * freq))
            vals = np.array([m.value for m in ch.messages], dtype=np.float64)

            padded = np.pad(vals, (w // 2, w - 1 - w // 2), mode="edge")
            windows = np.lib.stride_tricks.sliding_window_view(padded, w)
            smoothed = np.mean(windows, axis=1)[:len(vals)]

            data_log.add_channel(smooth_name, ch.units, float, ch.decimals)
            data_log.channels[smooth_name].messages = [
                Message(ch.messages[i].timestamp, float(smoothed[i])) for i in range(len(vals))
            ]


def derive_cg_accel_lateral(data_log, force=False):
    if not force and CH_CG_ACCEL_LAT in data_log.channels:
        return
    if CH_GROUND_SPEED not in data_log.channels or CH_YAW_RATE not in data_log.channels:
        return
    spd_chan = data_log.channels[CH_GROUND_SPEED]
    yaw_chan = data_log.channels[CH_YAW_RATE]
    n = len(spd_chan.messages)
    if n < 1:
        return
    t_arr = np.array([m.timestamp for m in spd_chan.messages])
    vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
    yr_rad = np.radians([m.value for m in yaw_chan.messages])
    ay = (vx_ms * yr_rad) / 9.80665
    data_log.add_channel(CH_CG_ACCEL_LAT, "G", float, 2)
    data_log.channels[CH_CG_ACCEL_LAT].messages = [Message(t_arr[i], ay[i]) for i in range(n)]


def derive_cg_accel_longitudinal(data_log, force=False):
    if not force and CH_CG_ACCEL_LON in data_log.channels:
        return
    if CH_GROUND_SPEED not in data_log.channels:
        return
    spd_chan = data_log.channels[CH_GROUND_SPEED]
    n = len(spd_chan.messages)
    if n < 2:
        return
    t_arr = np.array([m.timestamp for m in spd_chan.messages])
    vx_ms = np.array([m.value / 3.6 if spd_chan.units == "km/h" else m.value for m in spd_chan.messages])
    ax = np.gradient(vx_ms, t_arr) / 9.80665
    data_log.add_channel(CH_CG_ACCEL_LON, "G", float, 2)
    data_log.channels[CH_CG_ACCEL_LON].messages = [Message(t_arr[i], ax[i]) for i in range(n)]


def calculate_g_sum(data_log):
    if CH_CG_ACCEL_LON not in data_log.channels or CH_CG_ACCEL_LAT not in data_log.channels:
        return
    ax_msgs = data_log.channels[CH_CG_ACCEL_LON].messages
    ay_msgs = data_log.channels[CH_CG_ACCEL_LAT].messages
    n = min(len(ax_msgs), len(ay_msgs))
    if n < 2:
        return
    time_g = [ax_msgs[i].timestamp for i in range(n)]
    ax = np.array([ax_msgs[i].value for i in range(n)])
    ay_g = np.array([ay_msgs[i].value for i in range(n)])
    g_sum = np.sqrt(ax ** 2 + ay_g ** 2)
    data_log.add_channel(CH_G_COMBINED, "G", float, 2)
    data_log.channels[CH_G_COMBINED].messages = [Message(time_g[i], g_sum[i]) for i in range(n)]


def derive_brake_pos(data_log):
    if CH_BRAKE_PRESS not in data_log.channels or CH_BRAKE_POS in data_log.channels:
        return
    press_chan = data_log.channels[CH_BRAKE_PRESS]
    n = len(press_chan.messages)
    if n < 1:
        return
    time_p = [m.timestamp for m in press_chan.messages]
    press_vals = np.array([m.value for m in press_chan.messages])
    if press_chan.units == "bar":
        press_vals *= 100.0
    bpos = np.clip(press_vals / 96.0, 0.0, 100.0)
    data_log.add_channel(CH_BRAKE_POS, "%", float, 2)
    data_log.channels[CH_BRAKE_POS].messages = [Message(time_p[i], bpos[i]) for i in range(n)]


def calculate_input_rates(data_log):
    calculate_rate(data_log, CH_STEERING_ANGLE, "deg/s")
    calculate_rate(data_log, CH_THROTTLE_POS, "%/s")
    calculate_rate(data_log, CH_BRAKE_POS, "%/s")


def mirror_throttle_accel(data_log):
    if CH_THROTTLE_POS not in data_log.channels and CH_ACCELERATOR_POS in data_log.channels:
        src = data_log.channels[CH_ACCELERATOR_POS]
        data_log.add_channel(CH_THROTTLE_POS, "%", float, 2)
        data_log.channels[CH_THROTTLE_POS].messages = [Message(m.timestamp, m.value) for m in src.messages]


def calculate_rate(data_log, channel_name, unit):
    """ Helper to calculate the rate of change for a channel. """
    if channel_name in data_log.channels:
        chan = data_log.channels[channel_name]
        vals = np.array([m.value for m in chan.messages])
        times = np.array([m.timestamp for m in chan.messages])
        if len(times) < 2:
            return
        rate = np.zeros(len(times))
        dt = np.diff(times)
        dt[dt == 0] = 0.001  # Prevent div by zero
        rate[1:] = np.diff(vals) / dt

        new_name = channel_name + " Rate"
        data_log.add_channel(new_name, unit, float, 2)
        data_log.channels[new_name].messages = [Message(times[i], rate[i]) for i in range(len(times))]


def derive_yaw_rate_from_gps_heading(data_log):
    if CH_YAW_RATE in data_log.channels:
        return
    gps_h_chan = data_log.channels.get(CH_GPS_HEADING)
    if not gps_h_chan or len(gps_h_chan.messages) < 2:
        return
    times = np.array([m.timestamp for m in gps_h_chan.messages])
    headings = np.array([m.value for m in gps_h_chan.messages])
    h_unwrapped = np.unwrap(np.radians(headings))
    h_deg = np.degrees(h_unwrapped)
    yaw_rate_val = -np.gradient(h_deg, times)
    freq = gps_h_chan.avg_frequency()
    w = max(1, int(0.2 * freq))
    if len(yaw_rate_val) >= w and w > 1:
        padded = np.pad(yaw_rate_val, (w // 2, w - 1 - w // 2), mode="edge")
        yaw_rate_val = np.mean(np.lib.stride_tricks.sliding_window_view(padded, w), axis=1)[:len(yaw_rate_val)]
    spd_chan = data_log.channels.get(CH_GROUND_SPEED)
    if spd_chan and len(spd_chan.messages) == len(times):
        v_vals = np.array([m.value for m in spd_chan.messages])
        if spd_chan.units == "mph":
            v_vals *= 1.60934
        yaw_rate_val[v_vals < 5.0] = 0.0
    yaw_rate_val = np.clip(yaw_rate_val, -150.0, 150.0)
    data_log.add_channel(CH_YAW_RATE, "deg/s", float, 2)
    data_log.channels[CH_YAW_RATE].messages = [Message(times[i], yaw_rate_val[i]) for i in range(len(times))]
    data_log.metadata["yaw_rate_source"] = "gps_heading_derivative"


# Backward compatibility aliases
_derive_smoothed_accel = derive_smoothed_accel
_derive_cg_accel_lateral = derive_cg_accel_lateral
_derive_cg_accel_longitudinal = derive_cg_accel_longitudinal
_calculate_g_sum = calculate_g_sum
_derive_brake_pos = derive_brake_pos
_calculate_input_rates = calculate_input_rates
_mirror_throttle_accel = mirror_throttle_accel
_calculate_rate = calculate_rate
_derive_yaw_rate_from_gps_heading = derive_yaw_rate_from_gps_heading
