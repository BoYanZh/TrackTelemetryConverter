"""Parser for VBO telemetry log files.

Extracted from the original DataLog.vbo_log methods with identical behavior."""

from __future__ import annotations

import datetime

from ..channels import CH_ACCELERATOR_POS, CHANNEL_ALIASES
from ..models import Message
from ..provenance import record_channel_provenance


def _parse_vbo_latlon(val_str, is_lon=False):
    try:
        val_s = str(val_str).strip()
        sign = -1.0 if val_s.startswith("-") else 1.0
        val = float(val_s.lstrip("+-")) / 60.0
        if is_lon and sign > 0 and val > 50.0:
            sign = -1.0
        return sign * val
    except Exception:
        return 0.0


def _parse_vbo_time(val_str):
    try:
        t_float = float(val_str)
        hh = int(t_float // 10000)
        mm = int((t_float % 10000) // 100)
        ss = t_float % 100
        return hh * 3600.0 + mm * 60.0 + ss
    except ValueError:
        return 0.0


def parse_vbo_log(data_log, log_lines, target_lap=None):
    """ Creates channels populated with messages from a Racelogic .vbo log file. """
    data_log.clear()
    data_log.laps_info = {}

    if not log_lines:
        return

    sections = {}
    curr_sec = None
    for line in log_lines:
        line_s = line.strip()
        if line_s.startswith("[") and line_s.endswith("]"):
            curr_sec = line_s[1:-1]
            sections[curr_sec] = []
        elif curr_sec:
            sections[curr_sec].append(line_s)

    # Extract datetime from header line 1 (e.g. File created on 20/01/2026 at 07:31:48)
    if log_lines and "File created on" in log_lines[0]:
        parts = log_lines[0].strip().split()
        if len(parts) >= 6:
            d_str, t_str = parts[3], parts[5]
            try:
                data_log.datetime = datetime.datetime.strptime(f"{d_str} {t_str}", "%d/%m/%Y %H:%M:%S")
            except Exception:
                print(f"WARNING: Could not parse VBO header date: {d_str} {t_str}")

    cols_raw = sections.get("column names", [""])[0].split()
    data_lines = [line for line in sections.get("data", []) if line]

    if not cols_raw or not data_lines:
        print("ERROR: Invalid VBO file, missing [column names] or [data] section")
        return

    time_idx = cols_raw.index("time") if "time" in cols_raw else -1
    if time_idx < 0:
        print("ERROR: VBO file missing 'time' column")
        return

    col_idx_map = {}
    # Several VBO writers emit the same signal twice (raw + `-calc`/`-canbus`
    # variants). One sample per channel per row; later columns win (refined
    # values sort last). The winner is recorded instead of being implicit.
    raw_columns = {}
    for idx, col_name in enumerate(cols_raw):
        if col_name in CHANNEL_ALIASES:
            out_name, unit, dec = CHANNEL_ALIASES[col_name]
            if out_name not in data_log.channels:
                data_log.add_channel(out_name, unit, float, dec)
            raw_columns.setdefault(out_name, []).append(col_name)
            # AiM BrakePress sources read in bar while the canonical channel
            # is kPa (cf. xrk_parser pressure conversion).
            factor = 100.0 if col_name == "BrakePress" else 1.0
            is_latlon = col_name in ("lat", "long")
            is_lon = (col_name == "long")
            col_idx_map[idx] = (out_name, is_latlon, is_lon, factor)

    for out_name, raws in raw_columns.items():
        if len(raws) > 1:
            record_channel_provenance(
                data_log, out_name, raws[-1],
                raw_columns=list(raws),
                winner=raws[-1],
                rule="last column wins (refined values sort last)",
            )
    pedal_raw = None
    for raw in raw_columns.get(CH_ACCELERATOR_POS, []):
        if "accelerator" in raw.lower():
            pedal_raw = raw
            break
    if pedal_raw is not None:
        record_channel_provenance(
            data_log, CH_ACCELERATOR_POS, pedal_raw,
            power_semantics="accelerator_pedal",
            semantics_basis=(
                "RaceChrono PID 10071 documents accelerator_pos as Accelerator "
                "Pedal Pos; throttle body is the separate PID 10025 "
                "(docs/research/racechrono-pid-tables.md)"
            ),
        )

    t0 = None
    for line_s in data_lines:
        parts = line_s.split()
        if len(parts) >= len(cols_raw):
            t_sec = _parse_vbo_time(parts[time_idx])
            if t0 is None:
                t0 = t_sec
            t_rel = t_sec - t0
            if t_rel < 0:
                t_rel += 86400.0  # Handle midnight wrapping
            row_vals = {}
            for idx, (out_name, is_latlon, is_lon, factor) in col_idx_map.items():
                try:
                    val_str = parts[idx]
                    if is_latlon:
                        v_val = _parse_vbo_latlon(val_str, is_lon=is_lon)
                    else:
                        v_val = float(val_str) * factor
                    # Several VBO writers emit the same signal twice (raw +
                    # `-calc`/`-canbus` variants). One sample per channel per
                    # row; later columns win (refined values sort last).
                    row_vals[out_name] = v_val
                except ValueError:
                    pass
            for out_name, v_val in row_vals.items():
                data_log.channels[out_name].messages.append(Message(t_rel, v_val))
