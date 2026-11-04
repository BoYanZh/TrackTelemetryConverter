"""Parser for VBO telemetry log files.

Extracted from the original DataLog.vbo_log methods with identical behavior."""

from __future__ import annotations

import datetime

import numpy as np

from ..channels import CH_ACCELERATOR_POS, CHANNEL_ALIASES
from ..provenance import record_channel_provenance

#: Rows per chunk in the vectorized data pass (bounds the transient
#: split-string table while keeping numpy conversions amortized).
_VBO_CHUNK_ROWS = 20_000


def _safe_float_column(cells):
    """Convert string cells to (values, valid) with legacy float() semantics.

    C-speed ``astype`` fast path for clean columns; per-element fallback only
    for chunks containing bad cells (bad cells are skipped, like the legacy
    ``except ValueError: pass``).
    """
    str_arr = np.asarray(cells)
    try:
        return str_arr.astype(np.float64), np.ones(len(str_arr), dtype=bool)
    except ValueError:
        pass
    vals = np.empty(len(cells), dtype=np.float64)
    valid = np.zeros(len(cells), dtype=bool)
    for i, cell in enumerate(cells):
        try:
            vals[i] = float(cell)
            valid[i] = True
        except ValueError:
            vals[i] = np.nan
    return vals, valid


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

    # Group source columns by canonical channel, preserving column order:
    # later columns win per row (refined values sort last).
    groups = {}
    # Several VBO writers emit the same signal twice (raw + `-calc`/`-canbus`
    # variants). The winner is recorded instead of being implicit.
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
            groups.setdefault(out_name, []).append(
                (idx, is_latlon, (col_name == "long"), factor),
            )

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
    n_cols = len(cols_raw)
    buffers = {out_name: ([], []) for out_name in groups}
    for start in range(0, len(data_lines), _VBO_CHUNK_ROWS):
        rows = [line.split() for line in data_lines[start:start + _VBO_CHUNK_ROWS]]
        rows = [parts for parts in rows if len(parts) >= n_cols]
        if not rows:
            continue
        t_secs = [_parse_vbo_time(parts[time_idx]) for parts in rows]
        if t0 is None:
            t0 = t_secs[0]
        t_rel = np.asarray(t_secs, dtype=np.float64) - t0
        t_rel[t_rel < 0] += 86400.0  # Handle midnight wrapping
        for out_name, sources in groups.items():
            best = None
            best_valid = None
            for idx, is_latlon, is_lon, factor in sources:
                cells = [parts[idx] for parts in rows]
                if is_latlon:
                    # Never raises (0.0 fallback): always valid, like legacy.
                    arr = np.array(
                        [_parse_vbo_latlon(cell, is_lon=is_lon) for cell in cells],
                        dtype=np.float64,
                    )
                    valid = np.ones(len(rows), dtype=bool)
                else:
                    arr, valid = _safe_float_column(cells)
                    arr = arr * factor
                if best is None:
                    best, best_valid = arr, valid
                else:
                    # Later columns win, but only where they parsed.
                    best = np.where(valid, arr, best)
                    best_valid = best_valid | valid
            if best is None or best_valid is None:
                continue  # No source columns for this channel; nothing to store.
            ts_buf, val_buf = buffers[out_name]
            ts_buf.append(t_rel[best_valid])
            val_buf.append(best[best_valid])

    for out_name, (ts_parts, val_parts) in buffers.items():
        if ts_parts and out_name in data_log.channels:
            data_log.channels[out_name].set_samples(
                np.concatenate(ts_parts), np.concatenate(val_parts),
            )
