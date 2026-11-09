"""Parser for VBO telemetry log files.

Extracted from the original DataLog.vbo_log methods with identical behavior."""

from __future__ import annotations

import datetime
import re
from types import MappingProxyType

import numpy as np

from ..channels import (
    CH_ACCELERATOR_POS,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CHANNEL_ALIASES,
)
from ..lap_view import ChannelView, readonly
from ..provenance import record_channel_provenance

#: Rows per chunk in the vectorized data pass (bounds the transient
#: split-string table while keeping numpy conversions amortized).
_VBO_CHUNK_ROWS = 20_000

_DECLARED_UNITS = {
    "kmh", "km/h", "kph", "mph", "m/s", "mps", "m", "cm", "mm",
    "g", "mg", "bar", "kpa", "pa", "psi", "deg", "degree",
    "degrees", "rad", "rpm", "%", "c", "°c", "f", "°f", "s",
    "ms", "hz", "v", "volt", "a", "amp", "amps", "n", "nm", "ft",
}


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


def _vbo_column_keys(column_names):
    """Return stable per-column keys while preserving literal source names."""
    reserved = set(column_names)
    used = set()
    counts = {}
    keys = []
    for name in column_names:
        count = counts.get(name, 0) + 1
        counts[name] = count
        if count == 1:
            key = name
        else:
            suffix = count
            key = f"{name}#{suffix}"
            while key in reserved or key in used:
                suffix += 1
                key = f"{name}#{suffix}"
        keys.append(key)
        used.add(key)
    return keys


def _vbo_declared_unit(header):
    """Read a known unit suffix from a VBO [header] line, if declared."""
    if not header:
        return ""
    text = header.strip()
    bracketed = re.search(r"(?:\(([^()]*)\)|\[([^\[\]]*)\])\s*$", text)
    candidate = (bracketed.group(1) or bracketed.group(2)).strip() if bracketed else ""
    if not candidate:
        parts = text.split()
        candidate = parts[-1] if len(parts) > 1 else ""
    return candidate if candidate.casefold() in _DECLARED_UNITS else ""


def _selected_vbo_columns(raw_channels, column_names, keys):
    """Select physical columns, allowing original names to select duplicates."""
    if raw_channels is None:
        return ()
    if isinstance(raw_channels, str):
        if raw_channels.casefold() == "all":
            return tuple(range(len(column_names)))
        requested = {raw_channels}
    else:
        try:
            requested = set(raw_channels)
        except TypeError as exc:
            raise TypeError(
                "raw_channels must be None, 'all', a column name, or an iterable",
            ) from exc
        if any(not isinstance(name, str) for name in requested):
            raise TypeError("raw_channels column names must be strings")
    selected = set()
    for requested_name in requested:
        if requested_name in column_names:
            selected.update(
                idx for idx, source_name in enumerate(column_names)
                if source_name == requested_name
            )
        else:
            selected.update(idx for idx, key in enumerate(keys) if key == requested_name)
    return tuple(idx for idx in range(len(column_names)) if idx in selected)


def _vbo_sample_rate(timestamps):
    """Infer a rate from distinct finite elapsed timestamps, when possible."""
    finite = np.asarray(timestamps, dtype=np.float64)
    finite = np.unique(finite[np.isfinite(finite)])
    deltas = np.diff(finite)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    return float(1.0 / np.median(deltas)) if len(deltas) else None


def _record_vbo_raw_inventory(data_log, column_names, keys, header_lines):
    headers_aligned = len(header_lines) == len(column_names)
    data_log.metadata["vbo_raw_columns"] = [
        {
            "raw_name": raw_name,
            "key": keys[idx],
            "column_index": idx,
            "source_header": header_lines[idx] if headers_aligned else None,
            "source_units": _vbo_declared_unit(header_lines[idx]) if headers_aligned else "",
        }
        for idx, raw_name in enumerate(column_names)
    ]


def _append_vbo_raw_chunk(indices, rows, time_idx, elapsed, buffers, source_times, elapsed_parts):
    if not indices:
        return
    elapsed_parts.append(elapsed)
    source_time, _ = _safe_float_column([parts[time_idx] for parts in rows])
    source_times.append(source_time)
    for idx in indices:
        values, _ = _safe_float_column([parts[idx] for parts in rows])
        buffers[idx].append(values)


def _populate_vbo_raw_channels(
    data_log, column_names, keys, selected_indices, headers, buffers,
    source_time_parts, elapsed_parts,
):
    if not selected_indices:
        return
    elapsed = np.concatenate(elapsed_parts) if elapsed_parts else np.empty(0, dtype=np.float64)
    source_times = (
        np.concatenate(source_time_parts)
        if source_time_parts else np.empty(0, dtype=np.float64)
    )
    rate = _vbo_sample_rate(elapsed)
    monotonic = bool(
        len(elapsed) < 2
        or (np.all(np.isfinite(elapsed)) and np.all(np.diff(elapsed) >= 0)),
    )
    for idx in selected_indices:
        raw_name = column_names[idx]
        key = keys[idx]
        header = headers[idx] if len(headers) == len(column_names) else None
        units = _vbo_declared_unit(header) if header is not None else ""
        values = (
            np.concatenate(buffers[idx])
            if buffers[idx] else np.empty(0, dtype=np.float64)
        )
        provenance = MappingProxyType({
            "source": "VBO",
            "source_time_encoding": "HHMMSS.fraction",
            "dtype": "<f8",
            "raw_name": raw_name,
            "column_index": idx,
            "source_header": header,
            "source_units": units,
            "value_origin": "numeric_text",
            "series_sample_rate_hz": rate,
        })
        data_log.raw_channels[key] = ChannelView(
            key, units, readonly(elapsed), readonly(values), rate, provenance,
            readonly(source_times), monotonic,
        )


def parse_vbo_log(data_log, log_lines, target_lap=None, *, raw_channels=None):
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

    raw_keys = _vbo_column_keys(cols_raw) if raw_channels is not None else ()
    selected_raw_indices = _selected_vbo_columns(raw_channels, cols_raw, raw_keys)
    header_lines = [line for line in sections.get("header", []) if line]
    if raw_channels is not None:
        _record_vbo_raw_inventory(data_log, cols_raw, raw_keys, header_lines)

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
    selected_raw_buffers = {idx: [] for idx in selected_raw_indices}
    source_time_parts = []
    elapsed_parts = []
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
        _append_vbo_raw_chunk(
            selected_raw_indices, rows, time_idx, t_rel, selected_raw_buffers,
            source_time_parts, elapsed_parts,
        )
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

    _populate_vbo_raw_channels(
        data_log, cols_raw, raw_keys, selected_raw_indices, header_lines,
        selected_raw_buffers, source_time_parts, elapsed_parts,
    )
    _load_vbo_laptiming_traps(data_log, sections)


#: Sanity bounds for [laptiming] gate coordinates (decimal degrees).
_VBO_GATE_MAX_LAT_DEG = 90.0
_VBO_GATE_MAX_LON_DEG = 180.0
#: A parsed gate farther than this from the file's own GPS trace is not the
#: driven track (stale map) and is skipped.
_VBO_GATE_MAX_ANCHOR_DEG = 0.5
#: |minutes| above this must be longitude (latitude caps at 90 * 60).
_VBO_LON_MINUTES_FLOOR = 90.0 * 60.0


def _resolve_gate_pair(first_raw, second_raw, anchor):
    """Assign ``(lat_deg, lon_deg)`` for one gate endpoint pair.

    Writers disagree on field order (both long,lat and lat,long observed),
    so each pair is resolved independently: impossible latitudes are
    rejected, then the file's own GPS centroid picks the winner, else a
    minutes-magnitude rule (``|lon|`` > 90 * 60 settles it). Returns
    ``(lat, lon)`` in decimal degrees, or ``None`` when unusable.
    """
    options = []
    for lat_raw, lon_raw in ((first_raw, second_raw), (second_raw, first_raw)):
        lat = _parse_vbo_latlon(lat_raw, is_lon=False)
        lon = _parse_vbo_latlon(lon_raw, is_lon=True)
        if abs(lat) > _VBO_GATE_MAX_LAT_DEG or abs(lon) > _VBO_GATE_MAX_LON_DEG:
            continue
        options.append((lat_raw, lon_raw, lat, lon))
    if not options:
        return None
    if anchor is not None:
        best, best_dist = None, None
        for opt in options:
            dist = abs(opt[2] - anchor[0]) + abs(opt[3] - anchor[1])
            if best is None or dist < best_dist:
                best, best_dist = opt, dist
        if best_dist is not None and best_dist <= _VBO_GATE_MAX_ANCHOR_DEG:
            return best[2], best[3]
        return None
    if len(options) == 1:
        return options[0][2], options[0][3]

    def _magnitude(raw):
        try:
            return abs(float(str(raw).strip().lstrip("+-")))
        except ValueError:
            return 0.0

    big = [opt for opt in options if _magnitude(opt[1]) > _VBO_LON_MINUTES_FLOOR]
    if len(big) == 1:
        return big[0][2], big[0][3]
    return None


#: A gate encounter reading more parallel than this (degrees) is a drive-by,
#: not a line crossing, and is rejected as a trap.
_VBO_GATE_MIN_CROSS_ANGLE_DEG = 30.0
#: Encounters farther than this from the gate segment are not encounters.
_VBO_GATE_MAX_ENCOUNTER_M = 50.0


def _gate_is_crossed(lats, lons, lat1, lon1, lat2, lon2):
    """Whether the driven trace truly crosses a gate segment.

    RaceChrono maps sometimes carry along-track snippets (racing-line
    geometry) instead of crossable lines: a proximity-only check would
    "detect" those at the abeam moment with a systematic time error. A real
    crossing needs the closest approach on the segment span with the course
    roughly perpendicular to the gate. Returns ``True`` only then.
    """
    try:
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
    except (TypeError, ValueError):
        return False
    if lats.shape != lons.shape or lats.size < 2:
        return False
    if not all(np.isfinite(v) for v in (lat1, lon1, lat2, lon2)):
        return False
    mlat, mlon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    ax, ay = (lon1 - mlon) * 85670.0, (lat1 - mlat) * 111000.0
    bx, by = (lon2 - mlon) * 85670.0, (lat2 - mlat) * 111000.0
    dx, dy = bx - ax, by - ay
    if dx * dx + dy * dy < 1.0:
        return False
    ex = (lons - mlon) * 85670.0
    ey = (lats - mlat) * 111000.0
    denom = dx * dx + dy * dy
    u = ((ex - ax) * dx + (ey - ay) * dy) / denom
    closest = np.hypot(ex - (ax + np.clip(u, 0.0, 1.0) * dx),
                       ey - (ay + np.clip(u, 0.0, 1.0) * dy))
    i = int(np.argmin(closest))
    if closest[i] >= _VBO_GATE_MAX_ENCOUNTER_M:
        return False
    if not 0.0 <= float(u[i]) <= 1.0:
        return False
    vx = np.gradient(ex)
    vy = np.gradient(ey)
    course = float(np.degrees(np.arctan2(vx[i], vy[i])) % 360.0)
    gate_hdg = float(np.degrees(np.arctan2(dx, dy)) % 360.0)
    delta = abs((course - gate_hdg + 180.0) % 360.0 - 180.0)
    return min(delta, 180.0 - delta) > _VBO_GATE_MIN_CROSS_ANGLE_DEG


def _load_vbo_laptiming_traps(data_log, sections):
    """Load [laptiming] Start/Split gates into ``data_log.traps``.

    RaceChrono exports the driven track's gates here
    (``Start lon1 lat1 lon2 lat2 ¬ name``); field order is resolved per
    endpoint pair because writers disagree. A gate becomes a trap only if
    the driven trace truly crosses its segment (see
    :func:`_gate_is_crossed`): along-track snippets that a proximity check
    would mistime are skipped. Gate midpoints become traps so the existing
    GPS-crossing detector cuts laps and sector splits with no further
    changes. Files without the section keep ``traps`` untouched.
    """
    raw_lines = sections.get("laptiming", [])
    if not raw_lines:
        return
    anchor = None
    trace = None
    try:
        lat_ch = data_log.channels.get(CH_GPS_LATITUDE)
        lon_ch = data_log.channels.get(CH_GPS_LONGITUDE)
        if (
            lat_ch is not None
            and lon_ch is not None
            and len(lat_ch.values)
            and len(lon_ch.values)
        ):
            lats = np.asarray(lat_ch.values, dtype=float)
            lons = np.asarray(lon_ch.values, dtype=float)
            if lats.shape == lons.shape and lats.size and np.all(np.isfinite(lats)) \
                    and np.all(np.isfinite(lons)):
                trace = (lats, lons)
                anchor = (float(np.mean(lats)), float(np.mean(lons)))
                if not (np.isfinite(anchor[0]) and np.isfinite(anchor[1])):
                    anchor = None
    except (TypeError, ValueError, AttributeError):
        anchor = None
    if trace is None:
        # Without a GPS trace gates can be neither ordered reliably nor
        # verified as crossings -- and nothing downstream could detect
        # them anyway.
        return
    seen = set()
    split_no = 0
    for raw_line in raw_lines:
        line = str(raw_line).strip()
        if not line:
            continue
        head, _, name = line.partition("¬")
        name = name.strip()
        tokens = head.split()
        if len(tokens) < 5:
            continue
        kind = tokens[0].strip().lower()
        endpoints = []
        valid = True
        for first_raw, second_raw in ((tokens[1], tokens[2]), (tokens[3], tokens[4])):
            resolved = _resolve_gate_pair(first_raw, second_raw, anchor)
            if resolved is None:
                valid = False
                break
            endpoints.append(resolved)
        if not valid:
            continue
        (lat1, lon1), (lat2, lon2) = endpoints
        key = (round((lat1 + lat2) / 2, 6), round((lon1 + lon2) / 2, 6))
        if key in seen:
            continue
        seen.add(key)
        if not _gate_is_crossed(trace[0], trace[1], lat1, lon1, lat2, lon2):
            continue
        if not name:
            if kind == "start":
                name = "Start/Finish"
            elif kind == "split":
                split_no += 1
                name = f"Split {split_no}"
            else:
                name = tokens[0].strip() or "Split"
        data_log.traps.append({
            "name": name,
            "lat": (lat1 + lat2) / 2,
            "lon": (lon1 + lon2) / 2,
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
        })
