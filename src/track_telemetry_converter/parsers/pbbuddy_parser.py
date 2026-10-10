"""Parser for PBBUDDY telemetry log files.

Extracted from the original DataLog.pbbuddy_log methods with identical behavior."""

from __future__ import annotations

import datetime

from ..channels import (
    CH_GPS_ALTITUDE,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GROUND_SPEED,
    CH_LAP_NUMBER,
)
from ..models import Message


def _split_csv_line(line):
    import csv

    try:
        return [p.strip().strip('"').strip("'") for p in next(csv.reader([line]))]
    except Exception:  # pragma: no cover - csv.reader rarely raises; split fallback
        return [p.strip().strip('"').strip("'") for p in line.split(",")]


def _find_data_header(log_lines):
    """Return the index of the Time/... header row, or -1 if absent."""
    for i, line in enumerate(log_lines):
        line_clean = line.strip().strip('"').strip("'")
        if not line_clean:
            continue
        parts = _split_csv_line(line_clean)
        if len(parts) >= 3 and parts[0].lower() in ("time", "time (s)", "timestamp"):
            return i
    print("ERROR: Could not locate PB Buddy CSV data header line.")
    return -1


def _parse_metadata_block(data_log, log_lines, data_header_idx):
    """Process Key-Value metadata lines before the data header."""
    for line in log_lines[:data_header_idx]:
        line_s = line.strip()
        if not line_s:
            continue
        parts = _split_csv_line(line_s)
        if len(parts) == 2:
            key, val = parts[0], parts[1]
            data_log.metadata[key] = val
            if key == "Track name":
                data_log.metadata["venue_name"] = val
            elif key == "Date":
                data_log.metadata["date"] = val
            elif key == "Time":
                data_log.metadata["time"] = val
            elif key == "Session name":
                data_log.metadata["session"] = val


def _parse_session_datetime(data_log):
    """Extract session datetime from epoch metadata or Date/Time fields."""
    dt = None
    if "Session start, seconds since epoch" in data_log.metadata:
        try:
            epoch = float(data_log.metadata["Session start, seconds since epoch"])
            offset_sec = 0.0
            if "Timezone offset, milliseconds" in data_log.metadata:
                offset_sec = float(data_log.metadata["Timezone offset, milliseconds"]) / 1000.0
            dt = datetime.datetime.fromtimestamp(epoch + offset_sec, tz=datetime.timezone.utc).replace(tzinfo=None)
        except Exception:
            print("WARNING: Could not parse PB Buddy session epoch metadata")

    if dt is None:
        import contextlib

        date_str = data_log.metadata.get("date") or data_log.metadata.get("Date") or data_log.metadata.get("Session start date, local timezone, YYYYMMDD")
        time_str = data_log.metadata.get("time") or data_log.metadata.get("Time")
        if date_str and time_str:
            dt_str = f"{date_str} {time_str}".strip()
            for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                with contextlib.suppress(Exception):
                    dt = datetime.datetime.strptime(dt_str, fmt)
                    break
            if dt is None:
                print(f"WARNING: Could not parse PB Buddy date/time: {dt_str!r}")

    if dt:
        data_log.datetime = dt


def _build_channel_index(data_log, headers, units):
    """Map column positions to (source name, output channel) pairs."""
    mapping = {
        "Time": ("Time", "s", 3),
        CH_GPS_LATITUDE: (CH_GPS_LATITUDE, "deg", 7),
        CH_GPS_LONGITUDE: (CH_GPS_LONGITUDE, "deg", 7),
        "GPS Speed": (CH_GROUND_SPEED, "km/h", 5),  # Convert m/s -> km/h
        CH_GPS_HEADING: (CH_GPS_HEADING, "deg", 3),
        CH_GPS_ALTITUDE: (CH_GPS_ALTITUDE, "m", 3),
        # Lap count column - added by PB Buddy on request (Timur, 2026-08)
        "Lap Count": (CH_LAP_NUMBER, "", 0),
        "Lap": (CH_LAP_NUMBER, "", 0),
        CH_LAP_NUMBER: (CH_LAP_NUMBER, "", 0),
    }
    chan_indices = {}
    for idx, h in enumerate(headers):
        if h in mapping:
            out_name, out_unit, out_dec = mapping[h]
            data_log.add_channel(out_name, out_unit, float, out_dec)
            chan_indices[idx] = (h, out_name)
        elif h != "Time":
            unit = units[idx] if idx < len(units) else ""
            data_log.add_channel(h, unit, float, 3)
            chan_indices[idx] = (h, h)
    return chan_indices


def _parse_data_rows(data_log, log_lines, start_idx, headers, chan_indices):
    """Append samples for data rows, skipping comments and bad values."""
    for line in log_lines[start_idx:]:
        line_s = line.strip()
        if not line_s or line_s.startswith("//") or line_s.startswith("#"):
            continue
        parts = line_s.split(",")
        if len(parts) >= len(headers):
            try:
                t = float(parts[0])
                for idx, (orig_name, out_name) in chan_indices.items():
                    val = float(parts[idx])
                    if orig_name == "GPS Speed":
                        val = val * 3.6  # m/s -> km/h
                    data_log.channels[out_name].messages.append(Message(t, val))
            except ValueError:
                continue


def parse_pbbuddy_log(data_log, log_lines, target_lap=None):
    """ Creates channels populated with messages from a PB Buddy CSV log file. """
    data_log.clear()
    data_log.laps_info = {}

    if not log_lines:
        return

    # 1. Scan for Data Header Line (contains >= 3 columns and starts with Time/timestamp)
    data_header_idx = _find_data_header(log_lines)
    if data_header_idx == -1:
        return

    # 2. Process all lines BEFORE Data Header Line as Key-Value Metadata
    _parse_metadata_block(data_log, log_lines, data_header_idx)

    # Extract datetime from PB Buddy epoch + timezone offset or Date/Time fields
    _parse_session_datetime(data_log)

    # 3. Read Headers and Units Row
    header_line = log_lines[data_header_idx].strip()
    headers = [h.strip().strip('"').strip("'") for h in header_line.split(",")]

    units = []
    if data_header_idx + 1 < len(log_lines):
        next_line = log_lines[data_header_idx + 1].strip()
        unit_parts = [u.strip().strip('"').strip("'") for u in next_line.split(",")]
        if unit_parts and unit_parts[0].lower() in ("sec", "s", "seconds", "m/s", "deg", "m"):
            units = unit_parts

    chan_indices = _build_channel_index(data_log, headers, units)

    # 4. Parse Data Rows
    start_data_idx = data_header_idx + (2 if units else 1)
    _parse_data_rows(data_log, log_lines, start_data_idx, headers, chan_indices)
