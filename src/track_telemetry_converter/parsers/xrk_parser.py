"""Parser for AIM XRK / XRZ telemetry log files.

Uses the `libxrk` library (https://github.com/m3rlin45/libxrk) to decode the
binary AIM data logger format and populates a DataLog with standard MoTeC
channel names.  XRZ files (zlib-compressed XRK) are handled automatically by
libxrk.

Requires: pip install libxrk
"""

from __future__ import annotations

import datetime

import numpy as np

from ..channels import CHANNEL_ALIASES


def _unit_to_canonical(unit):
    """Normalize a libxrk unit string to a canonical MoTeC unit."""
    unit_l = (unit or "").strip().lower()
    return {
        "km/h": "km/h", "kph": "km/h", "kmh": "km/h", "m/s": "m/s",
        "mph": "mph",
        "deg": "deg", "°": "deg", "rpm": "rpm", "g": "G", "m": "m",
        "c": "C", "°c": "C", "f": "F", "°f": "F",
        "bar": "bar", "v": "V", "%": "%",
    }.get(unit_l, unit or "")


def _convert_values(values, source_unit, target_unit):
    """Convert a numeric XRK array from its declared unit to the alias unit."""
    source = _unit_to_canonical(source_unit)
    target = _unit_to_canonical(target_unit)
    result = np.asarray(values, dtype=np.float64)
    if not source or source == target:
        return result
    if source == "m/s" and target == "km/h":
        return result * 3.6
    if source == "mph" and target == "km/h":
        return result * 1.609344
    if source == "km/h" and target == "m/s":
        return result / 3.6
    if source == "F" and target == "C":
        return (result - 32.0) * (5.0 / 9.0)
    return result


def _dedupe_samples(timestamps, values):
    """Sort samples and keep the last value for each duplicate timestamp."""
    order = np.argsort(timestamps, kind="stable")
    timestamps = np.asarray(timestamps, dtype=np.float64)[order]
    values = np.asarray(values, dtype=np.float64)[order]
    if len(timestamps) < 2:
        return timestamps, values
    keep = np.concatenate([timestamps[1:] != timestamps[:-1], [True]])
    return timestamps[keep], values[keep]


def _parse_log_datetime(meta):
    for key in ("log_datetime", "datetime", "Log Date/Time", "date_time"):
        raw = meta.get(key)
        if raw:
            try:
                if isinstance(raw, (int, float)) and raw > 1e8:
                    timestamp = float(raw)
                    if timestamp > 1e11:
                        timestamp /= 1000.0
                    return datetime.datetime.fromtimestamp(timestamp)
                for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
                    try:
                        return datetime.datetime.strptime(str(raw)[:19], fmt)
                    except ValueError:
                        continue
            except (ValueError, TypeError, OSError):
                pass

    raw_date = meta.get("Log Date") or meta.get("log_date")
    raw_time = meta.get("Log Time") or meta.get("log_time") or "00:00:00"
    if raw_date:
        combined = f"{raw_date} {raw_time}"
        for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                    "%d/%m/%Y %H:%M:%S"):
            try:
                return datetime.datetime.strptime(combined, fmt)
            except ValueError:
                continue
    return None


def parse_xrk_log(data_log, xrk_file_path, target_lap=None):
    """ Creates channels populated with messages from an AIM .xrk/.xrz log file. """
    try:
        from libxrk import aim_xrk
        from libxrk.base import ChannelMetadata
    except ImportError:
        print("ERROR: 'libxrk' package is required for XRK/XRZ log processing.")
        print("  Install with: pip install libxrk")
        return

    data_log.clear()
    data_log.laps_info = {}

    log = aim_xrk(xrk_file_path)

    # ---- metadata ----
    meta = getattr(log, "metadata", {}) or {}
    venue = str(meta.get("venue", meta.get("Venue", meta.get("track", "")))).strip()
    if venue:
        data_log.metadata["venue_name"] = venue
    driver = str(meta.get("driver", meta.get("Driver", meta.get("racer", "")))).strip()
    if driver:
        data_log.metadata["driver"] = driver
    vehicle = str(meta.get("vehicle", meta.get("Vehicle", meta.get("car", "")))).strip()
    if vehicle:
        data_log.metadata["vehicle_id"] = vehicle
    comment = str(meta.get("comment", meta.get("Comment", ""))).strip()
    if comment:
        data_log.metadata["short_comment"] = comment

    data_log.datetime = _parse_log_datetime(meta)

    raw_channels = log.channels or {}
    laps = getattr(log, "laps", None)

    # AIM channel and lap timecodes share one log-relative millisecond clock.
    # Preserve their offsets by selecting one global origin for the session.
    # See docs/research/libxrk-0.13-timestamp-semantics.md for details on libxrk>=0.13 clock unwrapping.
    origins_ms = []
    for table in raw_channels.values():
        if table is None or table.num_rows == 0:
            continue
        try:
            timecodes = table.column("timecodes").to_numpy()
            if len(timecodes):
                origins_ms.append(float(np.nanmin(timecodes)))
        except (KeyError, ValueError, TypeError):
            continue
    if laps is not None and laps.num_rows > 0:
        try:
            lap_starts = [v for v in laps.column("start_time").to_pylist() if v is not None]
            if lap_starts:
                origins_ms.append(float(min(lap_starts)))
        except (KeyError, ValueError, TypeError):
            pass
    global_t0_ms = min(origins_ms) if origins_ms else 0.0
    aliases_casefold = {str(key).strip().casefold(): value for key, value in CHANNEL_ALIASES.items()}

    # ---- channels ----
    for ch_name, table in raw_channels.items():
        if ch_name is None or table is None or table.num_rows == 0:
            continue
        if ch_name.lower() == "timecodes":
            continue

        alias = CHANNEL_ALIASES.get(ch_name)
        if alias is None:
            alias = aliases_casefold.get(ch_name.strip().casefold())
        if alias is None:
            continue

        out_name, unit, dec = alias
        try:
            meta_obj = ChannelMetadata.from_channel_table(table)
        except Exception:
            meta_obj = None

        timecodes_ms = np.asarray(table.column("timecodes").to_numpy(), dtype=np.float64)
        values = np.asarray(table.column(ch_name).to_numpy(zero_copy_only=False), dtype=np.float64)
        valid = np.isfinite(timecodes_ms) & np.isfinite(values)
        if not np.any(valid):
            continue
        timecodes_ms = timecodes_ms[valid]
        values = values[valid]

        source_unit = getattr(meta_obj, "units", "") if meta_obj is not None else ""
        values = _convert_values(values, source_unit, unit)
        timestamps = (timecodes_ms - global_t0_ms) / 1000.0
        timestamps, values = _dedupe_samples(timestamps, values)

        if out_name not in data_log.channels:
            dec_pts = getattr(meta_obj, "dec_pts", None) if meta_obj is not None else None
            dec_use = dec if dec_pts is None else int(dec_pts)
            data_log.add_channel(out_name, unit, float, dec_use)

        data_log.channels[out_name].set_samples(timestamps, values)

    # ---- laps ----
    laps_info = {"laps": [], "total_laps": 0, "fastest_time": 0.0}
    try:
        if laps is not None and laps.num_rows > 0:
            lap_nums = laps.column("num").to_pylist()
            starts_ms = laps.column("start_time").to_pylist()
            ends_ms = laps.column("end_time").to_pylist()
            beacons = []
            for n, s, e in zip(lap_nums, starts_ms, ends_ms):
                if s is None or e is None:
                    continue
                start_s = (float(s) - global_t0_ms) / 1000.0
                end_s = (float(e) - global_t0_ms) / 1000.0
                dur = end_s - start_s
                if dur > 0:
                    laps_info["laps"].append({
                        "lap_num": int(n), "start_time": start_s,
                        "end_time": end_s, "duration": dur, "type": "Timed",
                    })
                    for boundary in (start_s, end_s):
                        if not any(abs(boundary - existing[0]) <= 1e-6 for existing in beacons):
                            beacons.append((boundary, "Start/Finish"))
                    if laps_info["fastest_time"] == 0.0 or dur < laps_info["fastest_time"]:
                        laps_info["fastest_time"] = dur
            laps_info["total_laps"] = len(laps_info["laps"])
            laps_info["beacons"] = sorted(beacons, key=lambda item: item[0])
    except Exception as e:
        print(f"WARNING: Failed to parse XRK laps: {e}")

    # lap filtering: keep channels within the requested lap window
    if target_lap is not None and str(target_lap).lower() != "all":
        try:
            lap_no = int(target_lap)
            lap_recs = [lap for lap in laps_info["laps"] if lap["lap_num"] == lap_no]
            if lap_recs:
                t_lo = lap_recs[0]["start_time"]
                t_hi = lap_recs[0]["end_time"]
                for name in list(data_log.channels):
                    ch = data_log.channels[name]
                    ch.messages = [m for m in ch.messages if t_lo <= m.timestamp <= t_hi]
            else:
                print(f"WARNING: Lap '{target_lap}' not found in XRK file; keeping all laps")
        except (ValueError, IndexError):
            print(f"WARNING: Lap '{target_lap}' not found in XRK file; keeping all laps")

    data_log.laps_info = laps_info
