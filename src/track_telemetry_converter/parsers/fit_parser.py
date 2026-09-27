"""Parser for Garmin .fit telemetry log files.

Uses the `fitparse` library to decode Garmin's binary FIT protocol.  Handles
both single-activity files and multi-sport/multi-session files, extracting
standard channels (GPS, speed, heart rate, power, cadence, altitude) and
lap boundaries from the `record`, `lap`, `session`, and `event` messages.

Requires: pip install fitparse
"""

from __future__ import annotations

import datetime

from ..models import Message

# Garmin FIT stores angles as semicircles; 2^31 = 180 degrees
_SEMICIRCLE = 2 ** 31


def _to_deg(semicircles):
    return float(semicircles) * 180.0 / _SEMICIRCLE


def _fit_field(msg, name):
    """Return a field's value, or None if the field is absent/None."""
    for f in msg.fields:
        if f.name == name and f.value is not None:
            return f.value
    return None


# record-field name -> (canonical channel, unit, decimals, convert_fn)
_FIT_CHANNEL_MAP = {
    "position_lat":        ("GPS Latitude",  "deg", 7, _to_deg),
    "position_long":       ("GPS Longitude", "deg", 7, _to_deg),
    "enhanced_speed":      ("Ground Speed",  "km/h", 2, lambda v: v * 3.6),
    "speed":               ("Ground Speed",  "km/h", 2, lambda v: v * 3.6),
    "enhanced_altitude":   ("GPS Altitude",  "m", 2, None),
    "altitude":            ("GPS Altitude",  "m", 2, None),
    "heart_rate":          ("Heart Rate",    "bpm", 0, None),
    "power":               ("Power",         "W", 0, None),
    "cadence":             ("Cadence",       "rpm", 0, None),
    "distance":            ("Distance",      "m", 2, None),
    "temperature":         ("Air Temp",      "C", 1, None),
    "vertical_oscillation": ("Vertical Osc", "mm", 1, None),
    "stance_time":         ("Stance Time",   "ms", 0, None),
    "left_right_balance":  ("L/R Balance",   "%", 0, None),
    "cycle_length":        ("Cycle Length",  "m", 2, None),
}


def parse_fit_log(data_log, fit_file_path, target_lap=None):
    """ Creates channels populated with messages from a Garmin .fit log file. """
    try:
        import fitparse
        from fitparse.processors import UTC_REFERENCE
    except ImportError:
        print("ERROR: 'fitparse' package is required for .fit log processing.")
        print("  Install with: pip install fitparse")
        return

    class _NaiveUtcDataProcessor(fitparse.FitFileDataProcessor):
        """Preserve fitparse's naive-UTC API without deprecated conversions."""

        @staticmethod
        def _to_naive_utc(value):
            return datetime.datetime.fromtimestamp(
                UTC_REFERENCE + value,
                datetime.timezone.utc,
            ).replace(tzinfo=None)

        def process_type_date_time(self, field_data):
            value = field_data.value
            if value is not None and value >= 0x10000000:
                field_data.value = self._to_naive_utc(value)
                field_data.units = None

        def process_type_local_date_time(self, field_data):
            if field_data.value is not None:
                field_data.value = self._to_naive_utc(field_data.value)
                field_data.units = None

    data_log.clear()
    data_log.laps_info = {}

    fit = fitparse.FitFile(
        fit_file_path,
        data_processor=_NaiveUtcDataProcessor(),
    )

    # ---- metadata ----
    for msg in fit.get_messages("file_id"):
        serial = _fit_field(msg, "serial_number")
        if serial is not None:
            data_log.metadata["vehicle_id"] = "Garmin S/N %s" % serial
        mfr = _fit_field(msg, "manufacturer")
        if mfr:
            data_log.metadata["long_comment"] = "Garmin FIT (manufacturer: %s)" % mfr
        created = _fit_field(msg, "time_created")
        if isinstance(created, datetime.datetime):
            data_log.datetime = created

    for msg in fit.get_messages("sport"):
        sport = _fit_field(msg, "sport")
        if sport:
            data_log.metadata["event_name"] = str(sport).title()

    # ---- channels ----
    # FIT files can contain more than one record message for the same timestamp.
    # Merge their fields so every output channel has a strictly increasing time
    # axis. Keeping duplicate timestamps would make derivative math divide by 0.
    records_by_time = {}
    for msg in fit.get_messages("record"):
        fields = {f.name: f.value for f in msg.fields if f.value is not None}
        if not fields:
            continue
        ts = fields.get("timestamp")
        if ts is None:
            continue
        if ts in records_by_time:
            records_by_time[ts].update(fields)
        else:
            records_by_time[ts] = fields

    records = sorted(records_by_time.items(), key=lambda item: item[0])
    t0 = records[0][0] if records else None

    for ts, fields in records:
        t_rel = (ts - t0).total_seconds()
        # prefer enhanced_* variants over legacy fields
        if "enhanced_speed" in fields:
            fields.pop("speed", None)
        if "enhanced_altitude" in fields:
            fields.pop("altitude", None)
        # fractional_cadence is the sub-RPM component, not a replacement for
        # the integer cadence field.
        fractional_cadence = fields.pop("fractional_cadence", None)
        if fractional_cadence is not None and fields.get("cadence") is not None:
            fields["cadence"] = float(fields["cadence"]) + float(fractional_cadence)
        for fit_name, (ch_name, unit, dec, conv) in _FIT_CHANNEL_MAP.items():
            val = fields.get(fit_name)
            if val is None:
                continue
            if ch_name not in data_log.channels:
                data_log.add_channel(ch_name, unit, float, dec)
            v = conv(val) if conv else float(val)
            data_log.channels[ch_name].messages.append(Message(t_rel, v))

    # ---- laps (from 'lap' messages when present, else 'event' boundaries) ----
    laps_info = {"laps": [], "total_laps": 0, "fastest_time": 0.0}
    beacons = []

    lap_msgs = list(fit.get_messages("lap"))
    if lap_msgs:
        for i, m in enumerate(lap_msgs):
            st = _fit_field(m, "start_time")
            et = _fit_field(m, "timestamp")
            dur = _fit_field(m, "total_timer_time")
            if st is None or et is None:
                continue
            start_s = (st - t0).total_seconds() if t0 else 0.0
            end_s = (et - t0).total_seconds() if t0 else 0.0
            if end_s <= start_s:
                continue
            lap_duration = float(dur) if dur is not None and float(dur) > 0 else end_s - start_s
            laps_info["laps"].append({
                "lap_num": i + 1, "start_time": start_s, "end_time": end_s,
                "duration": lap_duration, "type": "Timed",
            })
            if laps_info["fastest_time"] == 0.0 or lap_duration < laps_info["fastest_time"]:
                laps_info["fastest_time"] = lap_duration
    else:
        # fallback: split at event markers where event is a lap trigger
        prev_ts = None
        for m in fit.get_messages("event"):
            ev = _fit_field(m, "event")
            et = _fit_field(m, "event_type")
            ts = _fit_field(m, "timestamp")
            if ev in ("lap", "start", "stop_all", "marker") and ts is not None:
                t_s = (ts - t0).total_seconds() if t0 else 0.0
                if prev_ts is not None:
                    dur = t_s - prev_ts
                    if dur > 0:
                        laps_info["laps"].append({
                            "lap_num": len(laps_info["laps"]) + 1,
                            "start_time": prev_ts, "end_time": t_s,
                            "duration": dur, "type": "Timed",
                        })
                        if laps_info["fastest_time"] == 0.0 or dur < laps_info["fastest_time"]:
                            laps_info["fastest_time"] = dur
                prev_ts = t_s

    # Marker timestamps must be actual record-time boundaries. Timer duration
    # can exclude pauses and is valid for lap timing, but not as a time-axis
    # coordinate. Deduplicate contiguous lap start/end boundaries.
    for lap in laps_info["laps"]:
        for boundary in (lap["start_time"], lap["end_time"]):
            if not any(abs(boundary - existing[0]) <= 1e-6 for existing in beacons):
                beacons.append((boundary, "Start/Finish"))
    beacons.sort(key=lambda item: item[0])

    laps_info["total_laps"] = len(laps_info["laps"])
    laps_info["beacons"] = sorted(set(beacons))

    # ---- lap filtering ----
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
                print(f"WARNING: Lap '{target_lap}' not found in .fit file; keeping all laps")
        except (ValueError, IndexError):
            print(f"WARNING: Lap '{target_lap}' not found in .fit file; keeping all laps")

    data_log.laps_info = laps_info
