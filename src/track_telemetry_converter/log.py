from __future__ import annotations

import datetime
import math

import numpy as np

from .channels import (
    CH_CG_ACCEL_LAT,
    CH_ENGINE_RPM,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GROUND_SPEED,
    CH_RUNNING_TIME,
    TRACK_BEACONS,
)
from .models import Channel


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


class DataLog(object):
    channels: dict[str, Channel]
    """ Container for storing log data which contains a set of channels with time series data."""

    def __init__(self, name=""):
        self.name = name
        self.channels = {}
        self.datetime = None
        self.datetime_utc = None
        self.time_origin_epoch_ms = None
        self.session_created_epoch_ms = None
        self.metadata = {}
        self.traps = []

    def clear(self):
        self.channels = {}
        self.datetime = None
        self.datetime_utc = None
        self.time_origin_epoch_ms = None
        self.session_created_epoch_ms = None
        self.metadata = {}
        self.traps = []

    def add_channel(self, name, units, data_type, decimals, initial_message=None):
        if any(k in name.lower() for k in ["latitude", "longitude", "lat", "lon"]):
            decimals = max(decimals, 7)
        msg = [] if not initial_message else [initial_message]
        self.channels[name] = Channel(name, units, data_type, decimals, msg)

    def start(self):
        """ Returns the earliest timestamp from all existing channels [s]. """
        t = math.inf
        for name, channel in self.channels.items():
            t = min(t, channel.start())

        return t if t != math.inf else 0.0

    def end(self):
        """ Returns the latest timestamp from all existing channels [s]. """
        end = -math.inf
        for name, channel in self.channels.items():
            end = max(end, channel.end())

        return end if end != -math.inf else 0.0

    def duration(self):
        """ Returns the duration of the log [s]. """
        s = self.start()
        e = self.end()
        return max(0.0, e - s)

    def detect_native_frequency(self) -> float:
        """ Detects the primary native sampling frequency of the log data. """
        # 1. Check device_update_rate channel if present
        if "device_update_rate" in self.channels:
            vals = [m.value for m in self.channels["device_update_rate"].messages if m.value > 0]
            if vals:
                rate = float(np.mean(vals))
                for std_rate in [10.0, 20.0, 25.0, 50.0, 100.0]:
                    if abs(rate - std_rate) / std_rate <= 0.10:
                        return std_rate

        # 2. Inspect key physical channels
        candidates = [CH_GROUND_SPEED, CH_GPS_LATITUDE, CH_RUNNING_TIME, CH_CG_ACCEL_LAT, CH_ENGINE_RPM]
        raw_freq = 0.0
        for name in candidates:
            if name in self.channels and len(self.channels[name].messages) > 1:
                raw_freq = self.channels[name].avg_frequency()
                if raw_freq > 0:
                    break
        if raw_freq <= 0:
            all_freqs = [c.avg_frequency() for c in self.channels.values() if len(c.messages) > 1]
            raw_freq = max(all_freqs) if all_freqs else 25.0

        # Snap to standard logging rates (up to 50Hz for auto-detection)
        for std_rate in [10.0, 20.0, 25.0, 50.0]:
            if abs(raw_freq - std_rate) / std_rate <= 0.10:
                return std_rate

        if raw_freq > 50.0:
            return 50.0

        return max(1.0, round(raw_freq, 1))

    def resample(self, frequency="auto", mask_interp_gaps=False):
        """ Resamples all channels such that all messages occur at a fixed frequency.

        frequency: float, int, or 'auto' (detects native sample rate, e.g. 20Hz, 25Hz, 50Hz).
        mask_interp_gaps: bool, if True sets interpolated values across sample gaps (>1s) to NaN.
        Returns the frequency used for resampling.
        """
        if isinstance(frequency, str):
            if frequency.lower() == "auto":
                frequency = self.detect_native_frequency()
            else:
                frequency = float(frequency)
        elif frequency is None:
            frequency = self.detect_native_frequency()
        elif isinstance(frequency, (int, float)):
            if frequency <= 0:
                frequency = self.detect_native_frequency()
            else:
                frequency = float(frequency)
        else:
            frequency = self.detect_native_frequency()

        start = self.start()
        end = self.end()
        for channel_name in self.channels:
            self.channels[channel_name].resample(start, end, frequency, mask_interp_gaps=mask_interp_gaps)
        return frequency

    def detect_beacons(self, min_speed_kmh=30.0, min_time_sec=15.0, min_lap_sec=40.0):
        """ Detects trap / sector crossing timestamps from GPS coordinates and traps metadata. """
        traps = list(getattr(self, "traps", []) or [])

        if not traps:
            venue = self.metadata.get("venue_name", "").lower().replace("_", " ").replace("-", " ").replace(",", " ")
            if "ca 9 n" in venue or "ca-9 n" in venue or "highway 9 north" in venue:
                traps = TRACK_BEACONS["CA-9 N"]
            elif "ca 9" in venue or "ca-9" in venue or "highway 9" in venue or "saratoga" in venue:
                traps = TRACK_BEACONS["CA-9 S"]
            elif "laguna" in venue or "seca" in venue:
                traps = [TRACK_BEACONS["WeatherTech Raceway Laguna Seca"]]
            elif "sonoma" in venue:
                traps = [TRACK_BEACONS["Sonoma Raceway"]]
            elif any(k in venue for k in ["thunderhill", "thunder hill", "thill", "thunderhil"]):
                if any(k in venue for k in ["5 mile", "5 miles", "5mi"]):
                    traps = [TRACK_BEACONS["Thunderhill 5 Mile Double Bypass"]]
                elif "cyclone" in venue:
                    traps = [TRACK_BEACONS["Thunderhill East Cyclone"]]
                elif "west" not in venue:
                    traps = [TRACK_BEACONS["Thunderhill East Bypass"]]

        if isinstance(traps, dict):
            traps = [traps]

        if not traps:
            return []
        if CH_GPS_LATITUDE not in self.channels or CH_GPS_LONGITUDE not in self.channels:
            return []

        lat_m = self.channels[CH_GPS_LATITUDE].messages
        lon_m = self.channels[CH_GPS_LONGITUDE].messages
        if not lat_m or not lon_m or len(lat_m) != len(lon_m):
            return []

        spd_m = self.channels.get(CH_GROUND_SPEED)
        spd_vals = np.array([m.value for m in spd_m.messages]) if spd_m else None
        hdg_m = self.channels.get(CH_GPS_HEADING)
        hdg_vals = np.array([m.value for m in hdg_m.messages]) if hdg_m else None

        lat = np.array([m.value for m in lat_m])
        lon = np.array([m.value for m in lon_m])
        times = np.array([m.timestamp for m in lat_m])
        dur = self.duration()

        beacons = []
        for t in traps:
            t_lat = t["lat"]
            t_lon = t["lon"]
            t_name = t.get("name", "Start/Finish")
            t_hdg = t.get("heading_deg", None)

            # Flat earth distance approximation (meters around ~35-45 deg lat)
            d_lat = (lat - t_lat) * 111000.0
            d_lon = (lon - t_lon) * 85670.0
            dist = np.sqrt(d_lat**2 + d_lon**2)

            cand_indices = []
            for i in range(1, len(dist) - 1):
                if dist[i] > 25.0:
                    continue
                if spd_vals is not None and spd_vals[i] < min_speed_kmh:
                    continue
                if times[i] < min_time_sec or (dur > 0 and (dur - times[i]) < 5.0):
                    continue
                if t_hdg is not None and hdg_vals is not None:
                    ang_diff = np.abs((hdg_vals[i] - t_hdg + 180) % 360 - 180)
                    if ang_diff > 35.0:
                        continue

                if dist[i] <= dist[i - 1] and dist[i] <= dist[i + 1]:
                    cand_indices.append(i)

            last_t = -999.0
            for idx in cand_indices:
                t_val = times[idx]
                if t_val - last_t >= min_lap_sec:
                    beacons.append((float(t_val), t_name))
                    last_t = t_val

        beacons.sort(key=lambda x: x[0])
        return beacons

    def calculate_math_channels(self, g_source="auto", gear_ratio_thresholds=None):
        from .derived import calculate_math_channels as _calc
        _calc(
            self,
            g_source=g_source,
            gear_ratio_thresholds=gear_ratio_thresholds,
        )

    def _extract_datetime_from_text(self, log_lines, file_path=""):
        import os
        import re

        # 1. Filename pattern: session_YYYYMMDD_HHMMSS or YYYYMMDD_HHMM
        if file_path:
            filename = os.path.basename(file_path)
            match = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})?", filename)
            if match:
                groups = match.groups()
                y, m, d, hh, mm = map(int, groups[:5])
                ss = int(groups[5]) if groups[5] else 0
                try:
                    return datetime.datetime(y, m, d, hh, mm, ss)
                except ValueError:
                    pass

        if not log_lines:
            return None

        date_str = ""
        time_str = ""

        import csv

        # 2. Check header and data lines
        for line in log_lines[:50]:
            line_clean = line.strip()
            if not line_clean:
                continue

            # Check candump format: (1630268615.800257)
            if line_clean.startswith("(") and ")" in line_clean:
                try:
                    stamp = float(line_clean.split(")")[0][1:])
                    if stamp > 1e8:
                        return datetime.datetime.fromtimestamp(stamp)
                except ValueError:
                    pass

            try:
                parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
            except Exception:
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]

            if len(parts) >= 2:
                k = parts[0].lower()
                v = parts[1]
                if k in ["date", "log date", "created"]:
                    date_str = v
                elif k in ["time", "log time"] and (":" in v or "am" in v.lower() or "pm" in v.lower()):
                    time_str = v

            if parts and parts[0]:
                try:
                    stamp = float(parts[0])
                    if stamp > 1e8:
                        return datetime.datetime.fromtimestamp(stamp)
                except ValueError:
                    pass

        if date_str:
            dt_raw = f"{date_str} {time_str}".strip()
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%y %H:%M:%S",
                "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%A, %B %d, %Y %I:%M %p",
                "%A, %B %d, %Y %H:%M",
                "%Y-%m-%d",
                "%m/%d/%y",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(dt_raw, fmt)
                except ValueError:
                    pass

        return None

    def _extract_metadata(self, log_lines=None, file_path=""):
        import os
        metadata = {
            "driver": "",
            "vehicle_id": "",
            "venue_name": "",
            "event_name": "",
            "event_session": "",
            "short_comment": ""
        }

        filename = os.path.basename(file_path) if file_path else ""

        # 1. Filename pattern (_&_ split)
        if filename and "_&_" in filename:
            parts = filename.replace(".csv", "").replace(".rcz", "").split("_&_")
            if len(parts) >= 4:
                metadata["venue_name"] = parts[0]
                metadata["vehicle_id"] = parts[1]
                metadata["driver"] = parts[2]
                metadata["event_session"] = parts[3]

        # 2. Check header lines if available
        if log_lines:
            for line in log_lines[:30]:
                line_clean = line.strip().strip('"').strip("'")
                parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
                if len(parts) >= 2:
                    k = parts[0].lower()
                    v = parts[1]
                    if not v:
                        continue
                    if k in ["driver", "racer"]:
                        metadata["driver"] = v
                    elif k in ["vehicle", "car"]:
                        metadata["vehicle_id"] = v
                    elif k in ["session"]:
                        metadata["event_session"] = v
                    elif k in ["event", "championship"]:
                        metadata["event_name"] = v
                    elif k in ["venue", "track"]:
                        metadata["venue_name"] = v
                    elif k in ["comment", "notes"]:
                        metadata["short_comment"] = v

        # 3. Fallback venue inference from filename
        if not metadata["venue_name"] and filename:
            fn_lower = filename.lower()
            if "thunder" in fn_lower:
                metadata["venue_name"] = "Thunderhill Raceway Park"
            elif "laguna" in fn_lower:
                metadata["venue_name"] = "Laguna Seca"
            elif "sonoma" in fn_lower:
                metadata["venue_name"] = "Sonoma Raceway"

        self.metadata.update({k: v for k, v in metadata.items() if v})

    def from_can_log(self, log_lines, can_db):
        from .parsers.can_parser import parse_can_log
        parse_can_log(self, log_lines, can_db)

    def from_csv_log(self, log_lines):
        from .parsers.csv_parser import parse_csv_log
        parse_csv_log(self, log_lines)

    def from_racechrono_log(self, log_lines, target_lap=None):
        from .parsers.racechrono_csv_parser import parse_racechrono_log
        parse_racechrono_log(self, log_lines, target_lap=target_lap)

    def from_ibt_log(self, ibt_file_path):
        from .parsers.ibt_parser import parse_ibt_log
        parse_ibt_log(self, ibt_file_path)

    def from_vbo_log(self, log_lines, target_lap=None):
        from .parsers.vbo_parser import parse_vbo_log
        parse_vbo_log(self, log_lines, target_lap=target_lap)

    def from_pbbuddy_log(self, log_lines, target_lap=None):
        from .parsers.pbbuddy_parser import parse_pbbuddy_log
        parse_pbbuddy_log(self, log_lines, target_lap=target_lap)

    def from_rcz_log(self, rcz_file_path, target_lap=None, target_stint=None,
                     min_lap_sec=15.0, mask_interp_gaps=False,
                     target_session=None):
        from .parsers.rcz_parser import parse_rcz_log
        parse_rcz_log(
            self,
            rcz_file_path,
            target_lap=target_lap,
            target_stint=target_stint,
            min_lap_sec=min_lap_sec,
            mask_interp_gaps=mask_interp_gaps,
            target_session=target_session,
        )

    def from_accessport_log(self, log_lines):
        from .parsers.accessport_parser import parse_accessport_log
        parse_accessport_log(self, log_lines)

    def from_xrk_log(self, xrk_file_path, target_lap=None):
        from .parsers.xrk_parser import parse_xrk_log
        parse_xrk_log(self, xrk_file_path, target_lap=target_lap)

    def from_fit_log(self, fit_file_path, target_lap=None):
        from .parsers.fit_parser import parse_fit_log
        parse_fit_log(self, fit_file_path, target_lap=target_lap)

    @staticmethod
    def __parse_can_log_line(line):
        try:
            stamp, bus, msg = line.split()
            stamp = float(stamp[1:-1])
            can_id, data = msg.split("#")
            can_id = int(can_id, 16)
            data = bytearray.fromhex(data)
            return stamp, bus, can_id, data
        except (ValueError, IndexError):
            return None, None, None, None

    def __str__(self):
        output = "Log: %s, Duration: %f s" % (self.name, (self.end() - self.start()))
        for channel_name, channel_data in self.channels.items():
            output += "\n\t%s" % channel_data
        return output
