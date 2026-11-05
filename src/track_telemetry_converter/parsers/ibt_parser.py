"""Parser for IBT telemetry log files.

Canonical export channels and optional native sample views share one time axis."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from ..channels import (
    CH_LAP_NUMBER,
    CH_YAW_RATE,
    IBT_BRAKE_PRESS_MAP,
    IBT_CHANNEL_MAP,
    IBT_WHEEL_SPEED_MAP,
)
from ..lap_view import ChannelView, readonly
from ..provenance import record_yaw_source


class _IbtGeometry(NamedTuple):
    """Header/buffer geometry parsed from the .ibt file head."""

    tick_rate: int
    buf_offset: int
    n_ticks: int
    sess_info_len: int
    sess_info_offset: int
    num_vars: int
    var_header_offset: int
    buf_len: int


def _parse_ibt_timing(raw_file):
    """Parse irsdk_header + bufInfo; return geometry or None (prints why)."""
    import struct as _struct

    if len(raw_file) < 112:
        print("ERROR: .ibt file too small to be valid")
        return None

    # --- 1. Parse irsdk_header (12 ints = 48 bytes) ---
    # ver, status, tickRate, sessionInfoUpdate, sessionInfoLen, sessionInfoOffset,
    # numVars, varHeaderOffset, numBuf, bufLen, pad[2]
    (ver, status, tick_rate,
     sess_info_update, sess_info_len, sess_info_offset,
     num_vars, var_header_offset,
     num_buf, buf_len, _pad0, _pad1) = _struct.unpack_from("<12i", raw_file, 0)

    # --- 2. Read bufInfo[0] (first data buffer descriptor, at byte 48) ---
    # struct irsdk_bufInfo { int tickCount; int bufOffset; int pad[2]; }
    # In .ibt disk files this is written at offset 48 directly after the 12-int header.
    buf_tick_count, buf_offset = _struct.unpack_from("<2i", raw_file, 48)

    if buf_offset <= 0 or buf_offset >= len(raw_file):
        print("ERROR: Invalid .ibt data buffer offset")
        return None

    if tick_rate <= 0 or buf_len <= 0 or num_vars < 0:
        print("ERROR: Invalid .ibt tick rate or buffer geometry")
        return None
    if (var_header_offset < 0 or var_header_offset + num_vars * 144 > buf_offset
            or sess_info_len < 0 or sess_info_offset < 0
            or sess_info_offset + sess_info_len > buf_offset):
        print("ERROR: Invalid .ibt metadata offsets")
        return None
    n_ticks = (len(raw_file) - buf_offset) // buf_len
    if n_ticks <= 0:
        print("ERROR: .ibt file has no data ticks")
        return None

    return _IbtGeometry(
        tick_rate=tick_rate, buf_offset=buf_offset, n_ticks=n_ticks,
        sess_info_len=sess_info_len, sess_info_offset=sess_info_offset,
        num_vars=num_vars, var_header_offset=var_header_offset,
        buf_len=buf_len,
    )


def _parse_filename_start(ibt_file_path):
    """Return the recording-start datetime from the iRacing filename, or None.

    iRacing names telemetry files ``<car> <track> YYYY-MM-DD HH-MM-SS.ibt``
    with the session start time, e.g. ``mx5 ... 2026-11-04 10-28-59.ibt``.
    """
    import datetime as _dt
    import os as _os
    import re as _re

    match = _re.search(
        r"(\d{4})-(\d{2})-(\d{2}) (\d{2})-(\d{2})-(\d{2})",
        _os.path.basename(ibt_file_path),
    )
    if not match:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    try:
        return _dt.datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def _parse_ibt_session_info(data_log, raw_file, geo, ibt_file_path=""):
    """Extract venue/driver/car/datetime from the YAML session block."""
    # --- 3. Parse YAML Session Info ---
    sess_yaml = raw_file[geo.sess_info_offset: geo.sess_info_offset + geo.sess_info_len].decode("latin-1", errors="ignore")

    venue = ""
    driver = ""
    car = ""
    weekend_date = ""
    sess_dt = None

    for line in sess_yaml.splitlines():
        ls = line.strip()
        if ls.startswith("TrackDisplayName:"):
            venue = ls.split(":", 1)[1].strip()
        elif ls.startswith("UserName:"):
            driver = ls.split(":", 1)[1].strip()
        elif ls.startswith("CarScreenName:"):
            car = ls.split(":", 1)[1].strip()
        elif ls.startswith("Date:"):
            weekend_date = ls.split(":", 1)[1].strip()

    # Session start precedence: the iRacing filename timestamp is the
    # actual recording start (verified: start + duration == file mtime on
    # real files); the YAML WeekendDate is only the season weekend date
    # (often months off) and is kept as a renamed-file fallback.
    sess_dt = _parse_filename_start(ibt_file_path)
    if sess_dt is None and weekend_date:
        try:
            import datetime as _dt
            sess_dt = _dt.datetime.strptime(weekend_date, "%Y-%m-%d")
        except Exception:
            print(f"WARNING: Could not parse IBT session date: {weekend_date!r}")

    data_log.metadata["venue_name"] = venue
    data_log.metadata["driver"] = driver
    data_log.metadata["vehicle_id"] = car
    if sess_dt:
        data_log.datetime = sess_dt
        try:
            import datetime as _dt
            data_log.datetime_utc = sess_dt.astimezone(_dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            pass


def parse_ibt_log(data_log, ibt_file_path, *, raw_channels=None):
    """ Creates channels populated with messages from an iRacing native .ibt binary telemetry file.

    The .ibt format consists of:
      - irsdk_header (112 bytes): ver, status, tickRate, sessionInfoUpdate,
        sessionInfoLen, sessionInfoOffset, numVars, varHeaderOffset,
        numBuf, bufLen, + 2 pad ints
      - irsdk_diskSubHeader (32 bytes at offset 112): sessionStartDate,
        sessionStartTime, sessionEndTime, lapCount, recordCount
      - irsdk_varHeader array (144 bytes each): type, offset, count,
        countAsTime, name[32], desc[64], unit[32]
      - YAML session info block (UTF-8, sessionInfoLen bytes)
      - Data buffer: n_ticks * bufLen bytes, one tick row per sample

    irsdk var types: 0=char, 1=bool, 2=int32, 3=uint32, 4=float32, 5=float64
    Reference: https://sajax.github.io/irsdkdocs/
    """
    data_log.clear()
    data_log.laps_info = {}

    with open(ibt_file_path, "rb") as f:
        raw_file = f.read()

    geo = _parse_ibt_timing(raw_file)
    if geo is None:
        return
    dt = 1.0 / geo.tick_rate

    # --- 3. Parse YAML Session Info ---
    _parse_ibt_session_info(data_log, raw_file, geo, ibt_file_path)

    # --- 4. Parse irsdk_varHeader array ---
    # Each varHeader: type(i), offset(i), count(i), countAsTime(i),
    #                 name[32s], desc[64s], unit[32s]  => 4+4+4+4+32+64+32 = 144 bytes
    var_meta, raw_data, times = _parse_ibt_vars(raw_file, geo, dt)
    native_time = _extract_ibt_var(var_meta, raw_data, geo.buf_len, geo.n_ticks, "SessionTime")
    data_log.metadata["ibt"] = {
        "sample_rate": geo.tick_rate,
        "variables": {name: meta.as_dict() for name, meta in var_meta.items()},
    }
    if native_time is not None:
        data_log.source_timestamps = readonly(native_time)
        delta = np.diff(native_time)
        usable = np.isfinite(delta) & (delta > 0)
        times = np.r_[0.0, np.cumsum(np.where(usable, delta, dt))]
        data_log.metadata["ibt"]["time_repaired"] = bool(not np.all(usable))
        data_log.metadata["ibt"]["time_source"] = "SessionTime"
    else:
        data_log.metadata["ibt"]["time_source"] = "tick_rate"

    if raw_channels is not None:
        names = var_meta if raw_channels == "all" else (
            [raw_channels] if isinstance(raw_channels, str) else raw_channels
        )
        for name in names:
            if name not in var_meta:
                continue
            meta = var_meta[name]
            values = _native_ibt_var(meta, raw_data, geo.buf_len, geo.n_ticks)
            data_log.raw_channels[name] = ChannelView(
                name, meta.units, readonly(times), readonly(values), float(geo.tick_rate),
                MappingProxyType({
                    "source": "IBT", "variable": name, **meta.as_dict(),
                    "subsample_rate": float(geo.tick_rate * meta.count) if meta.count_as_time else None,
                }),
            )

    for ibt_name, ch_name, units, dec, conv in IBT_CHANNEL_MAP:
        _add_ibt_channel(data_log, var_meta, raw_data, geo, times, ibt_name, ch_name, units, dec, conv)

    for ibt_name, ch_name in IBT_WHEEL_SPEED_MAP:
        _add_ibt_channel(data_log, var_meta, raw_data, geo, times, ibt_name, ch_name, "km/h", 2, lambda x: x * 3.6)

    for ibt_name, ch_name in IBT_BRAKE_PRESS_MAP:
        _add_ibt_channel(data_log, var_meta, raw_data, geo, times, ibt_name, ch_name, "kPa", 2, lambda x: x * 100.0)

    if CH_YAW_RATE in data_log.channels:
        record_yaw_source(data_log, "ibt_recorded_YawRate")

    lap_arr = _extract_ibt_var(var_meta, raw_data, geo.buf_len, geo.n_ticks, "Lap")
    from .ibt_laps import build_ibt_laps

    signals = {
        key: _extract_ibt_var(var_meta, raw_data, geo.buf_len, geo.n_ticks, name)
        for key, name in {
            "lap_dist": "LapDistPct", "session_num": "SessionNum",
            "on_track": "IsOnTrack", "lap_completed": "LapCompleted",
            "pit_road": "OnPitRoad",
        }.items()
    }
    build_ibt_laps(data_log, lap_arr, times, session_time=native_time, **signals)
    data_log.channel_metadata[CH_LAP_NUMBER] = {
        "source": "IBT", "variable": "Lap", "sample_rate": float(geo.tick_rate),
        "derived": True,
    }

    # Cleanup empty channels
    empty = [name for name, ch in data_log.channels.items() if not ch.messages]
    for name in empty:
        del data_log.channels[name]


@dataclass(frozen=True)
class IbtVariable:
    """Recorded variable header, including its native type and array width."""

    dtype: np.dtype
    offset: int
    count: int
    units: str
    type_code: int
    count_as_time: bool
    description: str

    def as_dict(self):
        return {
            "dtype": self.dtype.str, "type_code": self.type_code,
            "count": self.count, "count_as_time": self.count_as_time,
            "description": self.description, "units": self.units,
        }


def _parse_ibt_vars(raw_file, geo, dt):
    """Parse var headers; return (var_meta, raw_data, times)."""
    import struct as _struct

    _ibt_type_map = {0: "S1", 1: "?", 2: "<i4", 3: "<u4", 4: "<f4", 5: "<f8"}

    var_meta = {}  # name -> validated native variable header
    for i in range(geo.num_vars):
        base = geo.var_header_offset + i * 144
        if base + 144 > len(raw_file):
            break
        vtype, voffset, vcount = _struct.unpack_from("<3i", raw_file, base)
        vname = raw_file[base + 16: base + 48].split(b"\x00")[0].decode("latin-1")
        vunit = raw_file[base + 112: base + 144].split(b"\x00")[0].decode("latin-1")
        if vtype not in _ibt_type_map:
            continue
        dtype = np.dtype(_ibt_type_map[vtype])
        if not vname or vcount <= 0 or voffset < 0 or voffset + dtype.itemsize * vcount > geo.buf_len:
            continue
        desc = raw_file[base + 48:base + 112].split(b"\x00")[0].decode("latin-1")
        var_meta[vname] = IbtVariable(
            dtype, voffset, vcount, vunit, vtype, bool(raw_file[base + 12]), desc,
        )

    raw_data = memoryview(raw_file)[geo.buf_offset: geo.buf_offset + geo.n_ticks * geo.buf_len]
    times = np.arange(geo.n_ticks, dtype=np.float64) * dt
    return var_meta, raw_data, times


def _extract_ibt_var(var_meta, raw_data, buf_len, n_ticks, var_name):
    if var_name not in var_meta:
        return None
    meta = var_meta[var_name]
    if meta.count != 1 or meta.type_code == 0:
        return None
    return _native_ibt_var(meta, raw_data, buf_len, n_ticks).astype(np.float64)


def _native_ibt_var(meta, raw_data, buf_len, n_ticks):
    shape = (n_ticks,) if meta.count == 1 else (n_ticks, meta.count)
    strides = (buf_len,) if meta.count == 1 else (buf_len, meta.dtype.itemsize)
    return np.ndarray(shape, dtype=meta.dtype, buffer=raw_data, offset=meta.offset, strides=strides)


def _add_ibt_channel(data_log, var_meta, raw_data, geo, times, ibt_name, ch_name, units, decimals, convert=None):
    arr = _extract_ibt_var(var_meta, raw_data, geo.buf_len, geo.n_ticks, ibt_name)
    if arr is None:
        return
    if ibt_name == "YawNorth" and arr.max() <= 0.0:
        return
    data_log.add_channel(ch_name, units, float, decimals)
    vals = convert(arr) if convert else arr
    data_log.channels[ch_name].set_samples(times, np.asarray(vals, dtype=np.float64))
    data_log.channel_metadata[ch_name] = {
        "source": "IBT", "variable": ibt_name, "sample_rate": float(geo.tick_rate),
        "native_units": var_meta[ibt_name].units, "converted": convert is not None,
    }
