"""Parser for IBT telemetry log files.

Extracted from the original DataLog.ibt_log methods with identical behavior."""

from __future__ import annotations

import numpy as np

from ..channels import CH_LAP_NUMBER, IBT_BRAKE_PRESS_MAP, IBT_CHANNEL_MAP, IBT_WHEEL_SPEED_MAP
from ..models import Message


def parse_ibt_log(data_log, ibt_file_path):
    """ Creates channels populated with messages from an iRacing native .ibt binary telemetry file.

    The .ibt format consists of:
      - irsdk_header (48 bytes): ver, status, tickRate, sessionInfoUpdate,
        sessionInfoLen, sessionInfoOffset, numVars, varHeaderOffset,
        numBuf, bufLen, + 2 pad ints
      - irsdk_diskSubHeader (32 bytes at offset 48): sessionStartDate,
        sessionStartTime, sessionEndTime, lapCount, recordCount
      - irsdk_varHeader array (144 bytes each): type, offset, count,
        countAsTime, name[32], desc[64], unit[32]
      - YAML session info block (UTF-8, sessionInfoLen bytes)
      - Data buffer: n_ticks * bufLen bytes, one tick row per sample

    irsdk var types: 0=char, 1=bool, 2=int32, 3=uint32, 4=float32, 5=float64
    Reference: https://sajax.github.io/irsdkdocs/
    """
    import struct as _struct

    data_log.clear()
    data_log.laps_info = {}

    with open(ibt_file_path, "rb") as f:
        raw_file = f.read()

    if len(raw_file) < 112:
        print("ERROR: .ibt file too small to be valid")
        return

    # --- 1. Parse irsdk_header (12 ints = 48 bytes) ---
    # ver, status, tickRate, sessionInfoUpdate, sessionInfoLen, sessionInfoOffset,
    # numVars, varHeaderOffset, numBuf, bufLen, pad[2]
    (ver, status, tick_rate,
     sess_info_update, sess_info_len, sess_info_offset,
     num_vars, var_header_offset,
     num_buf, buf_len, _pad0, _pad1) = _struct.unpack_from("12i", raw_file, 0)

    # --- 2. Read bufInfo[0] (first data buffer descriptor, at byte 48) ---
    # struct irsdk_bufInfo { int tickCount; int bufOffset; int pad[2]; }
    # In .ibt disk files this is written at offset 48 directly after the 12-int header.
    buf_tick_count, buf_offset = _struct.unpack_from("2i", raw_file, 48)

    if buf_offset <= 0 or buf_offset >= len(raw_file):
        print("ERROR: Invalid .ibt data buffer offset")
        return

    n_ticks = (len(raw_file) - buf_offset) // buf_len
    if n_ticks <= 0:
        print("ERROR: .ibt file has no data ticks")
        return

    dt = 1.0 / tick_rate

    # --- 3. Parse YAML Session Info ---
    sess_yaml = raw_file[sess_info_offset: sess_info_offset + sess_info_len].decode("latin-1", errors="ignore")

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

    # Parse session datetime from YAML WeekendDate field (e.g. "2026-08-07")
    if weekend_date:
        try:
            import datetime as _dt
            sess_dt = _dt.datetime.strptime(weekend_date, "%Y-%m-%d")
        except Exception:
            pass

    data_log.metadata["venue_name"] = venue
    data_log.metadata["driver"] = driver
    data_log.metadata["vehicle_id"] = car
    if sess_dt:
        data_log.datetime = sess_dt

    # --- 4. Parse irsdk_varHeader array ---
    # Each varHeader: type(i), offset(i), count(i), countAsTime(i),
    #                 name[32s], desc[64s], unit[32s]  => 4+4+4+4+32+64+32 = 144 bytes
    _ibt_type_map = {0: np.int8, 1: np.int8, 2: np.int32, 3: np.uint32, 4: np.float32, 5: np.float64}

    var_meta = {}  # name -> (numpy_dtype, byte_offset_in_tick, count)
    for i in range(num_vars):
        base = var_header_offset + i * 144
        if base + 144 > len(raw_file):
            break
        vtype, voffset, vcount = _struct.unpack_from("3i", raw_file, base)
        vname = raw_file[base + 16: base + 48].split(b"\x00")[0].decode("latin-1")
        vunit = raw_file[base + 112: base + 144].split(b"\x00")[0].decode("latin-1")
        dtype = _ibt_type_map.get(vtype, np.float32)
        var_meta[vname] = (dtype, voffset, vcount, vunit)

    raw_data = raw_file[buf_offset: buf_offset + n_ticks * buf_len]
    times = [i * dt for i in range(n_ticks)]

    def _extract(var_name):
        if var_name not in var_meta:
            return None
        dtype, voff, vcount, vunit = var_meta[var_name]
        try:
            return np.ndarray((n_ticks,), dtype=dtype, buffer=raw_data,
                              offset=voff, strides=(buf_len,)).astype(np.float64)
        except Exception:
            return None

    def _add_ch(ibt_name, ch_name, units, decimals, convert=None):
        arr = _extract(ibt_name)
        if arr is None:
            return
        if ibt_name == "YawNorth" and arr.max() <= 0.0:
            return
        data_log.add_channel(ch_name, units, float, decimals)
        vals = convert(arr) if convert else arr
        ch = data_log.channels[ch_name]
        for i in range(n_ticks):
            ch.messages.append(Message(times[i], float(vals[i])))

    for ibt_name, ch_name, units, dec, conv in IBT_CHANNEL_MAP:
        _add_ch(ibt_name, ch_name, units, dec, conv)

    for ibt_name, ch_name in IBT_WHEEL_SPEED_MAP:
        _add_ch(ibt_name, ch_name, "km/h", 2, lambda x: x * 3.6)

    for ibt_name, ch_name in IBT_BRAKE_PRESS_MAP:
        _add_ch(ibt_name, ch_name, "kPa", 2, lambda x: x * 100.0)

    # --- 6. Lap detection: raw Lap counter transitions ---
    # Every forward increment of the iRacing 'Lap' counter represents a
    # real crossing of the S/F timing line (including quick-reset laps).
    # Backward transitions (e.g. 16->0) are quick-reset drops and are
    # automatically skipped because prev_lap only updates on a beacon hit.
    lap_arr = _extract("Lap")
    data_log.add_channel(CH_LAP_NUMBER, "", float, 0)

    if lap_arr is not None and len(lap_arr) > 1:
        beacons = []
        prev_lap = lap_arr[0]
        for i in range(1, len(lap_arr)):
            new_val = lap_arr[i]
            if new_val > prev_lap and new_val > 0:
                beacons.append((times[i], f"Lap {int(new_val)}"))
                prev_lap = new_val

        data_log.laps_info["beacons"] = beacons

        # Build lap_number channel (0 before first S/F crossing, then 1,2,3...)
        lap_nums = np.zeros(n_ticks, dtype=int)
        label_counter = 0
        prev_lap_val = lap_arr[0]
        for i in range(1, n_ticks):
            v = lap_arr[i]
            if v > prev_lap_val and v > 0:
                label_counter += 1
                prev_lap_val = v
            lap_nums[i] = label_counter

        for i in range(n_ticks):
            data_log.channels[CH_LAP_NUMBER].messages.append(Message(times[i], float(lap_nums[i])))

        # Build laps_info structure (every segment between beacons is a lap)
        lap_items = []
        total_dur = times[-1] if len(times) > 0 else 0.0
        fastest_lap = None
        fastest_dur = float("inf")

        if beacons:
            first_t = beacons[0][0]
            if first_t > 0:
                lap_items.append({
                    "type": "Out Lap",
                    "lap_label": "Out Lap",
                    "lap_num": 1,
                    "start_time": 0.0,
                    "end_time": first_t,
                    "duration": first_t,
                    "stint": 0
                })

            for ci in range(len(beacons) - 1):
                s_t = beacons[ci][0]
                e_t = beacons[ci + 1][0]
                dur = e_t - s_t
                lap_num = len(lap_items) + 1
                lap_items.append({
                    "type": "Timed",
                    "lap_label": str(lap_num),
                    "lap_num": lap_num,
                    "start_time": s_t,
                    "end_time": e_t,
                    "duration": dur,
                    "stint": 0
                })
                if dur < fastest_dur:
                    fastest_dur = dur
                    fastest_lap = lap_num

            last_t = beacons[-1][0]
            remaining = total_dur - last_t
            if remaining > 0:
                lap_num = len(lap_items) + 1
                lap_items.append({
                    "type": "In Lap",
                    "lap_label": "In Lap",
                    "lap_num": lap_num,
                    "start_time": last_t,
                    "end_time": total_dur,
                    "duration": remaining,
                    "stint": 0
                })

        data_log.laps_info["laps"] = lap_items
        data_log.laps_info["total_laps"] = len(lap_items)
        data_log.laps_info["fastest_lap"] = fastest_lap if fastest_lap is not None else 1
        data_log.laps_info["fastest_time"] = fastest_dur if fastest_dur != float("inf") else 0.0
    else:
        data_log.channels[CH_LAP_NUMBER].messages = [Message(times[0], 1.0)]

    # Cleanup empty channels
    empty = [name for name, ch in data_log.channels.items() if not ch.messages]
    for name in empty:
        del data_log.channels[name]
