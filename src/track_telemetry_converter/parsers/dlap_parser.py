"""Parser for Dragy `.dlap` telemetry archives.

A `.dlap` file is an outer ZIP containing a track-map PNG and one AES-128-CBC
encrypted `<groupId>.group` blob. Decrypting the blob yields an inner ZIP with:

- `loopdata_*.txt`: 25 Hz GPS/IMU CSV, 27 columns (see `_LOOPDATA_CHANNELS`).
  Column 1 is GPS seconds-of-week (not milliseconds); the session clock is
  rebased to zero from its first sample. Only a subset carries telemetry;
  the remaining columns are diagnostics that Dragy's own frontend also
  ignores.
- `*.json`: session metadata with per-lap `startDataIndex`/`endDataIndex`
  row ranges into the loopdata file.
- `*.obd`: OBD CSV (`t,hz,rpm,acc_pos,brake,oil,coolant,map`) on a
  session-relative seconds time base. OBD coverage may be partial (the dongle
  can drop mid-session); samples outside the recorded window stay `NaN`.
- `*.obin` / `*.txt` / `*.plist`: ELM327 raw log, app debug log, pedal
  calibration; informational only.

Column semantics were reverse-engineered from Dragy's web frontend
(`processLine`/`CsvParseGps` for the loopdata columns, `decryptFile` +
`unZip`/`decodeGroup` for the AES layer, `interpolateFrame` for the
per-point field set) and cross-validated against a paired
Dragy-exported `.vbo` of the same session (speed/lat/lon/alt/heading/sats
r > 0.998, OBD rpm r > 0.999 at zero offset).

Notes and gotchas:
- Column 1 is GPS seconds-of-week, not milliseconds: 602837 at session
  start is Saturday 23:26:59 UTC (day 6 SOW 602818 + 19 leap seconds).
- Columns 16 and 18-24 are signal/fix diagnostics the frontend never
  reads; they are skipped here.
- OBD coverage may end mid-session (observed: dongle falls silent while
  BLE stays up; Dragy's own VBO writer forward-fills stale values from
  there on, this importer records NaN instead).

The AES key/IV are shipped inside Dragy's public web frontend and are needed
to read the user's own file; decryption requires `pycryptodome`
(`pip install .[dlap]`).
"""

from __future__ import annotations

import datetime
import io
import json
import zipfile

import numpy as np

from ..channels import (
    CH_ACCELERATOR_POS,
    CH_BRAKE_POS,
    CH_BRAKE_PRESS,
    CH_CG_ACCEL_LON,
    CH_COOLANT_TEMP,
    CH_CORR_DIST,
    CH_ENGINE_OIL_TEMP,
    CH_ENGINE_RPM,
    CH_GPS_ACCURACY,
    CH_GPS_ALTITUDE,
    CH_GPS_HEADING,
    CH_GPS_LATITUDE,
    CH_GPS_LONGITUDE,
    CH_GPS_SATS,
    CH_GROUND_SPEED,
    CH_MANIFOLD_PRESS,
)
from ..interpolation import _mask_interp_outside_range

# AES-128-CBC key/IV as embedded in Dragy's public web frontend
# (`static KEY` / `static IV`, NUL-padded to 16 bytes).
_DLAP_AES_KEY = bytes.fromhex("6932466c655a6e640000000000000000")
_DLAP_AES_IV = bytes.fromhex("69336576386c656e0000000000000000")

# loopdata column index -> (canonical channel, units, decimals).
# Column 0 is per-lap time, 1 is GPS seconds-of-week, 15 is a row index,
# 25/26 are epoch/obd timestamps; 16 and 18-24 are diagnostics ignored by
# Dragy's own frontend (fix type, signal quality) and skipped here as well.
_LOOPDATA_CHANNELS = {
    2: (CH_GROUND_SPEED, "km/h", 2),
    3: (CH_CG_ACCEL_LON, "G", 4),
    4: (CH_GPS_ALTITUDE, "m", 2),
    5: (CH_GPS_LATITUDE, "deg", 7),
    6: (CH_GPS_LONGITUDE, "deg", 7),
    7: (CH_CORR_DIST, "m", 2),
    8: (CH_GPS_ACCURACY, "m", 2),
    9: (CH_GPS_SATS, "", 0),
    10: (CH_GPS_HEADING, "deg", 2),
    11: (CH_BRAKE_POS, "%", 2),
}

# *.obd pid name -> (canonical channel, units, decimals). The pid order comes
# from the session JSON (`obd.pidEnums`); this is the fallback when absent.
_OBD_PID_CHANNELS = {
    "PID_RPM": (CH_ENGINE_RPM, "rpm", 0),
    "PID_ACC_POS": (CH_ACCELERATOR_POS, "%", 2),
    "PID_BRAKE_PRESS": (CH_BRAKE_PRESS, "kPa", 2),
    "PID_OIL_TEMP": (CH_ENGINE_OIL_TEMP, "C", 2),
    "PID_COOLANT_TEMP": (CH_COOLANT_TEMP, "C", 2),
    "PID_MAP": (CH_MANIFOLD_PRESS, "kPa", 2),
}
_FALLBACK_PID_ORDER = (
    "PID_RPM",
    "PID_ACC_POS",
    "PID_BRAKE_PRESS",
    "PID_OIL_TEMP",
    "PID_COOLANT_TEMP",
    "PID_MAP",
)


def _new_aes_cipher():
    try:
        from Crypto.Cipher import AES
    except ImportError as exc:
        raise RuntimeError(
            "Dragy .dlap import requires the 'pycryptodome' package "
            '(pip install ".[dlap]").',
        ) from exc
    return AES.new(_DLAP_AES_KEY, AES.MODE_CBC, _DLAP_AES_IV)


def _open_inner_zip(dlap_path):
    """Decrypt the `.group` entry of a `.dlap` archive; return inner files."""
    try:
        outer = zipfile.ZipFile(dlap_path, "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid .dlap file (not a ZIP archive): {dlap_path}") from exc
    with outer:
        group_names = [n for n in outer.namelist() if n.endswith(".group")]
        if not group_names:
            raise ValueError(f"Invalid .dlap file (no .group entry): {dlap_path}")
        if len(group_names) > 1:
            print(f"WARNING: Multiple .group entries, using {group_names[0]}")
        encrypted = outer.read(group_names[0])
    if len(encrypted) % 16 != 0 or not encrypted:
        raise ValueError("Invalid .dlap file (corrupt .group payload)")
    plain = _new_aes_cipher().decrypt(encrypted)
    pad = plain[-1]
    if not 1 <= pad <= 16 or plain[-pad:] != bytes([pad]) * pad:
        raise ValueError(
            "Failed to decrypt .dlap file (bad key/padding); "
            "the file may come from a newer Dragy app version.",
        )
    plain = plain[:-pad]
    if plain[:2] != b"PK":
        raise ValueError("Failed to decrypt .dlap file (not an inner ZIP archive)")
    try:
        inner = zipfile.ZipFile(io.BytesIO(plain), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError("Failed to decrypt .dlap file (bad inner archive)") from exc
    return inner


def _read_inner_text(inner, names):
    for name in names:
        try:
            return inner.read(name).decode("utf-8", errors="ignore"), name
        except KeyError:
            continue
    return None, None


def _parse_loopdata(text):
    """Parse loopdata CSV into a float matrix; blanks become NaN."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append([float(v) if v.strip() else np.nan for v in line.split(",")])
        except ValueError:
            continue
    if not rows:
        raise ValueError("Invalid .dlap file (empty loopdata)")
    width = max(len(r) for r in rows)
    mat = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        mat[i, : len(row)] = row
    return mat


def _parse_obd_pids(obd_meta):
    if isinstance(obd_meta, dict):
        raw = obd_meta.get("pidEnums") or obd_meta.get("pids")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(p).upper() for p in parsed]
            except ValueError:
                pass
        elif isinstance(raw, list):
            return [str(p).upper() for p in raw]
    return list(_FALLBACK_PID_ORDER)


def parse_dlap_log(data_log, dlap_file_path, target_lap=None):
    """Populate channels from a Dragy `.dlap` archive."""
    data_log.clear()
    data_log.laps_info = {}

    inner = _open_inner_zip(dlap_file_path)
    with inner:
        names = inner.namelist()
        loop_text, loop_name = _read_inner_text(
            inner, [n for n in names if n.startswith("loopdata")],
        )
        if loop_text is None:
            raise ValueError("Invalid .dlap file (no loopdata_*.txt entry)")
        meta_text, _ = _read_inner_text(inner, [n for n in names if n.endswith(".json")])
        obd_text, _ = _read_inner_text(inner, [n for n in names if n.endswith(".obd")])

    loop = _parse_loopdata(loop_text)
    if loop.shape[0] < 2 or loop.shape[1] < 26:
        raise ValueError("Invalid .dlap file (loopdata too short)")

    dev_sow = loop[:, 1]
    t0 = float(np.nanmin(dev_sow))
    t_rel = dev_sow - t0

    epoch_s = loop[:, 25]
    if np.isfinite(epoch_s).any():
        origin_s = float(epoch_s[np.isfinite(epoch_s)][0])
        data_log.time_origin_epoch_ms = int(origin_s * 1000)
        data_log.datetime_utc = datetime.datetime.fromtimestamp(
            origin_s, datetime.timezone.utc,
        ).replace(tzinfo=None)
        data_log.datetime = datetime.datetime.fromtimestamp(origin_s)

    for col, (ch_name, unit, dec) in _LOOPDATA_CHANNELS.items():
        if col >= loop.shape[1]:
            continue
        vals = loop[:, col].copy()
        if ch_name == CH_BRAKE_POS:
            vals = vals * 100.0
        if ch_name == CH_GPS_SATS:
            vals = np.where(np.isfinite(vals), np.round(vals), np.nan)
        mask = np.isfinite(t_rel) & np.isfinite(vals)
        if not np.any(mask):
            continue
        if ch_name not in data_log.channels:
            data_log.add_channel(ch_name, unit, float, dec)
        data_log.channels[ch_name].set_samples(t_rel[mask], vals[mask])

    meta = json.loads(meta_text) if meta_text else {}
    _apply_metadata(data_log, meta, dlap_file_path)
    _parse_dlap_obd(data_log, obd_text, meta, t_rel)
    _parse_dlap_laps(data_log, meta, t_rel, target_lap=target_lap)


def _apply_metadata(data_log, meta, dlap_file_path):
    venue = meta.get("trackTitle") or ""
    if not venue and isinstance(meta.get("track"), dict):
        venue = meta["track"].get("name", "")
    if venue:
        data_log.metadata["venue_name"] = venue
    obd_meta = meta.get("obd")
    vehicle = meta.get("brandName", "")
    if not vehicle and isinstance(obd_meta, dict):
        vehicle = obd_meta.get("brandName", "")
    if vehicle:
        data_log.metadata["vehicle_id"] = vehicle
    title = meta.get("noteTrackTitle") or meta.get("trackTitle", "")
    if title:
        data_log.metadata["event_session"] = title
    if not data_log.metadata.get("venue_name"):
        data_log._extract_metadata(file_path=dlap_file_path)


def _parse_dlap_obd(data_log, obd_text, meta, t_rel):
    if not obd_text:
        return
    rows = []
    for line in obd_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append([float(v) if v.strip() else np.nan for v in line.split(",")])
        except ValueError:
            continue
    if not rows:
        return
    obd = np.array(rows, dtype=np.float64)
    if obd.shape[1] < 3:
        return
    obd_meta = meta.get("obd") if isinstance(meta, dict) else None
    try:
        version = float(obd_meta.get("version", 0)) if isinstance(obd_meta, dict) else 0
    except (TypeError, ValueError):
        version = 0
    first_value_col = 2 if version > 2 else 1
    pids = _parse_obd_pids(obd_meta)
    obd_t = obd[:, 0]
    for i, pid in enumerate(pids):
        mapping = _OBD_PID_CHANNELS.get(pid)
        if mapping is None:
            continue
        col = first_value_col + i
        if col >= obd.shape[1]:
            continue
        ch_name, unit, dec = mapping
        mask = np.isfinite(obd_t) & np.isfinite(obd[:, col])
        if not np.any(mask):
            continue
        grid = np.interp(t_rel, obd_t[mask], obd[mask][:, col])
        grid = _mask_interp_outside_range(grid, t_rel, obd_t[mask])
        if ch_name not in data_log.channels:
            data_log.add_channel(ch_name, unit, float, dec)
        data_log.channels[ch_name].set_samples(t_rel, grid)


def _parse_dlap_laps(data_log, meta, t_rel, target_lap=None):
    laps = meta.get("laps") if isinstance(meta, dict) else None
    laps_info = {"laps": [], "total_laps": 0, "fastest_time": 0.0}
    if not laps:
        data_log.laps_info = laps_info
        return
    beacons = []
    for lap in laps:
        try:
            start_idx = int(lap.get("startDataIndex", -1))
            end_idx = int(lap.get("endDataIndex", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= start_idx < end_idx < len(t_rel):
            continue
        start_s = float(t_rel[start_idx])
        end_s = float(t_rel[end_idx])
        if not end_s > start_s:
            continue
        try:
            lap_num = int(lap.get("idx", len(laps_info["laps"]) + 1))
        except (TypeError, ValueError):
            lap_num = len(laps_info["laps"]) + 1
        duration = end_s - start_s
        laps_info["laps"].append({
            "lap_num": lap_num, "start_time": start_s, "end_time": end_s,
            "duration": duration, "type": "Timed",
        })
        if laps_info["fastest_time"] == 0.0 or duration < laps_info["fastest_time"]:
            laps_info["fastest_time"] = duration
    for lap in laps_info["laps"]:
        for boundary in (lap["start_time"], lap["end_time"]):
            if not any(abs(boundary - existing[0]) <= 1e-6 for existing in beacons):
                beacons.append((boundary, "Start/Finish"))
    beacons.sort(key=lambda item: item[0])
    # Consecutive timed laps share a Start/Finish crossing on adjacent
    # samples (endDataIndex of lap N, startDataIndex of lap N+1); merge
    # near-duplicate boundaries so the LDX gets one marker per crossing.
    merged = []
    for time, name in beacons:
        if merged and time - merged[-1][0] <= 0.5:
            continue
        merged.append((time, name))
    beacons = merged
    laps_info["total_laps"] = len(laps_info["laps"])
    laps_info["beacons"] = sorted(set(beacons))

    if target_lap is not None and str(target_lap).lower() != "all":
        try:
            lap_no = int(target_lap)
            lap_recs = [lap for lap in laps_info["laps"] if lap["lap_num"] == lap_no]
            if lap_recs:
                t_lo = lap_recs[0]["start_time"]
                t_hi = lap_recs[0]["end_time"]
                for name in list(data_log.channels):
                    ch = data_log.channels[name]
                    keep = (ch.timestamps >= t_lo) & (ch.timestamps <= t_hi)
                    ch.set_samples(ch.timestamps[keep], ch.values[keep])
            else:
                print(f"WARNING: Lap '{target_lap}' not found in .dlap file; keeping all laps")
        except (ValueError, IndexError):
            print(f"WARNING: Lap '{target_lap}' not found in .dlap file; keeping all laps")

    data_log.laps_info = laps_info
