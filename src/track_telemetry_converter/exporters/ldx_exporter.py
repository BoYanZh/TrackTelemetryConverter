"""MoTeC .ldx XML beacon/lap export.

Byte-canonical with MoTeC i2: 1-space indent, ``/>`` self-close, LF line
endings, ``<?xml version="1.0"?>`` declaration, unpadded ``d/M/yyyy`` /
``H:m:s`` stamps, no ``<Laps>`` block, and a ``Fastest Lap`` detail. i2
normalizes third-party .ldx files to exactly this shape on load, so matching
it keeps our outputs stable when opened in i2 (no silent rewrite, mtime kept).
"""

from __future__ import annotations

import math
import re

from .xml_utils import indent_xml

LDX_XML_DECLARATION = '<?xml version="1.0"?>'


def _clean_ascii(text):
    if not text:
        return ""
    s = str(text).replace("–", "-").replace("—", "-").replace("’", "'").replace("”", '"').replace("“", '"')
    return s.encode("ascii", errors="ignore").decode("ascii")


def write_ldx(motec_log, filename, laps_info=None, beacons=None):
    """Writes an XML .ldx file with lap markers, sector beacons, and metadata.

    Lap beacons render as MarkerGroup "Beacons" (Index 3). Sector splits
    render as MarkerGroup "Sections" (Index 4) from two sources merged here:
    GPS trap crossings whose names are not Start/Finish (``beacons`` param,
    e.g. RCZ/TrackAddict T1/T2 traps) and game sector splits carried in
    ``laps_info["sector_beacons"]`` (e.g. ACTI Last Sector Time latches
    "S1"/"S2", same cum time base as the lap beacons).
    """
    import xml.etree.ElementTree as ET

    root = ET.Element("LDXFile", {
        "Locale": "English_United States.1252",
        "DefaultLocale": "C",
        "Version": "1.6",
    })
    layers = ET.SubElement(root, "Layers")
    layer = ET.SubElement(layers, "Layer")

    marker_block = ET.SubElement(layer, "MarkerBlock")
    beacon_group = ET.SubElement(marker_block, "MarkerGroup", {"Name": "Beacons", "Index": "3"})

    detected_lap_beacons = []
    split_beacons = []

    if beacons:
        for item in beacons:
            b_time = item[0] if isinstance(item, (tuple, list)) else item
            b_name = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else "Beacon"
            b_name_clean = _clean_ascii(b_name)
            b_name_lower = b_name_clean.strip().lower()
            if b_name_lower in ("start/finish", "sf", "beacon") or b_name_lower.startswith("start") or b_name_lower.startswith("finish") or b_name_lower.startswith("lap"):
                detected_lap_beacons.append((b_time, b_name_clean))
            else:
                split_beacons.append((b_time, b_name_clean))

    # Prefer source lap timing over GPS crossing detection.
    lap_beacons = []
    b_from_laps = laps_info.get("beacons", []) if laps_info else []
    if b_from_laps:
        lap_beacons = [ (t, _clean_ascii(n)) for t, n in b_from_laps ]
    else:
        laps = laps_info.get("laps", []) if laps_info else []
        timed_laps = [
            lap
            for lap in laps
            if str(lap.get("type", "Timed")).strip().lower() == "timed"
            and lap.get("start_time") is not None
            and lap.get("end_time") is not None
            and float(lap["end_time"]) > float(lap["start_time"])
        ]
        if timed_laps:
            boundaries = [float(timed_laps[0]["start_time"])]
            boundaries.extend(float(lap["end_time"]) for lap in timed_laps)
            for boundary in boundaries:
                if not lap_beacons or abs(boundary - lap_beacons[-1][0]) > 1e-9:
                    lap_beacons.append((boundary, "Start/Finish"))
        else:
            lap_beacons = detected_lap_beacons

    for b_time, _ in lap_beacons:
        t_us = b_time * 1e6
        ET.SubElement(beacon_group, "Marker", {
            "Version": "100",
            "ClassName": "BCN",
            "Name": "Finish",
            "Flags": "77",
            "Time": f"{t_us:.17e}",
        })

    # GPS trap splits (split_beacons) and game sector splits (laps_info,
    # e.g. ACTI "S1"/"S2") merge into one Sections group.

    sector_from_laps = []
    if laps_info:
        for key in ("sector_beacons", "splits", "sectors"):
            vals = laps_info.get(key)
            if not vals:
                continue
            for item in vals:
                try:
                    t = float(item[0]) if isinstance(item, (tuple, list)) else float(item)
                except (TypeError, ValueError, IndexError):
                    continue
                if not math.isfinite(t):
                    continue
                name = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else "Split"
                name = _clean_ascii(name) or "Split"
                sector_from_laps.append((t, name))

    merged_sections = sorted(split_beacons + sector_from_laps, key=lambda kv: kv[0])
    deduped_sections = []
    for t, name in merged_sections:
        if deduped_sections and abs(t - deduped_sections[-1][0]) <= 1e-9:
            continue
        deduped_sections.append((t, name))

    if deduped_sections:
        section_group = ET.SubElement(marker_block, "MarkerGroup", {"Name": "Sections", "Index": "4"})
        for b_time, b_name in deduped_sections:
            t_us = b_time * 1e6
            ET.SubElement(section_group, "Marker", {
                "Version": "100",
                "ClassName": "BCN",
                "Name": _clean_ascii(b_name),
                "Flags": "77",
                "Time": f"{t_us:.17e}",
            })

    ET.SubElement(layer, "RangeBlock")

    # NOTE: i2 drops the <Laps> block on load (it rebuilds laps from the
    # Beacons itself), so emitting it guarantees a rewrite. Keep lap timing
    # in the beacons only.

    details = ET.SubElement(layers, "Details")

    def add_detail_str(string_id, val):
        ET.SubElement(details, "String", {"Id": string_id, "Value": _clean_ascii(val)})

    add_detail_str("Event", getattr(motec_log, "event_name", ""))
    add_detail_str("Venue", getattr(motec_log, "venue_name", getattr(getattr(motec_log, "head", None), "venue", "")))
    add_detail_str("Venue Category", "")
    add_detail_str("Driver", getattr(motec_log, "driver", getattr(getattr(motec_log, "head", None), "driver", "")))
    add_detail_str("Team", "")
    add_detail_str("Vehicle Id", getattr(motec_log, "vehicle_id", getattr(getattr(motec_log, "head", None), "vehicleid", "")))
    add_detail_str("Vehicle Number", "")
    add_detail_str("Vehicle Desc", getattr(motec_log, "vehicle_comment", ""))
    add_detail_str("Engine Id", "")
    add_detail_str("Session", getattr(motec_log, "event_session", ""))
    add_detail_str("Start Lap", "")
    add_detail_str("Short Comment", getattr(motec_log, "short_comment", ""))
    add_detail_str("Long Comment", getattr(motec_log, "long_comment", ""))

    dt = getattr(motec_log, "datetime", getattr(getattr(motec_log, "head", None), "datetime", None))
    if dt:
        # i2 canonical stamps are unpadded d/M/yyyy and H:m:s (no strftime
        # %-flags: those are platform-specific, so format explicitly).
        ET.SubElement(details, "DateTime", {
            "Id": "Log Date",
            "Value": f"{dt.day}/{dt.month}/{dt.year}",
        })
        ET.SubElement(details, "DateTime", {
            "Id": "Log Time",
            "Value": f"{dt.hour}:{dt.minute}:{dt.second}",
        })

    add_detail_str("Sky", "")
    add_detail_str("Wind Direction", "")
    add_detail_str("Weather Comment", "")
    add_detail_str("Vehicle Type", getattr(motec_log, "vehicle_type", ""))
    add_detail_str("Vehicle Drive Type", "")
    add_detail_str("Vehicle Comment", getattr(motec_log, "vehicle_comment", ""))

    counted_laps = 0
    if laps_info:
        for lap_data in laps_info.get("laps", []) or []:
            try:
                dur = float(lap_data.get("duration", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if dur > 0 and math.isfinite(dur):
                counted_laps += 1
    # i2 recomputes Total Laps on load as (lap beacons + 1): the beacons in
    # the file plus the out-lap that precedes the first flying lap. Writing
    # any other number guarantees a rewrite. Fall back to counted records,
    # then total_laps, when there are no beacons to count from.
    if lap_beacons:
        total_laps = len(lap_beacons) + 1
    elif counted_laps > 0:
        total_laps = counted_laps
    else:
        total_laps = laps_info.get("total_laps", 1) if laps_info else 1
    # i2 recomputes fastest lap from the beacon markers (min consecutive
    # difference, trailing partial excluded by construction). MoTeC times are
    # integer microseconds: i2 truncates the parsed us value to int, then
    # integer-divides by 1000. A 131949999.99999999us lap shows as 2:11.949,
    # never 2:11.950 (float seconds-domain math would round the other way).
    # Derive it from the exact decimals we emit (parse our own microsecond
    # strings back) so the doubles match i2's bitwise.
    fastest_time_sec = laps_info.get("fastest_time", 0.0) if laps_info else 0.0
    try:
        fastest_time_sec = float(fastest_time_sec or 0.0)
    except (TypeError, ValueError):
        fastest_time_sec = 0.0
    fastest_from_beacons = False
    if len(lap_beacons) >= 2:
        try:
            us = [float(f"{t * 1e6:.17e}") for t, _ in lap_beacons]
        except (TypeError, ValueError):
            us = []
        best_us = None
        for i in range(len(us) - 1):
            dur_us = us[i + 1] - us[i]
            if math.isfinite(dur_us) and dur_us > 0 and (best_us is None or dur_us < best_us):
                best_us = dur_us
        if best_us is not None:
            total_ms = int(best_us) // 1000
            m, rem_ms = divmod(total_ms, 60000)
            s, ms = divmod(rem_ms, 1000)
            fastest_time_str = f"{m}:{s:02d}.{ms:03d}"
            fastest_from_beacons = True
    if not fastest_from_beacons:
        if fastest_time_sec > 0:
            total_ms = math.floor(max(0.0, fastest_time_sec) * 1000)
            m, rem_ms = divmod(total_ms, 60000)
            s, ms = divmod(rem_ms, 1000)
            fastest_time_str = f"{m}:{s:02d}.{ms:03d}"
        else:
            fastest_time_str = "0:00.000"
    fastest_lap = laps_info.get("fastest_lap", None) if laps_info else None
    if fastest_lap is not None:
        try:
            fastest_lap = str(int(fastest_lap))
        except (TypeError, ValueError):
            fastest_lap = str(fastest_lap)

    add_detail_str("Total Laps", str(total_laps))
    add_detail_str("Fastest Time", fastest_time_str)
    if fastest_lap is not None:
        add_detail_str("Fastest Lap", fastest_lap)

    write_canonical_ldx(root, filename)


def write_canonical_ldx(root, filename):
    """Serialize an LDX tree exactly the way i2 does (LF, 1-space, ``/>``)."""
    import xml.etree.ElementTree as ET

    indent_xml(root, space=" ")
    body = ET.tostring(root, encoding="unicode")
    body = "\n".join(re.sub(r" />$", "/>", line) for line in body.split("\n"))
    if not body.endswith("\n"):
        body += "\n"
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(LDX_XML_DECLARATION + "\n" + body)


def dumps_canonical_ldx(root):
    """Canonical LDX bytes (for rewriters like video_sync that keep the tree)."""
    import xml.etree.ElementTree as ET

    indent_xml(root, space=" ")
    body = ET.tostring(root, encoding="unicode")
    body = "\n".join(re.sub(r" />$", "/>", line) for line in body.split("\n"))
    if not body.endswith("\n"):
        body += "\n"
    return (LDX_XML_DECLARATION + "\n" + body).encode("utf-8")
