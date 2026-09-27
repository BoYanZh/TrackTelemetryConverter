"""MoTeC .ldx XML beacon/lap export."""

from __future__ import annotations

from .xml_utils import indent_xml


def _clean_ascii(text):
    if not text:
        return ""
    s = str(text).replace("–", "-").replace("—", "-").replace("’", "'").replace("”", '"').replace("“", '"')
    return s.encode("ascii", errors="ignore").decode("ascii")


def write_ldx(motec_log, filename, laps_info=None, beacons=None):
    """ Writes an XML .ldx file containing lap markers, sector beacons, and metadata details. """
    import xml.etree.ElementTree as ET

    root = ET.Element("LDXFile", {
        "Locale": "English_United States.1252",
        "DefaultLocale": "C",
        "Version": "1.6"
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

    for idx, (b_time, b_name) in enumerate(lap_beacons):
        t_us = b_time * 1e6
        ET.SubElement(beacon_group, "Marker", {
            "Version": "100",
            "ClassName": "BCN",
            "Name": "Finish",
            "Flags": "77",
            "Time": f"{t_us:.17e}"
        })

    if split_beacons:
        section_group = ET.SubElement(marker_block, "MarkerGroup", {"Name": "Sections", "Index": "4"})
        for idx, (b_time, b_name) in enumerate(split_beacons):
            t_us = b_time * 1e6
            ET.SubElement(section_group, "Marker", {
                "Version": "100",
                "ClassName": "BCN",
                "Name": _clean_ascii(b_name),
                "Flags": "77",
                "Time": f"{t_us:.17e}"
            })

    ET.SubElement(layer, "RangeBlock")

    # Generate Laps block if lap info is available
    if laps_info and "laps" in laps_info and laps_info["laps"]:
        laps_elem = ET.SubElement(layer, "Laps")
        for lap_data in laps_info["laps"]:
            dur_s = float(lap_data.get("duration", 0.0))
            if dur_s > 0:
                dur_us = dur_s * 1e6
                lap_num = str(lap_data.get("lap_num", 1))
                ET.SubElement(laps_elem, "Lap", {
                    "Id": lap_num,
                    "Time": f"{dur_us:.17e}"
                })

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
        date_str = dt.strftime("%d/%m/%Y")
        time_str = dt.strftime("%H:%M:%S")
        ET.SubElement(details, "DateTime", {"Id": "Log Date", "Value": date_str})
        ET.SubElement(details, "DateTime", {"Id": "Log Time", "Value": time_str})

    add_detail_str("Sky", "")
    add_detail_str("Wind Direction", "")
    add_detail_str("Weather Comment", "")
    add_detail_str("Vehicle Type", getattr(motec_log, "vehicle_type", ""))
    add_detail_str("Vehicle Drive Type", "")
    add_detail_str("Vehicle Comment", getattr(motec_log, "vehicle_comment", ""))

    total_laps = laps_info.get("total_laps", 1) if laps_info else 1
    fastest_time_sec = laps_info.get("fastest_time", 0.0) if laps_info else 0.0

    if fastest_time_sec > 0:
        m, s = divmod(fastest_time_sec, 60)
        fastest_time_str = f"{int(m)}:{s:06.3f}"
    else:
        fastest_time_str = "0:00.000"

    add_detail_str("Total Laps", str(total_laps))
    add_detail_str("Fastest Time", fastest_time_str)

    indent_xml(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(xml_str)
