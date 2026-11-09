"""Byte-canonical LDX checks: output must match MoTeC i2's native shape.

i2 normalizes third-party .ldx files to this shape on load (1-space indent,
``/>`` self-close, LF endings, ``<?xml version="1.0"`` declaration, unpadded
``d/M/yyyy`` / ``H:m:s`` stamps, no ``<Laps>`` block, ``Fastest Lap`` detail).
Emitting it directly keeps our files stable when opened in i2.
"""

import datetime
import xml.etree.ElementTree as ET

from track_telemetry_converter.motec import MotecLog


def _write(tmp_path):
    ml = MotecLog()
    ml.initialize()
    ml.event_name = "E"
    ml.venue_name = "V"
    ml.driver = "D"
    ml.vehicle_id = "car"
    ml.datetime = datetime.datetime(2026, 11, 8, 22, 3, 5)
    laps_info = {
        "beacons": [(0.0, "Start/Finish"), (30.0, "Start/Finish"), (55.0, "Start/Finish")],
        "sector_beacons": [(6.0, "S1"), (13.0, "S2")],
        "laps": [
            {"lap_num": 1, "duration": 30.0, "type": "Timed",
             "start_time": 0.0, "end_time": 30.0},
            {"lap_num": 2, "duration": 25.0, "type": "Timed",
             "start_time": 30.0, "end_time": 55.0},
        ],
        "total_laps": 2,
        "fastest_time": 25.0,
        "fastest_lap": 2,
    }
    path = str(tmp_path / "out.ldx")
    ml.write_ldx(path, laps_info, beacons=[])
    with open(path, "rb") as f:
        raw = f.read()
    return raw, ET.fromstring(raw.decode("utf-8"))


def test_ldx_is_byte_canonical(tmp_path):
    raw, root = _write(tmp_path)
    text = raw.decode("utf-8")
    lines = text.split("\n")
    assert lines[0] == '<?xml version="1.0"?>'
    assert b"\r" not in raw
    assert text.endswith("</LDXFile>\n")
    # 1-space indent, never 2-space.
    assert "\n <Layers>" in text
    assert "\n  <Layers>" not in text
    # Self-closes have no space: i2 style.
    assert " />" not in text
    # i2 drops <Laps> on load, so we must not emit it; fastest lap is a detail.
    assert root.findall(".//Laps") == []
    details = {e.get("Id"): e.get("Value") for e in root.findall(".//Details/*")}
    assert details["Fastest Lap"] == "2"
    assert details["Fastest Time"] == "0:25.000"
    # i2 counts Total Laps as lap beacons + 1 (the out-lap); 3 beacons -> 4.
    assert details["Total Laps"] == "4"
    assert details["Log Date"] == "8/11/2026"
    assert details["Log Time"] == "22:3:5"
    # Content itself is untouched: beacons + sections survive.
    groups = {
        g.get("Name"): [m.get("Name") for m in g.findall("Marker")]
        for g in root.findall(".//MarkerGroup")
    }
    assert groups["Beacons"] == ["Finish", "Finish", "Finish"]
    assert groups["Sections"] == ["S1", "S2"]


def test_ldx_fastest_time_truncates_to_millisecond(tmp_path):
    # Float dust just under an ms boundary (130.6999999...s) must render as
    # 2:10.699 like i2, never round up to 2:10.700.
    import datetime

    ml = MotecLog()
    ml.initialize()
    ml.datetime = datetime.datetime(2026, 11, 8, 22, 11, 52)
    laps_info = {
        "beacons": [(0.0, "Start/Finish"), (130.7, "Start/Finish")],
        "laps": [{"lap_num": 1, "duration": 130.69999999999999, "type": "Timed",
                  "start_time": 0.0, "end_time": 130.69999999999999}],
        "total_laps": 1,
        "fastest_time": 130.69999999999999,
        "fastest_lap": 1,
    }
    path = str(tmp_path / "out.ldx")
    ml.write_ldx(path, laps_info, beacons=[])
    with open(path, "rb") as f:
        text = f.read().decode("utf-8")
    assert 'Id="Fastest Time" Value="2:10.699"' in text
    assert "2:10.700" not in text
