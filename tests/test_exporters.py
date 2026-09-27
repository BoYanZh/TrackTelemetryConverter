"""Tests for exporters (CSV, XML utils, GPX, KML)."""

import csv
import os
import tempfile
import xml.etree.ElementTree as ET

from conftest import _read_lines

from track_telemetry_converter.exporters.csv_export import write_csv
from track_telemetry_converter.exporters.xml_utils import indent_xml
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.models import Message


def test_csv_export():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    log.resample(20)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        assert write_csv(log, tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2, "CSV should have header + data rows"
        header = lines[0].strip().split(",")
        assert "Time (s)" in header
        assert len(header) >= 2, "CSV should have at least one channel column"
    finally:
        os.remove(tmp_path)


def test_csv_export_uses_timestamp_union_and_handles_empty_channels():
    log = DataLog()
    log.add_channel("Engine RPM", "rpm", float, 0)
    log.channels["Engine RPM"].messages = [
        Message(i / 100.0, 1000.0 + i) for i in range(101)
    ]
    for name, base in (("GPS Latitude", 37.0), ("GPS Longitude", -122.0)):
        log.add_channel(name, "deg", float, 7)
        log.channels[name].messages = [
            Message(i / 10.0, base + i * 1e-5) for i in range(11)
        ]
    log.add_channel("Empty", "", float, 0)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        assert write_csv(log, tmp_path)
        with open(tmp_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) - 1 == 101
        assert rows[0][:3] == ["Time (s)", "GPS Latitude", "GPS Longitude"]
        assert all(row[-1] == "" for row in rows[1:])
    finally:
        os.remove(tmp_path)


def test_xml_indent_fallback_for_python_38(monkeypatch):
    root = ET.Element("root")
    ET.SubElement(root, "child").text = "value"
    monkeypatch.delattr(ET, "indent", raising=False)

    indent_xml(root, space="  ")

    assert ET.tostring(root, encoding="unicode") == (
        "<root>\n  <child>value</child>\n</root>"
    )
