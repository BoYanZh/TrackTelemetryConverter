"""Tests for the TrackAddict / RaceRender-Data CSV parser."""

import numpy as np
from conftest import _read_lines

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers import auto_detect_log_type


def test_trackaddict_autodetect():
    assert auto_detect_log_type("tests/fixtures/trackaddict_sample.csv") == "TRACKADDICT"
    assert auto_detect_log_type("tests/fixtures/trackaddict_metric_sample.csv") == "TRACKADDICT"


def _raw_rows(fixture):
    lines = _read_lines(fixture)
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('"Time"'))
    headers = [h.strip().strip('"') for h in lines[header_idx].strip().split(",")]
    rows = []
    for line in lines[header_idx + 1:]:
        if line.strip():
            rows.append([v.strip() for v in line.strip().split(",")])
    return headers, rows


def test_trackaddict_imperial_units_convert():
    log = DataLog()
    log.from_trackaddict_log(_read_lines("trackaddict_sample.csv"))
    assert log.channels["Ground Speed"].units == "km/h"
    assert log.channels["Coolant Temp"].units == "C"
    assert log.channels["Engine RPM"].units == "rpm"
    assert log.metadata["vehicle_id"] == "2002 Chevrolet Camaro Z28"

    headers, rows = _raw_rows("trackaddict_sample.csv")
    mph = float(rows[10][headers.index("Speed (MPH)")])
    assert log.channels["Ground Speed"].values[10] == mph * 1.60934
    fahr = float(rows[10][headers.index("Engine Coolant Temp (F) *OBD")])
    assert log.channels["Coolant Temp"].values[10] == (fahr - 32.0) * 5.0 / 9.0
    psi = float(rows[10][headers.index("Barometric Pressure (PSI)")])
    assert log.channels["Barometric Pressure"].values[10] == psi * 6.89476
    assert log.channels["GPS Latitude"].values[10] == float(rows[10][headers.index("Latitude")])


def test_trackaddict_metric_units_preserved():
    log = DataLog()
    log.from_trackaddict_log(_read_lines("trackaddict_metric_sample.csv"))
    assert log.channels["Ground Speed"].units == "km/h"
    headers, rows = _raw_rows("trackaddict_metric_sample.csv")
    kmh = float(rows[10][headers.index("Speed (Km/h)")])
    assert log.channels["Ground Speed"].values[10] == kmh
    assert log.datetime is not None
    assert (log.datetime.year, log.datetime.month, log.datetime.day) == (2019, 6, 5)


def test_trackaddict_traps_from_comments():
    log = DataLog()
    log.from_trackaddict_log(_read_lines("trackaddict_sample.csv"))
    assert len(log.traps) == 5
    sf = log.traps[0]
    assert sf["name"] == "Start/Finish"
    assert abs(sf["lat"] - 41.455209) < 1e-6
    assert abs(sf["lon"] - -88.126891) < 1e-6
    assert [t["name"] for t in log.traps[1:]] == ["Split 1", "Split 2", "Split 3", "Split 4"]

    log = DataLog()
    log.from_trackaddict_log(_read_lines("trackaddict_metric_sample.csv"))
    assert len(log.traps) == 1
    assert log.traps[0]["name"] == "Start/Finish"


def test_trackaddict_laps_from_lap_column():
    lines = [
        '"Time","Lap","Latitude","Longitude","Speed (Km/h)"\n',
        "0.0,0,41.0,-88.0,50.0\n",
        "10.0,0,41.0,-88.0,50.0\n",
        "20.0,0,41.0,-88.0,50.0\n",
        "30.0,0,41.0,-88.0,50.0\n",
        "40.0,1,41.0,-88.0,50.0\n",
        "50.0,1,41.0,-88.0,50.0\n",
        "60.0,1,41.0,-88.0,50.0\n",
        "70.0,1,41.0,-88.0,50.0\n",
        "80.0,1,41.0,-88.0,50.0\n",
        "90.0,2,41.0,-88.0,50.0\n",
        "100.0,2,41.0,-88.0,50.0\n",
        "110.0,2,41.0,-88.0,50.0\n",
        "120.0,2,41.0,-88.0,50.0\n",
        "130.0,2,41.0,-88.0,50.0\n",
    ]
    log = DataLog()
    log.from_trackaddict_log(lines)
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap", "Timed", "Timed"]
    assert log.laps_info["total_laps"] == 3
    assert log.laps_info["fastest_time"] == 40.0


def test_trackaddict_pedal_maps_to_accelerator():
    lines = [
        '"Time","Latitude","Longitude",'
        '"Accelerator Pedal (%) *OBD","Throttle Position (%) *OBD"\n',
        "0.0,41.0,-88.0,10.0,12.0\n",
        "1.0,41.1,-88.1,20.0,22.0\n",
    ]
    log = DataLog()
    log.from_trackaddict_log(lines)
    assert np.array_equal(log.channels["Accelerator Pos"].values, [10.0, 20.0])
    assert np.array_equal(log.channels["Throttle Pos"].values, [12.0, 22.0])
    prov = log.metadata["channel_provenance"]
    assert prov["Accelerator Pos"]["power_semantics"] == "accelerator_pedal"


def test_trackaddict_target_lap_and_errors(capsys):
    lines = [
        '"Time","Lap","Latitude","Longitude","Speed (Km/h)"\n',
        "0.0,1,41.0,-88.0,50.0\n",
        "30.0,1,41.0,-88.0,60.0\n",
        "60.0,2,41.0,-88.0,70.0\n",
        "90.0,2,41.0,-88.0,80.0\n",
    ]
    log = DataLog()
    log.from_trackaddict_log(lines, target_lap="2")
    assert np.array_equal(log.channels["Ground Speed"].values, [70.0, 80.0])

    log = DataLog()
    log.from_trackaddict_log(lines, target_lap="9")
    assert "WARNING" in capsys.readouterr().out

    empty = DataLog()
    empty.from_trackaddict_log([])
    assert len(empty.channels) == 0
    no_header = DataLog()
    no_header.from_trackaddict_log(["foo,bar\n", "1,2\n"])
    assert len(no_header.channels) == 0
    assert "ERROR" in capsys.readouterr().out
