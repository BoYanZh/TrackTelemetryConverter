"""Tests for Garmin FIT format."""

import datetime
import os
import warnings

import numpy as np
from conftest import EXAMPLES, _dedup_channels, _import_or_skip

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers.fit_parser import _fit_field, _to_deg


def test_fit_smoke():
    _import_or_skip("fitparse")

    log = DataLog()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        log.from_fit_log(os.path.join(EXAMPLES, "garmin_sample.fit"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.duration() > 0
    _dedup_channels(log)


def test_fit_unique_timestamps_finite_math_and_lap_boundaries():
    _import_or_skip("fitparse")

    log = DataLog()
    log.from_fit_log(os.path.join(EXAMPLES, "garmin_sample.fit"))
    speed = log.channels["Ground Speed"]
    assert np.all(np.diff(speed.timestamps) > 0), "FIT timestamps must be strictly increasing"

    log.calculate_math_channels()
    for name in ("CG Accel Longitudinal", "CG Accel Long Smooth"):
        assert np.all(np.isfinite(log.channels[name].values)), f"{name} contains NaN/Inf"

    laps = log.laps_info["laps"]
    assert len(log.laps_info["beacons"]) == len(laps) + 1
    for previous, current in zip(laps, laps[1:], strict=False):
        assert previous["end_time"] == current["start_time"]


class _FakeFitField:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.units = "s"


class _FakeFitMsg:
    def __init__(self, fields):
        self.fields = [_FakeFitField(k, v) for k, v in fields.items()]


class _NoFieldsMsg:
    fields = []


class _FakeFitFile:
    def __init__(self, by_type):
        self._by_type = by_type

    def get_messages(self, name):
        return iter(self._by_type.get(name, []))


def _stub_fit(monkeypatch, by_type):
    fitparse = _import_or_skip("fitparse")
    monkeypatch.setattr(fitparse, "FitFile", lambda *a, **k: _FakeFitFile(by_type))


def _record(ts_sec, **fields):
    base = datetime.datetime(2026, 1, 1)
    rec = {"timestamp": base + datetime.timedelta(seconds=ts_sec)}
    rec.update(fields)
    return _FakeFitMsg(rec)


def test_fit_field_helpers():
    assert _to_deg(2**31) == 180.0
    msg = _FakeFitMsg({"a": 1, "b": None})
    assert _fit_field(msg, "a") == 1
    assert _fit_field(msg, "b") is None
    assert _fit_field(msg, "missing") is None


def test_fit_event_fallback_laps_and_record_skips(monkeypatch):
    _import_or_skip("fitparse")
    base = datetime.datetime(2026, 1, 1)
    _stub_fit(monkeypatch, {
        "record": [
            _NoFieldsMsg(),
            _FakeFitMsg({}),
            _FakeFitMsg({"speed": 5.0}),
            _record(0.0, enhanced_speed=10.0, speed=9.0, cadence=80,
                    fractional_cadence=0.5),
            _record(1.0, enhanced_speed=11.0),
            _record(1.0, heart_rate=150),
        ],
        "event": [
            _FakeFitMsg({"event": "lap", "event_type": "start",
                         "timestamp": base}),
            _FakeFitMsg({"event": "lap", "event_type": "stop_all",
                         "timestamp": base + datetime.timedelta(seconds=30)}),
            _FakeFitMsg({"event": "timer", "event_type": "start",
                         "timestamp": base}),
        ],
    })
    log = DataLog()
    log.from_fit_log("stub.fit")
    assert list(log.channels["Ground Speed"].values) == [36.0, 39.6]
    assert list(log.channels["Cadence"].values) == [80.5]
    assert list(log.channels["Heart Rate"].values) == [150.0]
    assert log.laps_info["total_laps"] == 1
    assert log.laps_info["laps"][0]["duration"] == 30.0
    assert len(log.laps_info["beacons"]) == 2


def test_fit_lap_message_skips_and_target_lap(monkeypatch, capsys):
    _import_or_skip("fitparse")
    base = datetime.datetime(2026, 1, 1)
    records = [_record(float(i), enhanced_speed=10.0 + i) for i in range(6)]
    laps = [
        _FakeFitMsg({}),
        _FakeFitMsg({"start_time": base,
                     "timestamp": base + datetime.timedelta(seconds=2),
                     "total_timer_time": 2.0}),
        _FakeFitMsg({"start_time": base + datetime.timedelta(seconds=2),
                     "timestamp": base + datetime.timedelta(seconds=2)}),
        _FakeFitMsg({"start_time": base + datetime.timedelta(seconds=2),
                     "timestamp": base + datetime.timedelta(seconds=5)}),
    ]
    _stub_fit(monkeypatch, {"record": records, "lap": laps})
    log = DataLog()
    log.from_fit_log("stub.fit", target_lap="2")
    assert log.laps_info["total_laps"] == 2
    assert log.duration() <= 3.1
    log_all = DataLog()
    log_all.from_fit_log("stub.fit", target_lap="99")
    assert "not found" in capsys.readouterr().out
    assert log_all.duration() > log.duration()


def test_fit_local_date_time_processor(monkeypatch):
    fitparse = _import_or_skip("fitparse")
    captured = {}

    def _capture(*a, **k):
        captured["proc"] = k.get("data_processor")
        return _FakeFitFile({})

    monkeypatch.setattr(fitparse, "FitFile", _capture)
    DataLog().from_fit_log("stub.fit")
    proc = captured["proc"]
    assert proc is not None
    fd = _FakeFitField("x", 86400)
    proc.process_type_local_date_time(fd)
    assert isinstance(fd.value, datetime.datetime)
    assert fd.units is None
    proc.process_type_local_date_time(_FakeFitField("x", None))


def test_fit_requires_fitparse(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "fitparse", None)
    monkeypatch.setitem(sys.modules, "fitparse.processors", None)
    log = DataLog()
    log.from_fit_log("dummy.fit")
    assert "fitparse" in capsys.readouterr().out
    assert len(log.channels) == 0


def test_fit_sport_all_none_fields_and_bad_target(monkeypatch, capsys):
    _import_or_skip("fitparse")
    _stub_fit(monkeypatch, {
        "file_id": [_FakeFitMsg({"serial_number": 123})],
        "sport": [_FakeFitMsg({"sport": "cycling"})],
        "record": [
            _FakeFitMsg({"speed": None}),
            _record(0.0, enhanced_speed=10.0),
            _record(60.0, enhanced_speed=12.0),
        ],
    })
    log = DataLog()
    log.from_fit_log("stub.fit")
    assert log.metadata["vehicle_id"] == "Garmin S/N 123"
    assert log.metadata["event_name"] == "Cycling"
    bad = DataLog()
    bad.from_fit_log("stub.fit", target_lap="abc")
    assert "not found" in capsys.readouterr().out
