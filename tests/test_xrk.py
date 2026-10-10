"""Tests for AIM XRK / XRZ format."""

import datetime as dtmod
import os

import numpy as np
from conftest import EXAMPLES, _dedup_channels, _import_or_skip

import track_telemetry_converter.parsers.xrk_parser as xrk_parser
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers.xrk_parser import (
    _convert_values,
    _dedupe_samples,
    _parse_log_datetime,
    _unit_to_canonical,
)


def test_xrk_smoke():
    _import_or_skip("libxrk")

    log = DataLog()
    log.from_xrk_log(os.path.join(EXAMPLES, "aim_sample.xrk"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.duration() > 0
    _dedup_channels(log)


def test_xrk_units_time_origin_and_datetime():
    _import_or_skip("libxrk")

    log = DataLog()
    log.from_xrk_log(os.path.join(EXAMPLES, "aim_sample.xrk"))

    speed = log.channels["Ground Speed"]
    rpm = log.channels["Engine RPM"]
    gps = log.channels["GPS Latitude"]
    assert speed.units == "km/h"
    assert max(speed.values) > 100.0, "XRK m/s speed was labelled km/h without conversion"
    assert rpm.start() == 0.0
    assert 0.0 < gps.start() < 0.1, "GPS must retain its shared-clock offset"
    assert abs(gps.end() - rpm.end()) < 0.2, "GPS and logger clocks must stay aligned"
    assert np.all(np.isfinite(gps.timestamps))
    assert np.all(np.diff(gps.timestamps) > 0), "XRK timestamps must be unique and monotonic"
    assert log.datetime is not None
    assert log.datetime.strftime("%Y-%m-%d %H:%M:%S") == "2016-01-23 12:09:04"


def test_xrk_convert_values_units():
    assert _unit_to_canonical("kph") == "km/h"
    assert _unit_to_canonical("°C") == "C"
    assert _unit_to_canonical("bogus") == "bogus"
    assert list(_convert_values([10.0], "m/s", "km/h")) == [36.0]
    assert list(_convert_values([10.0], "mph", "km/h")) == [16.09344]
    assert list(_convert_values([36.0], "km/h", "m/s")) == [10.0]
    assert list(_convert_values([212.0], "F", "C")) == [100.0]
    assert list(_convert_values([5.0], "km/h", "km/h")) == [5.0]
    assert list(_convert_values([5.0], "V", "V")) == [5.0]
    assert list(_convert_values([5.0], "", "km/h")) == [5.0]
    assert list(_convert_values([100.0], "bar", "kPa")) == [100.0]


def test_xrk_dedupe_samples():
    ts, vs = _dedupe_samples(
        np.array([2.0, 0.0, 1.0, 1.0]), np.array([20.0, 0.0, 10.0, 11.0])
    )
    assert list(ts) == [0.0, 1.0, 2.0]
    assert list(vs) == [0.0, 11.0, 20.0]
    ts, vs = _dedupe_samples(np.array([1.0]), np.array([7.0]))
    assert list(ts) == [1.0] and list(vs) == [7.0]


def test_xrk_parse_log_datetime():
    from track_telemetry_converter.parsers.xrk_parser import _parse_log_datetime

    epoch_s = _parse_log_datetime({"datetime": 1453551744})
    epoch_ms = _parse_log_datetime({"datetime": 1453551744000})
    assert epoch_s is not None and epoch_ms is not None
    assert epoch_s.strftime("%Y-%m-%d") == epoch_ms.strftime("%Y-%m-%d")
    text_dt = _parse_log_datetime({"Log Date/Time": "01/23/2016 12:09:04"})
    assert text_dt is not None
    assert text_dt.strftime("%Y-%m-%d %H:%M:%S") == "2016-01-23 12:09:04"
    split_dt = _parse_log_datetime({"Log Date": "01/23/2016", "Log Time": "12:09:04"})
    assert split_dt is not None
    assert split_dt.strftime("%H:%M:%S") == "12:09:04"
    dmy_dt = _parse_log_datetime({"Log Date": "23/01/2016", "Log Time": "12:09:04"})
    assert dmy_dt is not None
    assert dmy_dt.strftime("%Y-%m-%d") == "2016-01-23"
    assert _parse_log_datetime({}) is None
    assert _parse_log_datetime({"datetime": "not a date"}) is None


def test_xrk_parse_log_datetime_oserror(monkeypatch):
    class _BoomDT(dtmod.datetime):
        @classmethod
        def fromtimestamp(cls, *a, **k):
            raise OSError("boom")

    monkeypatch.setattr(xrk_parser.datetime, "datetime", _BoomDT)
    assert _parse_log_datetime({"datetime": 1453551744}) is None


def test_xrk_requires_libxrk(monkeypatch, capsys):
    import sys

    monkeypatch.setitem(sys.modules, "libxrk", None)
    monkeypatch.setitem(sys.modules, "libxrk.base", None)
    log = DataLog()
    log.from_xrk_log("dummy.xrk")
    assert "libxrk" in capsys.readouterr().out
    assert len(log.channels) == 0


class _FakeXrkCol:
    def __init__(self, vals):
        self._vals = list(vals)

    def __len__(self):
        return len(self._vals)

    def to_numpy(self, zero_copy_only=True):
        import numpy as np

        return np.array(self._vals, dtype=np.float64)

    def to_pylist(self):
        return list(self._vals)


class _FakeXrkTable:
    def __init__(self, cols):
        self._cols = cols
        self.num_rows = len(next(iter(cols.values()))) if cols else 0

    def column(self, name):
        if name not in self._cols:
            raise KeyError(name)
        return self._cols[name]


class _FakeXrkLog:
    def __init__(self, channels, metadata, laps=None):
        self.channels = channels
        self.metadata = metadata
        self.laps = laps


def _stub_xrk(monkeypatch, log_obj):
    libxrk = _import_or_skip("libxrk")
    monkeypatch.setattr(libxrk, "aim_xrk", lambda path: log_obj)


def test_xrk_stub_log_branches(monkeypatch, capsys):
    _import_or_skip("libxrk")
    channels = {
        "rpm": _FakeXrkTable({
            "timecodes": _FakeXrkCol([0.0, 1000.0, 2000.0, 2000.0]),
            "rpm": _FakeXrkCol([3000.0, 4000.0, 5000.0, 5100.0]),
        }),
        "mystery": _FakeXrkTable({
            "timecodes": _FakeXrkCol([0.0]),
            "mystery": _FakeXrkCol([1.0]),
        }),
        "empty": _FakeXrkTable({}),
        "Engine RPM": _FakeXrkTable({"Engine RPM": _FakeXrkCol([1.0])}),
        "GPS Speed": _FakeXrkTable({
            "timecodes": _FakeXrkCol([0.0, 1000.0]),
            "GPS Speed": _FakeXrkCol([float("nan"), float("nan")]),
        }),
        "timecodes": _FakeXrkTable({
            "timecodes": _FakeXrkCol([0.0]),
        }),
    }
    laps = _FakeXrkTable({
        "num": _FakeXrkCol([1.0, 2.0, 3.0]),
        "start_time": _FakeXrkCol([0.0, 2000.0, 5000.0]),
        "end_time": _FakeXrkCol([2000.0, None, 4000.0]),
    })
    _stub_xrk(monkeypatch, _FakeXrkLog(
        channels,
        {"vehicle": "GR86", "driver": "T", "comment": "stub"},
        laps=laps,
    ))
    log = DataLog()
    log.from_xrk_log("stub.xrk")
    assert list(log.channels["Engine RPM"].values) == [3000.0, 4000.0, 5100.0]
    assert "mystery" not in log.channels
    assert "Ground Speed" not in log.channels
    assert log.metadata["vehicle_id"] == "GR86"
    assert log.metadata["driver"] == "T"
    assert log.laps_info["total_laps"] == 1
    assert log.laps_info["fastest_time"] == 2.0

    filtered = DataLog()
    _stub_xrk(monkeypatch, _FakeXrkLog(
        channels,
        {},
        laps=laps,
    ))
    filtered.from_xrk_log("stub.xrk", target_lap="99")
    assert "not found" in capsys.readouterr().out

    hit = DataLog()
    hit.from_xrk_log("stub.xrk", target_lap="1")
    assert list(hit.channels["Engine RPM"].values) == [3000.0, 4000.0, 5100.0]
    bad_target = DataLog()
    bad_target.from_xrk_log("stub.xrk", target_lap="abc")
    assert "not found" in capsys.readouterr().out


def test_xrk_stub_laps_failure(monkeypatch, capsys):
    _import_or_skip("libxrk")

    class _BadLaps(_FakeXrkTable):
        def __init__(self):
            self._cols = {}

        @property
        def num_rows(self):
            return 1

        def column(self, name):
            raise RuntimeError("nope")

    _stub_xrk(monkeypatch, _FakeXrkLog(
        {"rpm": _FakeXrkTable({
            "timecodes": _FakeXrkCol([0.0]),
            "rpm": _FakeXrkCol([3000.0]),
        })},
        {},
        laps=_BadLaps(),
    ))
    log = DataLog()
    log.from_xrk_log("stub.xrk")
    assert "Failed to parse XRK laps" in capsys.readouterr().out
    assert log.laps_info["total_laps"] == 0
