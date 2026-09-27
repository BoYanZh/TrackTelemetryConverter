"""Tests for CAN log format with DBC."""

import os

from conftest import EXAMPLES, _dedup_channels, _import_or_skip, _read_lines

from track_telemetry_converter.log import DataLog


def test_can_smoke():
    cantools = _import_or_skip("cantools")

    log = DataLog()
    dbc_path = os.path.join(EXAMPLES, "sample_can_spec.dbc")
    can_db = cantools.database.load_file(dbc_path)
    log.from_can_log(_read_lines("can_sample.log"), can_db)
    assert len(log.channels) >= 1, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_can_resample():
    cantools = _import_or_skip("cantools")

    log = DataLog()
    dbc_path = os.path.join(EXAMPLES, "sample_can_spec.dbc")
    can_db = cantools.database.load_file(dbc_path)
    log.from_can_log(_read_lines("can_sample.log"), can_db)
    log.resample(20)
    assert log.duration() > 0
