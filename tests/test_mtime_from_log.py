"""Tests for --mtime-from-log output timestamp alignment."""

import datetime
import os
import tempfile

from conftest import EXAMPLES, _run_cli_in_process

from track_telemetry_converter.conversion import _log_end_epoch_sec
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.models import Message
from track_telemetry_converter.parsers.ibt_parser import _parse_filename_start


def _output_pair(output):
    return (output, os.path.splitext(output)[0] + ".ldx")


def test_mtime_from_log_uses_recorded_log_end():
    # VBO fixture header: "File created on 01/01/2026 at 10:00:00" + ~150 s
    # of telemetry, so the aligned mtime must land back on 2026-01-01 and
    # well before the fixture's own (August 2026) file mtime.
    source = os.path.join(EXAMPLES, "vbo_sample.vbo")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "mtime.ld")
        result = _run_cli_in_process(
            [source, "VBO", "--output", output, "--force", "--mtime-from-log"],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Set output mtime to log end" in result.stdout
        for path in _output_pair(output):
            moment = datetime.datetime.fromtimestamp(os.path.getmtime(path))
            assert (moment.year, moment.month, moment.day) == (2026, 1, 1)
            assert os.path.getmtime(path) < os.path.getmtime(source)


def test_mtime_from_log_falls_back_to_source_mtime():
    # Tesla logs carry no wall-clock: outputs inherit the source mtime.
    source = os.path.join(EXAMPLES, "tesla_sample.csv")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "mtime.ld")
        result = _run_cli_in_process(
            [source, "TESLA", "--output", output, "--force", "--mtime-from-log"],
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "using source file mtime" in result.stdout
        for path in _output_pair(output):
            assert int(os.path.getmtime(path)) == int(os.path.getmtime(source))


def test_mtime_untouched_without_flag():
    source = os.path.join(EXAMPLES, "vbo_sample.vbo")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "plain.ld")
        before = datetime.datetime.now().timestamp()
        result = _run_cli_in_process([source, "VBO", "--output", output, "--force"])
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Set output mtime" not in result.stdout
        for path in _output_pair(output):
            assert os.path.getmtime(path) >= before - 1


def test_parse_filename_start():
    assert _parse_filename_start(
        "mx5 mx52016_lagunaseca 2026 2026-11-04 10-28-59.ibt",
    ) == datetime.datetime(2026, 11, 4, 10, 28, 59)
    assert _parse_filename_start("session_20241109_125402_sonoma.vbo") is None
    assert _parse_filename_start("not a timestamp.ibt") is None


def _log_with_samples(timestamps):
    log = DataLog()
    log.add_channel("chan", "unit", float, 2)
    for index, stamp in enumerate(timestamps):
        log.channels["chan"].messages.append(Message(stamp, float(index)))
    return log


def test_log_end_epoch_absolute_timestamps():
    # CAN-style absolute epoch channel stamps are already wall-clock.
    log = _log_with_samples([1_700_000_000.0, 1_700_000_060.0])
    assert _log_end_epoch_sec(log) == 1_700_000_060.0


def test_log_end_epoch_origin_and_datetime():
    log = _log_with_samples([0.0, 60.0])
    log.time_origin_epoch_ms = 1_700_000_000_000
    assert _log_end_epoch_sec(log) == 1_700_000_060.0

    log = _log_with_samples([0.0, 60.0])
    log.datetime = datetime.datetime(2026, 5, 15, 0, 0, 0)
    expected = log.datetime.timestamp() + 60.0
    assert abs(_log_end_epoch_sec(log) - expected) < 1e-6


def test_log_end_epoch_without_wall_clock():
    assert _log_end_epoch_sec(_log_with_samples([0.0, 1.0])) is None
    assert _log_end_epoch_sec(DataLog()) is None
