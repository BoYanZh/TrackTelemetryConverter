"""Tests for privacy-conscious telemetry diagnostic reports."""

import json
import warnings

import numpy as np

from track_telemetry_converter.diagnostics import build_diagnostic_report
from track_telemetry_converter.log import DataLog


def _log_with_channel(name, timestamps, values):
    log = DataLog()
    log.add_channel(name, "km/h", float, 2)
    log.channels[name].set_samples(timestamps, values)
    return log


def test_report_has_stable_schema_and_sorted_privacy_safe_channel_summaries():
    log = DataLog()
    log.time_origin_epoch_ms = 1_700_000_000_000
    log.add_channel("Zulu", "m/s", float, 2)
    log.channels["Zulu"].set_samples([2.0, 3.0], [10.0, 12.0])
    log.add_channel("Alpha", "km/h", float, 2)
    log.channels["Alpha"].set_samples([0.0, 0.5, 1.0], [42.0, 43.0, 44.0])
    before = {
        name: (channel.timestamps.copy(), channel.values.copy())
        for name, channel in log.channels.items()
    }

    report = build_diagnostic_report(
        log, source_format="RCZ", session="session-1", stint=2, lap="all", stages=["parse", "inspect"],
    )

    assert list(report) == [
        "schema_version", "source", "selection", "timebase", "channels", "stages", "warnings", "validation",
    ]
    assert report["schema_version"] == 1
    assert report["source"] == {"format": "RCZ"}
    assert report["selection"] == {"session": "session-1", "stint": 2, "lap": "all"}
    assert report["timebase"] == {"unit": "s", "origin_epoch_ms": 1_700_000_000_000}
    assert [channel["name"] for channel in report["channels"]] == ["Alpha", "Zulu"]
    assert report["channels"][0] == {
        "name": "Alpha",
        "units": "km/h",
        "sample_count": 3,
        "frequency_hz": 2.0,
        "time_min_s": 0.0,
        "time_max_s": 1.0,
        "invalid_fraction": 0.0,
        "time_invalid_fraction": 0.0,
        "gaps": {"count": 0, "max_s": None, "threshold_s": 1.0},
        "duplicate_timestamps": 0,
        "nonmonotonic_timestamps": 0,
        "length_mismatch": False,
    }
    assert report["stages"] == ["parse", "inspect"]
    assert report["warnings"] == []
    assert report["validation"] == {"valid": True, "errors": []}
    serialized = json.dumps(report, allow_nan=False, sort_keys=True)
    assert "42.0" not in serialized
    assert "GPS" not in serialized
    for name, channel in log.channels.items():
        np.testing.assert_array_equal(channel.timestamps, before[name][0])
        np.testing.assert_array_equal(channel.values, before[name][1])
    assert build_diagnostic_report(
        log, source_format="RCZ", session="session-1", stint=2, lap="all", stages=["parse", "inspect"],
    ) == report


def test_irregular_timebase_reports_median_frequency_gaps_duplicates_and_reversals():
    log = _log_with_channel(
        "Speed",
        [0.0, 0.5, 2.0, 2.0, 1.0],
        [1.0, float("nan"), float("inf"), 4.0, 5.0],
    )

    report = build_diagnostic_report(log, source_format="CSV")
    channel = report["channels"][0]

    assert channel["frequency_hz"] == 1.0
    assert channel["gaps"] == {"count": 1, "max_s": 1.5, "threshold_s": 1.0}
    assert channel["duplicate_timestamps"] == 1
    assert channel["nonmonotonic_timestamps"] == 1
    assert channel["invalid_fraction"] == 0.4
    assert report["warnings"] == [
        {"code": "CHANNEL_NONFINITE_VALUES", "channel": "Speed"},
        {"code": "CHANNEL_TIME_GAPS", "channel": "Speed"},
    ]
    assert report["validation"] == {
        "valid": False,
        "errors": [
            {"code": "CHANNEL_DUPLICATE_TIMESTAMPS", "channel": "Speed"},
            {"code": "CHANNEL_NONMONOTONIC_TIMESTAMPS", "channel": "Speed"},
        ],
    }
    json.dumps(report, allow_nan=False)


def test_nonfinite_timestamps_are_counted_and_excluded_from_time_statistics():
    log = _log_with_channel("Time", [0.0, float("nan"), float("inf"), 1.0], [1.0, 2.0, 3.0, 4.0])

    report = build_diagnostic_report(log, source_format="VBO")
    channel = report["channels"][0]

    assert channel["time_min_s"] == 0.0
    assert channel["time_max_s"] == 1.0
    assert channel["time_invalid_fraction"] == 0.5
    assert channel["frequency_hz"] is None
    assert report["validation"]["valid"] is False
    assert {error["code"] for error in report["validation"]["errors"]} == {
        "CHANNEL_NONFINITE_TIMESTAMPS",
    }
    json.dumps(report, allow_nan=False)


def test_length_mismatch_and_empty_channels_are_invalid():
    log = DataLog()
    log.add_channel("Mismatch", "", float, 2)
    log.channels["Mismatch"].set_samples([0.0, 1.0], [3.0])
    log.add_channel("Empty", "", float, 2)

    report = build_diagnostic_report(log, source_format="FIT")

    by_name = {channel["name"]: channel for channel in report["channels"]}
    assert by_name["Mismatch"]["length_mismatch"] is True
    assert by_name["Empty"]["sample_count"] == 0
    assert report["validation"]["valid"] is False
    assert report["validation"]["errors"] == [
        {"code": "CHANNEL_EMPTY", "channel": "Empty"},
        {"code": "CHANNEL_LENGTH_MISMATCH", "channel": "Mismatch"},
    ]
    json.dumps(report, allow_nan=False)


def test_single_sample_has_no_frequency_and_empty_log_has_a_validation_error():
    single = _log_with_channel("RPM", [3.0], [4000.0])
    single_channel = build_diagnostic_report(single, source_format="ACTI")["channels"][0]
    assert single_channel["frequency_hz"] is None
    assert single_channel["gaps"]["count"] == 0

    empty_report = build_diagnostic_report(DataLog(), source_format="ACTI")
    assert empty_report["channels"] == []
    assert empty_report["validation"] == {"valid": False, "errors": [{"code": "NO_CHANNELS"}]}
    json.dumps(empty_report, allow_nan=False)


def test_extreme_finite_time_deltas_remain_strict_json_without_runtime_warnings():
    tiny_delta = _log_with_channel("Tiny", [0.0, 1e-320], [1.0, 2.0])
    unrepresentable_delta = _log_with_channel("Huge", [-1e308, 1e308], [1.0, 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tiny_report = build_diagnostic_report(tiny_delta, source_format="CSV")
        huge_report = build_diagnostic_report(unrepresentable_delta, source_format="CSV")

    tiny_channel = tiny_report["channels"][0]
    assert tiny_channel["frequency_hz"] is None
    assert tiny_channel["gaps"]["count"] == 0
    huge_channel = huge_report["channels"][0]
    assert huge_channel["frequency_hz"] is None
    assert huge_channel["gaps"] == {"count": 1, "max_s": None, "threshold_s": 1.0}
    json.dumps(tiny_report, allow_nan=False)
    json.dumps(huge_report, allow_nan=False)
