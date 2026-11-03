"""CLI inspection returns privacy-safe diagnostics and never exports telemetry."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
from conftest import (
    _run_cli_in_process,
    _write_backup_rcz,
    _write_lapped_rcz,
    _write_minimal_rcz,
)

from track_telemetry_converter.inspection import inspect_file


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codes(report: dict) -> set[str]:
    return {item["code"] for item in report["validation"]["errors"]}


def test_inspect_cli_emits_strict_json_without_writing_exports():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "private source.rcz"
        _write_minimal_rcz(source)
        before = _sha256(source)
        result = _run_cli_in_process([str(source), "AUTO", "--inspect"])

        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["schema_version"] == 1
        assert report["source"] == {"format": "RCZ"}
        assert report["timebase"]["origin_epoch_ms"] == 1_700_000_000_000
        assert report["stages"] == ["parse"]
        assert report["channels"]
        assert not set(("values", "samples", "metadata", "source_path")) & set(report)
        encoded = json.dumps(report, allow_nan=False)
        assert str(source) not in encoded
        assert "Synthetic Test Track" not in encoded
        assert "222000000" not in encoded
        assert "CLI RCZ Test" not in encoded
        assert list(Path(tmp).iterdir()) == [source]
        assert _sha256(source) == before


@pytest.mark.parametrize(
    ("entry_name", "expected_code"),
    [
        ("channel-only", "RCZ_SESSION_MISSING"),
        ("unusable-timestamp", "RCZ_TIMESTAMP_MISSING"),
    ],
)
def test_inspect_malformed_rcz_reports_error_without_partial_outputs(entry_name, expected_code):
    import zipfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "malformed.rcz"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            if entry_name == "channel-only":
                archive.writestr(
                    "channel_1_300_0_2_1",
                    np.array([1, 2, 3], dtype="<i8").tobytes(),
                )
            else:
                first = 1_700_000_000_000
                archive.writestr(
                    "session.json",
                    json.dumps({"firstTimestamp": first, "timeCreated": first, "laps": []}),
                )
                archive.writestr(
                    "channel_1_300_0_2_1",
                    np.array([0, 48, 105, 141, 176], dtype="<i8").tobytes(),
                )
        before = _sha256(source)

        result = _run_cli_in_process([str(source), "RCZ", "--inspect"])
        report = json.loads(result.stdout)

        assert result.returncode != 0
        assert expected_code in _codes(report)
        assert str(source) not in result.stdout
        assert list(Path(tmp).iterdir()) == [source]
        assert _sha256(source) == before


def test_inspect_rejects_missing_and_unsupported_inputs_with_stable_codes(tmp_path):
    missing = inspect_file(str(tmp_path / "missing.csv"))
    assert _codes(missing) == {"INPUT_NOT_FOUND"}

    source = tmp_path / "sample.csv"
    source.write_text("time,value\n0,1\n1,2\n", encoding="utf-8")
    unsupported = inspect_file(str(source), "NOT_A_FORMAT")
    assert _codes(unsupported) == {"INPUT_FORMAT_UNSUPPORTED"}


def test_inspect_rejects_conversion_output_options_as_json():
    result = _run_cli_in_process(["missing.rcz", "RCZ", "--inspect", "--csv"])
    report = json.loads(result.stdout)

    assert result.returncode != 0
    assert _codes(report) == {"INSPECT_CONVERSION_OPTIONS_UNSUPPORTED"}


def test_inspect_debug_report_is_same_strict_json_and_respects_collision_guard():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "minimal.rcz"
        report_path = Path(tmp) / "inspect.json"
        _write_minimal_rcz(source)

        result = _run_cli_in_process([
            str(source),
            "RCZ",
            "--inspect",
            "--debug-report",
            str(report_path),
        ])
        assert result.returncode == 0
        assert json.loads(result.stdout) == json.loads(report_path.read_text(encoding="utf-8"))

        before = report_path.read_bytes()
        blocked = _run_cli_in_process([
            str(source),
            "RCZ",
            "--inspect",
            "--debug-report",
            str(report_path),
        ])
        assert blocked.returncode != 0
        blocked_report = json.loads(blocked.stdout)
        assert "DEBUG_REPORT_WRITE_FAILED" in _codes(blocked_report)
        assert str(source) not in json.dumps(blocked_report)
        assert str(report_path) not in json.dumps(blocked_report)
        assert report_path.read_bytes() == before

        forced = _run_cli_in_process([
            str(source),
            "RCZ",
            "--inspect",
            "--debug-report",
            str(report_path),
            "--force",
        ])
        assert forced.returncode == 0
        assert json.loads(forced.stdout) == json.loads(report_path.read_text(encoding="utf-8"))


def test_inspect_debug_report_never_replaces_source_even_with_force():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "minimal.rcz"
        _write_minimal_rcz(source)
        before = _sha256(source)

        result = _run_cli_in_process([
            str(source),
            "RCZ",
            "--inspect",
            "--debug-report",
            str(source),
            "--force",
        ])

        assert result.returncode != 0
        failure_report = json.loads(result.stdout)
        assert "DEBUG_REPORT_WRITE_FAILED" in _codes(failure_report)
        assert str(source) not in json.dumps(failure_report)
        assert _sha256(source) == before


def test_inspect_backup_requires_explicit_session_and_stint():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "backup.rcz"
        _write_backup_rcz(source)

        missing_session = inspect_file(str(source), "RCZ")
        assert "RCZ_SESSION_SELECTION_REQUIRED" in _codes(missing_session)

        multi_stint = inspect_file(
            str(source),
            "RCZ",
            session="session_20260102_1100",
        )
        assert "RCZ_STINT_SELECTION_REQUIRED" in _codes(multi_stint)

        selected = inspect_file(
            str(source),
            "RCZ",
            session="session_20260102_1100",
            stint=0,
        )
        assert selected["validation"]["valid"]
        assert selected["selection"] == {
            "session": "session_20260102_1100",
            "stint": 0,
            "lap": "all",
        }


def test_inspect_rcz_rejects_unknown_and_malformed_lap_selections():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "lapped.rcz"
        _write_lapped_rcz(source)

        valid = inspect_file(str(source), "RCZ", lap=2)
        assert valid["validation"]["valid"]
        assert valid["selection"]["lap"] == 2

        unknown = inspect_file(str(source), "RCZ", lap=999)
        assert "RCZ_LAP_UNKNOWN" in _codes(unknown)
        assert unknown["validation"]["errors"][0]["stage"] == "parse.selection"

        malformed = inspect_file(str(source), "RCZ", lap="lap-two")
        assert "RCZ_LAP_SELECTION_INVALID" in _codes(malformed)

        nonpositive = inspect_file(str(source), "RCZ", lap=0)
        assert "RCZ_LAP_SELECTION_INVALID" in _codes(nonpositive)
