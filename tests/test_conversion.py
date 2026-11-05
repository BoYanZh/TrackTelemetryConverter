"""Tests for the programmatic conversion service boundary."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
from conftest import EXAMPLES, _run_cli_in_process, _write_backup_rcz

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.conversion import (
    ConversionOptions,
    ConversionRequest,
    ConversionResult,
    convert,
)


def _request(source: Path, output: Path, **options) -> ConversionRequest:
    return ConversionRequest(
        source=str(source),
        log_type="AUTO",
        options=ConversionOptions(output=str(output), **options),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert_without_output(request: ConversionRequest) -> ConversionResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = convert(request)
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""
    return result


def _read_channel_data(path: Path):
    parsed = ldData.fromfile(str(path))
    return {
        channel.name.strip(): (channel.freq, np.asarray(channel.data).copy())
        for channel in parsed.channs
    }


def _read_ldx_markers(path: Path):
    root = ET.parse(path).getroot()
    return [
        (
            group.attrib.get("Name"),
            tuple(
                (marker.attrib.get("Name"), marker.attrib.get("Time"))
                for marker in group.findall("Marker")
            ),
        )
        for group in root.findall(".//MarkerGroup")
    ]


def test_programmatic_conversion_is_quiet_structured_and_preserves_collision_and_force(tmp_path):
    source = Path(EXAMPLES) / "racechrono_sample.csv"
    output = tmp_path / "service.ld"
    request = _request(source, output)

    result = _convert_without_output(request)

    assert result.status == "success"
    assert result.error is None
    assert set(result.output_paths) == {str(output), str(output.with_suffix(".ldx"))}
    assert all(Path(path).is_file() for path in result.output_paths)
    assert _read_channel_data(output)
    assert ET.parse(output.with_suffix(".ldx")).getroot().tag == "LDXFile"

    repeated = _convert_without_output(request)
    assert repeated.status == "skipped"
    assert repeated.error is not None
    assert repeated.error["code"] == "OUTPUT_EXISTS"

    forced = _convert_without_output(
        _request(source, output, force=True),
    )
    assert forced.status == "success"
    assert _read_channel_data(output)


@pytest.mark.parametrize(
    ("entries", "expected_code"),
    [
        (
            [("channel_1_300_0_1_1", np.array([1, 2], dtype="<i8").tobytes())],
            "RCZ_SESSION_MISSING",
        ),
        (
            [
                (
                    "session.json",
                    json.dumps({"firstTimestamp": 1_700_000_000_000, "laps": []}),
                ),
                (
                    "channel_1_300_0_2_1",
                    np.array([0, 48, 105, 141, 176], dtype="<i8").tobytes(),
                ),
            ],
            "RCZ_TIMESTAMP_MISSING",
        ),
    ],
    ids=["missing-session", "missing-usable-timestamp"],
)
def test_malformed_rcz_returns_structured_failure_without_partial_output_or_source_change(
    tmp_path, entries, expected_code,
):
    source = tmp_path / "malformed.rcz"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in entries:
            archive.writestr(name, contents)
    before = _sha256(source)
    output = tmp_path / "not-created" / "converted.ld"

    result = _convert_without_output(_request(source, output))

    assert result.status == "failed"
    assert result.error is not None
    assert result.error["code"] == expected_code
    assert result.error["stage"].startswith("parse.")
    assert not output.parent.exists()
    assert not output.exists()
    assert not output.with_suffix(".ldx").exists()
    assert _sha256(source) == before


def test_programmatic_and_cli_exports_match_ld_channels_and_ldx_markers(tmp_path):
    source = Path(EXAMPLES) / "racechrono_sample.csv"
    service_output = tmp_path / "service.ld"
    cli_output = tmp_path / "cli.ld"

    result = _convert_without_output(_request(source, service_output))
    cli_result = _run_cli_in_process(
        [str(source), "AUTO", "--output", str(cli_output)],
    )

    assert result.status == "success"
    assert cli_result.returncode == 0, cli_result.stdout + cli_result.stderr
    service_channels = _read_channel_data(service_output)
    cli_channels = _read_channel_data(cli_output)
    assert service_channels.keys() == cli_channels.keys()
    for name in service_channels:
        service_freq, service_values = service_channels[name]
        cli_freq, cli_values = cli_channels[name]
        assert service_freq == cli_freq
        np.testing.assert_allclose(service_values, cli_values, equal_nan=True)
    assert _read_ldx_markers(service_output.with_suffix(".ldx")) == _read_ldx_markers(
        cli_output.with_suffix(".ldx"),
    )


def test_output_paths_omit_requested_auxiliary_formats_when_no_file_is_written(tmp_path):
    source = Path(EXAMPLES) / "csv_sample.csv"
    output = tmp_path / "without-gps.ld"

    result = _convert_without_output(_request(source, output, gpx=True, kml=True))

    assert result.status == "success"
    assert set(result.output_paths) == {str(output), str(output.with_suffix(".ldx"))}
    assert not output.with_suffix(".gpx").exists()
    assert not output.with_suffix(".kml").exists()


def test_conversion_request_and_result_are_picklable():
    source = Path(EXAMPLES) / "racechrono_sample.csv"
    request = _request(
        source,
        Path("converted.ld"),
        frequency="25",
        vbo=True,
        vbo_columns="MyRPM=Engine RPM",
    )
    result = ConversionResult(
        status="failed",
        output_paths=("converted.ld",),
        messages=("parse failed",),
        error={"code": "TEST_FAILURE", "stage": "parse", "message": "bad input"},
    )

    # Locally created pickle payloads.
    assert pickle.loads(pickle.dumps(request)) == request  # noqa: S301
    assert pickle.loads(pickle.dumps(result)) == result  # noqa: S301


def test_selected_backup_stint_failure_is_not_hidden_by_existing_first_stint(tmp_path):
    source = tmp_path / "backup.rcz"
    session = "session_20260102_1100"
    _write_backup_rcz(source)
    output_base = tmp_path / "selected.ld"
    request = ConversionRequest(
        source=str(source),
        log_type="RCZ",
        options=ConversionOptions(output=str(output_base), session=session),
    )

    initial = convert(request)
    assert initial.status == "success"
    stint0 = tmp_path / "selected_stint0.ld"
    stint1 = tmp_path / "selected_stint1.ld"
    assert stint0.is_file() and stint0.with_suffix(".ldx").is_file()
    assert stint1.is_file() and stint1.with_suffix(".ldx").is_file()
    stint1.unlink()
    stint1.with_suffix(".ldx").unlink()

    missing_clock = f"sessions/{session}/resume_1/channel_1_300_0_1_1"
    with zipfile.ZipFile(source, "r") as archive:
        entries = [
            (item.filename, archive.read(item.filename))
            for item in archive.infolist()
            if item.filename != missing_clock
        ]
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in entries:
            archive.writestr(name, contents)

    result = convert(request)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error["code"] == "RCZ_TIMESTAMP_MISSING"
    assert stint0.is_file() and stint0.with_suffix(".ldx").is_file()
    assert not stint1.exists() and not stint1.with_suffix(".ldx").exists()


def test_from_namespace_preserves_acti_options():
    """Batch mode must not drop ACTI flags when round-tripping argparse.

    Regression test: --realign-sf reached single-file conversion but was
    silently lost in batch mode because ConversionOptions lacked the field.
    """
    import argparse

    from track_telemetry_converter.conversion import _namespace

    args = argparse.Namespace(
        log="x.ld",
        log_type="ACTI",
        track="thunderhill_auto",
        anchor="",
        realign_sf=True,
    )
    request = ConversionRequest.from_namespace(args)
    assert request.options.track == "thunderhill_auto"
    assert request.options.realign_sf is True
    assert getattr(_namespace(request), "realign_sf", False) is True



