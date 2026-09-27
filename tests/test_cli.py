"""Tests for CLI entrypoint, roundtrips, atomic writes, and file collision guards."""

import os
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import patch

from conftest import (
    EXAMPLES,
    _assert_cli_roundtrip,
    _import_or_skip,
    _run_cli_in_process,
    _sha256,
)

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.cli import _csv_output_path
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.output import atomic_write_motec_pair, ensure_output_targets


def test_cli_ld_ldx_roundtrip_and_overwrite_guard():
    source = os.path.join(EXAMPLES, "aim_solo_sample.csv")
    with tempfile.TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, "roundtrip.ld")
        command = [
            source, "RACECHRONO",
            "--output", output,
        ]

        first = _run_cli_in_process(command)
        assert first.returncode == 0, first.stdout + first.stderr

        ldx_path = os.path.splitext(output)[0] + ".ldx"
        assert os.path.isfile(output)
        assert os.path.isfile(ldx_path)

        parsed = ldData.fromfile(output)
        assert parsed.channs
        for channel in parsed.channs:
            values = channel.data
            assert len(values) == channel.data_len
        root = ET.parse(ldx_path).getroot()
        assert root.tag == "LDXFile"

        before = {path: _sha256(path) for path in (output, ldx_path)}
        blocked = _run_cli_in_process(command)
        assert blocked.returncode != 0
        assert "already exists" in (blocked.stdout + blocked.stderr).lower()
        after = {path: _sha256(path) for path in (output, ldx_path)}
        assert after == before

        forced = _run_cli_in_process(
            command + [
                "--force",
                "--vehicle_id",
                "Toyota GR86",
                "--vehicle_weight",
                "1275",
            ]
        )
        assert forced.returncode == 0, forced.stdout + forced.stderr
        parsed = ldData.fromfile(output)
        assert parsed.head.vehicleid == "Toyota GR86"


def test_cli_vbo_end_to_end():
    _assert_cli_roundtrip(os.path.join(EXAMPLES, "vbo_sample.vbo"), "VBO")


def test_cli_ibt_end_to_end():
    _assert_cli_roundtrip(os.path.join(EXAMPLES, "ibt_sample.ibt"), "IBT")


def test_cli_xrk_end_to_end():
    _import_or_skip("libxrk")
    _assert_cli_roundtrip(os.path.join(EXAMPLES, "aim_sample.xrk"), "XRK")


def test_cli_fit_end_to_end():
    _import_or_skip("fitparse")
    _assert_cli_roundtrip(os.path.join(EXAMPLES, "garmin_sample.fit"), "FIT")


def test_atomic_output_verification_failure_preserves_existing_files():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples([0.0, 1.0], [10.0, 20.0])
    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ld_path = os.path.join(tmp_dir, "existing.ld")
        ldx_path = os.path.join(tmp_dir, "existing.ldx")
        with open(ld_path, "wb") as output:
            output.write(b"old ld")
        with open(ldx_path, "wb") as output:
            output.write(b"old ldx")

        with patch("track_telemetry_converter.output.verify_motec_pair", side_effect=RuntimeError("bad staged file")):
            try:
                atomic_write_motec_pair(motec, log, ld_path, ldx_path)
            except RuntimeError as exc:
                assert "bad staged file" in str(exc)
            else:
                raise AssertionError("verification failure must abort output replacement")

        with open(ld_path, "rb") as output:
            assert output.read() == b"old ld"
        with open(ldx_path, "rb") as output:
            assert output.read() == b"old ldx"
        assert not [name for name in os.listdir(tmp_dir) if name.startswith(".motec_")]

        real_replace = os.replace
        replace_calls = 0

        def fail_second_replace(source, target):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls == 2:
                raise PermissionError("second target locked")
            return real_replace(source, target)

        with patch("track_telemetry_converter.output.os.replace", side_effect=fail_second_replace):
            try:
                atomic_write_motec_pair(motec, log, ld_path, ldx_path)
            except PermissionError as exc:
                assert "second target locked" in str(exc)
            else:
                raise AssertionError("second replace failure must abort the pair commit")

        with open(ld_path, "rb") as output:
            assert output.read() == b"old ld"
        with open(ldx_path, "rb") as output:
            assert output.read() == b"old ldx"
        assert not [name for name in os.listdir(tmp_dir) if name.startswith(".motec_")]


def test_atomic_write_failure_cleans_up_partial_staged_files():
    log = DataLog()
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples([0.0, 1.0], [10.0, 20.0])
    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ld_path = os.path.join(tmp_dir, "fresh_output.ld")
        ldx_path = os.path.join(tmp_dir, "fresh_output.ldx")

        with patch("track_telemetry_converter.output.verify_motec_pair", side_effect=RuntimeError("corruption")):
            try:
                atomic_write_motec_pair(motec, log, ld_path, ldx_path)
            except RuntimeError as exc:
                assert "corruption" in str(exc)
            else:
                raise AssertionError("must propagate error")

        assert not os.path.exists(ld_path)
        assert not os.path.exists(ldx_path)
        assert not [name for name in os.listdir(tmp_dir) if name.startswith(".motec_")]

        try:
            ensure_output_targets([ld_path], force=True, source_path=ld_path)
        except ValueError as exc:
            assert "input file" in str(exc)
        else:
            raise AssertionError("--force must never replace the source input")


def test_csv_auxiliary_export_does_not_replace_source():
    source = os.path.join("logs", "session.csv")
    assert _csv_output_path(os.path.join("logs", "session.ld"), source).endswith(
        "session_export.csv"
    )
