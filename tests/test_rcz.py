"""Tests for RaceChrono RCZ format and multi-session archives."""

import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import timezone

import numpy as np
from conftest import (
    _assert_cli_roundtrip,
    _run_cli_in_process,
    _write_backup_rcz,
    _write_lapped_rcz,
    _write_minimal_rcz,
)

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.output import atomic_write_motec_pair


def test_cli_rcz_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "minimal.rcz")
        _write_minimal_rcz(source)
        _assert_cli_roundtrip(source, "RCZ", expected_channels=("Pitch Angle",))


def test_rcz_preserves_canonical_absolute_time_origin():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "minimal.rcz")
        _write_minimal_rcz(source)

        log = DataLog()
        log.from_rcz_log(source)

        assert log.time_origin_epoch_ms == 1_700_000_000_000
        assert log.session_created_epoch_ms == 1_700_000_000_000
        assert log.datetime_utc.tzinfo == timezone.utc
        assert log.datetime_utc.timestamp() == 1_700_000_000.0
        assert log.rcz_metadata["timeOriginEpochMs"] == 1_700_000_000_000


def test_rcz_rejects_non_epoch_gps_channel_2_as_timestamp(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "bad_timestamp.rcz")
        first = 1_700_000_000_000
        session = {
            "firstTimestamp": first,
            "timeCreated": first,
            "trackName": "Synthetic Test Track",
            "laps": [],
        }
        not_epoch = np.array([0, 48, 105, 141, 176], dtype="<i8")
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("session.json", json.dumps(session))
            archive.writestr("channel_1_300_0_2_1", not_epoch.tobytes())

        log = DataLog()
        log.from_rcz_log(source)

        assert log.channels == {}
        assert log.time_origin_epoch_ms is None
        assert "Could not find timestamp channel" in capsys.readouterr().out


def test_rcz_int64_device_timestamps_survive_low32_rollover():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "rollover.rcz")
        wrap = 2**32
        base = ((1_700_000_000_000 // wrap) + 1) * wrap - 100
        timestamps = (base + np.array([0, 50, 100, 150, 200])).astype("<i8")
        speed = np.full(5, 10_000, dtype="<i4")
        lat_lon = np.column_stack((
            np.arange(5, dtype="<i4") + 222_000_000,
            np.arange(5, dtype="<i4") - 732_000_000,
        ))
        pitch = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="<f8")
        lateral_g_raw = np.array([1000, 2000, 3000, 4000, 5000], dtype="<i4")
        session = {
            "firstTimestamp": int(base),
            "timeCreated": int(base),
            "trackName": "Synthetic Test Track",
            "laps": [],
        }
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("session.json", json.dumps(session))
            archive.writestr("channel_1_300_0_1_1", timestamps.tobytes())
            archive.writestr("channel_1_300_0_4_0", speed.tobytes())
            archive.writestr("channel_1_300_0_3_1", lat_lon.tobytes())
            archive.writestr("channel_2_201_0_1_1", timestamps.tobytes())
            archive.writestr("channel_2_201_0_10_0", lateral_g_raw.tobytes())
            archive.writestr("channel_12_100_8_8_1_1", timestamps.tobytes())
            archive.writestr("channel2_12_100_8_8_3", pitch.tobytes())

        log = DataLog()
        log.from_rcz_log(source)

        channel = log.channels["Pitch Angle"]
        assert np.allclose(channel.timestamps, [0.0, 0.05, 0.10, 0.15, 0.20])
        assert np.allclose(channel.values, [-2.0, -1.0, 0.0, 1.0, 2.0])
        assert np.allclose(
            log.channels["CG Accel Lateral"].values,
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert log.time_origin_epoch_ms == int(base)


def test_cli_rcz_backup_lists_sessions_without_exporting():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)

        result = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--list-sessions",
            ]
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "session_20260101_1000" in result.stdout
        assert "Alpha Track" in result.stdout
        assert "60.0s" in result.stdout
        assert "session_20260102_1100" in result.stdout
        assert "0,1" in result.stdout
        assert not any(name.endswith((".ld", ".ldx")) for name in os.listdir(tmp_dir))


def test_rcz_backup_target_session_reads_nested_channels_and_track_storage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)

        log = DataLog()
        log.from_rcz_log(source, target_session="session_20260101_1000")

        assert "Ground Speed" in log.channels
        assert log.rcz_metadata["trackName"] == "Alpha Track"
        assert log.duration() == 60.0
        assert log.traps == [
            {
                "name": "Start",
                "lat": 37.0,
                "lon": -122.0,
                "type": 1,
            }
        ]

        fallback_log = DataLog()
        fallback_log.from_rcz_log(
            source,
            target_session="session_20260102_1100",
            target_stint=0,
        )
        assert "Ground Speed" in fallback_log.channels
        assert fallback_log.traps[0]["name"] == "Finish"


def test_cli_rcz_backup_requires_session_and_exports_selected_session():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)

        missing_selection = _run_cli_in_process([source, "RCZ"])
        assert missing_selection.returncode == 1
        assert "contains 2 sessions" in missing_selection.stdout
        assert "--session ID" in missing_selection.stdout
        assert not any(name.endswith((".ld", ".ldx")) for name in os.listdir(tmp_dir))

        selected = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "session_20260101_1000",
            ]
        )
        assert selected.returncode == 0, selected.stdout + selected.stderr
        ld_path = os.path.join(tmp_dir, "backup_session_20260101_1000.ld")
        ldx_path = os.path.splitext(ld_path)[0] + ".ldx"
        assert ldData.fromfile(ld_path).channs
        assert ET.parse(ldx_path).getroot().tag == "LDXFile"

        invalid = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "session_missing",
            ]
        )
        assert invalid.returncode == 1
        assert "Unknown RCZ session ID 'session_missing'" in invalid.stdout


def test_cli_rcz_backup_exports_all_with_preflight_and_stint_split():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)
        output_dir = os.path.join(tmp_dir, "backup_sessions")

        result = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "all",
            ]
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "3 succeeded, 0 failed" in result.stdout

        stems = [
            "session_20260101_1000",
            "session_20260102_1100_stint0",
            "session_20260102_1100_stint1",
        ]
        for stem in stems:
            ld_path = os.path.join(output_dir, stem + ".ld")
            ldx_path = os.path.join(output_dir, stem + ".ldx")
            parsed = ldData.fromfile(ld_path)
            running_time = next(
                channel for channel in parsed.channs
                if channel.name.strip() == "Running Time"
            )
            assert running_time.data[0] == 0
            assert ET.parse(ldx_path).getroot().tag == "LDXFile"

        blocked = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "all",
                "--output-dir",
                output_dir,
            ]
        )
        assert blocked.returncode == 1
        assert "output already exists" in blocked.stdout
        assert "Preflight failed; no sessions were exported" in blocked.stdout

        forced = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "all",
                "--output-dir",
                output_dir,
                "--force",
            ]
        )
        assert forced.returncode == 0, forced.stdout + forced.stderr


def test_cli_rcz_backup_rejects_ambiguous_all_options():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)
        cases = [
            (["--session", "all", "--output", "one.ld"], "--output"),
            (["--session", "all", "--stint", "0"], "--stint all"),
        ]
        for options, expected in cases:
            result = _run_cli_in_process(
                [source, "RCZ"] + options
            )
            assert result.returncode == 1
            assert expected in result.stdout


def test_cli_rcz_backup_all_continues_after_session_failure():
    import zipfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)
        with zipfile.ZipFile(source, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "sessions/session_20260103_broken/session.json",
                json.dumps(
                    {
                        "firstTimestamp": 1_700_200_000_000,
                        "latestTimestamp": 1_700_200_010_000,
                        "timeCreated": 1_700_200_000_000,
                        "lengthTime": 10_000,
                        "lapCount": 0,
                        "trackName": "Broken Track",
                        "laps": [],
                    }
                ),
            )

        output_dir = os.path.join(tmp_dir, "exports")
        result = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "all",
                "--output-dir",
                output_dir,
            ]
        )

        assert result.returncode == 1
        assert "3 succeeded, 1 failed" in result.stdout
        assert "session_20260103_broken stint 0 failed" in result.stdout
        assert os.path.isfile(
            os.path.join(output_dir, "session_20260101_1000.ld")
        )
        assert not os.path.exists(
            os.path.join(output_dir, "session_20260103_broken.ld")
        )


def test_cli_rcz_backup_selected_session_splits_multiple_stints():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)

        result = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--session",
                "session_20260102_1100",
            ]
        )

        assert result.returncode == 0, result.stdout + result.stderr
        expected_stem = os.path.join(
            tmp_dir, "backup_session_20260102_1100_stint"
        )
        for stint in (0, 1):
            assert os.path.isfile(expected_stem + str(stint) + ".ld")
            assert os.path.isfile(expected_stem + str(stint) + ".ldx")


def test_rcz_target_lap_rebases_time_and_lap_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "lapped.rcz")
        _write_lapped_rcz(source)

        log = DataLog()
        log.from_rcz_log(source, target_lap="2")

        running_time = log.channels["Running Time"]
        assert running_time.timestamps[0] == 0.0
        assert running_time.values[0] == 0.0
        assert np.all(running_time.timestamps == running_time.values)
        assert set(log.channels["Lap Number"].values) == {2.0}

        laps = log.laps_info["laps"]
        assert len(laps) == 1
        assert laps[0]["type"] == "Timed"
        assert laps[0]["lap_num"] == 2
        assert laps[0]["start_time"] == 0.0
        assert laps[0]["end_time"] == 30.0
        assert log.laps_info["total_laps"] == 1
        assert log.laps_info["fastest_lap"] == 2
        assert log.laps_info["fastest_time"] == 30.0
        assert log.laps_info["session_duration"] == 30.0
        assert log.time_origin_epoch_ms == 1_700_000_020_000

        motec = MotecLog()
        motec.initialize()
        motec.add_all_channels(log)
        ld_path = os.path.join(tmp_dir, "lap2.ld")
        ldx_path = os.path.join(tmp_dir, "lap2.ldx")
        atomic_write_motec_pair(motec, log, ld_path, ldx_path)
        root = ET.parse(ldx_path).getroot()
        assert len(root.findall(".//Laps/Lap")) == 1
        assert [float(marker.get("Time")) for marker in root.findall(".//Marker")] == [
            0.0,
            30_000_000.0,
        ]


def test_rcz_min_lap_sec_controls_segment_filtering():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "lapped.rcz")
        _write_lapped_rcz(source)

        default_log = DataLog()
        default_log.from_rcz_log(source)
        assert [lap["type"] for lap in default_log.laps_info["laps"]] == [
            "Out Lap",
            "Timed",
            "Timed",
            "In Lap",
        ]

        filtered_log = DataLog()
        filtered_log.from_rcz_log(source, min_lap_sec=25.0)
        assert [lap["type"] for lap in filtered_log.laps_info["laps"]] == [
            "Timed",
            "Timed",
        ]


def test_rcz_high_start_speed_marks_partial_out_lap(capsys):
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "mid_lap.rcz")
        _write_lapped_rcz(source, start_speed_mps=20.0)

        log = DataLog()
        log.from_rcz_log(source)

        first_lap = log.laps_info["laps"][0]
        assert first_lap["type"] == "Partial Out Lap"
        assert first_lap["lap_label"] == "Partial Out Lap"
        assert "begins mid-lap at 72.0 km/h" in capsys.readouterr().out


def test_cli_rejects_invalid_min_lap_sec():
    result = _run_cli_in_process(
        [
            "missing.rcz",
            "RCZ",
            "--min-lap-sec",
            "0",
        ]
    )
    assert result.returncode == 1
    assert "--min-lap-sec must be a positive finite number" in result.stdout
