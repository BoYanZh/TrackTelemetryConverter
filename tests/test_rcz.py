"""Tests for RaceChrono RCZ format and multi-session archives."""

import json
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import timezone

import numpy as np
import pytest
from conftest import (
    _assert_cli_roundtrip,
    _run_cli_in_process,
    _write_backup_rcz,
    _write_lapped_rcz,
    _write_minimal_rcz,
)

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.errors import TelemetryParseError
from track_telemetry_converter.interpolation import _mask_interp_outside_range
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.output import atomic_write_motec_pair


def test_cli_rcz_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "minimal.rcz")
        _write_minimal_rcz(source)
        _assert_cli_roundtrip(source, "RCZ", expected_channels=("CG Accel Longitudinal",))


def test_rcz_preserves_canonical_absolute_time_origin():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "minimal.rcz")
        _write_minimal_rcz(source)

        log = DataLog()
        log.from_rcz_log(source)

        assert log.time_origin_epoch_ms == 1_700_000_000_000
        assert log.session_created_epoch_ms == 1_700_000_000_000
        assert log.datetime_utc is not None
        assert log.datetime_utc.tzinfo == timezone.utc
        assert log.datetime_utc.timestamp() == 1_700_000_000.0
        assert log.rcz_metadata["timeOriginEpochMs"] == 1_700_000_000_000


def test_rcz_missing_session_metadata_has_structured_error():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "missing_session.rcz")
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("channel_1_300_0_1_1", np.array([1, 2], dtype="<i8").tobytes())

        log = DataLog()
        with pytest.raises(TelemetryParseError) as caught:
            log.from_rcz_log(source)

        error = caught.value.to_dict()
        assert error["code"] == "RCZ_SESSION_MISSING"
        assert error["stage"] == "parse.session"
        assert error["source"] == source
        assert error["details"]["expected_entry"] == "session.json"
        assert log.channels == {}
        assert log.time_origin_epoch_ms is None


def test_rcz_rejects_non_epoch_gps_channel_2_as_timestamp():
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
        with pytest.raises(TelemetryParseError) as caught:
            log.from_rcz_log(source)

        assert log.channels == {}
        assert log.time_origin_epoch_ms is None
        assert caught.value.code == "RCZ_TIMESTAMP_MISSING"
        assert caught.value.stage == "parse.timebase"
        assert caught.value.to_dict()["source"] == source


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
        lon_accel_ms2 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="<f8")
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
            archive.writestr("channel2_12_100_8_8_3", lon_accel_ms2.tobytes())

        log = DataLog()
        log.from_rcz_log(source)

        channel = log.channels["CG Accel Longitudinal"]
        assert np.allclose(channel.timestamps, [0.0, 0.05, 0.10, 0.15, 0.20])
        assert np.allclose(
            channel.values, np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / 9.80665,
        )
        assert np.allclose(
            log.channels["CG Accel Lateral"].values,
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert log.time_origin_epoch_ms == int(base)


def test_rcz_outside_range_mask_keeps_endpoints_and_source_unchanged():
    source = np.array([0.04, 0.10, 0.20])
    target = np.array([0.0, 0.04, 0.10, 0.20, 0.24])
    interpolated = np.array([10.0, 10.0, 20.0, 30.0, 30.0])
    masked = _mask_interp_outside_range(interpolated, target, source)

    np.testing.assert_allclose(masked, [np.nan, 10, 20, 30, np.nan])
    np.testing.assert_array_equal(interpolated, [10, 10, 20, 30, 30])
    assert np.isnan(_mask_interp_outside_range([1, 2], [0, 1], [])).all()


@pytest.mark.parametrize("mask_interp_gaps", [False, True])
def test_rcz_sensor_boundaries_use_per_sample_timestamps(tmp_path, mask_interp_gaps):
    first = 1_700_000_000_000
    # Deliberately irregular GPS intervals: no synthetic fixed-Hz timebase.
    gps_ms = np.array([0, 40, 90, 140, 200, 260, 400, 500], dtype="<i8")
    gps = gps_ms + first
    imu_ms = np.array([40, 90, 200, 400], dtype="<i8")
    gyro_ms = np.array([0, 90, 260], dtype="<i8")
    can_ms = np.array([90, 200, 400], dtype="<i8")
    path = tmp_path / "uneven-sensor-windows.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        archive.writestr("channel_1_300_0_1_1", gps.tobytes())
        archive.writestr(
            "channel_1_300_0_4_0",
            np.full(gps.size, 10_000, dtype="<i4").tobytes(),
        )
        archive.writestr("channel_2_201_0_1_1", (first + imu_ms).tobytes())
        archive.writestr(
            "channel_2_201_0_10_0",
            np.array([1000, 2000, 3000, 4000], dtype="<i4").tobytes(),
        )
        archive.writestr("channel_3_202_0_1_1", (first + gyro_ms).tobytes())
        archive.writestr(
            "channel_3_202_0_14_0",
            np.array([1000, 2000, 3000], dtype="<i4").tobytes(),
        )
        for pid, values in (
            ("10024", [1000.0, 3000.0, 5000.0]),  # continuous RPM
            ("1004", [2.0, 3.0, 4.0]),  # discrete gear / ZOH
        ):
            archive.writestr(f"channel_12_100_{pid}_1_1", (first + can_ms).tobytes())
            archive.writestr(
                f"channel2_12_100_{pid}_{pid}_3",
                np.array(values, dtype="<f8").tobytes(),
            )
        archive.writestr("channel_4_101_0_1_1", (first + imu_ms).tobytes())
        archive.writestr(
            "channel2_4_101_0_1002_3",
            np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8").tobytes(),
        )

    log = DataLog()
    log.from_rcz_log(str(path), mask_interp_gaps=mask_interp_gaps)
    expected_times = gps_ms.astype(np.float64) / 1000.0
    assert log.time_origin_epoch_ms == first
    for name in ("Ground Speed", "Engine RPM", "Gear", "CG Accel Lateral",
                 "Chassis Yaw Rate", "Brake Pos"):
        np.testing.assert_array_equal(log.channels[name].timestamps, expected_times)

    rpm = log.channels["Engine RPM"].values
    np.testing.assert_allclose(rpm, [np.nan, np.nan, 1000, 1909.090909, 3000,
                                     3600, 5000, np.nan], atol=1e-5)
    gear = log.channels["Gear"].values
    np.testing.assert_allclose(gear, [np.nan, np.nan, 2, 2, 3, 3, 4, np.nan])
    imu = log.channels["CG Accel Lateral"].values
    np.testing.assert_allclose(imu, [np.nan, 0.1, 0.2, 0.245454545, 0.3,
                                     0.33, 0.4, np.nan], atol=1e-5)
    yaw = log.channels["Chassis Yaw Rate"].values
    np.testing.assert_allclose(yaw, [-1, -1.444444444, -2, -2.294117647,
                                     -2.647058824, -3, np.nan, np.nan], atol=1e-5)
    brake = log.channels["Brake Pos"].values
    np.testing.assert_allclose(brake, [np.nan, 10, 20, 24.545454545, 30,
                                       33, 40, np.nan], atol=1e-5)


def test_rcz_maps_four_wheel_speed_pids(tmp_path):
    """Wheel speed PIDs use raw m/s and map to FL/FR/RL/RR (km/h).

    PID identities verified on 2026-10 Thunder Hill East Bypass sessions:
    left = {197623, 132087}, driven (rear) axle = {132087, 99319}.
    """
    first = 1_700_000_000_000
    stamps_ms = (first + np.array([0, 100, 200, 300], dtype="<i8")).astype("<i8")
    path = tmp_path / "wheel-speeds.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        archive.writestr("channel_1_300_0_1_1", stamps_ms.tobytes())
        archive.writestr(
            "channel_1_300_0_4_0",
            np.full(4, 10_000, dtype="<i4").tobytes(),
        )
        for pid in ("197623", "164855", "132087", "99319"):
            archive.writestr(f"channel_12_200_{pid}_1_1", stamps_ms.tobytes())
            archive.writestr(
                f"channel2_12_200_{pid}_{pid}_3",
                np.array([10.0, 20.0, 30.0, 40.0], dtype="<f8").tobytes(),
            )

    log = DataLog()
    log.from_rcz_log(str(path))

    expected = {
        "Wheel Speed FL": "197623",
        "Wheel Speed FR": "164855",
        "Wheel Speed RL": "132087",
        "Wheel Speed RR": "99319",
    }
    assert set(expected) <= set(log.channels)
    for name in expected:
        channel = log.channels[name]
        assert channel.units == "km/h"
        np.testing.assert_allclose(
            channel.values, [36.0, 72.0, 108.0, 144.0], atol=1e-9,
        )


def test_rcz_maps_app_verified_pids(tmp_path):
    """PIDs from the RaceChrono v10.2.4 channel tables map to canonical channels.

    10058 fuel level (x2.00 vs pid 1020 in 10 corpus files), 1003 clutch_pos,
    14 Z-rate gyro (sole yaw source in 2021-22 files), 43 combined accel in
    m/s^2 (0 ~ 19.7 in 6 files).
    """
    first = 1_700_000_000_000
    stamps_ms = (first + np.array([0, 100, 200, 300], dtype="<i8")).astype("<i8")
    path = tmp_path / "app-pids.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        archive.writestr("channel_1_300_0_1_1", stamps_ms.tobytes())
        archive.writestr(
            "channel_1_300_0_4_0",
            np.full(4, 10_000, dtype="<i4").tobytes(),
        )
        pid_values = {
            "10058": [10.0, 20.0, 30.0, 40.0],
            "1020": [5.0, 10.0, 15.0, 20.0],
            "1003": [0.0, 50.0, 100.0, 25.0],
            "14": [-10.0, 5.0, 20.0, -30.0],
            "43": [4.903325, 9.80665, 14.709975, 19.6133],
        }
        for pid, values in pid_values.items():
            archive.writestr(f"channel_12_200_{pid}_1_1", stamps_ms.tobytes())
            archive.writestr(
                f"channel2_12_200_{pid}_{pid}_3",
                np.array(values, dtype="<f8").tobytes(),
            )

    log = DataLog()
    log.from_rcz_log(str(path))

    fuel = log.channels["Fuel Level"]
    assert fuel.units == "%"
    np.testing.assert_allclose(fuel.values, [10.0, 20.0, 30.0, 40.0], atol=1e-9)

    volume = log.channels["Fuel Volume"]
    assert volume.units == "L"
    np.testing.assert_allclose(volume.values, [5.0, 10.0, 15.0, 20.0], atol=1e-9)

    clutch = log.channels["Clutch Pos"]
    assert clutch.units == "%"
    np.testing.assert_allclose(clutch.values, [0.0, 50.0, 100.0, 25.0], atol=1e-9)

    yaw = log.channels["Chassis Yaw Rate"]
    assert yaw.units == "deg/s"
    np.testing.assert_allclose(yaw.values, [-10.0, 5.0, 20.0, -30.0], atol=1e-9)

    combined = log.channels["G Force Combined"]
    assert combined.units == "G"
    np.testing.assert_allclose(combined.values, [0.5, 1.0, 1.5, 2.0], atol=1e-9)


def test_cli_rcz_backup_lists_sessions_without_exporting():
    with tempfile.TemporaryDirectory() as tmp_dir:
        source = os.path.join(tmp_dir, "backup.rcz")
        _write_backup_rcz(source)

        result = _run_cli_in_process(
            [
                source,
                "RCZ",
                "--list-sessions",
            ],
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
            },
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
            ],
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
            ],
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
            ],
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
            ],
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
            ],
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
                [source, "RCZ"] + options,
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
                    },
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
            ],
        )

        assert result.returncode == 1
        assert "3 succeeded, 1 failed" in result.stdout
        assert "session_20260103_broken stint 0 failed" in result.stdout
        assert os.path.isfile(
            os.path.join(output_dir, "session_20260101_1000.ld"),
        )
        assert not os.path.exists(
            os.path.join(output_dir, "session_20260103_broken.ld"),
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
            ],
        )

        assert result.returncode == 0, result.stdout + result.stderr
        expected_stem = os.path.join(
            tmp_dir, "backup_session_20260102_1100_stint",
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
        assert [float(marker.get("Time", "0")) for marker in root.findall(".//Marker")] == [
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
        ],
    )
    assert result.returncode == 1
    assert "--min-lap-sec must be a positive finite number" in result.stdout



def test_rcz_imu_axes_and_can_yaw_use_consistent_vehicle_sign(tmp_path):
    """Synthetic device-2/3 RCZ uses G and CAN-positive yaw conventions."""
    first = 1_700_000_000_000
    ts = (first + np.arange(16, dtype="<i8") * 40).astype("<i8")
    path = tmp_path / "imu-axis-conventions.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_100_0_1_1", ts.tobytes())
        z.writestr("channel_1_100_0_4_0",
                   np.full(len(ts), 20_000, dtype="<i4").tobytes())
        z.writestr("channel_2_101_0_1_1", ts.tobytes())
        z.writestr("channel_2_101_0_9_0",
                   np.full(len(ts), -3200, dtype="<i4").tobytes())
        z.writestr("channel_2_101_0_10_0",
                   np.full(len(ts), 8500, dtype="<i4").tobytes())
        z.writestr("channel_2_101_0_11_0",
                   np.full(len(ts), 9700, dtype="<i4").tobytes())
        z.writestr("channel_3_102_0_1_1", ts.tobytes())
        z.writestr("channel_3_102_0_14_0",
                   np.full(len(ts), -12000, dtype="<i4").tobytes())

    gyro = DataLog()
    gyro.from_rcz_log(str(path), mask_interp_gaps=True)
    np.testing.assert_allclose(gyro.channels["CG Accel Lateral"].values, 0.85)
    np.testing.assert_allclose(gyro.channels["CG Accel Longitudinal"].values, -0.32)
    np.testing.assert_allclose(gyro.channels["CG Accel Vertical"].values, 0.97)
    assert "Lean Angle" not in gyro.channels
    np.testing.assert_allclose(gyro.channels["Chassis Yaw Rate"].values, 12.0)
    assert gyro.metadata["longitudinal_accel_source"] == "rcz_accelerometer_2_101"
    assert gyro.metadata["lateral_accel_source"] == "rcz_accelerometer_2_101"
    assert gyro.metadata["vertical_accel_source"] == "rcz_accelerometer_2_101"
    assert gyro.metadata["yaw_rate_source"] == "rcz_gyro_3_102"
    assert gyro.metadata["yaw_axis_sign_verified"] is True

    # CAN yaw uses the same positive-vehicle sign and supersedes device 3.
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("channel_12_200_51_1_1", ts.tobytes())
        z.writestr("channel2_12_200_51_51_3",
                   np.full(len(ts), 12.0, dtype="<f8").tobytes())
    can = DataLog()
    can.from_rcz_log(str(path), mask_interp_gaps=True)
    np.testing.assert_allclose(can.channels["Chassis Yaw Rate"].values, 12.0)
    assert can.metadata["yaw_rate_source"] == "rcz_can_12_200_pid_51"
    assert can.metadata["yaw_axis_sign_verified"] is True



def test_rcz_stint_zero_does_not_read_resume_can_channels(tmp_path):
    """Root-prefix parsing must not ingest later-stint channels."""
    start = 1_700_000_000_000
    early = start + np.arange(16, dtype="<i8") * 40
    later = start + 120_000 + np.arange(16, dtype="<i8") * 40
    path = tmp_path / "isolated-stints.rcz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": start, "timeCreated": start, "laps": [],
        }))
        for prefix, timestamps in (("", early), ("resume_1/", later)):
            z.writestr(prefix + "channel_1_100_0_1_1", timestamps.tobytes())
            z.writestr(prefix + "channel_1_100_0_4_0",
                       np.full(16, 15_000, dtype="<i4").tobytes())
        z.writestr("resume_1/channel_12_200_51_1_1", later.tobytes())
        z.writestr("resume_1/channel2_12_200_51_51_3",
                   np.full(16, 10.0, dtype="<f8").tobytes())

    root = DataLog()
    root.from_rcz_log(str(path), target_stint=0)
    assert "Chassis Yaw Rate" not in root.channels
    assert root.metadata.get("yaw_rate_source") is None

    resumed = DataLog()
    resumed.from_rcz_log(str(path), target_stint=1)
    np.testing.assert_allclose(resumed.channels["Chassis Yaw Rate"].values, 10)
    assert resumed.metadata["yaw_rate_source"] == "rcz_can_12_200_pid_51"


def test_rcz_gyro_xy_parsed_alongside_yaw(tmp_path):
    """Device-3 gyro X/Y (_12/_13) parse as roll/pitch rate; absence is a skip."""
    first = 1_700_000_000_000
    ts = (first + np.arange(8, dtype="<i8") * 40).astype("<i8")
    path = tmp_path / "gyro-xyz.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_100_0_1_1", ts.tobytes())
        z.writestr("channel_1_100_0_4_0",
                   np.full(len(ts), 20_000, dtype="<i4").tobytes())
        z.writestr("channel_3_102_0_1_1", ts.tobytes())
        z.writestr("channel_3_102_0_12_0",
                   np.full(len(ts), 4000, dtype="<i4").tobytes())
        z.writestr("channel_3_102_0_13_0",
                   np.full(len(ts), -6000, dtype="<i4").tobytes())
        z.writestr("channel_3_102_0_14_0",
                   np.full(len(ts), -12000, dtype="<i4").tobytes())

    log = DataLog()
    log.from_rcz_log(str(path), mask_interp_gaps=True)
    np.testing.assert_allclose(log.channels["Chassis Roll Rate"].values, -4.0)
    np.testing.assert_allclose(log.channels["Chassis Pitch Rate"].values, 6.0)
    np.testing.assert_allclose(log.channels["Chassis Yaw Rate"].values, 12.0)
    assert log.channels["Chassis Roll Rate"].units == "deg/s"
    assert log.channels["Chassis Pitch Rate"].units == "deg/s"

    # Legacy pack with Z only: no roll/pitch channels, yaw unchanged.
    legacy = tmp_path / "gyro-z-only.rcz"
    with zipfile.ZipFile(legacy, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_100_0_1_1", ts.tobytes())
        z.writestr("channel_1_100_0_4_0",
                   np.full(len(ts), 20_000, dtype="<i4").tobytes())
        z.writestr("channel_3_102_0_1_1", ts.tobytes())
        z.writestr("channel_3_102_0_14_0",
                   np.full(len(ts), -12000, dtype="<i4").tobytes())

    old = DataLog()
    old.from_rcz_log(str(legacy), mask_interp_gaps=True)
    assert "Chassis Roll Rate" not in old.channels
    assert "Chassis Pitch Rate" not in old.channels
    np.testing.assert_allclose(old.channels["Chassis Yaw Rate"].values, 12.0)


def _write_gps_aux_rcz(path, first, sats_raw, fix_raw=None, accuracy_mm=None,
                       battery_mpc=None, coord_prec_mm=None, alt_prec_mm=None,
                       gps_type="300"):
    """Write an RCZ whose GPS family carries the uncertain Aux value files."""
    import zipfile

    n = 5
    ts = (first + np.arange(n, dtype="<i8") * 40).astype("<i8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr(f"channel_1_{gps_type}_0_1_1", ts.tobytes())
        z.writestr(f"channel_1_{gps_type}_0_4_0",
                   np.full(n, 20_000, dtype="<i4").tobytes())
        payloads = {
            "30002_0": sats_raw, "30003_0": fix_raw, "30007_0": accuracy_mm,
            "46_0": battery_mpc, "30004_0": coord_prec_mm,
            "30005_0": alt_prec_mm,
        }
        for suffix, values in payloads.items():
            if values is None:
                continue
            z.writestr(f"channel_1_{gps_type}_0_{suffix}",
                       np.array(values, dtype="<i4").tobytes())


def test_rcz_gps_aux_promoted_to_canonical_names(tmp_path):
    """GPS value files 30002/30003/30007/46 parse under verified real names.

    Paired 2026-10-04 Thunder Hill East Bypass RCZ+CSV cross-check (all 5
    stints, 126,746 GPS samples): A↔satellites and B↔fix match exactly,
    C↔accuracy and D↔battery within CSV text rounding.
    """
    first = 1_700_000_000_000
    path = tmp_path / "gps-aux.rcz"
    _write_gps_aux_rcz(
        path, first,
        sats_raw=[14, 15, 15, 14, 13],
        fix_raw=[3, 3, 3, 3, 3],
        accuracy_mm=[1129, 1125, 1122, 1119, 1116],
        battery_mpc=[84000, 84000, 83000, 83000, 82000],
    )

    log = DataLog()
    log.from_rcz_log(str(path))

    sats = log.channels["GPS Satellites"]
    assert sats.units == ""
    np.testing.assert_allclose(sats.values, [14, 15, 15, 14, 13])
    np.testing.assert_allclose(
        log.channels["GPS Fix"].values, [3, 3, 3, 3, 3],
    )
    acc = log.channels["GPS Accuracy"]
    assert acc.units == "m"
    np.testing.assert_allclose(acc.values, [1.129, 1.125, 1.122, 1.119, 1.116])
    bat = log.channels["Device Battery"]
    assert bat.units == "%"
    np.testing.assert_allclose(bat.values, [84.0, 84.0, 83.0, 83.0, 82.0])
    # Old neutral names are gone; E/F absent here: skipped, not zero-filled.
    for stale in ("GPS Aux A", "GPS Aux B", "GPS Aux C", "GPS Aux D",
                  "GPS Aux E", "GPS Aux F"):
        assert stale not in log.channels
    # Existing channels untouched by the renamed companions.
    np.testing.assert_allclose(log.channels["Ground Speed"].values, 72.0)


def test_rcz_gps_aux_a_decodes_legacy_x1000_counts(tmp_path):
    """The 2026-02-19 pack stores satellite counts x1000 (3000 ~ 16000)."""
    first = 1_700_000_000_000
    path = tmp_path / "gps-aux-legacy.rcz"
    _write_gps_aux_rcz(
        path, first, sats_raw=[3000, 12000, 14000, 15000, 16000],
    )

    log = DataLog()
    log.from_rcz_log(str(path))

    np.testing.assert_allclose(
        log.channels["GPS Satellites"].values, [3, 12, 14, 15, 16],
    )


def test_rcz_gps_aux_precision_sentinel_stays_missing(tmp_path):
    """INT32_MAX precision samples are unknown, not two-million-metre errors."""
    first = 1_700_000_000_000
    path = tmp_path / "gps-aux-prec.rcz"
    _write_gps_aux_rcz(
        path, first,
        sats_raw=[14, 14, 14, 14, 14],
        coord_prec_mm=[1100, 2147483647, 900, 2147483647, 1500],
        alt_prec_mm=[2147483647, 1900, 1900, 1000, 2147483647],
    )

    log = DataLog()
    log.from_rcz_log(str(path))

    aux_e = log.channels["GPS Aux E"].values
    assert aux_e[0] == 1.1 and aux_e[2] == 0.9 and aux_e[4] == 1.5
    assert np.isnan(aux_e[1]) and np.isnan(aux_e[3])
    aux_f = log.channels["GPS Aux F"].values
    assert np.isnan(aux_f[0]) and np.isnan(aux_f[4])
    assert aux_f[1] == 1.9


def test_rcz_gps_aux_matches_paired_csv_export(tmp_path):
    """Miniature of the 2026-10-04 paired proof: synthetic RCZ GPS value
    files and a synthetic RaceChrono CSV export of the same samples land
    on identical canonical channels (the full-size corpus pair is ~265 MB
    and stays out of the repo)."""
    first = 1_700_000_000_000
    path = tmp_path / "gps-aux-paired.rcz"
    _write_gps_aux_rcz(
        path, first,
        sats_raw=[14, 15, 15, 14, 13],
        fix_raw=[3, 3, 3, 3, 3],
        accuracy_mm=[1129, 1125, 1122, 1119, 1116],
        battery_mpc=[84000, 84000, 83000, 83000, 82000],
    )
    rcz_log = DataLog()
    rcz_log.from_rcz_log(str(path))

    csv_lines = [
        "timestamp,satellites,fix_type,accuracy,device_battery_level\n",
        "unix time,sats,,m,%\n",
        "0.0,14,3,1.129,84\n",
        "0.04,15,3,1.125,84\n",
        "0.08,15,3,1.122,83\n",
        "0.12,14,3,1.119,83\n",
        "0.16,13,3,1.116,82\n",
    ]
    csv_log = DataLog()
    csv_log.from_racechrono_log(csv_lines)

    for name in ("GPS Satellites", "GPS Fix", "GPS Accuracy", "Device Battery"):
        np.testing.assert_allclose(
            rcz_log.channels[name].values, csv_log.channels[name].values,
            err_msg=f"paired RCZ/CSV mismatch on {name}",
        )
        assert rcz_log.channels[name].units == csv_log.channels[name].units


def test_rcz_secondary_imu_uses_suffixed_channels(tmp_path):
    """A second IMU set lands in Secondary channels; primary values unchanged."""
    first = 1_700_000_000_000
    ts = (first + np.arange(8, dtype="<i8") * 40).astype("<i8")
    path = tmp_path / "dual-imu.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(len(ts), 20_000, dtype="<i4").tobytes())
        # Primary box IMU (301/302).
        z.writestr("channel_2_301_0_1_1", ts.tobytes())
        z.writestr("channel_2_301_0_10_0",
                   np.full(len(ts), 8500, dtype="<i4").tobytes())
        z.writestr("channel_2_301_0_9_0",
                   np.full(len(ts), -3200, dtype="<i4").tobytes())
        z.writestr("channel_2_301_0_11_0",
                   np.full(len(ts), 9700, dtype="<i4").tobytes())
        z.writestr("channel_3_302_0_1_1", ts.tobytes())
        z.writestr("channel_3_302_0_14_0",
                   np.full(len(ts), -12000, dtype="<i4").tobytes())
        # Secondary phone IMU (101/102) at the same epochs.
        z.writestr("channel_2_101_0_1_1", ts.tobytes())
        z.writestr("channel_2_101_0_10_0",
                   np.full(len(ts), 4250, dtype="<i4").tobytes())
        z.writestr("channel_2_101_0_9_0",
                   np.full(len(ts), -1600, dtype="<i4").tobytes())
        z.writestr("channel_2_101_0_11_0",
                   np.full(len(ts), 4850, dtype="<i4").tobytes())
        z.writestr("channel_3_102_0_1_1", ts.tobytes())
        z.writestr("channel_3_102_0_14_0",
                   np.full(len(ts), -6000, dtype="<i4").tobytes())

    log = DataLog()
    log.from_rcz_log(str(path), mask_interp_gaps=True)

    # Primary channels keep the box values, not the phone's.
    np.testing.assert_allclose(log.channels["CG Accel Lateral"].values, 0.85)
    np.testing.assert_allclose(
        log.channels["CG Accel Longitudinal"].values, -0.32,
    )
    np.testing.assert_allclose(log.channels["Chassis Yaw Rate"].values, 12.0)
    assert log.metadata["lateral_accel_source"] == "rcz_accelerometer_2_301"
    assert log.metadata["yaw_rate_source"] == "rcz_gyro_3_302"
    # The extra set is preserved under Secondary names.
    np.testing.assert_allclose(
        log.channels["CG Accel Lateral Secondary"].values, 0.425,
    )
    np.testing.assert_allclose(
        log.channels["CG Accel Longitudinal Secondary"].values, -0.16,
    )
    np.testing.assert_allclose(
        log.channels["CG Accel Vertical Secondary"].values, 0.485,
    )
    np.testing.assert_allclose(
        log.channels["Chassis Yaw Rate Secondary"].values, 6.0,
    )
    assert log.metadata["secondary_imu_source"] == "rcz_secondary_2_101/3_102"

    # Single-set packs gain no Secondary channels (byte-identical output).
    single = tmp_path / "single-imu.rcz"
    with zipfile.ZipFile(single, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(len(ts), 20_000, dtype="<i4").tobytes())
        z.writestr("channel_2_301_0_1_1", ts.tobytes())
        z.writestr("channel_2_301_0_10_0",
                   np.full(len(ts), 8500, dtype="<i4").tobytes())
        z.writestr("channel_3_302_0_1_1", ts.tobytes())
        z.writestr("channel_3_302_0_14_0",
                   np.full(len(ts), -12000, dtype="<i4").tobytes())

    solo = DataLog()
    solo.from_rcz_log(str(single), mask_interp_gaps=True)
    assert not [name for name in solo.channels if "Secondary" in name]
    assert "secondary_imu_source" not in solo.metadata


def test_rcz_uptime_stream_check_reports_structure(tmp_path, capsys):
    """Uptime files are validated (monotonic, 1:1), never parsed as time."""
    from track_telemetry_converter.parsers.rcz_parser import (
        _check_uptime_stream,
    )

    good_ticks = np.array([0, 25, 50, 75, 100], dtype="<i4")
    good = np.column_stack(
        (good_ticks, np.zeros(5, dtype="<i4")),
    ).astype("<i4").tobytes()
    summary = _check_uptime_stream(good, 5)
    assert summary == {
        "samples": 5, "monotonic": True, "clock_match": True,
        "median_ticks_per_sample": 25.0,
    }

    bad_ticks = np.array([0, 25, 10, 75], dtype="<i4")
    bad = np.column_stack(
        (bad_ticks, np.zeros(4, dtype="<i4")),
    ).astype("<i4").tobytes()
    assert _check_uptime_stream(bad, 4)["monotonic"] is False
    assert _check_uptime_stream(good, 99)["clock_match"] is False
    assert _check_uptime_stream(b"\x00\x01", 0)["samples"] == 0

    # Parse-level: a corrupt device uptime warns but keeps every channel.
    first = 1_700_000_000_000
    ts = (first + np.arange(5, dtype="<i8") * 40).astype("<i8")
    path = tmp_path / "uptime.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(5, 20_000, dtype="<i4").tobytes())
        z.writestr("channel_1_300_0_2_1", bad)

    log = DataLog()
    log.from_rcz_log(str(path))
    assert "corrupt" in capsys.readouterr().out
    assert log.metadata["rcz_uptime_clocks"]["channel_1_300_0_2_1"]["monotonic"] is False
    np.testing.assert_allclose(log.channels["Ground Speed"].values, 72.0)


def test_rcz_pid43_timestamp_only_emits_nan_placeholder(tmp_path):
    """PID 43 with timestamps but no values degrades to NaN, not zeros."""
    first = 1_700_000_000_000
    ts = (first + np.arange(4, dtype="<i8") * 100).astype("<i8")
    path = tmp_path / "pid43-degraded.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(4, 10_000, dtype="<i4").tobytes())
        z.writestr("channel_12_200_43_1_1", ts.tobytes())

    log = DataLog()
    log.from_rcz_log(str(path))

    combined = log.channels["G Force Combined"]
    assert combined.units == "G"
    assert np.isnan(combined.values).all()
    assert log.metadata["g_combined_source"] == "rcz_pid43_timestamps_only"

    # Real companion values on another sub-device still win over the gap.
    both = tmp_path / "pid43-both.rcz"
    with zipfile.ZipFile(both, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(4, 10_000, dtype="<i4").tobytes())
        z.writestr("channel_12_100_43_1_1", ts.tobytes())
        z.writestr("channel2_12_100_43_43_3",
                   np.array([4.903325, 9.80665, 14.709975, 19.6133]).tobytes())
        z.writestr("channel_12_200_43_1_1", ts.tobytes())

    real = DataLog()
    real.from_rcz_log(str(both))
    np.testing.assert_allclose(
        real.channels["G Force Combined"].values, [0.5, 1.0, 1.5, 2.0],
        atol=1e-9,
    )
    assert "g_combined_source" not in real.metadata


def test_rcz_dev4_phone_rates_and_fahrenheit_temps(tmp_path):
    """Device-4 PIDs 49/50 are roll/pitch rate; its temps arrive in Fahrenheit."""
    first = 1_700_000_000_000
    ts = (first + np.arange(4, dtype="<i8") * 100).astype("<i8")
    path = tmp_path / "dev4.rcz"
    pid_values = {
        "49": [-19.9, 0.0, 29.6, -0.5],
        "50": [-18.6, 0.0, 14.6, -0.1],
        "10026": [154.4, 180.0, 201.2, 192.0],
        "10066": [143.6, 200.0, 246.2, 207.0],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_100_0_1_1", ts.tobytes())
        z.writestr("channel_1_100_0_4_0",
                   np.full(4, 10_000, dtype="<i4").tobytes())
        z.writestr("channel_4_101_0_1_1", ts.tobytes())
        for pid, values in pid_values.items():
            z.writestr(f"channel2_4_101_0_{pid}_3",
                       np.array(values, dtype="<f8").tobytes())

    log = DataLog()
    log.from_rcz_log(str(path))

    roll = log.channels["Chassis Roll Rate"]
    assert roll.units == "deg/s"
    np.testing.assert_allclose(roll.values, [-19.9, 0.0, 29.6, -0.5], atol=1e-9)
    pitch = log.channels["Chassis Pitch Rate"]
    assert pitch.units == "deg/s"
    np.testing.assert_allclose(pitch.values, [-18.6, 0.0, 14.6, -0.1], atol=1e-9)
    # Fahrenheit in, Celsius out: 154.4 F == 68.0 C, not 154 C coolant.
    coolant = log.channels["Coolant Temp"]
    assert coolant.units == "C"
    np.testing.assert_allclose(
        coolant.values,
        [(v - 32.0) * 5.0 / 9.0 for v in pid_values["10026"]],
        atol=1e-9,
    )
    assert coolant.values.max() < 100.0
    oil = log.channels["Engine Oil Temp"]
    np.testing.assert_allclose(
        oil.values,
        [(v - 32.0) * 5.0 / 9.0 for v in pid_values["10066"]],
        atol=1e-9,
    )


def test_rcz_traps_carry_full_fields_and_debug_unknowns(tmp_path, caplog):
    """Trap bearing/width/order survive in trap dicts; unknown keys log-debug."""
    import logging

    first = 1_700_000_000_000
    ts = (first + np.arange(4, dtype="<i8") * 100).astype("<i8")
    path = tmp_path / "traps.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("trackId.json", json.dumps({"track": {"traps": [
            {
                "name": "Start/Finish",
                "centerLatitude": 237230835,
                "centerLongitude": -733987259,
                "bearing": 180000,
                "width": 100000,
                "type": 3,
                "orderValue": 0,
                "uniDirectional": True,
                "futureKey": "not-yet-known",
            },
        ]}}))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(4, 10_000, dtype="<i4").tobytes())

    log = DataLog()
    with caplog.at_level(
        logging.DEBUG, logger="track_telemetry_converter.parsers.rcz_parser",
    ):
        log.from_rcz_log(str(path))

    (trap,) = log.traps
    assert trap["name"] == "Start/Finish"
    assert trap["bearing_deg"] == 180.0
    assert trap["width_m"] == 100.0
    assert trap["order"] == 0
    assert trap["uni_directional"] is True
    assert "futureKey" in caplog.text


def test_rcz_secondary_imu_covers_100_101_phone_scheme(tmp_path):
    """The 2026-03-07 phone scheme (accel 2_100, gyro 3_101) is a valid set.

    Primary stays on the 301/302 box; the phone set lands in Secondary
    channels. Before the probe lists grew "100"/"101", this exact layout
    parsed with no Secondary channels at all.
    """
    first = 1_700_000_000_000
    ts = (first + np.arange(8, dtype="<i8") * 40).astype("<i8")
    path = tmp_path / "dual-imu-100-101.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(len(ts), 20_000, dtype="<i4").tobytes())
        z.writestr("channel_2_301_0_1_1", ts.tobytes())
        z.writestr("channel_2_301_0_10_0",
                   np.full(len(ts), 8500, dtype="<i4").tobytes())
        z.writestr("channel_3_302_0_1_1", ts.tobytes())
        z.writestr("channel_3_302_0_14_0",
                   np.full(len(ts), -12000, dtype="<i4").tobytes())
        z.writestr("channel_2_100_0_1_1", ts.tobytes())
        z.writestr("channel_2_100_0_10_0",
                   np.full(len(ts), 4250, dtype="<i4").tobytes())
        z.writestr("channel_2_100_0_9_0",
                   np.full(len(ts), -1600, dtype="<i4").tobytes())
        z.writestr("channel_3_101_0_1_1", ts.tobytes())
        z.writestr("channel_3_101_0_14_0",
                   np.full(len(ts), -6000, dtype="<i4").tobytes())
        z.writestr("channel_3_101_0_12_0",
                   np.full(len(ts), 2000, dtype="<i4").tobytes())

    log = DataLog()
    log.from_rcz_log(str(path), mask_interp_gaps=True)

    np.testing.assert_allclose(log.channels["CG Accel Lateral"].values, 0.85)
    np.testing.assert_allclose(log.channels["Chassis Yaw Rate"].values, 12.0)
    np.testing.assert_allclose(
        log.channels["CG Accel Lateral Secondary"].values, 0.425,
    )
    np.testing.assert_allclose(
        log.channels["CG Accel Longitudinal Secondary"].values, -0.16,
    )
    np.testing.assert_allclose(
        log.channels["Chassis Yaw Rate Secondary"].values, 6.0,
    )
    np.testing.assert_allclose(
        log.channels["Chassis Roll Rate Secondary"].values, -2.0,
    )
    assert log.metadata["secondary_imu_source"] == "rcz_secondary_2_100/3_101"


def test_rcz_uptime_check_ignores_obd_pid_timestamps(tmp_path, capsys):
    """channel_12_*_2_1 files are per-PID OBD clocks, not uptime streams."""
    first = 1_700_000_000_000
    ts = (first + np.arange(5, dtype="<i8") * 40).astype("<i8")
    path = tmp_path / "obd-ts.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(5, 20_000, dtype="<i4").tobytes())
        # Well-formed device uptime (1:1, monotonic) stays silent.
        ticks = np.column_stack(
            (np.array([0, 25, 50, 75, 100], dtype="<i4"),
             np.zeros(5, dtype="<i4")),
        ).astype("<i4").tobytes()
        z.writestr("channel_1_300_0_2_1", ticks)
        # OBD per-PID timestamp file: must not warn, must not enter metadata.
        z.writestr("channel_12_200_1001_2_1", ts.tobytes())
        z.writestr("channel2_12_200_1001_1001_3",
                   np.full(5, 10.0, dtype="<f8").tobytes())

    log = DataLog()
    log.from_rcz_log(str(path))

    assert "corrupt" not in capsys.readouterr().out
    assert list(log.metadata["rcz_uptime_clocks"]) == ["channel_1_300_0_2_1"]


def test_rcz_session_notes_land_in_metadata(tmp_path):
    """session.json notes/description survive as short_comment metadata."""
    first = 1_700_000_000_000
    ts = (first + np.arange(4, dtype="<i8") * 100).astype("<i8")
    path = tmp_path / "notes.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
            "notes": "Evening session, new rear pads",
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(4, 10_000, dtype="<i4").tobytes())

    log = DataLog()
    log.from_rcz_log(str(path))

    assert log.metadata["short_comment"] == "Evening session, new rear pads"


def _write_fragment_rcz(path, first, gps_types=("300",), fragment=None,
                        laps=None):
    """Write an RCZ with the given GPS families plus an optional fragment."""
    n = 5
    ts = (first + np.arange(n, dtype="<i8") * 40).astype("<i8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first,
            "laps": laps if laps is not None else [],
        }))
        for gps_type in gps_types:
            z.writestr(f"channel_1_{gps_type}_0_1_1", ts.tobytes())
            z.writestr(f"channel_1_{gps_type}_0_4_0",
                       np.full(n, 20_000, dtype="<i4").tobytes())
        if fragment is not None:
            z.writestr("sessionfragment.json", json.dumps(fragment))


def test_rcz_fragment_declaration_prefers_declared_gps(tmp_path):
    """Fragment-declared GPS wins; absent declarations fall back to legacy."""
    first = 1_700_000_000_000

    # Legacy order without a fragment: 300 beats 100.
    plain = tmp_path / "dual-gps-plain.rcz"
    _write_fragment_rcz(plain, first, gps_types=("300", "100"))
    legacy = DataLog()
    legacy.from_rcz_log(str(plain))
    assert legacy.metadata["gps_source"] == "rcz_gps_1_300"

    # The same pack with a fragment declaring 100 resolves to 100.
    declared = tmp_path / "dual-gps-declared.rcz"
    _write_fragment_rcz(declared, first, gps_types=("300", "100"), fragment={
        "primaryGpsDeviceIndex": 100,
        "imuAccDevice": 101,
        "imuGyroDevice": 102,
    })
    log = DataLog()
    log.from_rcz_log(str(declared))
    assert log.metadata["gps_source"] == "rcz_gps_1_100"
    assert log.metadata["rcz_device_roles"] == {
        "gps": "100", "accel": "101", "gyro": "102",
    }
    np.testing.assert_allclose(
        log.channels["Ground Speed"].values,
        legacy.channels["Ground Speed"].values,
    )

    # Declared-but-absent family falls back to the present one.
    missing = tmp_path / "dual-gps-missing.rcz"
    _write_fragment_rcz(missing, first, gps_types=("300",), fragment={
        "primaryGpsDeviceIndex": 200,
    })
    fallback = DataLog()
    fallback.from_rcz_log(str(missing))
    assert fallback.metadata["gps_source"] == "rcz_gps_1_300"
    assert "Ground Speed" in fallback.channels


def test_rcz_single_device_fragment_is_channel_identical(tmp_path):
    """A fragment declaring the only present family changes no samples."""
    first = 1_700_000_000_000
    plain = tmp_path / "single-plain.rcz"
    _write_fragment_rcz(plain, first, gps_types=("300",))
    declared = tmp_path / "single-declared.rcz"
    _write_fragment_rcz(declared, first, gps_types=("300",), fragment={
        "primaryGpsDeviceIndex": 300,
    })

    base, with_fragment = DataLog(), DataLog()
    base.from_rcz_log(str(plain))
    with_fragment.from_rcz_log(str(declared))

    assert set(with_fragment.channels) == set(base.channels)
    for name in base.channels:
        np.testing.assert_array_equal(
            with_fragment.channels[name].timestamps,
            base.channels[name].timestamps,
        )
        np.testing.assert_array_equal(
            with_fragment.channels[name].values,
            base.channels[name].values,
        )


def test_rcz_fragment_bias_and_videos_land_in_metadata(tmp_path, caplog):
    """gyroBias is recorded for consumers (not applied); videos are counted."""
    import logging

    first = 1_700_000_000_000
    path = tmp_path / "fragment-meta.rcz"
    _write_fragment_rcz(path, first, gps_types=("100",), fragment={
        "primaryGpsDeviceIndex": 100,
        "imuAccDevice": 101,
        "imuGyroDevice": 102,
        "devices2": {"102": {"gyroBias": [-2.141, 0.741, 1.052]}},
        "videos": [{"name": "front.mp4"}, {"name": "rear.mp4"}],
    })

    log = DataLog()
    with caplog.at_level(
        logging.DEBUG, logger="track_telemetry_converter.parsers.rcz_parser",
    ):
        log.from_rcz_log(str(path))

    assert log.metadata["rcz_gyro_bias_declared"] == [-2.141, 0.741, 1.052]
    assert log.metadata["rcz_videos_count"] == 2
    assert "video" in caplog.text
    # Faithful reader: no channel carries a bias-corrected duplicate.
    assert "Ground Speed" in log.channels


def test_rcz_magnetometer_archived_and_zero_uptime_only_validated(tmp_path, capsys):
    """Device-8 28/29/30 archive raw; an all-zero _2_1 stores no channel."""
    first = 1_700_000_000_000
    n = 5
    ts = (first + np.arange(n, dtype="<i8") * 40).astype("<i8")
    mag_ts = (first + np.arange(n, dtype="<i8") * 80).astype("<i8")
    mag_x = np.array([1000, -2000, 3000, -4000, 5000], dtype="<i4")
    mag_y = np.array([-15000, 20000, -25000, 29999, -30000], dtype="<i4")
    mag_z = np.array([0, 100, -100, 200, -200], dtype="<i4")
    zeros = np.zeros((n, 2), dtype="<i4").tobytes()
    path = tmp_path / "mag.rcz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_300_0_1_1", ts.tobytes())
        z.writestr("channel_1_300_0_4_0",
                   np.full(n, 20_000, dtype="<i4").tobytes())
        z.writestr("channel_8_400_0_1_1", mag_ts.tobytes())
        z.writestr("channel_8_400_0_28_0", mag_x.tobytes())
        z.writestr("channel_8_400_0_29_0", mag_y.tobytes())
        z.writestr("channel_8_400_0_30_0", mag_z.tobytes())
        z.writestr("channel_8_400_0_2_1", zeros)

    log = DataLog()
    log.from_rcz_log(str(path))

    for name in ("Magnetic Field X", "Magnetic Field Y", "Magnetic Field Z"):
        assert log.channels[name].units == ""
        np.testing.assert_array_equal(
            log.channels[name].timestamps,
            log.channels["Ground Speed"].timestamps,
        )
    # Raw counts survive the resample at the shared epochs (GPS 0..160 ms
    # sits inside the mag 0..320 ms clock; shared epochs are exact:
    # GPS idx 0/2/4 = 0/80/160 ms = mag samples 0/1/2).
    np.testing.assert_allclose(
        log.channels["Magnetic Field X"].values[[0, 2, 4]],
        [1000, -2000, 3000],
    )
    np.testing.assert_allclose(
        log.channels["Magnetic Field Y"].values[[0, 2, 4]],
        [-15000, 20000, -25000],
    )
    # The degraded all-zero uptime companion is validated, never stored.
    assert "Magnetic Field" not in log.metadata.get("rcz_uptime_clocks", {})
    assert (
        log.metadata["rcz_uptime_clocks"]["channel_8_400_0_2_1"]["samples"] == n
    )
    assert not [
        name for name in log.channels if "uptime" in name.lower()
    ]
    assert "Ground Speed" in log.channels


def test_rcz_invalid_laps_are_excluded(tmp_path):
    """laps[].isInvalid=True never seeds reconstruction (else zero impact)."""
    first = 1_700_000_000_000
    laps = [
        {
            "number": 10,
            "sessionResume": 0,
            "startTimestamp": first + 20_000,
            "finishTimestamp": first + 50_000,
            "isInvalid": False,
        },
        {
            "number": 11,
            "sessionResume": 0,
            "startTimestamp": first + 50_000,
            "finishTimestamp": first + 80_000,
            "isInvalid": True,
        },
    ]
    flagged = tmp_path / "invalid-lap.rcz"
    _write_fragment_rcz(flagged, first, gps_types=("300",), laps=laps)
    clean = tmp_path / "clean-lap.rcz"
    _write_fragment_rcz(clean, first, gps_types=("300",), laps=[laps[0]])

    flagged_log, clean_log = DataLog(), DataLog()
    flagged_log.from_rcz_log(str(flagged))
    clean_log.from_rcz_log(str(clean))

    assert [lap["type"] for lap in flagged_log.laps_info["laps"]] == [
        lap["type"] for lap in clean_log.laps_info["laps"]
    ]
    assert flagged_log.laps_info["total_laps"] == clean_log.laps_info["total_laps"]
    # The invalid lap contributes no timed segment.
    assert not [
        lap for lap in flagged_log.laps_info["laps"]
        if lap.get("orig_num") == 11
    ]


def _write_suffix_probe_rcz(path, first, extra=None):
    """Write an RCZ covering every known channel_1_/2_/3_ probe suffix."""
    n = 5
    ts = (first + np.arange(n, dtype="<i8") * 40).astype("<i8")
    lat_lon = np.column_stack((
        np.full(n, 222_000_000, dtype="<i4"),
        np.full(n, -732_000_000, dtype="<i4"),
    ))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("session.json", json.dumps({
            "firstTimestamp": first, "timeCreated": first, "laps": [],
        }))
        z.writestr("channel_1_200_0_1_1", ts.tobytes())
        z.writestr("channel_1_200_0_4_0",
                   np.full(n, 20_000, dtype="<i4").tobytes())
        z.writestr("channel_1_200_0_3_1", lat_lon.tobytes())
        z.writestr("channel_1_200_0_5_0",
                   np.full(n, 100_000, dtype="<i4").tobytes())
        z.writestr("channel_1_200_0_6_0",
                   np.full(n, 90_000, dtype="<i4").tobytes())
        for suffix, values in (
            ("30002_0", [14] * n), ("30003_0", [3] * n),
            ("30007_0", [1129] * n), ("46_0", [84000] * n),
            ("30004_0", [1100] * n), ("30005_0", [1900] * n),
        ):
            z.writestr(f"channel_1_200_0_{suffix}",
                       np.array(values, dtype="<i4").tobytes())
        ticks = np.column_stack(
            (np.arange(n, dtype="<i4") * 25, np.zeros(n, dtype="<i4")),
        ).astype("<i4").tobytes()
        z.writestr("channel_1_200_0_2_1", ticks)
        z.writestr("channel_2_201_0_1_1", ts.tobytes())
        for suffix, raw in (("9_0", -3200), ("10_0", 8500), ("11_0", 9700)):
            z.writestr(f"channel_2_201_0_{suffix}",
                       np.full(n, raw, dtype="<i4").tobytes())
        z.writestr("channel_3_202_0_1_1", ts.tobytes())
        for suffix, raw in (("12_0", 4000), ("13_0", -6000), ("14_0", -12000)):
            z.writestr(f"channel_3_202_0_{suffix}",
                       np.full(n, raw, dtype="<i4").tobytes())
        # Excluded families: device 8/12/4 keep their own handling.
        mag_ts = (first + np.arange(n, dtype="<i8") * 80).astype("<i8")
        z.writestr("channel_8_400_0_1_1", mag_ts.tobytes())
        z.writestr("channel_8_400_0_28_0",
                   np.full(n, 1000, dtype="<i4").tobytes())
        z.writestr("channel_12_200_51_1_1", ts.tobytes())
        z.writestr("channel2_12_200_51_51_3",
                   np.full(n, 12.0, dtype="<f8").tobytes())
        if extra:
            for name, payload in extra.items():
                z.writestr(name, payload)


def test_rcz_known_suffixes_stay_silent(tmp_path, capsys):
    """A pack using every known 1/2/3 suffix warns about nothing."""
    first = 1_700_000_000_000
    path = tmp_path / "known-suffixes.rcz"
    _write_suffix_probe_rcz(path, first)

    log = DataLog()
    log.from_rcz_log(str(path))

    assert "unmapped channel suffixes" not in capsys.readouterr().out
    assert "Ground Speed" in log.channels
    assert "Chassis Yaw Rate" in log.channels


def test_rcz_unknown_suffix_warns_without_changing_parse(tmp_path, capsys):
    """A future suffix (e.g. 30008) prints WARNING; the rest is untouched."""
    first = 1_700_000_000_000
    n = 5
    base_path = tmp_path / "base.rcz"
    _write_suffix_probe_rcz(base_path, first)
    base = DataLog()
    base.from_rcz_log(str(base_path))
    capsys.readouterr()

    unknown_name = "channel_1_200_0_30008_0"
    extended_path = tmp_path / "extended.rcz"
    _write_suffix_probe_rcz(
        extended_path, first,
        extra={unknown_name: np.full(n, 1234, dtype="<i4").tobytes()},
    )
    extended = DataLog()
    extended.from_rcz_log(str(extended_path))

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert unknown_name in out
    assert "1_200" in out
    assert set(extended.channels) == set(base.channels)
    for name in base.channels:
        np.testing.assert_array_equal(
            extended.channels[name].timestamps, base.channels[name].timestamps,
        )
        np.testing.assert_array_equal(
            extended.channels[name].values, base.channels[name].values,
        )
