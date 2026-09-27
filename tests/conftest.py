"""Shared fixtures and test helpers for TrackTelemetryConverter test suite."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from unittest import SkipTest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read_lines(filename):
    path = os.path.join(EXAMPLES, filename)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.readlines()


def _dedup_channels(log):
    """MoTeC doesn't allow duplicate channel names, validate this invariant."""
    assert len(log.channels) == len(set(log.channels)), f"Duplicate channel names: {list(log.channels)}"


def _import_or_skip(module_name):
    """Import an optional test dependency without turning a skip into a pass."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise SkipTest(f"{module_name} not installed") from exc


def _sha256(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def _run_cli_in_process(args):
    """Run CLI main() in-process, capturing stdout/stderr and exit code."""
    from io import StringIO

    from track_telemetry_converter.cli import main

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = StringIO(), StringIO()
    code = 0
    try:
        res = main(args)
        if isinstance(res, int):
            code = res
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
    finally:
        stdout_val = sys.stdout.getvalue()
        stderr_val = sys.stderr.getvalue()
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return subprocess.CompletedProcess(args=args, returncode=code, stdout=stdout_val, stderr=stderr_val)


def _assert_cli_roundtrip(source, log_type, expected_channels=(), in_process=True):
    from track_telemetry_converter._vendor.ldparser import ldData

    with tempfile.TemporaryDirectory() as tmp_dir:
        output = os.path.join(tmp_dir, f"{log_type.lower()}_roundtrip.ld")
        cli_args = [
            source,
            log_type,
            "--output",
            output,
        ]
        if in_process:
            result = _run_cli_in_process(cli_args)
        else:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "track_telemetry_converter",
                ] + cli_args,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
        assert result.returncode == 0, result.stdout + result.stderr

        ldx_path = os.path.splitext(output)[0] + ".ldx"
        assert os.path.isfile(output)
        assert os.path.isfile(ldx_path)
        parsed = ldData.fromfile(output)
        assert parsed.channs
        assert all(channel.data_len > 0 for channel in parsed.channs)
        channel_names = {channel.name for channel in parsed.channs}
        assert set(expected_channels) <= channel_names
        assert ET.parse(ldx_path).getroot().tag == "LDXFile"


def _write_minimal_rcz(path):
    import zipfile

    first_timestamp = 1_700_000_000_000
    timestamps = np.arange(first_timestamp, first_timestamp + 500, 100, dtype="<i8")
    speed = np.array([10_000, 12_000, 14_000, 16_000, 18_000], dtype="<i4")
    obd_times = timestamps.copy()
    pitch = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="<f8")
    lat_lon = np.array(
        [
            [222_000_000, -732_000_000],
            [222_000_006, -731_999_994],
            [222_000_012, -731_999_988],
            [222_000_018, -731_999_982],
            [222_000_024, -731_999_976],
        ],
        dtype="<i4",
    )
    session = {
        "firstTimestamp": first_timestamp,
        "timeCreated": first_timestamp,
        "trackName": "Synthetic Test Track",
        "title": "CLI RCZ Test",
        "laps": [],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps(session))
        archive.writestr("channel_1_300_0_1_1", timestamps.tobytes())
        archive.writestr("channel_1_300_0_4_0", speed.tobytes())
        archive.writestr("channel_1_300_0_3_1", lat_lon.tobytes())
        archive.writestr("channel_12_100_8_8_1_1", obd_times.tobytes())
        archive.writestr("channel2_12_100_8_8_3", pitch.tobytes())


def _write_lapped_rcz(path, start_speed_mps=0.0):
    """Write a small RCZ with out, timed, timed, and in-lap segments."""
    import zipfile

    first_timestamp = 1_700_000_000_000
    timestamps = np.arange(
        first_timestamp,
        first_timestamp + 101_000,
        1_000,
        dtype="<i8",
    )
    speed = np.full(len(timestamps), round(start_speed_mps * 1000), dtype="<i4")
    lat_lon = np.column_stack(
        (
            np.arange(len(timestamps), dtype="<i4") + 222_000_000,
            np.arange(len(timestamps), dtype="<i4") - 732_000_000,
        )
    )
    session = {
        "firstTimestamp": first_timestamp,
        "timeCreated": first_timestamp,
        "trackName": "Synthetic Test Track",
        "title": "Lapped RCZ Test",
        "laps": [
            {
                "number": 10,
                "sessionResume": 0,
                "startTimestamp": first_timestamp + 20_000,
                "finishTimestamp": first_timestamp + 50_000,
            },
            {
                "number": 11,
                "sessionResume": 0,
                "startTimestamp": first_timestamp + 50_000,
                "finishTimestamp": first_timestamp + 80_000,
            },
            {
                "number": 12,
                "sessionResume": 0,
                "startTimestamp": first_timestamp + 80_000,
                "finishTimestamp": None,
            },
        ],
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("session.json", json.dumps(session))
        archive.writestr("channel_1_300_0_1_1", timestamps.tobytes())
        archive.writestr("channel_1_300_0_4_0", speed.tobytes())
        archive.writestr("channel_1_300_0_3_1", lat_lon.astype("<i4").tobytes())


def _write_backup_rcz(path):
    """Write a tiny RaceChrono backup containing two nested sessions."""
    import zipfile

    sessions = [
        {
            "id": "session_20260101_1000",
            "timeCreated": 1_700_000_000_000,
            "trackName": "Alpha Track",
            "trackLocalUuid": "track-alpha",
            "trackId": -1,
            "gps_type": "100",
            "stints": (0,),
        },
        {
            "id": "session_20260102_1100",
            "timeCreated": 1_700_100_000_000,
            "trackName": "Beta Track",
            "trackLocalUuid": None,
            "trackId": -2,
            "gps_type": "300",
            "stints": (0, 1),
        },
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "json/track_storage.json",
            json.dumps(
                {
                    "tracks": [
                        {
                            "id": -1,
                            "localUuid": "track-alpha",
                            "name": "Alpha Track",
                            "traps": [
                                {
                                    "name": "Start",
                                    "centerLatitude": 222_000_000,
                                    "centerLongitude": -732_000_000,
                                    "type": 1,
                                }
                            ],
                        },
                        {
                            "id": -2,
                            "localUuid": "track-beta",
                            "name": "Beta Track",
                            "traps": [
                                {
                                    "name": "Finish",
                                    "centerLatitude": 222_600_000,
                                    "centerLongitude": -731_400_000,
                                    "type": 2,
                                }
                            ],
                        },
                    ]
                }
            ),
        )
        for session_index, descriptor in enumerate(sessions):
            first_timestamp = descriptor["timeCreated"] + 10_000
            laps = [
                {
                    "number": stint + 1,
                    "sessionResume": stint,
                    "startTimestamp": first_timestamp + 20_000,
                    "finishTimestamp": first_timestamp + 50_000,
                }
                for stint in descriptor["stints"]
            ]
            session = {
                "firstTimestamp": first_timestamp,
                "latestTimestamp": first_timestamp + 60_000,
                "timeCreated": descriptor["timeCreated"],
                "lengthTime": 60_000,
                "lapCount": len(laps),
                "trackName": descriptor["trackName"],
                "trackLocalUuid": descriptor["trackLocalUuid"],
                "trackId": descriptor["trackId"],
                "laps": laps,
            }
            root = "sessions/%s/" % descriptor["id"]
            archive.writestr(root + "session.json", json.dumps(session))
            for stint in descriptor["stints"]:
                channel_root = root + ("resume_%s/" % stint if stint else "")
                timestamps = np.arange(
                    first_timestamp,
                    first_timestamp + 61_000,
                    1_000,
                    dtype="<i8",
                )
                speed = np.full(
                    len(timestamps), 10_000 + session_index * 1_000, dtype="<i4"
                )
                lat_lon = np.column_stack(
                    (
                        np.arange(len(timestamps), dtype="<i4") + 222_000_000,
                        np.arange(len(timestamps), dtype="<i4") - 732_000_000,
                    )
                )
                gps_type = descriptor["gps_type"]
                archive.writestr(
                    channel_root + "channel_1_%s_0_1_1" % gps_type,
                    timestamps.tobytes(),
                )
                archive.writestr(
                    channel_root + "channel_1_%s_0_4_0" % gps_type,
                    speed.tobytes(),
                )
                archive.writestr(
                    channel_root + "channel_1_%s_0_3_1" % gps_type,
                    lat_lon.astype("<i4").tobytes(),
                )
