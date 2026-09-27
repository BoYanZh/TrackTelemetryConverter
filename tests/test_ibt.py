"""Tests for iRacing IBT format."""

import os
import struct
import tempfile

from conftest import EXAMPLES, _dedup_channels

from track_telemetry_converter.log import DataLog


def test_ibt_smoke():
    log = DataLog()
    log.from_ibt_log(os.path.join(EXAMPLES, "ibt_sample.ibt"))

    assert "Ground Speed" in log.channels
    assert "Lap Number" in log.channels
    assert len(log.channels["Ground Speed"].messages) == 10

    spd_msgs = log.channels["Ground Speed"].messages
    assert abs(spd_msgs[-1].value - 45.0 * 3.6) < 0.1

    lap_msgs = log.channels["Lap Number"].messages
    assert lap_msgs[-1].value == 1.0

    assert log.metadata["venue_name"] == "Test Track"
    assert log.metadata["driver"] == "Test Driver"
    assert log.metadata["vehicle_id"] == "Test Car"
    assert log.datetime is not None

    beacons = log.laps_info.get("beacons", [])
    assert len(beacons) == 1
    assert "Lap 1" in beacons[0][1]

    _dedup_channels(log)


def test_ibt_invalid():
    with tempfile.NamedTemporaryFile(suffix=".ibt", delete=False) as tf:
        tf.write(b"\x00" * 10)
        tf.flush()
        tmp_path = tf.name

    try:
        log = DataLog()
        log.from_ibt_log(tmp_path)
        assert len(log.channels) == 0
    finally:
        os.unlink(tmp_path)

    header = struct.pack("12i", 2, 0, 10, 0, 0, 0, 0, 0, 0, 8, 0, 0)
    buf_info = struct.pack("4i", 0, 999999, 0, 0)
    buf = header + buf_info + b"\x00" * 200

    with tempfile.NamedTemporaryFile(suffix=".ibt", delete=False) as tf:
        tf.write(buf)
        tf.flush()
        tmp_path2 = tf.name

    try:
        log = DataLog()
        log.from_ibt_log(tmp_path2)
        assert len(log.channels) == 0
    finally:
        os.unlink(tmp_path2)
