"""Synthetic MoTeC LD parsing: mapping, units, laps and ACTI separation."""

import numpy as np
import pytest

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers import auto_detect_log_type


class _StubCol:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float64)
        self.dtype = self._values.dtype

    def to_numpy(self, dtype=None):
        return np.asarray(self._values, dtype=dtype) if dtype else self._values

    def __len__(self):
        return len(self._values)


class _StubDF:
    def __init__(self, cols):
        self._cols = dict(cols)
        self.columns = list(self._cols)

    def __getitem__(self, col):
        return _StubCol(self._cols[col])


def _write_ld(path, cols):
    ldData.frompd(_StubDF(cols)).write(str(path))  # type: ignore[arg-type]


def test_motec_ld_mapping_units_and_laps(tmp_path):
    n = 600
    path = tmp_path / "synthetic.ld"
    _write_ld(path, {
        "Ground Speed": np.full(n, 100.0),
        "Speed (m/s)": np.full(n, 30.0),
        "GPS Latitude": np.linspace(39.5400, 39.5500, n),
        "GPS Longitude": np.linspace(-122.3300, -122.3200, n),
        "Bearing": np.full(n, 90.0),
        "CG Accel Lateral": np.full(n, 0.5),
        "Chassis Yaw Rate": np.full(n, 5.0),
        "Lap Number": np.concatenate([np.full(200, 7.0), np.full(200, 8.0), np.full(200, 9.0)]),
        "Custom Chan": np.arange(n, dtype=float),
    })
    assert auto_detect_log_type(str(path)) == "MOTEC_LD"

    log = DataLog()
    log.from_motec_ld_log(str(path))
    assert log.channels["Ground Speed"].values[0] == pytest.approx(100.0)
    # Second speed source must not overwrite the first canonical channel.
    assert log.channels["Ground Speed"].units == "km/h"
    assert "Speed (m/s)" in log.channels
    assert log.channels["Speed (m/s)"].values[0] == pytest.approx(30.0)
    assert log.channels["GPS Heading"].values[0] == pytest.approx(90.0)
    assert "gps_heading_synthetic" not in log.metadata
    assert log.channels["CG Accel Lateral"].values[0] == pytest.approx(0.5)
    assert log.metadata["yaw_rate_source"].startswith("motec_ld_recorded_")
    assert log.metadata["lap_timing_semantics"].startswith("Recorded MoTeC LD lap-number")
    assert [lap["lap_num"] for lap in log.laps_info["laps"]] == [7, 8, 9]
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap", "Timed", "In Lap"]
    assert log.metadata["native_sample_rate_hz"]["Ground Speed"] == pytest.approx(10.0)


def test_motec_ld_unit_conversion_and_heading_synthesis(tmp_path):
    n = 300
    path = tmp_path / "units.ld"
    _write_ld(path, {
        "Speed (m/s)": np.full(n, 25.0),
        "Latitude": np.linspace(37.0, 37.01, n),
        "Longitude": np.linspace(-122.0, -121.99, n),
        "RPM (rpm)": np.full(n, 5000.0),
        "Lap #": np.concatenate([np.full(150, 1.0), np.full(150, 2.0)]),
    })
    log = DataLog()
    log.from_motec_ld_log(str(path))
    assert log.channels["Ground Speed"].values[0] == pytest.approx(90.0)
    assert log.channels["GPS Latitude"].values[0] == pytest.approx(37.0)
    assert log.metadata["gps_heading_synthetic"] is True
    assert np.all(np.isfinite(log.channels["GPS Heading"].values))
    assert log.metadata["yaw_rate_source"] == "gps_heading_derivative"
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Out Lap", "Partial"]


def test_motec_ld_rejects_acti_sim_logs(tmp_path):
    n = 200
    path = tmp_path / "sim.ld"
    _write_ld(path, {
        "Car Coord X": np.linspace(0, 100, n),
        "Car Coord Y": np.linspace(0, 50, n),
        "Ground Speed": np.full(n, 80.0),
    })
    assert auto_detect_log_type(str(path)) == "ACTI"
    log = DataLog()
    with pytest.raises(ValueError, match="ACTI"):
        log.from_motec_ld_log(str(path))


def test_motec_ld_laps_from_beacons():
    """Beacon crossings split out/timed/in laps without a lap channel."""
    from track_telemetry_converter.parsers import motec_ld_parser as m

    info = m._build_laps_from_beacons(
        [(10.0, "Start/Finish"), (30.0, "Start/Finish")], 46.0, 15.0,
    )
    assert [lap["type"] for lap in info["laps"]] == ["Out Lap", "Timed", "In Lap"]
    assert [b for b, _ in info["beacons"]] == pytest.approx([10.0, 30.0])


def test_motec_ld_drifting_lap_channel_falls_back_to_single_out_lap(tmp_path):
    """A resample-interpolated lap channel must not survive as Lap Number."""
    n = 300
    path = tmp_path / "drift.ld"
    _write_ld(path, {
        "Ground Speed": np.full(n, 100.0),
        "Lap Number": np.linspace(1.0, 2.0, n),
    })
    log = DataLog()
    log.from_motec_ld_log(str(path))
    assert log.laps_info["total_laps"] == 1
    assert log.laps_info["laps"][0]["type"] == "Out Lap"
    np.testing.assert_allclose(log.channels["Lap Number"].values, np.ones(n))
