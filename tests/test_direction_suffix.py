"""Tests for the opt-in --direction-suffix output naming (cw/ccw from GPS heading)."""

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from conftest import _run_cli_in_process

from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.parsers.acti_parser import heading_direction


def _loop(n=400, radius=300.0, clockwise=False):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if clockwise:
        t = -t
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _write_synthetic_ld(path, clockwise=False, laps=2, freq=20.0):
    """Minimal ACTI-shaped .ld: circular trajectory + speed + Lap Time sawtooth."""
    n_lap = 400
    n = n_lap * laps
    t = np.arange(n) / freq
    xy = np.vstack([_loop(n_lap, clockwise=clockwise) for _ in range(laps)])
    log = DataLog()
    log.add_channel("Car Coord X", "m", float, 3)
    log.channels["Car Coord X"].set_samples(t, xy[:, 0])
    log.add_channel("Car Coord Y", "m", float, 3)
    log.channels["Car Coord Y"].set_samples(t, xy[:, 1])
    log.add_channel("Ground Speed", "km/h", float, 2)
    log.channels["Ground Speed"].set_samples(t, np.full(n, 100.0))
    log.add_channel("Lap Time", "s", float, 3)
    log.channels["Lap Time"].set_samples(
        t, np.tile(np.linspace(0, n_lap / freq, n_lap), laps))
    ml = MotecLog()
    ml.driver = "Test Driver"
    ml.vehicle_id = "test_car"
    ml.venue_name = "Test Venue"
    ml.initialize()
    ml.add_all_channels(log)
    ml.write(path)
    return path


def _write_anchor(path, bidirectional=True):
    profile = {
        "name": "Test Loop",
        "ref_lat": 39.0,
        "ref_lon": -122.0,
        "theta_deg": 0.0,
        "dx_m": 0.0,
        "dy_m": 0.0,
    }
    if bidirectional:
        profile["bidirectional"] = True
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"test_loop": profile}, handle)
    return path


def test_heading_direction_cw_ramp():
    direction, total = heading_direction([0, 90, 180, 270, 180 + 180])
    assert direction == "cw"
    assert total == pytest.approx(360.0)


def test_heading_direction_ccw_ramp():
    direction, total = heading_direction([0, 270, 180, 90, 0])
    assert direction == "ccw"
    assert total == pytest.approx(-360.0)


def test_heading_direction_wrap_safe():
    # A full clockwise circle crossing the 0/360 seam must stay positive.
    seq = [350.0, 0.0, 10.0, 90.0, 180.0, 270.0, 350.0]
    direction, total = heading_direction(seq)
    assert direction == "cw"
    assert total == pytest.approx(360.0)
    # Same circle driven the other way.
    direction, total = heading_direction(list(reversed(seq)))
    assert direction == "ccw"
    assert total == pytest.approx(-360.0)


def test_heading_direction_indeterminate():
    assert heading_direction([0.0]) == (None, 0.0)
    assert heading_direction([]) == (None, 0.0)
    assert heading_direction([np.nan, np.nan])[0] is None
    direction, _ = heading_direction([0, 10, 20, 15])
    assert direction is None


def _convert_acti(tmp_path, name, clockwise, extra_args=()):
    src = str(tmp_path / name)
    _write_synthetic_ld(src, clockwise=clockwise)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    args = [src, "ACTI", "--track", "test_loop", "--anchor", anchor,
            *extra_args]
    return _run_cli_in_process(args), tmp_path


def test_cli_direction_suffix_ccw(tmp_path):
    result, tmp_path = _convert_acti(tmp_path, "ccw_run.ld", clockwise=False,
                                     extra_args=["--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(str(tmp_path / "ccw_run_ccw.ld"))
    assert os.path.isfile(str(tmp_path / "ccw_run_ccw.ldx"))
    assert "ccw" in result.stdout


def test_cli_direction_suffix_cw(tmp_path):
    result, tmp_path = _convert_acti(tmp_path, "cw_run.ld", clockwise=True,
                                     extra_args=["--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(str(tmp_path / "cw_run_cw.ld"))
    assert os.path.isfile(str(tmp_path / "cw_run_cw.ldx"))


def test_cli_direction_suffix_default_off(tmp_path):
    src = str(tmp_path / "plain_run.ld")
    _write_synthetic_ld(src)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    wanted = str(tmp_path / "plain_out.ld")
    result = _run_cli_in_process(
        [src, "ACTI", "--track", "test_loop", "--anchor", anchor,
         "--output", wanted])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(wanted)
    assert not os.path.exists(str(tmp_path / "plain_out_ccw.ld"))
    assert not os.path.exists(str(tmp_path / "plain_out_cw.ld"))
    assert "_ccw" not in result.stdout and "_cw" not in result.stdout


def test_cli_direction_suffix_explicit_output_wins(tmp_path):
    src = str(tmp_path / "run.ld")
    _write_synthetic_ld(src)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    wanted = str(tmp_path / "my_name.ld")
    result = _run_cli_in_process(
        [src, "ACTI", "--track", "test_loop", "--anchor", anchor,
         "--output", wanted, "--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(wanted)
    assert not os.path.exists(str(tmp_path / "my_name_ccw.ld"))
    assert not os.path.exists(str(tmp_path / "my_name_cw.ld"))
    assert "WARNING" in result.stdout and "--output" in result.stdout


def test_cli_direction_suffix_no_gps_warns_and_keeps_name(tmp_path):
    src = str(tmp_path / "nogps.csv")
    with open(src, "w", encoding="utf-8") as handle:
        handle.write("Time,Throttle Pos\n")
        for i in range(50):
            handle.write(f"{i * 0.1:.1f},{float(i):.1f}\n")
    result = _run_cli_in_process([src, "CSV", "--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(str(tmp_path / "nogps.ld"))
    assert not os.path.exists(str(tmp_path / "nogps_cw.ld"))
    assert not os.path.exists(str(tmp_path / "nogps_ccw.ld"))
    assert "WARNING" in result.stdout


def test_batch_direction_suffix(tmp_path):
    _write_synthetic_ld(str(tmp_path / "a.ld"), clockwise=False)
    _write_synthetic_ld(str(tmp_path / "b.ld"), clockwise=True)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    result = _run_cli_in_process(
        [str(tmp_path), "ACTI", "--track", "test_loop", "--anchor", anchor,
         "--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(str(tmp_path / "a_gps_ccw.ld"))
    assert os.path.isfile(str(tmp_path / "b_gps_cw.ld"))


def test_batch_output_dir_acti_writes_single_gps_name(tmp_path):
    """Batch ACTI + --output-dir + --direction-suffix lands one file per input.

    Regression test: the batch ``_gps`` stem composed with the ``_cw``/``_ccw``
    suffix must yield exactly one .ld/.ldx pair per input in the output dir,
    never a ``foo_cw`` twin next to ``foo_gps_cw``; a --force rerun keeps the
    same file set instead of regrowing twins.
    """
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    out_dir.mkdir()
    _write_synthetic_ld(str(src_dir / "a.ld"), clockwise=False)
    _write_synthetic_ld(str(src_dir / "b.ld"), clockwise=True)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    args = [
        str(src_dir), "ACTI", "--track", "test_loop", "--anchor", anchor,
        "--output-dir", str(out_dir), "--direction-suffix",
    ]
    result = _run_cli_in_process(args)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = ["a_gps_ccw.ld", "a_gps_ccw.ldx", "b_gps_cw.ld", "b_gps_cw.ldx"]
    assert sorted(os.listdir(str(out_dir))) == expected
    assert sorted(os.listdir(str(src_dir))) == ["a.ld", "b.ld"]
    rerun = _run_cli_in_process([*args, "--force"])
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert sorted(os.listdir(str(out_dir))) == expected


def _ldx_detail(ldx_path, detail_id):
    root = ET.parse(ldx_path).getroot()
    for elem in root.iter("String"):
        if elem.get("Id") == detail_id:
            value = elem.get("Value")
            assert value is not None, f"{detail_id!r} has no Value in {ldx_path}"
            return value
    raise AssertionError(f"{detail_id!r} detail missing in {ldx_path}")


def test_venue_suffix_ccw(tmp_path):
    result, tmp_path = _convert_acti(tmp_path, "vccw_run.ld", clockwise=False,
                                     extra_args=["--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    ldx = str(tmp_path / "vccw_run_ccw.ldx")
    assert os.path.isfile(ldx)
    assert _ldx_detail(ldx, "Venue") == "Test Loop CCW"
    # Only the venue carries direction; event/session metadata are untouched.
    assert _ldx_detail(ldx, "Event") == "Test Loop Track Day"


def test_venue_suffix_cw(tmp_path):
    result, tmp_path = _convert_acti(tmp_path, "vcw_run.ld", clockwise=True,
                                     extra_args=["--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    ldx = str(tmp_path / "vcw_run_cw.ldx")
    assert os.path.isfile(ldx)
    assert _ldx_detail(ldx, "Venue") == "Test Loop CW"
    assert _ldx_detail(ldx, "Event") == "Test Loop Track Day"


def test_venue_explicit_wins(tmp_path):
    result, tmp_path = _convert_acti(tmp_path, "vexp_run.ld", clockwise=False,
                                     extra_args=["--direction-suffix",
                                                 "--venue_name", "My Track"])
    assert result.returncode == 0, result.stdout + result.stderr
    ldx = str(tmp_path / "vexp_run_ccw.ldx")
    assert os.path.isfile(ldx)
    assert _ldx_detail(ldx, "Venue") == "My Track"
    assert "WARNING" in result.stdout and "--venue_name" in result.stdout


def test_venue_default_off(tmp_path):
    src = str(tmp_path / "voff_run.ld")
    _write_synthetic_ld(src)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    wanted = str(tmp_path / "voff_out.ld")
    result = _run_cli_in_process(
        [src, "ACTI", "--track", "test_loop", "--anchor", anchor,
         "--output", wanted])
    assert result.returncode == 0, result.stdout + result.stderr
    ldx = str(tmp_path / "voff_out.ldx")
    assert _ldx_detail(ldx, "Venue") == "Test Loop"


def test_venue_no_gps_unchanged(tmp_path):
    src = str(tmp_path / "vnogps.csv")
    with open(src, "w", encoding="utf-8") as handle:
        handle.write("Time,Throttle Pos\n")
        for i in range(50):
            handle.write(f"{i * 0.1:.1f},{float(i):.1f}\n")
    result = _run_cli_in_process([src, "CSV", "--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    venue = _ldx_detail(str(tmp_path / "vnogps.ldx"), "Venue")
    assert " CW" not in venue and " CCW" not in venue


def test_batch_venue_suffix(tmp_path):
    _write_synthetic_ld(str(tmp_path / "va.ld"), clockwise=False)
    _write_synthetic_ld(str(tmp_path / "vb.ld"), clockwise=True)
    anchor = _write_anchor(str(tmp_path / "anchors.json"))
    result = _run_cli_in_process(
        [str(tmp_path), "ACTI", "--track", "test_loop", "--anchor", anchor,
         "--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert _ldx_detail(str(tmp_path / "va_gps_ccw.ldx"), "Venue") == "Test Loop CCW"
    assert _ldx_detail(str(tmp_path / "vb_gps_cw.ldx"), "Venue") == "Test Loop CW"


def test_direction_suffix_skipped_for_single_direction_track(tmp_path):
    """--direction-suffix is a no-op for single-direction ACTI profiles.

    The flag stays global, but only bidirectional profiles take _cw/_ccw:
    sonoma-style tracks keep plain names with a notice, no venue suffix.
    """
    _write_synthetic_ld(str(tmp_path / "a.ld"), clockwise=False)
    anchor = _write_anchor(str(tmp_path / "anchors.json"), bidirectional=False)
    result = _run_cli_in_process(
        [str(tmp_path), "ACTI", "--track", "test_loop", "--anchor", anchor,
         "--direction-suffix"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert os.path.isfile(str(tmp_path / "a_gps.ld"))
    assert os.path.isfile(str(tmp_path / "a_gps.ldx"))
    assert not os.path.exists(str(tmp_path / "a_gps_ccw.ld"))
    assert not os.path.exists(str(tmp_path / "a_gps_cw.ld"))
    assert "not applicable" in result.stdout
    assert _ldx_detail(str(tmp_path / "a_gps.ldx"), "Venue") == "Test Loop"
