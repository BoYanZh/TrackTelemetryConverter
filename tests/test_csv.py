"""Tests for CSV, PB Buddy, Accessport, and RaceChrono CSV formats."""

import os
import tempfile

import numpy as np
from conftest import _dedup_channels, _read_lines

from track_telemetry_converter._vendor.ldparser import ldData
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.motec import MotecLog
from track_telemetry_converter.output import atomic_write_motec_pair


def test_csv_smoke():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))


def test_pbbuddy_smoke():
    log = DataLog()
    lines = _read_lines("pbbuddy_sample.csv")
    log.from_pbbuddy_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.metadata.get("venue_name") == "Thunderhill East (Cyclone)"
    assert log.duration() > 0


def test_aim_solo_smoke():
    log = DataLog()
    lines = _read_lines("aim_solo_sample.csv")
    log.from_racechrono_log(lines)
    _dedup_channels(log)
    assert len(log.channels) > 0
    assert "Ground Speed" in log.channels
    assert "CG Accel Lateral" in log.channels
    assert "GPS Latitude" in log.channels
    assert "GPS Longitude" in log.channels
    assert log.metadata.get("venue_name") == "Laguna Seca"
    assert log.datetime is not None
    assert log.datetime.strftime("%Y-%m-%d") == "2026-02-14"
    assert log.duration() > 0


def test_csv_resample_and_math():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    for ch in log.channels.values():
        assert len(ch.messages) > 0, f"Channel {ch.name} empty after resample"
    log.calculate_math_channels()


def test_accessport_smoke():
    log = DataLog()
    log.from_accessport_log(_read_lines("accessport_sample.csv"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_accessport_resample_and_math():
    log = DataLog()
    log.from_accessport_log(_read_lines("accessport_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    log.calculate_math_channels()


def test_racechrono_smoke():
    log = DataLog()
    log.from_racechrono_log(_read_lines("racechrono_sample.csv"))
    assert len(log.channels) >= 5, f"Too few channels: {len(log.channels)}"
    assert log.duration() > 0
    _dedup_channels(log)


def test_racechrono_resample_and_math():
    log = DataLog()
    log.from_racechrono_log(_read_lines("racechrono_sample.csv"))
    log.resample(20)
    assert log.duration() > 0
    log.calculate_math_channels()


def test_csv_middle_non_numeric_column_keeps_right_columns():
    log = DataLog()
    lines = [
        "time,colA,colB,colC\n",
        "0.0,1.0,XXX,100.0\n",
        "0.1,2.0,YYY,200.0\n",
        "0.2,3.0,ZZZ,300.0\n",
    ]
    log.from_csv_log(lines)
    assert "colA" in log.channels
    assert "colC" in log.channels
    assert "colB" not in log.channels
    col_c = log.channels["colC"].messages
    assert len(col_c) == 3, f"colC messages: {len(col_c)}"
    assert col_c[-1].value == 300.0


def test_auto_frequency_detection():
    log = DataLog()
    log.from_csv_log(_read_lines("csv_sample.csv"))
    freq = log.detect_native_frequency()
    assert freq > 0, f"Invalid auto frequency: {freq}"
    resampled_f = log.resample("auto")
    assert resampled_f == freq


def test_racechrono_long_names_prefer_canbus_and_roundtrip():
    lines = [
        "Time (s),Longitudinal acceleration (G) *calc,"
        "Longitudinal acceleration (G) *canbus,Combined acceleration (G) *canbus,"
        "Engine oil temperature (.C) *canbus\n",
        "s,G,G,G,C\n",
        "0.0,0.10,0.20,0.30,100.0\n",
        "1.0,0.11,0.21,0.31,101.0\n",
    ]
    log = DataLog()
    log.from_racechrono_log(lines)

    assert "Longitudinal acceleration (G) *calc" not in log.channels
    assert np.array_equal(log.channels["CG Accel Longitudinal"].values, [0.20, 0.21])
    assert np.array_equal(log.channels["G Force Combined"].values, [0.30, 0.31])
    assert np.array_equal(log.channels["Engine Oil Temp"].values, [100.0, 101.0])
    assert all(len(name.encode("ascii")) <= 32 for name in log.channels)

    motec = MotecLog()
    motec.initialize()
    motec.add_all_channels(log)
    with tempfile.TemporaryDirectory() as tmp_dir:
        ld_path = os.path.join(tmp_dir, "long_names.ld")
        ldx_path = os.path.join(tmp_dir, "long_names.ldx")
        atomic_write_motec_pair(motec, log, ld_path, ldx_path)
        parsed = ldData.fromfile(ld_path)
        assert [channel.name for channel in parsed.channs] == list(log.channels)


def test_racechrono_unit_conversions():
    lines = [
        "Time (s),speed,coolant_temp,brake_pressure,yaw_rate,speedv\n",
        "s,m/s,F,psi,deg/s,mph\n",
        "0.0,10.0,212.0,100.0,5.0,60.0\n",
        "1.0,20.0,32.0,200.0,6.0,60.0\n",
    ]
    log = DataLog()
    log.from_racechrono_log(lines)
    assert np.array_equal(log.channels["Ground Speed"].values, [36.0, 72.0])
    assert np.array_equal(log.channels["Coolant Temp"].values, [100.0, 0.0])
    assert np.allclose(
        log.channels["Brake Press"].values, [689.476, 1378.952],
    )
    assert np.array_equal(
        log.channels["Chassis Yaw Rate"].values, [-5.0, -6.0],
    )
    assert np.allclose(
        log.channels["Vehicle Speed"].values, [96.5604, 96.5604],
    )


def test_racechrono_bar_brake_and_gyro_fallback():
    log = DataLog()
    log.from_racechrono_log([
        "Time (s),brakepress,rpm\n",
        "s,bar,rpm\n",
        "0.0,1.0,3000.0\n",
        "1.0,2.0,4000.0\n",
    ])
    assert np.array_equal(log.channels["Brake Press"].values, [100.0, 200.0])

    log = DataLog()
    log.from_racechrono_log([
        "Time (s),z_rate_of_rotation,rpm\n",
        "s,deg/s,rpm\n",
        "0.0,2.0,3000.0\n",
        "1.0,4.0,4000.0\n",
    ])
    assert np.array_equal(
        log.channels["Chassis Yaw Rate"].values, [-2.0, -4.0],
    )

    log = DataLog()
    log.from_racechrono_log([
        "Time (s),y_rate_of_rotation,rpm\n",
        "s,deg/s,rpm\n",
        "0.0,3.0,3000.0\n",
        "1.0,6.0,4000.0\n",
    ])
    assert np.array_equal(
        log.channels["Chassis Yaw Rate"].values, [-3.0, -6.0],
    )

    log = DataLog()
    log.from_racechrono_log([
        "Time (s),Foo (bar),rpm\n",
        "s,bar,rpm\n",
        "0.0,1.5,3000.0\n",
        "1.0,2.5,4000.0\n",
    ])
    assert log.channels["Foo"].units == "bar"
    assert np.array_equal(log.channels["Foo"].values, [1.5, 2.5])


def test_racechrono_pedal_semantics_and_provenance():
    log = DataLog()
    log.from_racechrono_log([
        "Time (s),accelerator_pos,pps,yaw_rate,rpm,rpm\n",
        "s,%,%,deg/s,rpm,rpm\n",
        "0.0,10.0,20.0,5.0,3000.0,9999.0\n",
        "1.0,11.0,21.0,6.0,4000.0,8888.0\n",
    ])
    # accelerator_pos is the pedal (RaceChrono PID 10071), not the throttle.
    assert np.array_equal(log.channels["Accelerator Pos"].values, [10.0, 11.0])
    assert np.array_equal(log.channels["Throttle Pos"].values, [20.0, 21.0])
    prov = log.metadata["channel_provenance"]
    assert prov["Accelerator Pos"]["power_semantics"] == "accelerator_pedal"
    assert "10071" in prov["Accelerator Pos"]["semantics_basis"]
    assert prov["Throttle Pos"]["power_semantics"] == "unverified"
    # Recorded yaw keeps its negate convention but stays sign-unverified.
    assert np.array_equal(
        log.channels["Chassis Yaw Rate"].values, [-5.0, -6.0],
    )
    assert log.metadata["yaw_rate_source"] == "racechrono_recorded_yaw_rate"
    assert log.metadata["yaw_axis_sign_verified"] is False
    # First of the duplicate rpm columns wins; the loser is recorded.
    assert np.array_equal(log.channels["Engine RPM"].values, [3000.0, 4000.0])
    assert {
        "canonical": "Engine RPM", "kept_raw": "rpm", "dropped_raw": "rpm",
    } in log.metadata["dropped_duplicate_columns"]


def test_racechrono_alpha_laps_and_empty_channel_cleanup():
    lines = ["Time (s),speed,lap_number,rpm\n", "s,km/h,,rpm\n"]
    for i, lap in enumerate(["X", "X", "X", "Y"]):
        lines.append(f"{i * 10.0:.1f},50.0,{lap},\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    assert log.laps_info["total_laps"] == 1
    assert "Engine RPM" not in log.channels

    lines = ["\x00junk\n", "Time (s),speed,rpm\n", "s,m/s,rpm\n"]
    lines.append("0.0,10.0,3000.0\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    assert np.array_equal(log.channels["Ground Speed"].values, [36.0])


def test_racechrono_target_lap_and_errors(capsys):
    lines = [
        "Time (s),speed,lap_number\n",
        "s,km/h,\n",
        "0.0,36.0,1\n",
        "10.0,72.0,1\n",
        '20.0,100.0,"2"\n',
        "30.0,110.0,2\n",
    ]
    log = DataLog()
    log.from_racechrono_log(lines, target_lap="2")
    assert np.array_equal(
        log.channels["Ground Speed"].timestamps, [20.0, 30.0],
    )
    empty = DataLog()
    empty.from_racechrono_log([])
    assert len(empty.channels) == 0
    no_header = DataLog()
    no_header.from_racechrono_log(["foo,bar\n", "1,2\n"])
    assert len(no_header.channels) == 0
    assert "ERROR" in capsys.readouterr().out


def test_racechrono_beacon_markers_build_laps():
    lines = ["Beacon markers,30.0,90.0,150.0\n"]
    lines += ["Time (s),speed,coolant_temp\n", "s,km/h,C\n"]
    for i in range(16):
        lines.append(f"{i * 10.0:.1f},{50.0 + i:.1f},90.0\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    laps = log.laps_info["laps"]
    assert log.laps_info["total_laps"] == 3
    assert [lap["type"] for lap in laps] == ["Out Lap", "Timed", "In Lap"]
    assert laps[1]["duration"] == 60.0
    assert log.laps_info["fastest_time"] == 60.0


def test_pbbuddy_epoch_offset_datetime():
    lines = [
        '"Session start, seconds since epoch","1771107686"\n',
        '"Timezone offset, milliseconds","-28800000"\n',
        '"Track name","Thunderhill East (Cyclone)"\n',
        "Time,GPS Latitude,GPS Speed\n",
        "sec,deg,m/s\n",
        "0.0,39.5,10.0\n",
        "1.0,39.6,20.0\n",
    ]
    log = DataLog()
    log.from_pbbuddy_log(lines)
    assert log.datetime is not None
    assert log.datetime.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-14 14:21:26"
    assert np.array_equal(log.channels["Ground Speed"].values, [36.0, 72.0])


def test_pbbuddy_date_time_fallback_and_extras(capsys):
    lines = [
        '"Date","2/14/26"\n',
        '"Time","22:21:26"\n',
        "Time,GPS Latitude,Foo\n",
        "sec,deg,bar\n",
        "# comment line\n",
        "// another comment\n",
        "\n",
        "0.0,39.5,1.0\n",
        "1.0,39.6,2.0\n",
    ]
    log = DataLog()
    log.from_pbbuddy_log(lines)
    assert log.datetime is not None
    assert log.datetime.strftime("%Y-%m-%d %H:%M:%S") == "2026-02-14 22:21:26"
    assert log.channels["Foo"].units == "bar"
    assert np.array_equal(log.channels["Foo"].values, [1.0, 2.0])

    empty = DataLog()
    empty.from_pbbuddy_log([])
    assert len(empty.channels) == 0
    no_header = DataLog()
    no_header.from_pbbuddy_log(["foo,bar\n", "1,2\n"])
    assert len(no_header.channels) == 0
    assert "ERROR" in capsys.readouterr().out


def test_pbbuddy_bad_epoch_and_bad_row():
    lines = [
        "\x00junk\n",
        '"Session start, seconds since epoch","garbage"\n',
        "Time,GPS Latitude,GPS Speed\n",
        "sec,deg,m/s\n",
        "oops,39.5,10.0\n",
        "0.0,39.5,10.0\n",
        "1.0,39.6,20.0\n",
    ]
    log = DataLog()
    log.from_pbbuddy_log(lines)
    assert log.datetime is None
    assert np.array_equal(
        log.channels["GPS Latitude"].values, [39.5, 39.6],
    )


def test_racechrono_duplicate_canonical_and_row_skips():
    lines = [
        "Time (s),rpm,rpm,speed\n",
        "s,rpm,rpm,m/s\n",
        "0.0,3000.0,9999.0,10.0\n",
        "1.0,,20.0\n",
        "oops,4000.0,4000.0,30.0\n",
        '"3.0",abc,6000.0,50.0\n',
        '2.0,"5000.0",5000.0,40.0\n',
        '4.0,"",6000.0,60.0\n',
    ]
    log = DataLog()
    log.from_racechrono_log(lines)
    assert np.array_equal(log.channels["Engine RPM"].values, [3000.0, 5000.0])
    assert np.array_equal(
        log.channels["Ground Speed"].values, [36.0, 180.0, 144.0, 216.0],
    )


def test_racechrono_short_and_timed_laps():
    lines = ["Time (s),speed,lap_number\n", "s,km/h,\n"]
    sched = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7]
    for i, lap in enumerate(sched):
        lines.append(f"{i * 10.0:.1f},50.0,{lap}\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    laps = log.laps_info["laps"]
    assert [lap["type"] for lap in laps] == ["Out Lap", "Timed", "Timed", "In Lap"]
    assert log.laps_info["total_laps"] == 4
    assert log.laps_info["fastest_time"] == 20.0

    lines = ["Beacon markers,abc,5.0,5.5,50.0,110.0\n"]
    lines += ["Time (s),speed,lap_number\n", "s,km/h,\n"]
    for i in range(12):
        lines.append(f"{i * 10.0:.1f},50.0,1\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    types = [lap["type"] for lap in log.laps_info["laps"]]
    assert types == ["Timed", "In Lap"]
    assert log.laps_info["fastest_time"] == 44.5

    lines = ["Beacon markers,5.0,50.0,60.0\n"]
    lines += ["Time (s),speed,lap_number\n", "s,km/h,\n"]
    for i in range(12):
        lines.append(f"{i * 10.0:.1f},50.0,1\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    assert [lap["type"] for lap in log.laps_info["laps"]] == ["Timed"]


def test_racechrono_unlabeled_rows_do_not_form_bogus_lap():
    lines = ["timestamp,lap_number,speed\n", "s,,m/s\n"]
    for i in range(11):
        lines.append(f"{i * 10.0:.1f},1,10.0\n")
    for i in range(11, 21):
        lines.append(f"{i * 10.0:.1f},,10.0\n")
    for i in range(21, 51):
        lines.append(f"{i * 10.0:.1f},2,10.0\n")
    log = DataLog()
    log.from_racechrono_log(lines)
    laps = log.laps_info["laps"]
    assert all(lap["lap_label"] != "" for lap in laps)
    assert log.laps_info["total_laps"] == 2
    # Unlabeled rows still contribute channel samples.
    assert len(log.channels["Ground Speed"].messages) == 51
    timed = [lap for lap in laps if lap["type"] == "Timed"]
    bounds = [float(timed[0]["start_time"])] + [float(lap["end_time"]) for lap in timed]
    assert bounds == sorted(bounds)
