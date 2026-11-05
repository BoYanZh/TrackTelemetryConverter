"""Focused tests for conservative IBT lap segmentation."""

import numpy as np

from track_telemetry_converter.channels import CH_LAP_NUMBER
from track_telemetry_converter.log import DataLog
from track_telemetry_converter.parsers.ibt_laps import build_ibt_laps


def _build(lap, *, times=None, **signals):
    lap = np.asarray(lap, dtype=float)
    if times is None:
        times = np.arange(len(lap), dtype=float)
    log = DataLog()
    build_ibt_laps(log, lap, times, **signals)
    return log


def test_adjacent_confirmed_crossings_build_out_timed_and_in_laps():
    log = _build(
        [0, 0, 0, 1, 1, 1, 1, 2, 2],
        lap_dist=[0.2, 0.6, 0.99, 0.0, 0.4, 0.8, 0.99, 0.0, 0.2],
        lap_completed=[-1, -1, -1, 0, 0, 0, 0, 1, 1],
        session_time=np.arange(9, dtype=float),
    )

    assert log.laps_info["beacons"] == [(3.0, "Lap 1"), (7.0, "Lap 2")]
    laps = log.laps_info["laps"]
    assert [(lap["type"], lap["lap_num"], lap["source_lap_num"]) for lap in laps] == [
        ("Out Lap", 1, 0),
        ("Timed", 2, 1),
        ("In Lap", 3, 2),
    ]
    assert [lap["complete"] for lap in laps] == [False, True, False]
    assert all(lap["valid"] is None for lap in laps)
    assert log.laps_info["fastest_lap"] == 2
    assert log.laps_info["fastest_time"] == 4.0
    np.testing.assert_array_equal(log.channels[CH_LAP_NUMBER].values, [0, 0, 0, 1, 1, 1, 1, 2, 2])


def test_counter_reset_splits_stints_and_never_completes_across_reset():
    log = _build(
        [2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3],
        lap_dist=[0.1, 0.4, 0.6, 0.8, 0.99, 0.0, 0.3, 0.6, 0.99, 0.0, 0.3, 0.6, 0.99, 0.0],
        lap_completed=[1, 1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2],
    )

    assert [time for time, _ in log.laps_info["beacons"]] == [5.0, 9.0, 13.0]
    laps = log.laps_info["laps"]
    complete = [lap for lap in laps if lap["complete"]]
    assert [(lap["start_time"], lap["end_time"], lap["stint"]) for lap in complete] == [
        (5.0, 9.0, 1),
        (9.0, 13.0, 1),
    ]
    assert all(not lap["complete"] for lap in laps if lap["stint"] == 0)
    assert all(lap["source_lap_num"] is not None for lap in laps)


def test_positive_distance_teleport_breaks_lap_even_when_counters_continue():
    log = _build(
        [2, 3, 3, 3, 4, 4, 4, 4, 5, 5],
        lap_dist=[0.9, 0.0, 0.2, 0.993, 0.0, 0.3, 0.6, 0.99, 0.0, 0.3],
        lap_completed=[1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
    )

    assert [time for time, _ in log.laps_info["beacons"]] == [1.0, 4.0, 8.0]
    complete = [lap for lap in log.laps_info["laps"] if lap["complete"]]
    assert [(lap["start_time"], lap["end_time"]) for lap in complete] == [(4.0, 8.0)]
    assert log.laps_info["fastest_time"] == 4.0


def test_one_distance_wrap_cannot_confirm_neighboring_counter_edges_twice():
    log = _build(
        [0, 0, 0, 0, 1, 2, 2],
        lap_dist=[0.2, 0.4, 0.6, 0.8, 0.99, 0.0, 0.3],
        lap_completed=[-1, -1, -1, -1, 0, 1, 1],
    )

    assert log.laps_info["beacons"] == [(5.0, "Lap 2")]
    assert not any(lap["complete"] for lap in log.laps_info["laps"])


def test_unconfirmed_increment_and_counter_jump_create_only_partial_segments():
    no_wrap = _build(
        [0, 1, 1],
        lap_dist=[0.2, 0.3, 0.4],
        lap_completed=[-1, 0, 0],
    )
    assert no_wrap.laps_info["beacons"] == []
    assert all(not lap["complete"] for lap in no_wrap.laps_info["laps"])

    jump = _build(
        [1, 1, 3, 3, 4, 4, 4, 4, 5, 5],
        lap_dist=[0.2, 0.4, 0.2, 0.6, 0.99, 0.0, 0.3, 0.6, 0.99, 0.0],
        lap_completed=[0, 0, 2, 2, 3, 3, 3, 3, 4, 4],
    )
    assert [time for time, _ in jump.laps_info["beacons"]] == [4.0, 8.0]
    assert [(lap["start_time"], lap["end_time"]) for lap in jump.laps_info["laps"] if lap["complete"]] == [
        (4.0, 8.0),
    ]


def test_session_and_invalid_on_track_breaks_make_laps_partial():
    log = _build(
        [0, 0, 1, 1, 2, 2, 2],
        lap_dist=[0.4, 0.9, 0.0, 0.3, 0.6, 0.9, 0.99],
        lap_completed=[-1, -1, 0, 0, 1, 1, 1],
        session_num=[1, 1, 1, 2, 2, 2, 2],
        on_track=[1, 1, 1, 1, 0, 1, 1],
    )

    assert log.laps_info["beacons"] == [(2.0, "Lap 1")]
    assert all(not lap["complete"] for lap in log.laps_info["laps"])
    assert sorted({lap["stint"] for lap in log.laps_info["laps"]}) == [0, 1, 2, 3]
    assert any(lap["valid"] is False for lap in log.laps_info["laps"])


def test_native_time_reset_counter_invalidity_and_long_gap_split_stints():
    reset = _build(
        [0, 0, 1, 1],
        lap_dist=[0.2, 0.99, 0.0, 0.99],
        session_time=[10, 11, 0, 1],
    )
    assert reset.laps_info["beacons"] == []
    assert all(not lap["complete"] for lap in reset.laps_info["laps"])

    invalid = _build([0, 0, np.nan, 1, 1])
    assert all(not lap["complete"] for lap in invalid.laps_info["laps"])
    assert any(lap["valid"] is False for lap in invalid.laps_info["laps"])

    gap = _build(
        [0, 0, 1, 1],
        times=[0.0, 0.02, 2.0, 2.02],
        lap_dist=[0.2, 0.99, 0.0, 0.99],
        lap_completed=[-1, -1, 0, 0],
    )
    assert gap.laps_info["beacons"] == []
    assert all(not lap["complete"] for lap in gap.laps_info["laps"])

    negative_distance = _build(
        [0, 1, 1],
        lap_dist=[0.8, -0.1, 0.2],
        lap_completed=[-1, 0, 0],
    )
    assert negative_distance.laps_info["beacons"] == []


def test_pit_road_marks_lap_invalid_and_excludes_it_from_fastest():
    log = _build(
        [0, 0, 1, 1, 1, 1, 2, 2, 2],
        lap_dist=[0.2, 0.6, 0.99, 0.0, 0.3, 0.6, 0.99, 0.0, 0.2],
        lap_completed=[-1, -1, 0, 0, 0, 0, 1, 1, 1],
        pit_road=[0, 0, 0, 0, 1, 0, 0, 0, 0],
    )

    timed = [lap for lap in log.laps_info["laps"] if lap["complete"]]
    assert len(timed) == 1
    assert timed[0]["valid"] is False
    assert log.laps_info["fastest_lap"] is None
    assert log.laps_info["fastest_time"] == 0.0


def test_missing_counter_is_one_partial_segment_with_unknown_source_number():
    log = DataLog()
    build_ibt_laps(log, None, [0.0, 1.0, 2.0])

    assert log.laps_info["beacons"] == []
    assert log.laps_info["total_laps"] == 1
    assert log.laps_info["fastest_lap"] is None
    assert log.laps_info["laps"][0]["complete"] is False
    assert log.laps_info["laps"][0]["source_lap_num"] is None
    np.testing.assert_array_equal(log.channels[CH_LAP_NUMBER].values, [0, 0, 0])
