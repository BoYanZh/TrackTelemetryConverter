"""Build conservative lap segments from iRacing IBT sample arrays.

This module is kept separate from binary parsing so counter/reset behavior can
be tested directly. Times are the caller's monotonic elapsed-time axis.
"""

from __future__ import annotations

import math

import numpy as np

from ..channels import CH_LAP_NUMBER

_PCT_WRAP_HIGH = 0.75
_PCT_WRAP_LOW = 0.25
_CROSSING_SKEW_SAMPLES = 1
_DIST_DISCONTINUITY_THRESHOLD = 0.5
_LONG_GAP_FACTOR = 10.0
_MIN_LONG_GAP_SECONDS = 0.5


def _samples(values, count: int) -> np.ndarray | None:
    """Return a float vector aligned to ``count``, padding missing values with NaN."""
    if values is None:
        return None
    source = np.asarray(values, dtype=np.float64)
    if source.ndim == 0:
        return np.full(count, float(source), dtype=np.float64)
    source = source.reshape(-1)
    aligned = np.full(count, np.nan, dtype=np.float64)
    copy_count = min(count, len(source))
    aligned[:copy_count] = source[:copy_count]
    return aligned


def _integral_samples(values: np.ndarray | None, *, nonnegative: bool = False) -> np.ndarray:
    if values is None:
        return np.ones(0, dtype=bool)
    finite = np.isfinite(values)
    integral = finite & (np.abs(values - np.rint(values)) < 1e-6)
    if nonnegative:
        integral &= values >= 0
    return integral


def _long_gap_limit(values: np.ndarray) -> float:
    if len(values) < 2:
        return math.inf
    diffs = np.diff(values)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    if not len(positive):
        return math.inf
    return max(_MIN_LONG_GAP_SECONDS, float(np.median(positive)) * _LONG_GAP_FACTOR)


def _nearby_pair_ends(index: int, active: np.ndarray, break_before: np.ndarray):
    count = len(active)
    for end in range(
        index - _CROSSING_SKEW_SAMPLES,
        index + _CROSSING_SKEW_SAMPLES + 1,
    ):
        if (
            0 < end < count
            and active[end - 1]
            and active[end]
            and not break_before[end]
        ):
            yield end


def _find_increment(
    values: np.ndarray,
    index: int,
    active: np.ndarray,
    break_before: np.ndarray,
    used: set[int],
) -> int | None:
    """Find a unique +1 edge within one sample of a Lap transition."""
    matches = []
    for end in _nearby_pair_ends(index, active, break_before):
        before, after = values[end - 1], values[end]
        if end not in used and np.isfinite(before) and np.isfinite(after) and after - before == 1:
            matches.append(end)
    return min(matches, key=lambda end: (abs(end - index), end)) if matches else None


def _find_distance_wrap(
    values: np.ndarray,
    index: int,
    active: np.ndarray,
    break_before: np.ndarray,
    used: set[int],
) -> int | None:
    """Check for a near-zero wrap near the Lap edge, tolerating one sample skew.

    No SDK timing tolerance is specified. One sample is used because the
    observed 60 Hz IBTs update these channels on the same tick, and adjacent
    tick checks allow for discrete channel update order without widening the
    crossing window into meaningful track time.
    """
    matches = []
    for end in _nearby_pair_ends(index, active, break_before):
        before, after = values[end - 1], values[end]
        if (
            end not in used
            and
            np.isfinite(before)
            and np.isfinite(after)
            and before >= _PCT_WRAP_HIGH
            and after >= 0
            and after <= _PCT_WRAP_LOW
            and before - after > 0.5
        ):
            matches.append(end)
    return min(matches, key=lambda end: (abs(end - index), end)) if matches else None


def _completed_counter_static_nonnegative(
    values: np.ndarray, index: int, active: np.ndarray, break_before: np.ndarray,
) -> bool:
    """Allow a distance-confirmed initial crossing before LapCompleted advances."""
    indices = [index]
    if index > 0 and active[index - 1] and not break_before[index]:
        indices.append(index - 1)
    if (
        index + _CROSSING_SKEW_SAMPLES < len(values)
        and active[index + 1]
        and not break_before[index + 1]
    ):
        indices.append(index + 1)
    nearby = values[indices]
    return bool(
        len(nearby)
        and np.all(np.isfinite(nearby))
        and np.all(np.abs(nearby - np.rint(nearby)) < 1e-6)
        and np.all(nearby >= 0)
        and np.all(nearby == nearby[0]),
    )


def _crossing_is_confirmed(
    index: int,
    active: np.ndarray,
    break_before: np.ndarray,
    lap_dist: np.ndarray | None,
    lap_completed: np.ndarray | None,
    used_distance_wraps: set[int],
    used_completed_edges: set[int],
) -> tuple[bool, int | None, int | None]:
    distance_wrap = (
        _find_distance_wrap(lap_dist, index, active, break_before, used_distance_wraps)
        if lap_dist is not None
        else None
    )
    if lap_dist is not None and distance_wrap is None:
        return False, None, None

    if lap_completed is None:
        return True, distance_wrap, None
    completed_edge = _find_increment(
        lap_completed, index, active, break_before, used_completed_edges,
    )
    if completed_edge is not None:
        return True, distance_wrap, completed_edge
    # Some sessions begin with LapCompleted already at zero; a real distance
    # wrap still confirms that first line crossing. A stationary -1 baseline
    # does not confirm the green-flag Lap 0 -> 1 edge.
    static_completed = (
        distance_wrap is not None
        and _completed_counter_static_nonnegative(lap_completed, index, active, break_before)
    )
    return static_completed, distance_wrap, None


def _candidate_priority(index, active, break_before, lap_dist, lap_completed):
    offsets = []
    if lap_dist is not None:
        match = _find_distance_wrap(lap_dist, index, active, break_before, set())
        offsets.append(abs(match - index) if match is not None else math.inf)
    if lap_completed is not None:
        match = _find_increment(lap_completed, index, active, break_before, set())
        offsets.append(abs(match - index) if match is not None else math.inf)
    return (max(offsets, default=0), sum(offsets), index)


def _validity(
    start: int,
    stop: int,
    on_track: np.ndarray | None,
    pit_road: np.ndarray | None,
    invalid_samples: np.ndarray,
) -> bool | None:
    if np.any(invalid_samples[start:stop]):
        return False
    if on_track is not None and np.any(on_track[start:stop] <= 0):
        return False
    if pit_road is not None and np.any(np.isfinite(pit_road[start:stop]) & (pit_road[start:stop] > 0)):
        return False
    # The provided signals can identify known invalidating conditions, but do
    # not include an affirmative lap-valid flag.
    return None


def _active_samples(elapsed, lap, session, native_time, track, completed, dist):
    count = len(elapsed)
    time_valid = np.isfinite(elapsed)
    lap_valid = _integral_samples(lap, nonnegative=True) if lap is not None else np.ones(count, bool)
    completed_valid = _integral_samples(completed) if completed is not None else np.ones(count, bool)
    session_valid = np.isfinite(session) if session is not None else np.ones(count, bool)
    native_time_valid = np.isfinite(native_time) if native_time is not None else np.ones(count, bool)
    track_valid = np.isfinite(track) & (track > 0) if track is not None else np.ones(count, bool)
    distance_valid = np.isfinite(dist) & (dist >= 0) if dist is not None else np.ones(count, bool)
    active = (
        time_valid
        & lap_valid
        & completed_valid
        & session_valid
        & native_time_valid
        & track_valid
        & distance_valid
    )
    return time_valid, lap_valid, completed_valid, active, ~active


def _structural_breaks(elapsed, lap, lap_valid, session, native_time, completed, completed_valid, dist):
    count = len(elapsed)
    break_before = np.zeros(count, dtype=bool)
    break_before[0] = True
    if count <= 1:
        return break_before

    time_diffs = np.diff(elapsed)
    break_before[1:] |= ~np.isfinite(time_diffs) | (time_diffs <= 0)
    break_before[1:] |= time_diffs > _long_gap_limit(elapsed)

    if native_time is not None:
        session_diffs = np.diff(native_time)
        comparable = np.isfinite(session_diffs)
        break_before[1:] |= comparable & (session_diffs < 0)
        break_before[1:] |= comparable & (session_diffs > _long_gap_limit(native_time))

    if session is not None:
        same_session = np.isfinite(session[:-1]) & np.isfinite(session[1:])
        break_before[1:] |= same_session & (session[:-1] != session[1:])

    if lap is not None:
        lap_diff = np.diff(lap)
        pair_valid = lap_valid[:-1] & lap_valid[1:]
        break_before[1:] |= pair_valid & (lap_diff < 0)
        break_before[1:] |= pair_valid & (lap_diff > 1)

    if completed is not None:
        completed_diff = np.diff(completed)
        pair_valid = completed_valid[:-1] & completed_valid[1:]
        break_before[1:] |= pair_valid & (completed_diff < 0)
        break_before[1:] |= pair_valid & (completed_diff > 1)

    if dist is not None:
        distance_diff = np.diff(dist)
        distance_pair_valid = np.isfinite(dist[:-1]) & np.isfinite(dist[1:])
        # Large positive jumps are active resets/teleports, not a normal
        # start/finish wrap. Split even when Lap and SessionTime continue.
        break_before[1:] |= (
            distance_pair_valid & (distance_diff > _DIST_DISCONTINUITY_THRESHOLD)
        )
    return break_before


def _confirmed_crossings(lap, lap_valid, active, break_before, dist, completed):
    count = len(active)
    confirmed = np.zeros(count, dtype=bool)
    confirmed_distance_wraps: set[int] = set()
    if lap is None or count <= 1:
        return confirmed, break_before

    used_distance_wraps: set[int] = set()
    used_completed_edges: set[int] = set()
    lap_diff = np.diff(lap)
    candidate_indices = np.flatnonzero(
        (lap_diff == 1)
        & lap_valid[:-1]
        & lap_valid[1:]
        & active[:-1]
        & active[1:]
        & ~break_before[1:],
    ) + 1
    candidate_indices = sorted(
        (int(index) for index in candidate_indices),
        key=lambda index: _candidate_priority(index, active, break_before, dist, completed),
    )
    # Candidates are few compared with samples. Their corroboration checks
    # remain bounded to adjacent sample edges.
    for index in candidate_indices:
        is_confirmed, distance_wrap, completed_edge = _crossing_is_confirmed(
            index,
            active,
            break_before,
            dist,
            completed,
            used_distance_wraps,
            used_completed_edges,
        )
        if distance_wrap is not None:
            used_distance_wraps.add(distance_wrap)
        if completed_edge is not None:
            used_completed_edges.add(completed_edge)
        if is_confirmed:
            confirmed[index] = True
            if distance_wrap is not None:
                confirmed_distance_wraps.add(distance_wrap)
        else:
            break_before[index] = True

    if dist is not None:
        distance_diff = np.diff(dist)
        wrap_indices = np.flatnonzero(
            np.isfinite(dist[:-1])
            & np.isfinite(dist[1:])
            & (distance_diff < -_DIST_DISCONTINUITY_THRESHOLD),
        ) + 1
        for index in wrap_indices:
            if int(index) not in confirmed_distance_wraps:
                break_before[index] = True
    return confirmed, break_before


def build_ibt_laps(
    data_log,
    lap_arr,
    times,
    *,
    lap_dist=None,
    session_num=None,
    session_time=None,
    on_track=None,
    lap_completed=None,
    pit_road=None,
):
    """Populate the canonical lap channel and ``data_log.laps_info``.

    A reliable crossing requires an adjacent ``Lap`` +1 transition. If
    ``LapDistPct`` exists, it must also wrap near that edge. ``LapCompleted``
    adds corroboration when present. Counter resets, discontinuities, invalid
    samples, session changes, and track-unavailable intervals split stints so
    no complete lap can span them.
    """
    elapsed = np.asarray(times, dtype=np.float64).reshape(-1)
    count = len(elapsed)
    lap = _samples(lap_arr, count)
    dist = _samples(lap_dist, count)
    session = _samples(session_num, count)
    native_time = _samples(session_time, count)
    track = _samples(on_track, count)
    completed = _samples(lap_completed, count)
    pit = _samples(pit_road, count)
    if dist is not None:
        # Float32 LapDistPct samples can round a true zero slightly below 0.
        # Clamp only tiny numeric underflow; genuinely negative positions are
        # unavailable and split the continuity run.
        dist[(dist < 0) & (dist >= -1e-3)] = 0.0

    data_log.add_channel(CH_LAP_NUMBER, "", float, 0)
    lap_numbers = np.zeros(count, dtype=np.float64)
    beacons: list[tuple[float, str]] = []
    laps: list[dict] = []

    if count == 0:
        data_log.channels[CH_LAP_NUMBER].set_samples(elapsed, lap_numbers)
        data_log.laps_info = {
            "beacons": beacons,
            "laps": laps,
            "total_laps": 0,
            "fastest_lap": None,
            "fastest_time": 0.0,
        }
        return

    time_valid, lap_valid, completed_valid, active, invalid_samples = _active_samples(
        elapsed, lap, session, native_time, track, completed, dist,
    )
    # Elapsed-time and native SessionTime gaps use a robust median tick
    # interval; ten times that interval avoids splitting on ordinary jitter.
    break_before = _structural_breaks(
        elapsed, lap, lap_valid, session, native_time, completed,
        completed_valid, dist,
    )
    confirmed, break_before = _confirmed_crossings(
        lap, lap_valid, active, break_before, dist, completed,
    )

    # Separate active and invalid/unavailable spans, as well as explicit
    # structural breaks. Invalid spans remain visible as partial segments.
    starts = [0]
    for index in range(1, count):
        if break_before[index] or active[index] != active[index - 1]:
            starts.append(index)
    stops = starts[1:] + [count]

    for stint, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        if not time_valid[start]:
            continue
        run_crossings = np.flatnonzero(confirmed[start:stop]) + start if active[start] else np.array([], dtype=int)

        boundaries = [start, *[int(i) for i in run_crossings], stop]
        for segment_index, (segment_start, segment_stop) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True),
        ):
            if segment_start >= segment_stop:
                continue
            start_time = float(elapsed[segment_start])
            if segment_stop < count and time_valid[segment_stop]:
                end_time = float(elapsed[segment_stop])
            else:
                end_time = float(elapsed[segment_stop - 1])
            if end_time < start_time:
                continue

            complete = segment_index > 0 and segment_index < len(boundaries) - 2 and active[start]
            first_edge = segment_index == 0 and start == 0 and len(run_crossings) > 0
            last_edge = (
                segment_index == len(boundaries) - 2
                and stop == count
                and len(run_crossings) > 0
            )
            if complete:
                segment_type = "Timed"
                label = str(len(laps) + 1)
            elif first_edge:
                segment_type = "Out Lap"
                label = "Out Lap"
            elif last_edge:
                segment_type = "In Lap"
                label = "In Lap"
            else:
                segment_type = "Partial"
                label = "Partial"

            source_lap_num = None
            if lap is not None and lap_valid[segment_start]:
                source_lap_num = int(round(float(lap[segment_start])))

            lap_num = len(laps) + 1
            segment_valid = _validity(segment_start, segment_stop, track, pit, invalid_samples)
            duration = max(0.0, end_time - start_time)
            laps.append({
                "type": segment_type,
                "lap_label": label,
                "lap_num": lap_num,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "stint": stint,
                "complete": bool(complete),
                "valid": segment_valid,
                "source_lap_num": source_lap_num,
            })

    # Beacons are only confirmed Lap edges; source labels retain iRacing's
    # counter value while lap_num remains a unique extraction index.
    if lap is not None:
        for index in np.flatnonzero(confirmed):
            beacons.append((float(elapsed[index]), f"Lap {int(round(lap[index]))}"))

    # Keep the canonical channel as a cumulative crossing count. The unique
    # extraction key lives in each lap record's lap_num field.
    lap_numbers = np.cumsum(confirmed.astype(np.float64))

    eligible = [
        item for item in laps
        if item["complete"] and item["valid"] is not False
    ]
    fastest = min(eligible, key=lambda item: item["duration"]) if eligible else None

    data_log.channels[CH_LAP_NUMBER].set_samples(elapsed, lap_numbers)
    data_log.laps_info = {
        "beacons": beacons,
        "laps": laps,
        "total_laps": len(laps),
        "fastest_lap": fastest["lap_num"] if fastest else None,
        "fastest_time": fastest["duration"] if fastest else 0.0,
    }
