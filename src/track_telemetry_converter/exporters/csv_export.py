"""CSV telemetry export for TrackTelemetryConverter.

Writes all resampled channels as a tabular CSV with a header row of
canonical MoTeC channel names.  Designed to be imported into AIM
RaceStudio / RaceChrono RaceStudio CSV import wizards (column mapping
is done manually in the target application).

Time is exported as elapsed seconds since log start by default, or as
the log's wall-clock timestamp when --csv-wallclock is passed.
"""

from __future__ import annotations

import csv

import numpy as np

from ..channels import CH_GPS_LATITUDE, CH_GPS_LONGITUDE


def _ordered_channels(data_log):
    """Return channel names ordered by sample rate (highest first),
    with GPS coordinates moved to the front for easy mapping."""
    names = sorted(
        data_log.channels,
        key=lambda n: -data_log.channels[n].avg_frequency(),
    )
    priority = [
        name for name in (CH_GPS_LATITUDE, CH_GPS_LONGITUDE)
        if name in names
    ]
    return priority + [name for name in names if name not in priority]


def write_csv(data_log, csv_filename, wallclock=False):
    """Write all DataLog channels to a CSV file.

    Returns True on success, False if no channels are available.
    """
    if not data_log.channels:
        return False

    names = _ordered_channels(data_log)
    arrays = [
        (data_log.channels[n].timestamps, data_log.channels[n].values)
        for n in names
    ]
    nonempty = [(ts, vs) for ts, vs in arrays if len(ts) > 0]
    if not nonempty:
        return False

    # Fast path: every channel shares one strictly-increasing time grid
    # (always true right after DataLog.resample, which is when the CLI
    # exports). Then each row is a straight column slice formatted in C by
    # np.savetxt ('%.6g' matches f"{v:.6g}", including nan/inf/-0
    # spellings), in row chunks to bound memory. Falls back to the general
    # union-timeline loop below when grids differ or a channel is empty.
    grid = nonempty[0][0]
    common_grid = (
        len(grid) > 0
        and len(nonempty) == len(names)
        and bool(np.all(np.diff(grid) > 0))
        and all(
            len(ts) == len(grid) and np.array_equal(ts, grid)
            for ts, _ in nonempty
        )
    )

    t0_wall = 0.0
    if wallclock and getattr(data_log, "datetime", None) is not None:
        t0_wall = data_log.datetime.timestamp()

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        time_col = "Wall Time (s)" if wallclock else "Time (s)"
        writer.writerow([time_col] + names)

        if common_grid:
            fmts = ["%.3f"] + ["%.6g"] * len(nonempty)
            for start in range(0, len(grid), 50000):
                end = min(start + 50000, len(grid))
                chunk = np.empty((end - start, len(nonempty) + 1), dtype=np.float64)
                chunk[:, 0] = grid[start:end] + t0_wall
                for j, (_, vs) in enumerate(nonempty):
                    chunk[:, j + 1] = vs[start:end]
                np.savetxt(f, chunk, fmt=fmts, delimiter=",", newline="\r\n")
            return True

        timeline = np.unique(np.concatenate([ts for ts, _ in nonempty]))

        idx = {name: 0 for name in names}
        for t in timeline:
            t_out = t + t0_wall if wallclock else t
            row = [f"{t_out:.3f}"]
            for name in names:
                ch = data_log.channels[name]
                timestamps = ch.timestamps
                j = idx[name]
                if len(timestamps) == 0 or t < timestamps[0]:
                    row.append("")
                    continue
                # forward-fill: advance while next message time <= current
                while j + 1 < len(timestamps) and timestamps[j + 1] <= t:
                    j += 1
                idx[name] = j
                row.append(f"{ch.values[j]:.6g}")
            writer.writerow(row)

    return True
