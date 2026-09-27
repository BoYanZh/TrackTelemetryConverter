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
    channels = {n: data_log.channels[n] for n in names}

    nonempty_times = [
        channel.timestamps for channel in channels.values()
        if len(channel.timestamps) > 0
    ]
    if not nonempty_times:
        return False
    timeline = np.unique(np.concatenate(nonempty_times))

    t0_wall = 0.0
    if wallclock and getattr(data_log, "datetime", None) is not None:
        t0_wall = data_log.datetime.timestamp()

    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        time_col = "Wall Time (s)" if wallclock else "Time (s)"
        writer.writerow([time_col] + names)

        idx = {name: 0 for name in names}
        for t in timeline:
            t_out = t + t0_wall if wallclock else t
            row = [f"{t_out:.3f}"]
            for name in names:
                ch = channels[name]
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
