"""Parser for CSV telemetry log files.

Extracted from the original DataLog.csv_log methods with identical behavior."""

from __future__ import annotations

import numpy as np

#: Rows per chunk in the vectorized data pass. Bounds the transient
#: split-string table (~rows x columns Python strings) while keeping
#: numpy conversions large enough to amortize call overhead.
_CSV_CHUNK_ROWS = 20_000


def _max_text_decimals(cells):
    """Max decimals over the raw cell texts (legacy ``split(".")`` semantics).

    ``"2.50"`` -> 2, ``"3"`` -> 0, ``"1.5e-3"`` -> 4: everything after the
    last ``"."`` counts, exactly like the legacy per-cell scan. ``cells``
    is a unicode numpy array (already built for the float conversion).
    """
    if len(cells) == 0:
        return 0
    parts = np.char.rpartition(cells, ".")
    sep = parts[:, 1]
    tail = parts[:, 2]
    has_dot = sep == "."
    if not np.any(has_dot):
        return 0
    return int(np.char.str_len(tail[has_dot]).max())


def parse_csv_log(data_log, log_lines):
    """ Creates channels populated with messages from a CSV log file.

    This will create a channel for each column in the CSV file, with the name of that channel
    taken from the CSV header. All channels will be created without any units. Any non numeric data
    will be ignored, and that channel will be removed. The first column of data must be time

    log_lines: List, containing CSV log lines
    """
    data_log.clear()

    # 1. Dynamically locate Data Header Line
    data_header_idx = 0
    import csv
    for i, line in enumerate(log_lines):
        line_clean = line.strip().strip('"').strip("'")
        if not line_clean:
            continue
        try:
            parts = [p.strip().strip('"').strip("'") for p in next(csv.reader([line_clean]))]
        except Exception:
            parts = [p.strip().strip('"').strip("'") for p in line_clean.split(",")]
        if len(parts) >= 2 and parts[0].lower() in ("time", "time (s)", "timestamp"):
            data_header_idx = i
            break

    # Get the channel names, ignore the first column as it is assumed to be time
    header = log_lines[data_header_idx].strip("\n")
    channel_names = [name.strip().strip('"').strip("'") for name in header.split(",")[1:]]

    # MoTeC channel names are limited to 32 characters; longer header fields
    # are file metadata (e.g. COBB "AP Info" trailers), not data channels.
    kept_names = []
    for name in channel_names:
        if len(name) > 32:
            print(f"WARNING: Ignoring over-long CSV column (file metadata, not data): {name[:48]}...")
            continue
        kept_names.append(name)
    channel_names = kept_names

    # We'll keep a map of names and column numbers for easy channel lookups when parsing rows.
    # A duplicate header keeps the LAST column (dict overwrite), matching legacy behavior.
    i = 0
    channel_dict = {}
    for name in channel_names:
        data_log.add_channel(name, "", float, 0)

        channel_dict[name] = i
        i += 1

    # 2. Vectorized data pass over chunks (C-speed float conversion per
    # column). Outcomes match the legacy row loop exactly: rows with a
    # non-numeric time are skipped; a column with any non-numeric cell is
    # dropped entirely (with one WARNING); decimals are the max over the
    # raw cell texts of the kept rows.
    columns = list(channel_dict.items())
    n_cols = len(columns)
    alive = [True] * n_cols
    col_parts = [[] for _ in range(n_cols)]
    ts_parts = []
    decimals = [0] * n_cols
    total_rows = len(log_lines)

    for start in range(data_header_idx + 1, total_rows, _CSV_CHUNK_ROWS):
        chunk = log_lines[start:start + _CSV_CHUNK_ROWS]
        table = [row.strip("\n").split(",") for row in chunk]

        keep = []
        ts_vals = []
        for k, row in enumerate(table):
            try:
                ts_vals.append(float(row[0]))
            except ValueError:
                continue
            keep.append(k)
        if not keep:
            continue
        ts_parts.append(np.asarray(ts_vals, dtype=np.float64))

        for pos, (name, col) in enumerate(columns):
            if not alive[pos]:
                continue
            # Short rows raise IndexError here, as legacy `values[i + 1]` did.
            str_arr = np.asarray([table[k][col + 1] for k in keep])
            try:
                arr = str_arr.astype(np.float64)
            except ValueError:
                print(f"WARNING: Found non numeric values for channel {name}, removing channel")
                alive[pos] = False
                col_parts[pos] = []
                continue
            col_parts[pos].append(arr)
            chunk_dec = _max_text_decimals(str_arr)
            if chunk_dec > decimals[pos]:
                decimals[pos] = chunk_dec

    for pos, (name, _col) in enumerate(columns):
        if not alive[pos]:
            # Dropped on the first non-numeric cell, buffered values discarded.
            if name in data_log.channels:
                del data_log.channels[name]
            continue
        if ts_parts and col_parts[pos]:
            data_log.channels[name].set_samples(
                np.concatenate(ts_parts), np.concatenate(col_parts[pos]),
            )
            if decimals[pos] > data_log.channels[name].decimals:
                data_log.channels[name].decimals = decimals[pos]
        # No samples: the empty channel stays, exactly like the legacy loop.
