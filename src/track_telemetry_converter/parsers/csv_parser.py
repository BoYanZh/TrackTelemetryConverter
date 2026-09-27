"""Parser for CSV telemetry log files.

Extracted from the original DataLog.csv_log methods with identical behavior."""

from __future__ import annotations


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

    # We'll keep a map of names and column numbers for easy channel lookups when parsing rows
    i = 0
    channel_dict = {}
    for name in channel_names:
        data_log.add_channel(name, "", float, 0)

        channel_dict[name] = i
        i += 1

    chan_buffers = {name: ([], []) for name in channel_names}

    # Go through each line grabbing all the channel values
    for line in log_lines[data_header_idx + 1:]:
        line = line.strip("\n")
        values = line.split(",")

        # Timestamp is the first element
        try:
            t = float(values[0])
        except ValueError:
            continue

        invalid_channels = []
        for name, i in channel_dict.items():
            try:
                val = float(values[i + 1])
                ts_buf, val_buf = chan_buffers[name]
                ts_buf.append(t)
                val_buf.append(val)

                val_text_split = values[i + 1].split(".")
                decimals_present = 0 if len(val_text_split) == 1 else len(val_text_split[1])
                data_log.channels[name].decimals = max(decimals_present, data_log.channels[name].decimals)
            except ValueError:
                print("WARNING: Found non numeric values for channel %s, removing channel" % name)
                invalid_channels.append(name)

        for name in invalid_channels:
            del channel_dict[name]
            del data_log.channels[name]
            if name in chan_buffers:
                del chan_buffers[name]

    for name, (ts_buf, val_buf) in chan_buffers.items():
        if ts_buf and name in data_log.channels:
            data_log.channels[name].set_samples(ts_buf, val_buf)
