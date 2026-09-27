"""Parser for ACCESSPORT telemetry log files.

Extracted from the original DataLog.accessport_log methods with identical behavior."""

from __future__ import annotations


def parse_accessport_log(data_log, log_lines):
    """ Creates channels populated with messages from a COBB Accessport CSV log file.

    This will create a channel for each column in the CSV file, with the name and units of that
    channel taken from the CSV header. Any non numeric data will be ignored, and that channel
    will be removed.

    log_lines: List, containing CSV log lines
    """

    from .csv_parser import parse_csv_log

    parse_csv_log(data_log, log_lines)

    # Accessport logs have a column for AP info which is not of any value so we'll delete it
    for key in data_log.channels.keys():
        if "AP Info" in key:
            del data_log.channels[key]
            break

    # Update all the channel names and units
    for channel_name, channel in data_log.channels.items():
        # Channels have the format "Name (Units)"
        name, units = channel_name.split(" (")
        units = units[:-1]

        channel.name = name
        channel.units = units
