"""Parser for CAN telemetry log files.

Extracted from the original DataLog.can_log methods with identical behavior."""

from __future__ import annotations

from ..models import Message


def _parse_can_log_line(line):
    try:
        stamp, bus, msg = line.split()
        stamp = float(stamp[1:-1])
        can_id, data = msg.split("#")
        can_id = int(can_id, 16)
        data = bytearray.fromhex(data)
        return stamp, bus, can_id, data
    except (ValueError, IndexError):
        return None, None, None, None


def parse_can_log(data_log, log_lines, can_db):
    """ Creates channels populated with messages from a candump file and can database.

    This will create a channel for each entry in the database that has messages present in the
    log.

    log_lines: List, containing candump log lines (recorded with 'candump' with '-l')
    can_db: cantools.database
    """
    data_log.clear()
    data_log.datetime = data_log._extract_datetime_from_text(log_lines)

    # Cache all the frame ids in the database for quick lookups
    known_ids = set()
    for msg in can_db.messages:
        known_ids.add(msg.frame_id)

    for line in log_lines:
        stamp, bus, id, data = _parse_can_log_line(line)
        if stamp is None:
            continue

        if id not in known_ids:
            continue

        db_msg = can_db.get_message_by_frame_id(id)
        msg_decoded = can_db.decode_message(id, data)

        for msg, signal in zip(msg_decoded.items(), db_msg.signals):
            name = msg[0]
            value = msg[1]

            if name in data_log.channels:
                data_log.channels[name].messages.append(Message(stamp, value))
            else:
                data_log.add_channel(name, signal.unit, float, 3, Message(stamp, value))
