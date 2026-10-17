"""Shared channel-provenance helpers for telemetry parsers.

Every parser collapses vendor-specific raw sources into canonical MoTeC
channels. Without a record of that collapse, three XRK-class bugs recur in
other formats: pedal-vs-throttle relabeling, silent unit changes, and
order-dependent winners when several raw columns map to one channel.

Helpers here mirror the XRK parser's ``channel_provenance`` convention so
all formats record the same facts instead of dropping them.
"""

from __future__ import annotations

from .channels import CH_YAW_RATE


def record_channel_provenance(data_log, out_name, raw_name, **fields):
    """Record which raw source produced a canonical channel.

    Extra keyword fields (e.g. ``power_semantics``, ``semantics_basis``,
    ``source_units``) are merged into the channel's record.
    """
    provenance = data_log.metadata.setdefault("channel_provenance", {})
    record = provenance.setdefault(out_name, {"raw_name": raw_name})
    record.update(fields)
    return record


def record_dropped_duplicate(data_log, out_name, kept_raw, dropped_raw):
    """Record a raw column skipped because its canonical channel was taken."""
    dropped = data_log.metadata.setdefault("dropped_duplicate_columns", [])
    dropped.append({
        "canonical": out_name,
        "kept_raw": kept_raw,
        "dropped_raw": dropped_raw,
    })
    return dropped


def record_yaw_source(data_log, raw_name, *, verified=False):
    """Record which raw source supplied yaw rate and its sign status.

    ``verified`` is True only when the sign convention was checked against
    a documented vehicle reference; otherwise it stays False so a later
    reader knows the axis direction is unconfirmed.
    """
    if "yaw_rate_source" not in data_log.metadata:
        data_log.metadata["yaw_rate_source"] = raw_name
    data_log.metadata["yaw_axis_sign_verified"] = bool(verified)
    record_channel_provenance(data_log, CH_YAW_RATE, raw_name)
