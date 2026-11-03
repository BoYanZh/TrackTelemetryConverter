"""Privacy-conscious, JSON-safe summaries of parsed telemetry logs."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .log import DataLog

_GAP_THRESHOLD_SECONDS = 1.0


def _channel_diagnostics(channel: Any) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """Summarize one channel without exposing its sample values or mutating it."""
    timestamps = np.asarray(channel.timestamps, dtype=np.float64)
    values = np.asarray(channel.values, dtype=np.float64)
    finite_times = np.isfinite(timestamps)
    finite_values = np.isfinite(values)
    valid_times = timestamps[finite_times]

    # Finite endpoints can still have an unrepresentably large delta, while
    # tiny positive deltas can overflow when converted to frequency below.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        differences = np.diff(timestamps)
    adjacent_finite = finite_times[:-1] & finite_times[1:]
    adjacent_differences = differences[adjacent_finite]
    positive_differences = adjacent_differences[adjacent_differences > 0]
    gaps = positive_differences[positive_differences > _GAP_THRESHOLD_SECONDS]
    finite_positive_differences = positive_differences[np.isfinite(positive_differences)]
    duplicate_count = int(np.count_nonzero(adjacent_differences == 0))
    nonmonotonic_count = int(np.count_nonzero(adjacent_differences < 0))

    sample_count = int(timestamps.size)
    invalid_value_count = int(values.size - np.count_nonzero(finite_values))
    time_invalid_count = int(sample_count - np.count_nonzero(finite_times))
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    def issue(target: list[dict[str, str]], code: str) -> None:
        target.append({"code": code, "channel": str(channel.name)})

    if sample_count == 0 or values.size == 0:
        issue(errors, "CHANNEL_EMPTY")
    if timestamps.size != values.size:
        issue(errors, "CHANNEL_LENGTH_MISMATCH")
    if time_invalid_count:
        issue(errors, "CHANNEL_NONFINITE_TIMESTAMPS")
    if duplicate_count:
        issue(errors, "CHANNEL_DUPLICATE_TIMESTAMPS")
    if nonmonotonic_count:
        issue(errors, "CHANNEL_NONMONOTONIC_TIMESTAMPS")
    if invalid_value_count:
        issue(warnings, "CHANNEL_NONFINITE_VALUES")
    if gaps.size:
        issue(warnings, "CHANNEL_TIME_GAPS")

    frequency_hz = None
    if finite_positive_differences.size:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            candidate_frequency = float(1.0 / np.median(finite_positive_differences))
        if np.isfinite(candidate_frequency):
            frequency_hz = candidate_frequency
    max_gap = None
    if gaps.size and np.all(np.isfinite(gaps)):
        max_gap = float(np.max(gaps))
    report = {
        "name": str(channel.name),
        "units": str(channel.units),
        "sample_count": sample_count,
        "frequency_hz": frequency_hz,
        "time_min_s": float(np.min(valid_times)) if valid_times.size else None,
        "time_max_s": float(np.max(valid_times)) if valid_times.size else None,
        "invalid_fraction": float(invalid_value_count / values.size) if values.size else 0.0,
        "time_invalid_fraction": float(time_invalid_count / sample_count) if sample_count else 0.0,
        "gaps": {
            "count": int(gaps.size),
            "max_s": max_gap,
            "threshold_s": _GAP_THRESHOLD_SECONDS,
        },
        "duplicate_timestamps": duplicate_count,
        "nonmonotonic_timestamps": nonmonotonic_count,
        "length_mismatch": timestamps.size != values.size,
    }
    return report, warnings, errors


def build_diagnostic_report(
    data_log: DataLog,
    *,
    source_format: str,
    session: str | int | None = None,
    stint: str | int | None = None,
    lap: str | int | None = None,
    stages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a stable diagnostics report without including raw telemetry data."""
    channels = []
    warnings: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for channel in sorted(data_log.channels.values(), key=lambda item: item.name):
        report, channel_warnings, channel_errors = _channel_diagnostics(channel)
        channels.append(report)
        warnings.extend(channel_warnings)
        errors.extend(channel_errors)

    if not channels:
        errors.append({"code": "NO_CHANNELS"})

    return {
        "schema_version": 1,
        "source": {"format": str(source_format)},
        "selection": {"session": session, "stint": stint, "lap": lap},
        "timebase": {
            "unit": "s",
            "origin_epoch_ms": data_log.time_origin_epoch_ms,
        },
        "channels": channels,
        "stages": list(stages) if stages is not None else ["parse"],
        "warnings": sorted(warnings, key=lambda item: (item["channel"], item["code"])),
        "validation": {"valid": not errors, "errors": errors},
    }
