"""Read-only, unresampled views of DataLog samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


def readonly(array):
    """Return a non-writable view without changing the owner's array flags."""
    view = array.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class ChannelView:
    """Native samples with units, rate and source metadata, including IBT arrays."""

    name: str
    units: str
    timestamps: np.ndarray
    values: np.ndarray
    sample_rate: float | None
    provenance: Mapping[str, Any]
    source_timestamps: np.ndarray | None = None
    timestamps_monotonic: bool = True

    def between(self, start, end, *, include_end=False):
        if self.timestamps_monotonic:
            left = self.timestamps.searchsorted(start, side="left")
            right = self.timestamps.searchsorted(end, side="right" if include_end else "left")
            selection = slice(left, right)
        else:
            within_end = self.timestamps <= end if include_end else self.timestamps < end
            selection = np.isfinite(self.timestamps) & (self.timestamps >= start) & within_end
        return ChannelView(
            self.name, self.units, readonly(self.timestamps[selection]),
            readonly(self.values[selection]), self.sample_rate, self.provenance,
            readonly(self.source_timestamps[selection]) if self.source_timestamps is not None else None,
            self.timestamps_monotonic,
        )


@dataclass(frozen=True)
class LapView:
    """One recorded lap segment; times retain the session's elapsed origin."""

    info: Mapping[str, Any]
    channels: Mapping[str, ChannelView]
    raw_channels: Mapping[str, ChannelView]


def make_lap_view(log, info, channels=None, raw_channels=None):
    if isinstance(channels, str):
        channels = (channels,)
    if isinstance(raw_channels, str):
        raw_channels = (raw_channels,)
    start, end = info["start_time"], info["end_time"]
    include_end = info is log.laps_info.get("laps", [None])[-1]
    selected = {}
    for name in log.channels if channels is None else channels:
        if name not in log.channels:
            continue
        channel = log.channels[name]
        meta = log.channel_metadata.get(name, {})
        view = ChannelView(
            name, channel.units, readonly(channel.timestamps), readonly(channel.values),
            meta.get("sample_rate"), MappingProxyType(dict(meta)),
        )
        selected[name] = view.between(start, end, include_end=include_end)
    raw = {
        name: log.raw_channels[name].between(start, end, include_end=include_end)
        for name in log.raw_channels if raw_channels is None or name in raw_channels
    }
    return LapView(
        MappingProxyType(dict(info)), MappingProxyType(selected), MappingProxyType(raw),
    )
