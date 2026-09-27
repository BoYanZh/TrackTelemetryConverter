from __future__ import annotations

import numpy as np

# Largest sane sensor dropout interval (in milliseconds) before treating missing data
# as an intentional logging gap / dropout that should be masked as NaN.
DEFAULT_GAP_THRESHOLD_MS: float = 1000.0


def _interp_zoh(times_target, times_src, values_src):
    idx = np.searchsorted(times_src, times_target, side="right") - 1
    idx = np.clip(idx, 0, len(values_src) - 1)
    return values_src[idx]


def _mask_interp_gaps(values, times_target, times_src, gap_threshold_ms=DEFAULT_GAP_THRESHOLD_MS):
    """Set interpolated values to NaN where consecutive source samples are
    separated by more than gap_threshold_ms (dropped frames / gap artifacts).
    """
    if len(times_src) < 2:
        return values
    gap_ms = np.diff(times_src) * 1000.0
    gap_idx = np.where(gap_ms > gap_threshold_ms)[0]
    if len(gap_idx) == 0:
        return values
    mask = np.zeros(len(values), dtype=bool)
    for i in gap_idx:
        t0, t1 = times_src[i], times_src[i + 1]
        mask |= (times_target > t0) & (times_target < t1)
    values = np.array(values, dtype=np.float64, copy=True)
    values[mask] = np.nan
    return values
