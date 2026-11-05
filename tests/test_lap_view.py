"""Sample-window behavior for shared native channel views."""

from types import MappingProxyType

import numpy as np

from track_telemetry_converter.lap_view import ChannelView, readonly


def test_ordered_view_slices_share_values_and_exact_source_clock():
    times = np.array([0.0, 0.1, 0.1, 0.2])
    values = np.array([10, 11, 12, 13], dtype=np.int32)
    source = np.array([1000, 1100, 1100, 1200], dtype=np.int64)
    view = ChannelView(
        "sensor", "count", readonly(times), readonly(values), 10.0,
        MappingProxyType({}), readonly(source),
    )

    selected = view.between(0.1, 0.2)

    np.testing.assert_array_equal(selected.values, [11, 12])
    np.testing.assert_array_equal(selected.source_timestamps, [1100, 1100])
    assert np.shares_memory(selected.values, values)
    assert np.shares_memory(selected.source_timestamps, source)
    assert not selected.values.flags.writeable
    assert selected.source_timestamps is not None
    assert not selected.source_timestamps.flags.writeable
    assert times.flags.writeable and values.flags.writeable and source.flags.writeable
    inclusive = view.between(0.1, 0.2, include_end=True)
    np.testing.assert_array_equal(inclusive.values, [11, 12, 13])


def test_unordered_view_preserves_order_duplicates_and_source_clock_alignment():
    view = ChannelView(
        "sensor", "count", readonly(np.array([0.4, 0.2, 0.1, 0.1, np.nan, np.inf])),
        readonly(np.array([40, 20, 10, 11, 99, 98])), None, MappingProxyType({}),
        readonly(np.array([1400, 1200, 1100, 1100, -1, -2])), False,
    )

    selected = view.between(0.1, 0.2, include_end=True)

    np.testing.assert_array_equal(selected.timestamps, [0.2, 0.1, 0.1])
    np.testing.assert_array_equal(selected.values, [20, 10, 11])
    np.testing.assert_array_equal(selected.source_timestamps, [1200, 1100, 1100])
    assert not selected.values.flags.writeable
    assert selected.timestamps_monotonic is False
    all_timed = view.between(-np.inf, np.inf, include_end=True)
    np.testing.assert_array_equal(all_timed.values, [40, 20, 10, 11])
