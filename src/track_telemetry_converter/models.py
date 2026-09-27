"""Pure data models for telemetry channels and messages."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .channels import DISCRETE_CHANNELS
from .interpolation import _interp_zoh, _mask_interp_gaps


class Message(object):
    """ A single message in a time series of data. """

    def __init__(self, timestamp: float = 0.0, value: float = 0.0):
        self.timestamp = float(timestamp)
        self.value = float(value)

    def __str__(self):
        return ("t=%f, value=%f" % (self.timestamp, self.value))


class MessageListProxy(Sequence[Message]):
    """Zero-copy / lazy list proxy over Channel timestamps and values arrays."""
    def __init__(self, channel: Channel):
        self._channel = channel

    def __len__(self):
        return len(self._channel._timestamps)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            ts_sub = self._channel._timestamps[idx]
            val_sub = self._channel._values[idx]
            return [Message(t, v) for t, v in zip(ts_sub, val_sub)]
        return Message(self._channel._timestamps[idx], self._channel._values[idx])

    def __iter__(self):
        for t, v in zip(self._channel._timestamps, self._channel._values):
            yield Message(t, v)

    def __bool__(self):
        return len(self._channel._timestamps) > 0

    def append(self, msg: Message):
        self._channel.add_sample(msg.timestamp, msg.value)

    def extend(self, msgs):
        if not msgs:
            return
        ts = [m.timestamp for m in msgs]
        vs = [m.value for m in msgs]
        self._channel.add_samples(ts, vs)


class Channel(object):
    """ Represents a single channel of data containing a time series of values."""
    def __init__(self, name, units, data_type, decimals, messages=None):
        self.name = str(name).strip()
        self.units = str(units)
        self.data_type = data_type
        self.decimals = decimals
        self._timestamps = np.array([], dtype=np.float64)
        self._values = np.array([], dtype=np.float64)

        if messages:
            self.messages = messages

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @timestamps.setter
    def timestamps(self, arr):
        self._timestamps = np.ascontiguousarray(arr, dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, arr):
        self._values = np.ascontiguousarray(arr, dtype=np.float64)

    def set_samples(self, timestamps, values):
        self._timestamps = np.ascontiguousarray(timestamps, dtype=np.float64)
        self._values = np.ascontiguousarray(values, dtype=np.float64)

    def add_sample(self, timestamp: float, value: float):
        self._timestamps = np.append(self._timestamps, float(timestamp))
        self._values = np.append(self._values, float(value))

    def add_samples(self, timestamps, values):
        if len(timestamps) == 0:
            return
        ts = np.ascontiguousarray(timestamps, dtype=np.float64)
        vs = np.ascontiguousarray(values, dtype=np.float64)
        if len(self._timestamps) == 0:
            self._timestamps = ts
            self._values = vs
        else:
            self._timestamps = np.concatenate([self._timestamps, ts])
            self._values = np.concatenate([self._values, vs])

    @property
    def messages(self):
        return MessageListProxy(self)

    @messages.setter
    def messages(self, msgs):
        if isinstance(msgs, MessageListProxy):
            return
        if not msgs:
            self._timestamps = np.array([], dtype=np.float64)
            self._values = np.array([], dtype=np.float64)
        elif isinstance(msgs, (list, tuple)):
            if len(msgs) > 0 and isinstance(msgs[0], Message):
                self._timestamps = np.array([m.timestamp for m in msgs], dtype=np.float64)
                self._values = np.array([m.value for m in msgs], dtype=np.float64)
            else:
                self.set_samples([getattr(m, 'timestamp', 0.0) for m in msgs], [getattr(m, 'value', 0.0) for m in msgs])

    def start(self):
        return float(self._timestamps[0]) if len(self._timestamps) > 0 else math.inf

    def end(self):
        return float(self._timestamps[-1]) if len(self._timestamps) > 0 else -math.inf

    def avg_frequency(self):
        if len(self._timestamps) >= 2:
            dt = self.end() - self.start()
            return len(self._timestamps) / dt if dt > 0 else 0.0
        else:
            return 0.0

    def resample(self, start_time, end_time, frequency, mask_interp_gaps=False):
        if len(self._timestamps) == 0:
            return

        num_msgs = math.floor(frequency * (end_time - start_time))
        if num_msgs < 1:
            return
        dt_step = 1.0 / frequency

        src_t = self._timestamps
        src_v = self._values
        new_t = start_time + dt_step * np.arange(num_msgs)

        if self.name in DISCRETE_CHANNELS:
            new_v = _interp_zoh(new_t, src_t, src_v)
        else:
            new_v = np.interp(new_t, src_t, src_v)
            if mask_interp_gaps:
                new_v = _mask_interp_gaps(new_v, new_t, src_t)

        self._timestamps = new_t
        self._values = new_v

    def __str__(self):
        s = "Channel: %s, Units: %s, Decimals: %d, Messages: %d, Frequency: %.2f Hz"
        return s % (self.name, self.units, self.decimals, len(self._timestamps), self.avg_frequency())
