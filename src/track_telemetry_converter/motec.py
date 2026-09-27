from __future__ import annotations

import datetime
import struct
from typing import TYPE_CHECKING

import numpy as np

from ._vendor.ldparser import ldChan, ldData, ldEvent, ldHead, ldVehicle, ldVenue

if TYPE_CHECKING:
    from .models import Channel


class MotecLog(object):
    """ Handles generating a MoTeC .ld file from log data.

    This utilizes the ldparser library for packing all the meta data and channel data into the
    correct binary format. Some functionality and information (e.g. pointer constants below) was
    missing from the ldparser library, so this class servers as a wrapper to fill in the gaps.

    This operates on containers from the data_log library.
    """
    # Fixed byte offsets for MoTeC LD v1.x binary container format.
    # The MoTeC LD binary file layout consists of fixed-offset metadata blocks
    # followed by an array of channel metadata headers and subsequent raw sample data:
    #   - 0..1761: ldHead (root header with pointers to event, venue, and channel list)
    #   - 1762: ldVehicle (vehicle metadata: id, weight, type, comments)
    #   - 5078: ldVenue (venue / track metadata)
    #   - 8180: ldEvent (session and event metadata)
    #   - 11336: First ldChan channel metadata descriptor
    #   - 11336 + N * CHANNEL_HEADER_SIZE: Start of sequential channel sample data payloads
    # These offsets conform to the reverse-engineered MoTeC binary container layout implemented in ldparser.
    VEHICLE_PTR = 1762
    VENUE_PTR = 5078
    EVENT_PTR = 8180
    HEADER_PTR = 11336

    CHANNEL_HEADER_SIZE = struct.calcsize(ldChan.fmt)

    ld_channels: list[ldChan]

    def __init__(self):
        self.driver = ""
        self.vehicle_id = ""
        self.vehicle_weight = 0
        self.vehicle_type = ""
        self.vehicle_comment = ""
        self.venue_name = ""
        self.event_name = ""
        self.event_session = ""
        self.long_comment = ""
        self.short_comment = ""
        self.datetime = datetime.datetime.now()

        # File components from ldparser
        self.ld_header = None
        self.ld_channels = []

    def initialize(self):
        """ Initializes all the meta data for the motec log.

        This must be called before adding any channel data.
        """
        ld_vehicle = ldVehicle(self.vehicle_id, self.vehicle_weight, self.vehicle_type, \
            self.vehicle_comment)
        ld_venue = ldVenue(self.venue_name, self.VEHICLE_PTR, ld_vehicle)
        ld_event = ldEvent(self.event_name, self.event_session, self.long_comment, \
            self.VENUE_PTR, ld_venue)

        self.ld_header = ldHead(self.HEADER_PTR, self.HEADER_PTR, self.EVENT_PTR, ld_event, \
            self.driver, self.vehicle_id, self.venue_name, self.datetime, self.short_comment)

    def add_channel(self, log_channel: Channel):
        dtype = np.float32 if log_channel.data_type is float else np.int32
        freq = int(round(log_channel.avg_frequency()))

        ld_channel = ldChan(
            None,
            meta_ptr=0,
            prev_meta_ptr=0,
            next_meta_ptr=0,
            data_ptr=0,
            data_len=len(log_channel.values),
            dtype=dtype,
            freq=freq,
            shift=0,
            mul=1,
            scale=1,
            dec=0,
            name=log_channel.name,
            short_name="",
            unit=log_channel.units,
        )
        ld_channel._data = log_channel.values.astype(dtype)
        self.ld_channels.append(ld_channel)

    def add_all_channels(self, data_log):
        for channel in data_log.channels.values():
            self.add_channel(channel)

    def _finalize_pointers(self):
        n = len(self.ld_channels)
        if n == 0:
            return

        meta_base = self.HEADER_PTR
        for i, ch in enumerate(self.ld_channels):
            ch.meta_ptr = meta_base + i * self.CHANNEL_HEADER_SIZE
            ch.prev_meta_ptr = meta_base + (i - 1) * self.CHANNEL_HEADER_SIZE if i > 0 else 0
            ch.next_meta_ptr = meta_base + (i + 1) * self.CHANNEL_HEADER_SIZE if i < n - 1 else 0

        data_ptr = meta_base + n * self.CHANNEL_HEADER_SIZE
        self.ld_header.data_ptr = data_ptr
        for ch in self.ld_channels:
            ch.data_ptr = data_ptr
            data_ptr += ch._data.nbytes

        # Validate pointer layout integrity before serialization
        assert self.ld_header.data_ptr >= self.HEADER_PTR + n * self.CHANNEL_HEADER_SIZE, (
            "Data pointer overlaps with channel headers"
        )
        for i, ch in enumerate(self.ld_channels):
            assert ch.meta_ptr == meta_base + i * self.CHANNEL_HEADER_SIZE
            if i < n - 1:
                assert ch.next_meta_ptr == ch.meta_ptr + self.CHANNEL_HEADER_SIZE

    def write(self, filename):
        if not self.ld_channels:
            with open(filename, "wb") as f:
                self.ld_header.write(f, 0)
            return

        self._finalize_pointers()
        ld_data = ldData(self.ld_header, self.ld_channels)
        ld_data.channs[-1].next_meta_ptr = 0
        ld_data.write(filename)

    def write_ldx(self, filename, laps_info=None, beacons=None):
        from .exporters.ldx_exporter import write_ldx
        return write_ldx(self, filename, laps_info=laps_info, beacons=beacons)
    @staticmethod
    def _find_gps_channels(data_log):
        from .exporters.gpx_export import _find_gps_channels
        return _find_gps_channels(data_log)
    @staticmethod
    def _downsample_step(channel, target_hz=5.0):
        from .exporters.gpx_export import _downsample_step
        return _downsample_step(channel, target_hz=target_hz)
    @staticmethod
    def _iter_gps_trackpoints(data_log, target_hz=5.0):
        from .exporters.gpx_export import _iter_gps_trackpoints
        yield from _iter_gps_trackpoints(data_log, target_hz=target_hz)
    def write_gpx(self, gpx_filename, data_log):
        from .exporters.gpx_export import write_gpx
        return write_gpx(self, gpx_filename, data_log)
    def write_kml(self, kml_filename, data_log):
        from .exporters.gpx_export import write_kml
        return write_kml(self, kml_filename, data_log)
    def write_csv(self, csv_filename, data_log, wallclock=False):
        from .exporters.csv_export import write_csv
        return write_csv(data_log, csv_filename, wallclock=wallclock)
