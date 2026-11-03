"""Shared telemetry input loading for inspect and conversion."""
from __future__ import annotations

import contextlib
import io
import json
import os
from collections.abc import Callable

from .log import DataLog
from .parsers import auto_detect_log_type


def _legacy_parse(emit: Callable[[str], None], callback, *args, **kwargs):
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            return callback(*args, **kwargs)
    except SystemExit as exc:
        raise ValueError(f"Legacy parser exited with code {exc.code}") from exc
    finally:
        for line in output.getvalue().splitlines():
            emit(line)


def _read_input_lines(log_path):
    with open(log_path, encoding="utf-8", errors="ignore") as file:
        return file.readlines()


_TEXT_LOG_READERS = {
    "CSV": ("Extracting data...", "from_csv_log", False),
    "ACCESSPORT": ("Extracting data...", "from_accessport_log", False),
    "RACECHRONO": ("Extracting data...", "from_racechrono_log", True),
    "PBBUDDY": ("Extracting PB Buddy data...", "from_pbbuddy_log", True),
    "TRACKADDICT": ("Extracting TrackAddict data...", "from_trackaddict_log", True),
    "TESLA": ("Extracting Tesla Track Mode data...", "from_tesla_log", True),
    "VBO": ("Extracting Racelogic VBO data...", "from_vbo_log", True),
}
_PATH_LOG_READERS = {
    "FIT": ("Extracting Garmin .fit telemetry...", "from_fit_log", True),
    "XRK": ("Extracting AIM XRK/XRZ telemetry...", "from_xrk_log", True),
    "DLAP": ("Extracting Dragy DLAP data...", "from_dlap_log", True),
    "IBT": ("Extracting iRacing .ibt telemetry...", "from_ibt_log", False),
}


def _load_anchor_file(anchor_path):
    if not anchor_path:
        return None
    if not os.path.isfile(anchor_path):
        raise ValueError(f"Anchor file {anchor_path} does not exist")
    try:
        with open(anchor_path, encoding="utf-8") as handle:
            profiles = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid anchor file {anchor_path}: {exc}") from exc
    if not isinstance(profiles, dict):
        raise ValueError(f"Anchor file {anchor_path} must contain a JSON object")
    return profiles


def extract_log_data(args, active_type, data_log, stint_arg, target_session, *, emit=lambda _message: None):
    """Populate a DataLog and emit parser progress/warnings as plain messages."""
    emit("Loading log...")
    if active_type == "CAN":
        try:
            import cantools
        except ImportError as exc:
            raise ValueError("'cantools' package is required for CAN log processing. Install with: pip install cantools") from exc
        if not os.path.isfile(args.dbc or ""):
            raise ValueError(f"DBC file {args.dbc} does not exist")
        emit("Loading DBC...")
        can_db = cantools.database.load_file(args.dbc)
        emit("Extracting data...")
        _legacy_parse(emit, data_log.from_can_log, _read_input_lines(args.log), can_db)
        return
    if active_type == "RCZ":
        emit("Extracting RCZ data directly...")
        _legacy_parse(
            emit,
            data_log.from_rcz_log,
            args.log,
            target_lap=args.lap,
            target_stint=stint_arg,
            min_lap_sec=args.min_lap_sec,
            mask_interp_gaps=args.mask_interp_gaps,
            target_session=target_session,
        )
        return
    if active_type == "ACTI":
        from .parsers.acti_parser import ACTI_TRACK_PROFILES, THUNDERHILL_AUTO_KEY

        anchors = _load_anchor_file(args.anchor)
        profiles = dict(ACTI_TRACK_PROFILES)
        if anchors:
            for key, cfg in anchors.items():
                merged = dict(profiles.get(key, {}))
                merged.update(cfg)
                profiles[key] = merged
        valid_tracks = [*sorted(profiles), THUNDERHILL_AUTO_KEY]
        if not args.track or args.track not in [*profiles, THUNDERHILL_AUTO_KEY]:
            raise ValueError("ACTI input needs --track with one of: " + ", ".join(valid_tracks))
        if args.track == THUNDERHILL_AUTO_KEY:
            emit("Extracting ACTI sim telemetry (track profile: thunderhill_auto; per-file Car Coord X probe selects thunderhill_west/east_bypass)...")
        else:
            emit(f"Extracting ACTI sim telemetry (track profile: {args.track})...")
        _legacy_parse(
            emit,
            data_log.from_acti_log,
            args.log,
            target_lap=args.lap,
            track=args.track,
            anchors=profiles,
        )
        return
    if active_type in _TEXT_LOG_READERS:
        message, method, use_lap = _TEXT_LOG_READERS[active_type]
        emit(message)
        lines = _read_input_lines(args.log)
        callback = getattr(data_log, method)
        if use_lap:
            _legacy_parse(emit, callback, lines, target_lap=args.lap)
        else:
            _legacy_parse(emit, callback, lines)
        return
    if active_type in _PATH_LOG_READERS:
        message, method, use_lap = _PATH_LOG_READERS[active_type]
        emit(message)
        callback = getattr(data_log, method)
        if use_lap:
            _legacy_parse(emit, callback, args.log, target_lap=args.lap)
        else:
            _legacy_parse(emit, callback, args.log)
        return
    raise ValueError(f"Unsupported input type: {active_type}")


def load_log_data(args, active_type, stint_arg, target_session):
    """Parse one file and return (DataLog, ordered progress messages)."""
    messages = []
    data_log = DataLog()
    extract_log_data(
        args,
        active_type,
        data_log,
        stint_arg,
        target_session,
        emit=messages.append,
    )
    return data_log, messages



def detect_input_type(source, requested_type, *, emit=lambda _message: None):
    """Resolve AUTO and filename overrides while preserving sniff warnings."""
    requested_type = str(requested_type).upper()
    if requested_type == "AUTO" or str(source).lower().endswith(".rcz"):
        if str(source).lower().endswith(".rcz"):
            return "RCZ"
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                active_type = auto_detect_log_type(source)
        finally:
            for line in output.getvalue().splitlines():
                emit(line)
        if requested_type == "AUTO":
            emit(f"Auto-detected log type: {active_type}")
        return active_type
    return requested_type
