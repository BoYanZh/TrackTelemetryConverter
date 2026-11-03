"""Read telemetry inputs for diagnostics without changing or exporting them."""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from types import SimpleNamespace
from typing import Any

from .diagnostics import build_diagnostic_report
from .input_loading import detect_input_type, load_log_data
from .log import DataLog
from .output import ensure_output_targets
from .parsers import SUPPORTED_LOG_TYPES
from .rcz_archive import discover_rcz_sessions, get_rcz_stints


def _error_report(
    code: str,
    message: str,
    *,
    source_format: str = "UNKNOWN",
    stage: str = "inspect",
) -> dict[str, Any]:
    report = build_diagnostic_report(DataLog(), source_format=source_format, stages=[])
    report["validation"]["valid"] = False
    report["validation"]["errors"] = [
        {"code": code, "stage": stage, "message": message},
    ]
    report["warnings"] = []
    return report


def inspect_file(
    source: str,
    source_format: str = "AUTO",
    *,
    session: str | None = None,
    stint: str | int = "all",
    lap: str | int = "all",
    min_lap_sec: float = 15.0,
    mask_interp_gaps: bool = False,
    dbc: str | None = None,
    track: str = "",
    anchor: str = "",
) -> dict[str, Any]:
    """Return diagnostics for one input without resampling or writing converted files."""
    source = os.path.expanduser(os.fspath(source))
    if os.path.isdir(source):
        return _error_report(
            "INSPECT_DIRECTORY_UNSUPPORTED",
            "Inspect accepts one telemetry file.",
        )
    if not os.path.isfile(source):
        return _error_report("INPUT_NOT_FOUND", "Input file does not exist.", stage="parse")

    source_format = str(source_format).upper()
    active_format = "UNKNOWN"
    selected_session = session
    requested_lap = None
    try:
        # Format sniffers and parsers historically write progress to stdout. Keep
        # that text out of the JSON interface and out of the diagnostic payload.
        with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
            active_format = detect_input_type(source, source_format)
            if active_format not in SUPPORTED_LOG_TYPES:
                return _error_report(
                    "INPUT_FORMAT_UNSUPPORTED",
                    "Input format is not supported.",
                    source_format=active_format,
                    stage="parse",
                )

            if active_format == "RCZ":
                if lap is not None and str(lap).lower() != "all":
                    try:
                        requested_lap = int(lap)
                    except (TypeError, ValueError):
                        return _error_report(
                            "RCZ_LAP_SELECTION_INVALID",
                            "Choose a numeric RCZ lap number or 'all'.",
                            source_format=active_format,
                            stage="parse.selection",
                        )
                    if requested_lap <= 0:
                        return _error_report(
                            "RCZ_LAP_SELECTION_INVALID",
                            "RCZ lap numbers must be positive integers.",
                            source_format=active_format,
                            stage="parse.selection",
                        )

                backup_sessions = discover_rcz_sessions(source)
                if backup_sessions:
                    if not session or str(session).lower() == "all":
                        return _error_report(
                            "RCZ_SESSION_SELECTION_REQUIRED",
                            "Choose one session ID to inspect.",
                            source_format=active_format,
                            stage="parse.session",
                        )
                    selected = next(
                        (item for item in backup_sessions if item.session_id == session),
                        None,
                    )
                    if selected is None:
                        return _error_report(
                            "RCZ_SESSION_UNKNOWN",
                            "The selected RCZ session does not exist.",
                            source_format=active_format,
                            stage="parse.session",
                        )
                elif session:
                    return _error_report(
                        "RCZ_SESSION_SELECTION_INVALID",
                        "Session selection is only valid for RCZ backup archives.",
                        source_format=active_format,
                        stage="parse.session",
                    )

                stints = get_rcz_stints(source, target_session=selected_session)
                if len(stints) > 1 and str(stint).lower() == "all":
                    return _error_report(
                        "RCZ_STINT_SELECTION_REQUIRED",
                        "Choose one stint to inspect.",
                        source_format=active_format,
                        stage="parse.session",
                    )
                if str(stint).lower() != "all":
                    try:
                        requested_stint = int(stint)
                    except (TypeError, ValueError):
                        return _error_report(
                            "RCZ_STINT_UNKNOWN",
                            "The selected RCZ stint does not exist.",
                            source_format=active_format,
                            stage="parse.session",
                        )
                    if requested_stint not in stints:
                        return _error_report(
                            "RCZ_STINT_UNKNOWN",
                            "The selected RCZ stint does not exist.",
                            source_format=active_format,
                            stage="parse.session",
                        )

            args = SimpleNamespace(
                log=source,
                log_type=active_format,
                lap=lap,
                min_lap_sec=min_lap_sec,
                mask_interp_gaps=mask_interp_gaps,
                dbc=dbc,
                track=track,
                anchor=anchor,
                session=selected_session,
                g_source="auto",
                gear_ratio_thresholds=None,
                gearbox=None,
            )
            data_log, _messages = load_log_data(
                args,
                active_format,
                str(stint),
                selected_session,
            )
    except SystemExit:
        return _error_report(
            "INPUT_PARSE_FAILED",
            "The input could not be parsed.",
            source_format=active_format,
            stage="parse",
        )
    except Exception as exc:
        from .errors import TelemetryParseError

        if isinstance(exc, TelemetryParseError):
            safe_messages = {
                "RCZ_SESSION_MISSING": "Invalid RCZ file, missing session.json.",
                "RCZ_TIMESTAMP_MISSING": "Could not find timestamp channel in RCZ file.",
            }
            message = safe_messages.get(exc.code, "The input could not be parsed.")
            stage = exc.stage
            code = exc.code
        else:
            code = "INPUT_READ_FAILED"
            stage = "parse"
            message = "The input could not be read."
        return _error_report(
            code, message, source_format=active_format, stage=stage,
        )

    if requested_lap is not None:
        available_laps = {
            int(item["lap_num"])
            for item in data_log.laps_info.get("laps", [])
            if item.get("lap_num") is not None
        }
        if requested_lap not in available_laps:
            return _error_report(
                "RCZ_LAP_UNKNOWN",
                "The selected RCZ lap does not exist in the selected stint.",
                source_format=active_format,
                stage="parse.selection",
            )

    return build_diagnostic_report(
        data_log,
        source_format=active_format,
        session=selected_session,
        stint=stint,
        lap=lap,
        stages=["parse"],
    )


def _protect_report_target(report_path: str, source_path: str) -> None:
    source_real = os.path.normcase(os.path.realpath(source_path))
    target_real = os.path.normcase(os.path.realpath(report_path))
    if source_real == target_real:
        raise ValueError("refusing to replace input file with debug report")
    if os.path.exists(report_path) and os.path.exists(source_path):
        try:
            if os.path.samefile(report_path, source_path):
                raise ValueError("refusing to replace input file with debug report")
        except OSError:
            pass


def write_debug_report(
    report: dict[str, Any],
    report_path: str,
    source_path: str,
    *,
    force: bool = False,
) -> None:
    """Write strict JSON, protecting the source and existing reports."""
    report_path = os.path.expanduser(os.fspath(report_path))
    source_path = os.path.expanduser(os.fspath(source_path))
    _protect_report_target(report_path, source_path)
    ensure_output_targets([report_path], force=force, source_path=source_path)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    parent = os.path.dirname(os.path.abspath(report_path)) or os.getcwd()

    if force:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="\n",
                dir=parent,
                prefix=".debug-report-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            _protect_report_target(report_path, source_path)
            os.replace(temp_path, report_path)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        descriptor = os.open(
            report_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
