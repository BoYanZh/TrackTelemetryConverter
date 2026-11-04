"""Application service for verified telemetry conversion."""
from __future__ import annotations

import argparse
import os
import zipfile
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

from .diagnostics import build_diagnostic_report
from .errors import TelemetryParseError
from .input_loading import detect_input_type, load_log_data
from .motec import MotecLog
from .output import atomic_write_motec_pair, ensure_output_targets
from .rcz_archive import discover_rcz_sessions, get_rcz_stints


def _normalize_venue(name):
    from .metadata import normalize_venue

    return normalize_venue(name)

def _output_paths(args, out_base):
    if out_base:
        ld_filename = os.path.splitext(out_base)[0] + ".ld"
    else:
        input_dir, input_filename = os.path.split(args.log)
        input_stem = os.path.splitext(input_filename)[0]
        ld_filename = os.path.join(input_dir, input_stem + ".ld")
    ldx_filename = os.path.splitext(ld_filename)[0] + ".ldx"
    targets = [ld_filename, ldx_filename]
    if getattr(args, "gpx", False):
        targets.append(os.path.splitext(ld_filename)[0] + ".gpx")
    if getattr(args, "kml", False):
        targets.append(os.path.splitext(ld_filename)[0] + ".kml")
    if getattr(args, "csv", False):
        targets.append(_csv_output_path(ld_filename, args.log))
    if getattr(args, "vbo", False):
        targets.append(_vbo_output_path(ld_filename, args.log))
    if getattr(args, "dlap", False):
        targets.append(_dlap_output_path(ld_filename, args.log))
    return ld_filename, ldx_filename, targets


def _csv_output_path(ld_filename, source_path):
    csv_filename = os.path.splitext(ld_filename)[0] + ".csv"
    if os.path.normcase(os.path.abspath(csv_filename)) == os.path.normcase(
        os.path.abspath(source_path),
    ):
        csv_filename = os.path.splitext(ld_filename)[0] + "_export.csv"
    return csv_filename


def _vbo_output_path(ld_filename, source_path):
    vbo_filename = os.path.splitext(ld_filename)[0] + ".vbo"
    if os.path.normcase(os.path.abspath(vbo_filename)) == os.path.normcase(
        os.path.abspath(source_path),
    ):
        vbo_filename = os.path.splitext(ld_filename)[0] + "_export.vbo"
    return vbo_filename


def _dlap_output_path(ld_filename, source_path):
    dlap_filename = os.path.splitext(ld_filename)[0] + ".dlap"
    if os.path.normcase(os.path.abspath(dlap_filename)) == os.path.normcase(
        os.path.abspath(source_path),
    ):
        dlap_filename = os.path.splitext(ld_filename)[0] + "_export.dlap"
    return dlap_filename


def _direction_suffix_applies(active_type, data_log):
    """Whether --direction-suffix may rename this conversion.

    ACTI tracks only take ``_cw``/``_ccw`` when their resolved profile is
    marked bidirectional (shipped: thunderhill_west, buttonwillow_1a_ccw;
    custom anchors can set it; thunderhill_auto follows the resolved
    layout). Single-direction tracks such as sonoma keep plain names even
    with the flag. Other input types keep the flag's generic GPS-heading
    behavior.
    """
    if active_type != "ACTI":
        return True
    return bool(getattr(data_log, "metadata", {}).get("acti_bidirectional", False))


def _direction_suffix_candidate_sets(args, out_base, ld_filename, output_targets):
    """Both _cw/_ccw output sets, or None when the suffix won't apply.

    Mirrors ``_maybe_apply_direction_suffix`` naming (minus its guard/print
    side effects) so the incremental path can skip before parsing: direction
    is content-derived, so a complete previous set means this exact file was
    already converted.
    """
    if not getattr(args, "direction_suffix", False):
        return None
    if out_base is None and getattr(args, "output", None) \
            and not getattr(args, "_batch_mode", False):
        return None  # explicit --output keeps the user filename (suffix ignored)
    stem, _ext = os.path.splitext(ld_filename)
    if stem.endswith("_cw") or stem.endswith("_ccw"):
        return None
    return [
        [
            (f"{stem}_{direction}" + target[len(stem):])
            if target.startswith(stem) else target
            for target in output_targets
        ]
        for direction in ("cw", "ccw")
    ]


def _raise_if_direction_outputs_exist(args, out_base, ld_filename, output_targets):
    """Skip-before-parse for --direction-suffix reruns.

    The existing guard only sees the unsuffixed names, so reruns fully parse
    every file just to discover the suffixed outputs already exist. When a
    complete _cw or _ccw set from a previous run is present, raise the same
    FileExistsError the post-parse guard would raise (the caller maps it to
    a skip). The plain (unsuffixed) guard is deferred to post-parse when
    --direction-suffix uses default naming, so directionless files (no GPS
    heading: suffix never applies) re-parse every rerun. A complete
    unsuffixed set means the file already converted exactly as a rerun
    would. Caveat: the single transitional run that first ENABLES
    --direction-suffix needs --force, otherwise files that only have
    legacy unsuffixed outputs keep them instead of gaining _cw/_ccw twins.
    """
    if getattr(args, "force", False):
        return
    candidate_sets = _direction_suffix_candidate_sets(
        args, out_base, ld_filename, output_targets,
    )
    if not candidate_sets:
        return
    for candidates in candidate_sets:
        if candidates and all(os.path.exists(path) for path in candidates):
            raise FileExistsError(
                f"output already exists: {candidates[0]}; pass --force to replace it",
            )
    if output_targets and all(os.path.exists(path) for path in output_targets):
        raise FileExistsError(
            f"output already exists: {output_targets[0]}; pass --force to replace it",
        )


def _detect_track_direction(data_log):
    """Classify loop direction from the log's GPS Heading channel.

    Returns ``(direction, total_deg)`` with direction ``"cw"`` / ``"ccw"``,
    or ``(None, 0.0)`` when the log carries no usable GPS Heading.
    """
    from .channels import CH_GPS_HEADING
    from .parsers.acti_parser import heading_direction

    channels = getattr(data_log, "channels", {})
    channel = channels.get(CH_GPS_HEADING)
    if channel is None:
        return None, 0.0
    return heading_direction(channel.values)


def _maybe_apply_direction_suffix(args, data_log, ld_filename, ldx_filename,
                                  output_targets, output_override=None,
                                  direction=None, total=0.0, emit=lambda _message: None):
    """Append the opt-in _cw/_ccw suffix to planned output paths.

    ``direction``/``total`` arrive precomputed by the caller (which also owns
    the no-GPS warning); ``None`` simply keeps the names unchanged. Returns
    the (possibly unchanged) ``(ld, ldx, targets)`` triple. The user's
    explicit --output filename always wins: it is kept unchanged with a
    warning. New targets are preflight-checked so the existing-output guard
    still applies.
    """
    if output_override is None and getattr(args, "output", None) \
            and not getattr(args, "_batch_mode", False):
        emit("WARNING: --direction-suffix ignored: --output is explicit; "
              "keeping user filename")
        return ld_filename, ldx_filename, output_targets
    if direction is None:
        return ld_filename, ldx_filename, output_targets
    stem, _ext = os.path.splitext(ld_filename)
    if stem.endswith("_cw") or stem.endswith("_ccw"):
        return ld_filename, ldx_filename, output_targets
    new_stem = f"{stem}_{direction}"
    new_ld = new_stem + ".ld"
    new_ldx = new_stem + ".ldx"
    new_targets = [
        (new_stem + target[len(stem):]) if target.startswith(stem) else target
        for target in output_targets
    ]
    ensure_output_targets(
        new_targets,
        force=getattr(args, "force", False),
        source_path=args.log,
    )
    emit(f"Detected track direction: {direction} "
          f"(heading winding {total:+.0f} deg); output -> {new_ld}")
    return new_ld, new_ldx, new_targets


def _detect_gps_venue(data_log):
    """Return a GPS-detected venue name, or "" when GPS is unusable/unknown."""
    from .channels import CH_GPS_LATITUDE, CH_GPS_LONGITUDE
    from .metadata import detect_venue_from_gps

    channels = getattr(data_log, "channels", {})
    lat_ch = channels.get(CH_GPS_LATITUDE)
    lon_ch = channels.get(CH_GPS_LONGITUDE)
    if lat_ch is None or lon_ch is None:
        return ""
    lats = [m.value for m in lat_ch.messages]
    lons = [m.value for m in lon_ch.messages]
    return detect_venue_from_gps(lats, lons)


def _resolve_venue_name(args, data_log, emit=lambda _message: None):
    """Resolve the venue through strings, then GPS area detection.

    Precedence: explicit --venue_name > file metadata / RCZ track > filename.
    An explicit --venue_name is never overridden. When the string result
    names no known track, GPS area detection fills in (e.g. AiM free text
    like "new track 01"); anything still unknown passes through unchanged.
    """
    from .metadata import is_meaningful_venue, venue_facility

    meta = getattr(data_log, "metadata", {})

    venue_name = args.venue_name if args.venue_name else meta.get("venue_name", "")
    if not venue_name and hasattr(data_log, "rcz_metadata") and data_log.rcz_metadata.get("trackName"):
        venue_name = data_log.rcz_metadata.get("trackName")

    if not venue_name:
        filename = os.path.basename(args.log)
        venue_name = _normalize_venue(filename)
        if venue_name == filename:
            venue_name = _normalize_venue(meta.get("event_session", meta.get("Session", "")))
    else:
        venue_name = _normalize_venue(venue_name)

    if not args.venue_name:
        gps_venue = _detect_gps_venue(data_log)
        if gps_venue and (not is_meaningful_venue(venue_name)
                          or venue_facility(venue_name) != venue_facility(gps_venue)):
            emit(f"GPS-detected venue: {gps_venue} (strings gave {venue_name!r})")
            venue_name = gps_venue
    return venue_name


def _build_motec_log(args, data_log, direction=None, emit=lambda _message: None):
    """Resolve venue/event/driver metadata and build the MoTeC log.

    ``direction`` ("cw"/"ccw"/None) appends " CW"/" CCW" to the venue only
    (never to event/session metadata, which derive from the plain venue).
    An explicit --venue_name always wins and is never suffixed.
    """
    meta = getattr(data_log, "metadata", {})

    venue_name = _resolve_venue_name(args, data_log, emit)

    event_name = args.event_name if args.event_name else meta.get("event_name", "")
    if not event_name and venue_name:
        event_name = venue_name.replace("_", " ").title() + " Track Day" if not venue_name.endswith("Track Day") else venue_name

    if direction in ("cw", "ccw") and not getattr(args, "venue_name", ""):
        suffix = " CW" if direction == "cw" else " CCW"
        if venue_name and not venue_name.upper().endswith((" CW", " CCW")):
            venue_name = f"{venue_name}{suffix}"

    data_log.metadata["venue_name"] = venue_name

    driver_name = args.driver if args.driver else meta.get("driver", meta.get("Racer", ""))
    if not driver_name:
        driver_name = "Unknown Driver"

    vehicle_name = args.vehicle_id if args.vehicle_id else meta.get(
        "vehicle_id", meta.get("Vehicle", ""),
    )

    motec_log = MotecLog()
    if data_log.datetime is not None:
        motec_log.datetime = data_log.datetime
    motec_log.driver = driver_name
    motec_log.vehicle_id = vehicle_name
    motec_log.vehicle_weight = args.vehicle_weight
    motec_log.vehicle_type = args.vehicle_type
    motec_log.vehicle_comment = args.vehicle_comment
    motec_log.venue_name = venue_name
    motec_log.event_name = event_name
    motec_log.event_session = args.event_session if args.event_session else meta.get("event_session", "")
    motec_log.long_comment = args.long_comment if args.long_comment else meta.get("long_comment", meta.get("Comment", ""))
    motec_log.short_comment = args.short_comment if args.short_comment else meta.get("short_comment", "")

    motec_log.initialize()
    motec_log.add_all_channels(data_log)
    return motec_log


def _write_all_outputs(args, motec_log, data_log, ld_filename, ldx_filename, output_targets, emit=lambda _message: None):
    """Write the .ld/.ldx pair plus every requested auxiliary export."""
    emit("Saving MoTeC log...")
    output_dir = os.path.dirname(ld_filename)
    if output_dir and not os.path.isdir(output_dir):
        emit(f"Directory '{output_dir}' does not exist, will create it")
        os.makedirs(output_dir, exist_ok=True)

    ensure_output_targets(
        output_targets,
        force=getattr(args, "force", False),
        source_path=args.log,
    )
    try:
        atomic_write_motec_pair(motec_log, data_log, ld_filename, ldx_filename)
        written_paths = [ld_filename, ldx_filename]
        emit(f"Saved .ld file: {ld_filename}")
        emit(f"Saved .ldx file: {ldx_filename}")
    except PermissionError as exc:
        raise PermissionError(
            f"output is locked; close the file in MoTeC i2 and retry: {ld_filename}",
        ) from exc

    if args.gpx:
        gpx_filename = os.path.splitext(ld_filename)[0] + ".gpx"
        if motec_log.write_gpx(gpx_filename, data_log):
            written_paths.append(gpx_filename)
            emit(f"Saved .gpx track file: {gpx_filename}")

    if args.kml:
        kml_filename = os.path.splitext(ld_filename)[0] + ".kml"
        if motec_log.write_kml(kml_filename, data_log):
            written_paths.append(kml_filename)
            emit(f"Saved .kml Google Earth file: {kml_filename}")

    if args.csv:
        csv_filename = _csv_output_path(ld_filename, args.log)
        if motec_log.write_csv(csv_filename, data_log, wallclock=getattr(args, "csv_wallclock", False)):
            written_paths.append(csv_filename)
            emit(f"Saved .csv data file: {csv_filename}")

    if getattr(args, "vbo", False):
        vbo_filename = _vbo_output_path(ld_filename, args.log)
        columns = getattr(args, "vbo_columns_parsed", None)
        if motec_log.write_vbo(vbo_filename, data_log, columns):
            written_paths.append(vbo_filename)
            emit(f"Saved .vbo track file: {vbo_filename}")

    if getattr(args, "dlap", False):
        dlap_filename = _dlap_output_path(ld_filename, args.log)
        if motec_log.write_dlap(dlap_filename, data_log):
            written_paths.append(dlap_filename)
            emit(f"Saved .dlap track file: {dlap_filename}")

    emit("Done!")
    return written_paths


@dataclass(frozen=True)
class ConversionOptions:
    output: str | None = None
    dbc: str | None = None
    track: str = ""
    anchor: str = ""
    session: str | None = None
    stint: str = "all"
    lap: str = "all"
    min_lap_sec: float = 15.0
    mask_interp_gaps: bool = False
    frequency: str = "auto"
    g_source: str = "auto"
    gear_ratio_thresholds: tuple[float, ...] | None = None
    gearbox: str | None = None
    direction_suffix: bool = False
    driver: str = ""
    vehicle_id: str = ""
    vehicle_weight: int = 0
    vehicle_type: str = ""
    vehicle_comment: str = ""
    venue_name: str = ""
    event_name: str = ""
    event_session: str = ""
    long_comment: str = ""
    short_comment: str = ""
    force: bool = False
    gpx: bool = False
    kml: bool = False
    csv: bool = False
    csv_wallclock: bool = False
    vbo: bool = False
    vbo_columns: str | None = None
    dlap: bool = False
    output_dir: str | None = None
    batch_mode: bool = False


@dataclass(frozen=True)
class ConversionRequest:
    source: str
    log_type: str = "AUTO"
    options: ConversionOptions = field(default_factory=ConversionOptions)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> ConversionRequest:
        option_names = {item.name for item in fields(ConversionOptions)}
        values = {
            name: getattr(args, name)
            for name in option_names
            if hasattr(args, name)
        }
        if "_batch_mode" in vars(args):
            values["batch_mode"] = args._batch_mode
        return cls(
            source=os.fspath(args.log),
            log_type=str(args.log_type),
            options=ConversionOptions(**values),
        )


@dataclass(frozen=True)
class ConversionResult:
    status: Literal["success", "skipped", "failed"]
    output_paths: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    error: dict[str, str] | None = None
    diagnostics: dict[str, Any] | None = None


def _namespace(request: ConversionRequest) -> argparse.Namespace:
    values = asdict(request.options)
    batch_mode = values.pop("batch_mode", False)
    values.update(log=request.source, log_type=request.log_type)
    args = argparse.Namespace(**values)
    args._batch_mode = batch_mode
    return args


def _failure(exc: Exception, messages: list[str]) -> ConversionResult:
    code = exc.code if isinstance(exc, TelemetryParseError) else type(exc).__name__.upper()
    stage = exc.stage if isinstance(exc, TelemetryParseError) else "convert"
    error = {"code": code, "stage": stage, "message": str(exc)}
    return ConversionResult(
        status="failed",
        messages=tuple(messages),
        error=error,
    )


def _convert_one(
    request: ConversionRequest,
    *,
    stint_override: str | None = None,
    output_override: str | None = None,
    session_override: str | None = None,
) -> ConversionResult:
    messages: list[str] = []
    emit = messages.append
    args = _namespace(request)
    try:
        stint_arg = stint_override if stint_override is not None else args.stint
        out_base = output_override if output_override is not None else args.output
        target_session = session_override if session_override is not None else args.session
        if str(target_session).lower() == "all":
            target_session = None

        active_type = detect_input_type(args.log, args.log_type, emit=emit)
        args.vbo_columns_parsed = None
        if args.vbo_columns:
            from .exporters.vbo_export import parse_vbo_columns

            args.vbo_columns_parsed = parse_vbo_columns(args.vbo_columns)

        if active_type == "RCZ" and str(stint_arg).lower() == "all":
            try:
                stints = get_rcz_stints(args.log, target_session=target_session)
            except Exception as exc:
                emit(f"Warning: Failed checking stints, falling back to default export: {exc}")
                stints = []
            if len(stints) > 1:
                emit(f"Auto-detected {len(stints)} stints in RCZ log: {stints}")
                base_output = out_base if out_base else args.log
                base_name, _ = os.path.splitext(base_output)
                results = []
                for st in stints:
                    st_out = f"{base_name}_stint{st}"
                    emit(f"\n=== Exporting Stint {st} -> {st_out} ===")
                    result = _convert_one(
                        request,
                        stint_override=str(st),
                        output_override=st_out,
                        session_override=target_session,
                    )
                    results.append(result)
                    messages.extend(result.messages)
                failed = next((item for item in results if item.status == "failed"), None)
                skipped = next((item for item in results if item.status == "skipped"), None)
                status = "failed" if failed else "skipped" if skipped else "success"
                problem = failed or skipped
                if status == "success":
                    emit("Auto-stint split export complete.")
                return ConversionResult(
                    status=status,
                    output_paths=tuple(path for item in results for path in item.output_paths),
                    messages=tuple(messages),
                    error=problem.error if problem else None,
                )

        ld_filename, ldx_filename, output_targets = _output_paths(args, out_base)
        if not (getattr(args, "direction_suffix", False) and out_base is None):
            ensure_output_targets(
                output_targets,
                force=getattr(args, "force", False),
                source_path=args.log,
            )
        if getattr(args, "direction_suffix", False):
            # Skip-before-parse: a complete _cw/_ccw (or plain) set from a
            # previous run means this exact file was already converted; the
            # post-parse guard would only discover that after full parsing.
            _raise_if_direction_outputs_exist(args, out_base, ld_filename, output_targets)

        data_log, parse_messages = load_log_data(args, active_type, stint_arg, target_session)
        messages.extend(parse_messages)
        if not data_log.channels:
            raise ValueError("Failed to find any channels in log data")

        emit(f"Parsed {data_log.duration():.1f}s log with {len(data_log.channels)} channels:")
        for channel in data_log.channels.values():
            emit(f"\t{channel}")

        resample_freq = data_log.resample(
            args.frequency,
            mask_interp_gaps=args.mask_interp_gaps,
        )
        emit(f"Resampled channels at {resample_freq:.1f} Hz...")
        emit(f"Calculating math channels (g-source: {args.g_source})...")
        data_log.calculate_math_channels(
            g_source=args.g_source,
            gear_ratio_thresholds=args.gear_ratio_thresholds,
            gearbox=args.gearbox,
        )
        emit("Converting to MoTeC log...")

        direction, total = None, 0.0
        if getattr(args, "direction_suffix", False):
            if not _direction_suffix_applies(active_type, data_log):
                track = getattr(data_log, "metadata", {}).get("acti_track", active_type)
                emit(f"direction suffix not applicable to '{track}'; keeping plain names "
                     "(only bidirectional tracks such as thunderhill_west take _cw/_ccw)")
            else:
                direction, total = _detect_track_direction(data_log)
                if direction is None:
                    emit(
                        "WARNING: --direction-suffix skipped: no usable GPS Heading "
                        "(no GPS data or indeterminate direction)",
                    )
                elif getattr(args, "venue_name", ""):
                    emit(
                        "WARNING: --direction-suffix venue ignored: --venue_name is explicit; "
                        "keeping user value",
                    )

        motec_log = _build_motec_log(args, data_log, direction=direction, emit=emit)
        if getattr(args, "direction_suffix", False):
            ld_filename, ldx_filename, output_targets = _maybe_apply_direction_suffix(
                args,
                data_log,
                ld_filename,
                ldx_filename,
                output_targets,
                output_override=output_override,
                direction=direction,
                total=total,
                emit=emit,
            )

        stages = ["parse", "resample", "derive", "motec", "write"]
        diagnostics = build_diagnostic_report(
            data_log,
            source_format=active_type,
            session=target_session,
            stint=stint_arg,
            lap=args.lap,
            stages=stages,
        )
        written_paths = _write_all_outputs(
            args,
            motec_log,
            data_log,
            ld_filename,
            ldx_filename,
            output_targets,
            emit=emit,
        )
        return ConversionResult(
            status="success",
            output_paths=tuple(written_paths),
            messages=tuple(messages),
            diagnostics=diagnostics,
        )
    except FileExistsError as exc:
        return ConversionResult(
            status="skipped",
            messages=tuple(messages),
            error={"code": "OUTPUT_EXISTS", "stage": "write.preflight", "message": str(exc)},
        )
    except Exception as exc:
        return _failure(exc, messages)


def _backup_all(request: ConversionRequest, sessions) -> ConversionResult:
    args = _namespace(request)
    options = request.options
    if options.output:
        return _failure(ValueError("--output cannot be used with --session all; use --output-dir"), [])
    if str(options.stint).lower() != "all":
        return _failure(ValueError("--session all requires --stint all"), [])

    output_dir = options.output_dir or os.path.splitext(request.source)[0] + "_sessions"
    jobs = []
    all_targets = []
    for session in sessions:
        for stint in session.stints:
            stem = session.session_id
            if len(session.stints) > 1:
                stem += f"_stint{stint}"
            output_base = os.path.join(output_dir, stem)
            _, _, targets = _output_paths(args, output_base)
            jobs.append((session.session_id, stint, output_base))
            all_targets.extend(targets)
    try:
        ensure_output_targets(all_targets, force=options.force, source_path=request.source)
    except FileExistsError as exc:
        return ConversionResult(
            status="skipped",
            messages=(f"ERROR: {exc}", "Preflight failed; no sessions were exported."),
            error={"code": "OUTPUT_EXISTS", "stage": "write.preflight", "message": str(exc)},
        )
    except (OSError, ValueError) as exc:
        return ConversionResult(
            status="failed",
            messages=(f"ERROR: {exc}", "Preflight failed; no sessions were exported."),
            error={
                "code": type(exc).__name__.upper(),
                "stage": "write.preflight",
                "message": str(exc),
            },
        )

    results = []
    messages = []
    for session_id, stint, output_base in jobs:
        messages.append(f"\n=== Exporting {session_id} stint {stint} ===")
        result = _convert_one(
            request,
            stint_override=str(stint),
            output_override=output_base,
            session_override=session_id,
        )
        results.append(result)
        messages.extend(result.messages)
        if result.status != "success":
            message = result.error["message"] if result.error else result.status
            messages.append(f"ERROR: {session_id} stint {stint} failed: {message}")

    failed = [result for result in results if result.status not in {"success", "skipped"}]
    skipped = [result for result in results if result.status == "skipped"]
    status = "failed" if failed else (
        "skipped"
        if skipped and not any(result.status == "success" for result in results)
        else "success"
    )
    summary = (
        f"\nRCZ backup export complete: {len(results) - len(failed) - len(skipped)} succeeded, "
        f"{len(failed)} failed, {len(skipped)} skipped."
    )
    return ConversionResult(
        status=status,
        output_paths=tuple(path for result in results for path in result.output_paths),
        messages=tuple(messages + [summary]),
        error=(failed or skipped)[0].error if failed or skipped else None,
    )


def convert(request: ConversionRequest) -> ConversionResult:
    """Convert one source using explicit options, returning paths and notices."""
    options = request.options
    if not os.path.isfile(request.source):
        return _failure(FileNotFoundError(f"log file {request.source} does not exist"), [])

    detection_messages = []
    try:
        active_type = detect_input_type(
            request.source,
            request.log_type,
            emit=detection_messages.append,
        )
    except Exception as exc:
        return _failure(exc, detection_messages)
    request = replace(request, log_type=active_type)

    if active_type == "RCZ":
        try:
            sessions = discover_rcz_sessions(request.source)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            return _failure(ValueError(f"Failed to inspect RCZ archive: {exc}"), detection_messages)
        if sessions:
            if str(options.session).lower() == "all":
                result = _backup_all(request, sessions)
                return replace(result, messages=tuple(detection_messages) + result.messages)
            if not options.session:
                result = _failure(
                    ValueError(
                        f"RaceChrono backup contains {len(sessions)} sessions; use "
                        "--list-sessions, --session ID, or --session all.",
                    ),
                    detection_messages,
                )
                return result
            selected = next((item for item in sessions if item.session_id == options.session), None)
            if selected is None:
                return _failure(
                    ValueError(f"Unknown RCZ session ID '{options.session}'."),
                    detection_messages,
                )
            if options.output_dir:
                return _failure(
                    ValueError("--output-dir is only valid with --session all"),
                    detection_messages,
                )
            output_base = options.output or f"{os.path.splitext(request.source)[0]}_{selected.session_id}"
            request = replace(
                request,
                options=replace(options, output=output_base, session=selected.session_id),
            )
        elif options.session:
            return _failure(
                ValueError("--session is only valid for a multi-session RCZ backup"),
                detection_messages,
            )
    if active_type == "CAN" and not (options.dbc and os.path.isfile(options.dbc)):
        return _failure(ValueError(f"DBC file {options.dbc} does not exist"), detection_messages)

    result = _convert_one(request)
    return replace(result, messages=tuple(detection_messages) + result.messages)
