"""Write an explicit RaceChrono/video alignment into a MoTeC LDX sidecar."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class VideoSyncError(ValueError):
    """The LD, source RCZ, alignment, or LDX cannot be trusted for export."""


@dataclass(frozen=True)
class VideoAlignment:
    """Explicit alignment values supplied by an upstream video editor."""

    rcz_path: str | Path
    video_path: str | Path
    stint: int | None = None
    offset: float = 0.0
    rate: float = 1.0
    video_start: float = 0.0
    video_start_utc: float | None = None


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise VideoSyncError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoSyncError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise VideoSyncError(f"{field} must be a finite number")
    return number


def _series(channel: Any) -> tuple[np.ndarray, np.ndarray]:
    times = getattr(channel, "times", None)
    if times is None:
        times = getattr(channel, "timestamps", None)
    values = getattr(channel, "values", None)
    if times is None or values is None:
        raise VideoSyncError("telemetry channel is missing time or value samples")
    times_array = np.asarray(times, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    count = min(times_array.size, values_array.size)
    return times_array[:count], values_array[:count]


def verify_ld_source(
    running_time_start: float,
    ld_channels: dict[str, Any],
    source_channels: dict[str, Any],
) -> dict[str, float]:
    """Check that LD samples share the selected RCZ timeline and content."""
    tolerances = {
        "Ground Speed": 0.1,
        "Engine RPM": 2.0,
        "GPS Latitude": 0.00001,
        "GPS Longitude": 0.00001,
    }
    errors: dict[str, float] = {}
    for name, tolerance in tolerances.items():
        if name not in ld_channels or name not in source_channels:
            continue
        ld_times, ld_values = _series(ld_channels[name])
        source_times, source_values = _series(source_channels[name])
        if len(ld_times) < 10 or len(source_times) < 10:
            continue

        keep = np.isfinite(ld_times) & np.isfinite(ld_values)
        indices = np.flatnonzero(keep)
        if indices.size > 500:
            indices = indices[np.linspace(0, indices.size - 1, 500).astype(int)]
        sample_times = running_time_start + ld_times[indices]

        finite_source = np.isfinite(source_times) & np.isfinite(source_values)
        valid_source_times = source_times[finite_source]
        valid_source_values = source_values[finite_source]
        if len(valid_source_times) < 2:
            continue
        order = np.argsort(valid_source_times)
        valid_source_times = valid_source_times[order]
        valid_source_values = valid_source_values[order]
        valid_source_times, unique_indices = np.unique(
            valid_source_times, return_index=True,
        )
        valid_source_values = valid_source_values[unique_indices]

        finite_ld_times = ld_times[np.isfinite(ld_times)]
        if finite_ld_times.size:
            source_step = float(np.median(np.diff(valid_source_times)))
            coverage_tolerance = max(0.05, source_step * 1.5)
            outside = max(
                float(valid_source_times[0] - np.min(running_time_start + finite_ld_times)),
                float(np.max(running_time_start + finite_ld_times) - valid_source_times[-1]),
                0.0,
            )
            if outside > coverage_tolerance:
                raise VideoSyncError(
                    f"LD {name} time range extends {outside:g}s outside the selected RCZ "
                    f"channel coverage (tolerance {coverage_tolerance:g}s)",
                )

        inside = (sample_times >= valid_source_times[0]) & (
            sample_times <= valid_source_times[-1]
        )
        if np.count_nonzero(inside) < 10:
            continue
        expected = np.interp(
            sample_times[inside], valid_source_times, valid_source_values,
        )
        actual = ld_values[indices][inside]
        error = float(np.max(np.abs(actual - expected)))
        if not math.isfinite(error) or error > tolerance:
            raise VideoSyncError(
                f"LD does not match the selected RCZ at Running Time {running_time_start:g}: "
                f"{name} max sample error {error:g} exceeds {tolerance:g}",
            )
        errors[name] = error

    if "Ground Speed" not in errors or not any(
        name in errors for name in ("Engine RPM", "GPS Latitude", "GPS Longitude")
    ):
        raise VideoSyncError(
            "cannot verify LD against RCZ: need at least 10 matching Ground Speed "
            "samples and one matching RPM/GPS channel",
        )
    return errors


def _load_ld(ld_path: Path) -> tuple[float, dict[str, Any]]:
    """Load native LD Running Time and decoded channels for source checks."""
    from ._vendor.ldparser import ldData
    from .log import DataLog

    try:
        binary = ldData.fromfile(str(ld_path))
    except Exception as exc:
        raise VideoSyncError(f"cannot read MoTeC LD: {exc}") from exc

    running = next(
        (
            channel
            for channel in binary.channs
            if str(getattr(channel, "name", "")).strip().casefold() == "running time"
        ),
        None,
    )
    if running is None:
        raise VideoSyncError("LD has no Running Time channel")
    raw_running_values = np.asarray(running.data)
    running_values = np.asarray(raw_running_values, dtype=np.float64)
    if running_values.size < 20 or not np.all(np.isfinite(running_values)):
        raise VideoSyncError("LD Running Time channel is empty, short, or non-finite")
    frequency = float(getattr(running, "freq", 0.0) or 0.0)
    if not math.isfinite(frequency) or frequency <= 0:
        raise VideoSyncError("LD Running Time must increase at a known sample rate")
    precision = (
        float(np.finfo(raw_running_values.dtype).eps)
        if raw_running_values.dtype.kind == "f"
        else 0.0
    )
    quantization_tolerance = max(
        1e-6,
        precision * max(1.0, float(np.max(np.abs(running_values)))) * 2.0,
    )
    axis_tolerance = min(0.1 / frequency, quantization_tolerance)
    expected_values = running_values[0] + np.arange(running_values.size) / frequency
    if np.max(np.abs(running_values - expected_values)) > axis_tolerance:
        raise VideoSyncError("LD Running Time is not a regular native-rate time axis")

    log = DataLog()
    try:
        log.from_motec_ld_log(str(ld_path))
    except Exception as exc:
        raise VideoSyncError(f"cannot decode LD channels: {exc}") from exc
    return float(running_values[0]), log.channels


def _load_rcz(rcz_path: Path, stint: int | None):
    from .log import DataLog

    log = DataLog()
    try:
        log.from_rcz_log(str(rcz_path), target_stint=stint)
    except Exception as exc:
        raise VideoSyncError(f"cannot load source RCZ: {exc}") from exc
    return log


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):  # comments and processing instructions
        return ""
    return tag.rsplit("}", 1)[-1]


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(parent) if _local_name(child.tag) == name]


def _existing_xml_path(value: str, base: Path) -> str:
    path = Path(value)
    return _path_key(path if path.is_absolute() else base / path)


def _read_ldx(path: Path) -> ET.ElementTree[ET.Element]:
    try:
        parser = ET.XMLParser(  # noqa: S314 - local user-selected LDX file
            target=ET.TreeBuilder(insert_comments=True),
        )
        tree = ET.parse(path, parser=parser)  # noqa: S314 - local user-selected LDX file
    except (OSError, ET.ParseError) as exc:
        raise VideoSyncError(f"cannot read LDX {path}: {exc}") from exc
    if _local_name(tree.getroot().tag) != "LDXFile":
        raise VideoSyncError("expected a MoTeC LDXFile document")
    return tree


def _rebase_relative_sources(
    tree: ET.ElementTree[ET.Element], source_directory: Path,
) -> None:
    for source in tree.getroot().iter():
        if _local_name(source.tag) != "Source":
            continue
        filename = source.get("FileName")
        if filename and not Path(filename).is_absolute():
            source.set("FileName", str((source_directory / filename).resolve()))


def _same_sync(
    source: ET.Element,
    source_ldx_directory: Path,
    video_path: Path,
    data_time_us: float,
) -> bool:
    filename = source.get("FileName")
    if (
        not filename
        or _existing_xml_path(filename, source_ldx_directory) != _path_key(video_path)
    ):
        return False
    sync_points = _children(source, "SyncPoints")
    if len(sync_points) != 1 or sync_points[0].get("Balance") != "2":
        return False
    points = _children(sync_points[0], "SyncPoint")
    if len(points) != 1:
        return False
    point = points[0]
    try:
        current_data_time = float(point.get("DataTime", "nan"))
        current_video_time = float(point.get("VideoTime", "nan"))
    except ValueError:
        return False
    return (
        math.isclose(current_data_time, data_time_us, rel_tol=0.0, abs_tol=0.5)
        and current_video_time == 0.0
        and point.get("Manual") == "1"
    )


def _write_sync_point(
    tree: ET.ElementTree[ET.Element],
    source_ldx_directory: Path,
    video_path: Path,
    data_time_us: float,
    view_name: str,
) -> bool:
    root = tree.getroot()
    videos_nodes = _children(root, "Videos")
    if len(videos_nodes) > 1:
        raise VideoSyncError("LDX contains multiple Videos sections")
    videos = videos_nodes[0] if videos_nodes else ET.SubElement(root, "Videos")
    views = _children(videos, "View")

    matches: list[tuple[ET.Element, ET.Element]] = []
    for view in views:
        for source in _children(view, "Source"):
            filename = source.get("FileName")
            if filename and _existing_xml_path(filename, source_ldx_directory) == _path_key(video_path):
                matches.append((view, source))
    if len(matches) > 1:
        raise VideoSyncError("LDX has ambiguous views for this video path")

    if matches:
        view, source = matches[0]
    else:
        named_views = [view for view in views if view.get("Name") == view_name]
        if len(named_views) > 1:
            raise VideoSyncError(f"LDX has multiple views named {view_name!r}")
        if named_views:
            view = named_views[0]
            sources = _children(view, "Source")
            if any(item.get("FileName") for item in sources):
                raise VideoSyncError(
                    f"MoTeC view {view_name!r} already points to another video; "
                    "choose an unused view name",
                )
            source = sources[0] if sources else ET.SubElement(view, "Source")
        else:
            view = ET.SubElement(videos, "View", {"Name": view_name})
            source = ET.SubElement(view, "Source")

    if _same_sync(source, source_ldx_directory, video_path, data_time_us):
        return False

    view.set("Name", view.get("Name") or view_name)
    source.set("FileName", str(video_path))
    for sync_points in _children(source, "SyncPoints"):
        source.remove(sync_points)
    sync_points = ET.SubElement(source, "SyncPoints", {"Balance": "2"})
    ET.SubElement(
        sync_points,
        "SyncPoint",
        {
            "DataTime": format(data_time_us, ".12g"),
            "VideoTime": "0",
            "Manual": "1",
        },
    )
    return True


def _backup_name(path: Path) -> Path:
    candidate = Path(str(path) + ".bak")
    index = 1
    while candidate.exists():
        candidate = Path(str(path) + f".bak.{index}")
        index += 1
    return candidate


def _atomic_save(tree: ET.ElementTree[ET.Element], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            tree.write(stream, encoding="utf-8", xml_declaration=True)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temp_name).replace(path)
    finally:
        try:
            Path(temp_name).unlink()
        except FileNotFoundError:
            pass


def export_motec_sync(
    ld_path: str | Path,
    alignment: VideoAlignment,
    ldx_path: str | Path | None = None,
    output_path: str | Path | None = None,
    view_name: str = "Camera View 1",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate alignment and write a manual ``VideoTime=0`` LDX sync point.

    Alignment uses ``RCZ time = source video time + offset``. For an excerpt
    cut from that source, ``video_start`` is the source time represented by
    excerpt time zero. MoTeC's LD time axis starts at ``Running Time[0]``.
    """
    ld_file = Path(ld_path).expanduser().absolute()
    rcz_file = Path(alignment.rcz_path).expanduser().absolute()
    video_file = Path(alignment.video_path).expanduser().absolute()
    for label, path in (("LD", ld_file), ("RCZ", rcz_file), ("video", video_file)):
        if not path.is_file():
            raise VideoSyncError(f"{label} not found: {path}")

    if alignment.stint is not None and (
        isinstance(alignment.stint, bool)
        or not isinstance(alignment.stint, int)
        or alignment.stint < 0
    ):
        raise VideoSyncError("stint must be a non-negative integer or None")
    offset = _finite_number(alignment.offset, "offset")
    rate = _finite_number(alignment.rate, "rate")
    video_start = _finite_number(alignment.video_start, "video_start")
    video_start_utc = (
        None
        if alignment.video_start_utc is None
        else _finite_number(alignment.video_start_utc, "video_start_utc")
    )
    if not math.isclose(rate, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise VideoSyncError("MoTeC video sync export currently requires rate 1.0")
    if video_start < 0:
        raise VideoSyncError("video_start must be non-negative")
    if not isinstance(view_name, str) or not view_name.strip():
        raise VideoSyncError("view_name must be a non-empty string")
    if not isinstance(dry_run, bool):
        raise VideoSyncError("dry_run must be a boolean")

    running_start, ld_channels = _load_ld(ld_file)
    source_log = _load_rcz(rcz_file, alignment.stint)
    sample_errors = verify_ld_source(running_start, ld_channels, source_log.channels)

    origin_ms = getattr(source_log, "time_origin_epoch_ms", None)
    origin_s: float | None = None
    if origin_ms is not None:
        origin_s = _finite_number(origin_ms, "RCZ time origin") / 1000.0
    if video_start_utc is not None:
        if origin_s is None:
            raise VideoSyncError(
                "cannot validate video_start_utc because the selected RCZ has no absolute time origin",
            )
        current_offset = video_start_utc - origin_s
        if not math.isclose(current_offset, offset, rel_tol=0.0, abs_tol=0.01):
            raise VideoSyncError(
                "video_start_utc does not match the supplied offset and selected RCZ origin "
                f"(offset {offset:.6f}s, current {current_offset:.6f}s)",
            )

    data_time_us = (offset + video_start - running_start) * 1_000_000.0
    if not math.isfinite(data_time_us):
        raise VideoSyncError("mapped MoTeC DataTime is non-finite")
    source_ldx = (
        Path(ldx_path).expanduser().absolute()
        if ldx_path is not None
        else ld_file.with_suffix(".ldx")
    )
    destination = (
        Path(output_path).expanduser().absolute()
        if output_path is not None
        else source_ldx
    )
    if source_ldx.suffix.casefold() != ".ldx" or destination.suffix.casefold() != ".ldx":
        raise VideoSyncError("LDX source and output paths must use the .ldx extension")
    for input_path in (ld_file, rcz_file, video_file):
        if _path_key(destination) == _path_key(input_path):
            raise VideoSyncError("LDX output cannot overwrite the LD, RCZ, or video input")

    if (
        destination.is_file()
        and _path_key(destination) != _path_key(source_ldx)
    ):
        raise VideoSyncError(
            "output LDX already exists separately from the selected source LDX; "
            "pass that output as --ldx to update it",
        )
    read_from = source_ldx
    if not read_from.is_file():
        raise VideoSyncError(f"LDX sidecar not found: {read_from}")
    tree = _read_ldx(read_from)
    sync_changed = _write_sync_point(
        tree,
        read_from.parent,
        video_file,
        data_time_us,
        view_name.strip(),
    )
    if read_from.parent != destination.parent:
        _rebase_relative_sources(tree, read_from.parent)
    changed = sync_changed or not destination.exists()

    backup: Path | None = None
    if not dry_run and changed:
        if destination.exists():
            backup = _backup_name(destination)
            shutil.copy2(destination, backup)
        _atomic_save(tree, destination)

    return {
        "ld": str(ld_file),
        "rcz": str(rcz_file),
        "video": str(video_file),
        "stint": alignment.stint,
        "running_time_start": running_start,
        "offset": offset,
        "video_start": video_start,
        "video_start_utc": video_start_utc,
        "rcz_time_origin": origin_s,
        "data_time_us": data_time_us,
        "sample_errors": sample_errors,
        "output": str(destination),
        "backup": str(backup) if backup else None,
        "changed": changed,
        "dry_run": dry_run,
    }
