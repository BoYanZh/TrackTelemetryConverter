"""Verified, recoverable output helpers for MoTeC files."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from ._vendor.ldparser import ldData


def ensure_output_targets(targets, force=False, source_path=None):
    """Reject accidental source replacement and existing outputs."""
    source = os.path.abspath(source_path) if source_path else None
    normalized = [os.path.abspath(path) for path in targets]
    if len(normalized) != len(set(os.path.normcase(path) for path in normalized)):
        raise ValueError("multiple requested outputs resolve to the same file")
    for path in normalized:
        if source and os.path.normcase(path) == os.path.normcase(source):
            raise ValueError(
                f"refusing to replace input file: {path}; choose a different --output"
            )
        if os.path.exists(path) and not force:
            raise FileExistsError(f"output already exists: {path}; pass --force to replace it")


def verify_motec_pair(ld_path, ldx_path, expected_channels):
    """Read generated files back and verify binary/XML structure and data."""
    parsed = ldData.fromfile(ld_path)
    if len(parsed.channs) != len(expected_channels):
        raise RuntimeError(
            f"MoTeC verification failed: expected {len(expected_channels)} channels, "
            f"found {len(parsed.channs)}"
        )

    for expected, actual in zip(expected_channels, parsed.channs):
        if actual.name.strip() != expected.name.strip():
            raise RuntimeError(
                f"MoTeC verification failed: channel {expected.name!r} read back as "
                f"{actual.name!r}"
            )
        actual_values = np.asarray(actual.data)
        expected_values = np.asarray(expected._data)
        if len(actual_values) != len(expected_values):
            raise RuntimeError(
                f"MoTeC verification failed: channel {actual.name!r} has "
                f"{len(actual_values)} samples, expected {len(expected_values)}"
            )
        if actual.freq != expected.freq:
            raise RuntimeError(
                f"MoTeC verification failed: channel {actual.name!r} frequency "
                f"{actual.freq} != {expected.freq}"
            )
        if not np.array_equal(actual_values, expected_values, equal_nan=True):
            raise RuntimeError(
                f"MoTeC verification failed: channel {actual.name!r} data changed on write"
            )

    root = ET.parse(ldx_path).getroot()
    if root.tag != "LDXFile":
        raise RuntimeError(f"LDX verification failed: unexpected root element {root.tag!r}")
    for marker_group in root.findall(".//MarkerGroup"):
        times = [float(marker.get("Time")) for marker in marker_group.findall("Marker")]
        if any(not math.isfinite(value) for value in times):
            raise RuntimeError("LDX verification failed: marker time is not finite")
        if times != sorted(times):
            raise RuntimeError("LDX verification failed: marker times are not sorted")
    for lap in root.findall(".//Lap"):
        duration = float(lap.get("Time", "0"))
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("LDX verification failed: lap duration must be positive")


def _temporary_path(target, suffix):
    directory = os.path.dirname(os.path.abspath(target)) or os.getcwd()
    handle, path = tempfile.mkstemp(prefix=".motec_", suffix=suffix, dir=directory)
    os.close(handle)
    return path


def _commit_files(pairs):
    """Replace verified files and restore previous targets if a replace fails."""
    backups = {}
    committed = []
    try:
        for _, target in pairs:
            if os.path.exists(target):
                backup = _temporary_path(target, ".backup")
                shutil.copy2(target, backup)
                backups[target] = backup

        for staged, target in pairs:
            os.replace(staged, target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            backup = backups.get(target)
            if backup and os.path.exists(backup):
                os.replace(backup, target)
                backups.pop(target, None)
            elif os.path.exists(target):
                os.remove(target)
        raise
    finally:
        for backup in backups.values():
            if os.path.exists(backup):
                os.remove(backup)


def atomic_write_motec_pair(motec_log, data_log, ld_path, ldx_path):
    """Write, read back, then atomically replace each final MoTeC output."""
    temp_ld = _temporary_path(ld_path, ".ld.tmp")
    temp_ldx = _temporary_path(ldx_path, ".ldx.tmp")
    try:
        motec_log.write(temp_ld)
        beacons = data_log.detect_beacons()
        motec_log.write_ldx(
            temp_ldx,
            getattr(data_log, "laps_info", None),
            beacons=beacons,
        )
        verify_motec_pair(temp_ld, temp_ldx, motec_log.ld_channels)
        _commit_files(((temp_ld, ld_path), (temp_ldx, ldx_path)))
    finally:
        for path in (temp_ld, temp_ldx):
            if os.path.exists(path):
                os.remove(path)
