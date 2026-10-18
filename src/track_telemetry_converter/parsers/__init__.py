"""Input format parsers and registry for TrackTelemetryConverter."""
from __future__ import annotations

import re
from typing import NamedTuple


class ParserInfo(NamedTuple):
    name: str
    extensions: tuple[str, ...]
    description: str


SUPPORTED_LOG_TYPES = (
    "CAN",
    "CSV",
    "ACCESSPORT",
    "RACECHRONO",
    "RCZ",
    "PBBUDDY",
    "TRACKADDICT",
    "TESLA",
    "VBO",
    "DLAP",
    "IBT",
    "XRK",
    "FIT",
    "ACTI",
)

#: Convertible input extensions for CLI batch (AUTO) mode. Anything else
#: (video, configs, images, ...) is skipped without attempting conversion. `.ld`/`.ldx` are outputs (ACTI
#: `.ld` is only claimed by an explicit ACTI batch, never AUTO) and `.dbc`
#: is a sidecar, so none of them appear here.
SUPPORTED_INPUT_EXTENSIONS = frozenset({
    ".rcz",
    ".csv",
    ".vbo",
    ".dlap",
    ".ibt",
    ".xrk",
    ".xrz",
    ".fit",
    ".log",
    ".can",
    ".trc",
    ".asc",
})

#: Extensions the converter writes that are never inputs themselves.
OUTPUT_EXTENSIONS = frozenset({".ld", ".ldx"})

#: Sibling sidecar extension (passed via --dbc, never converted directly).
DBC_EXTENSION = ".dbc"


def _sniff_ld_for_acti(file_path: str) -> bool:
    """Header-only probe: Car Coord X/Y present without GPS Latitude means ACTI."""
    try:
        from .._vendor.ldparser import ldChan, ldHead

        with open(file_path, "rb") as handle:
            head = ldHead.fromfile(handle)
            names = set()
            meta_ptr = head.meta_ptr
            while meta_ptr:
                chan = ldChan.fromfile(file_path, meta_ptr)
                names.add(chan.name)
                meta_ptr = chan.next_meta_ptr
                if len(names) > 400:
                    break
    except Exception:
        return False
    # Converter outputs carry GPS but never raw Car Coord channels.
    return "Car Coord X" in names and "Car Coord Y" in names and "GPS Latitude" not in names


def auto_detect_log_type(file_path: str) -> str:
    """Auto-detect telemetry log format from file extension and header signature."""
    path_lower = file_path.lower()
    if path_lower.endswith(".ibt"):
        return "IBT"
    if path_lower.endswith(".rcz"):
        return "RCZ"
    if path_lower.endswith((".xrk", ".xrz")):
        return "XRK"
    if path_lower.endswith(".fit"):
        return "FIT"
    if path_lower.endswith(".dlap"):
        return "DLAP"
    if path_lower.endswith(".ld") and _sniff_ld_for_acti(file_path):
        return "ACTI"
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        if re.search(r"^\(\d+\.\d+\)\s+\S+\s+[0-9A-Fa-f]+#", sample, re.MULTILINE):
            return "CAN"
        if path_lower.endswith(".vbo") or "[header]" in sample or "[column names]" in sample:
            return "VBO"
        if "# RaceRender Data" in sample:
            return "TRACKADDICT"
        if "Elapsed Time (ms)" in sample:
            return "TESLA"
        if "RaceChrono" in sample or "RaceStudio" in sample or "Solo" in sample or "GPS_LatAcc" in sample or "AiM CSV File" in sample:
            return "RACECHRONO"
        if "Session ID" in sample or "Track ID" in sample or "PB Buddy" in sample:
            return "PBBUDDY"
        if "Time,GPS Latitude" in sample:
            return "PBBUDDY"
        if "Time (sec)" in sample or "AP Info" in sample:
            return "ACCESSPORT"
    except Exception:
        print("WARNING: Could not sniff log format; falling back to CSV")
    return "CSV"
