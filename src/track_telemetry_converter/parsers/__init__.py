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
    "VBO",
    "DLAP",
    "IBT",
    "XRK",
    "FIT",
)


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
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        if re.search(r"^\(\d+\.\d+\)\s+\S+\s+[0-9A-Fa-f]+#", sample, re.MULTILINE):
            return "CAN"
        if path_lower.endswith(".vbo") or "[header]" in sample or "[column names]" in sample:
            return "VBO"
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
