"""Input format parsers and registry for TrackTelemetryConverter."""
from __future__ import annotations

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
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(2048)
        if path_lower.endswith(".vbo") or "[header]" in sample or "[column names]" in sample:
            return "VBO"
        if "RaceChrono" in sample or "RaceStudio" in sample or "Solo" in sample or "GPS_LatAcc" in sample:
            return "RACECHRONO"
        if "Session ID" in sample or "Track ID" in sample or "PB Buddy" in sample:
            return "PBBUDDY"
        if "Time,GPS Latitude" in sample:
            return "PBBUDDY"
        if "Time," in sample and "RPM" in sample:
            return "ACCESSPORT"
    except Exception:
        pass
    return "CSV"
