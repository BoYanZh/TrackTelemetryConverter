"""Read-only discovery helpers for RaceChrono RCZ archives."""

from __future__ import annotations

import json
import posixpath
import zipfile
from dataclasses import dataclass


@dataclass(frozen=True)
class RczSession:
    session_id: str
    prefix: str
    time_created: int
    track_name: str
    duration_sec: float
    lap_count: int
    stints: tuple


def discover_rcz_sessions(rcz_path):
    """Return sorted descriptors for sessions nested in a RaceChrono backup."""
    sessions = []
    with zipfile.ZipFile(rcz_path, "r") as archive:
        names = archive.namelist()
        for name in names:
            parts = name.split("/")
            if (
                len(parts) == 3
                and parts[0] == "sessions"
                and parts[1].startswith("session_")
                and parts[2] == "session.json"
            ):
                metadata = json.loads(archive.read(name).decode("utf-8"))
                prefix = posixpath.dirname(name) + "/"
                stints = {
                    int(lap.get("sessionResume", 0))
                    for lap in metadata.get("laps", [])
                }
                for member in names:
                    if not member.startswith(prefix + "resume_"):
                        continue
                    resume_name = member[len(prefix):].split("/", 1)[0]
                    try:
                        stints.add(int(resume_name[len("resume_"):]))
                    except ValueError:
                        continue
                if not stints:
                    stints.add(0)
                duration_ms = metadata.get("lengthTime")
                if duration_ms is None:
                    first = metadata.get("firstTimestamp", 0)
                    latest = metadata.get("latestTimestamp", first)
                    duration_ms = max(0, latest - first)
                sessions.append(
                    RczSession(
                        session_id=parts[1],
                        prefix=prefix,
                        time_created=int(metadata.get("timeCreated") or 0),
                        track_name=metadata.get("trackName") or "Unknown",
                        duration_sec=float(duration_ms) / 1000.0,
                        lap_count=int(
                            metadata.get("lapCount", len(metadata.get("laps", [])))
                        ),
                        stints=tuple(sorted(stints)),
                    )
                )
    return sorted(sessions, key=lambda item: (item.time_created, item.prefix))
