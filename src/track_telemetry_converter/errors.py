"""Structured errors from telemetry conversion stages.

Core parsers raise these errors; CLI entry points decide how to display them.
"""

from __future__ import annotations

from typing import Any


class TelemetryParseError(ValueError):
    """A parser failure with a stable machine-readable code and context."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        stage: str,
        source: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.source = source
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": str(self),
            "source": self.source,
            "details": self.details,
        }
