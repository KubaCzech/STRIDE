"""Data types and schemas for drift descriptions."""

from dataclasses import dataclass


@dataclass
class DriftDescription:
    """Represents characteristics of a detected concept drift event."""

    error_rate_at_warning: float | None = None
    error_rate_at_detection: float | None = None
    drift_duration: int | None = None
    drift_start_index: int | None = None
    drift_end_index: int | None = None
    error_rate_at_peak: float | None = None
    detected_at: int | None = None
