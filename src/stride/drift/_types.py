"""Data types and schemas for drift descriptions."""

from dataclasses import dataclass


@dataclass
class DriftDescription:
    """Represents characteristics of a detected concept drift event.

    Attributes:
        error_rate_at_warning: Error rate calculated at initial warning threshold.
        error_rate_at_detection: Error rate when drift alarm was triggered.
        drift_duration: Total count of stream samples between onset and detection.
        drift_start_index: Stream sample index pinpointing drift onset.
        drift_end_index: Stream sample index where drift concluded or stabilized.
        error_rate_at_peak: Maximum observed error rate across the drift event.
        detected_at: Stream sample index where detector emitted drift signal.
    """

    error_rate_at_warning: float | None = None
    error_rate_at_detection: float | None = None
    drift_duration: int | None = None
    drift_start_index: int | None = None
    drift_end_index: int | None = None
    error_rate_at_peak: float | None = None
    detected_at: int | None = None
