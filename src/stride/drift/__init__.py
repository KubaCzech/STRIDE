"""Drift detection methods, sequential error descriptors, and drift event tracking."""

from river.drift import ADWIN

from .adwin import DualADWIN
from .binary_descriptor import (
    BinaryErrorDriftDescriptor,
    DriftDescription,
    generate_error_stream,
    run_drift_detection,
    visualization_and_correction,
)

__all__ = [
    "ADWIN",
    "DualADWIN",
    "BinaryErrorDriftDescriptor",
    "DriftDescription",
]
