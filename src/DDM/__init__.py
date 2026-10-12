"""Compatibility bridge for legacy DDM imports."""

from stride.drift import ADWIN, BinaryErrorDriftDescriptor, DriftDescription, DualADWIN

__all__ = ["ADWIN", "DualADWIN", "BinaryErrorDriftDescriptor", "DriftDescription"]
