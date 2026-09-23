"""
STRIDE: Explainable AI and Drift Detection Framework.

A comprehensive research framework for detecting, characterizing, and explaining
concept drift in machine learning pipelines.
"""

from . import common, datasets, drift, models, plotting, xai
from .exceptions import (
    DimensionalityError,
    DriftDetectionError,
    OptionalDependencyError,
    StrideError,
)

# Top-level convenient access to core estimators and analyzers
from .drift import BinaryErrorDriftDescriptor
from .models import MODELS, BaseModel, MLPModel, RandomForestModel
from .xai import (
    ClusterBasedDriftDetector,
    DecisionBoundaryDriftAnalyzer,
    DescriptiveStatisticsDriftDetector,
    FeatureImportanceDriftAnalyzer,
)

# Subpackage compatibility aliases
DDM = drift
clustering = xai.clustering
decision_boundary = xai.boundary
feature_importance = xai.importance
recurrence = xai.recurrence
descriptive_statistics = xai.stats

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Subpackages
    "common",
    "datasets",
    "drift",
    "models",
    "plotting",
    "xai",
    # Core Analyzers
    "BinaryErrorDriftDescriptor",
    "ClusterBasedDriftDetector",
    "DecisionBoundaryDriftAnalyzer",
    "DescriptiveStatisticsDriftDetector",
    "FeatureImportanceDriftAnalyzer",
    # Models
    "BaseModel",
    "MLPModel",
    "RandomForestModel",
    "MODELS",
    # Exceptions
    "StrideError",
    "DriftDetectionError",
    "DimensionalityError",
    "OptionalDependencyError",
    # Aliases
    "DDM",
    "clustering",
    "decision_boundary",
    "feature_importance",
    "recurrence",
    "descriptive_statistics",
]
