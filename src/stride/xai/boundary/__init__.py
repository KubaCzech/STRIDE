"""Decision boundary migration analysis, neural projections, and disagreement trees."""

from .analysis import DecisionBoundaryDriftAnalyzer
from .disagreement import compute_disagreement_analysis

try:
    from .ssnp import SSNP
except ImportError:
    pass

try:
    from .visualization import (
        plot_categorical_drift_map,
        plot_decision_boundary,
        plot_decision_boundary_shift,
    )
except ImportError:
    pass

__all__ = [
    "DecisionBoundaryDriftAnalyzer",
    "compute_disagreement_analysis",
]
