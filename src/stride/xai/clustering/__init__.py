"""Class-conditional X-means clustering, reference cluster tracking and permutation-tested drift detection."""

from .clustering import ClusterBasedDriftDetector
from .matching import assign_to_reference, map_new_clusters_to_old, merge_clusters
from .statistics import calculate_shift_significance, compare_desc_stats_for_clusters, compute_desc_stats_for_clusters
from .xmeans import bic_score, reshape_clusters, run_xmeans

try:
    from .visualization import (
        plot_centers_shift,
        plot_clustering_heatmap,
        plot_clusters_by_class,
        plot_drift_clustered,
    )
except ImportError:
    pass

__all__ = [
    "ClusterBasedDriftDetector",
    "assign_to_reference",
    "bic_score",
    "calculate_shift_significance",
    "map_new_clusters_to_old",
    "merge_clusters",
    "compute_desc_stats_for_clusters",
    "compare_desc_stats_for_clusters",
    "run_xmeans",
    "reshape_clusters",
]
