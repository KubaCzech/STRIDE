"""Dynamic clustering, X-means BIC splitting, and optimal Hungarian assignment."""

from .clustering import ClusterBasedDriftDetector
from .matching import map_new_clusters_to_old, merge_clusters
from .statistics import compare_desc_stats_for_clusters, compute_desc_stats_for_clusters
from .xmeans import reshape_clusters, run_xmeans

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
    "map_new_clusters_to_old",
    "merge_clusters",
    "compute_desc_stats_for_clusters",
    "compare_desc_stats_for_clusters",
    "run_xmeans",
    "reshape_clusters",
]
