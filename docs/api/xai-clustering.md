# xAI: Clustering Dynamics

The `stride.xai.clustering` subpackage discovers class-conditional sub-populations with X-means, tracks them between two stream windows, and decides on drift with a permutation test.

---

## Overview

Clustering dynamics track multimodal feature deformations and density shifts across sliding stream windows. Clusters are discovered within each class with BIC-based X-means splitting; the detection window is assigned to the reference clusters, so STRIDE identifies cluster displacement, birth, and extinction on a common partition. See the [user guide](../user-guide/clustering-dynamics.md) for the method.

```python
from stride.xai.clustering import (
    ClusterBasedDriftDetector,
    assign_to_reference,
    run_xmeans,
)
```

### Key Components

| Class / Function | Description |
| :--- | :--- |
| `ClusterBasedDriftDetector` | Orchestrator: per-class X-means, cluster tracking, the four drift criteria, drift strength and permutation test. |
| `run_xmeans` | Executes X-means clustering with dynamic $K \in [k_{\text{init}}, k_{\max}]$ selection via BIC; optional restarts keep the best BIC. |
| `bic_score` | BIC of a hard clustering under the spherical Gaussian model of Pelleg & Moore. |
| `assign_to_reference` | Assigns a new window to reference clusters and detects appeared and disappeared clusters. |
| `map_new_clusters_to_old` | Minimum-weight bipartite matching between two sets of centroids (Kuhn-Munkres), used by `matching="independent"`. |
| `merge_clusters` | Combines per-class cluster results into globally unique cluster IDs. |
| `compute_desc_stats_for_clusters` | Computes mean, median and standard deviation per cluster and feature in both windows. |
| `calculate_shift_significance` | Two-sample test statistics for centroid, statistic and dispersion changes, scaled by cluster size. |

---

## API Specification

::: stride.xai.clustering
