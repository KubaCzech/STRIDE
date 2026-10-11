# xAI: Clustering Dynamics

The `stride.xai.clustering` subpackage provides tools for discovering streaming sub-populations dynamically via X-means and tracking centroid migrations using optimal Hungarian bipartite matching.

---

## Overview

Clustering dynamics track multimodal feature deformations and density shifts across sliding stream windows. By pairing Bayesian Information Criterion (BIC) cluster splitting with Hungarian centroid assignment, STRIDE identifies cluster displacement, birth, and extinction without supervised labels.

```python
from stride.xai.clustering import (
    ClusterBasedDriftDetector,
    map_new_clusters_to_old,
    run_xmeans,
)
```

### Key Components

| Class / Function | Description |
| :--- | :--- |
| `ClusterBasedDriftDetector` | Orchestrator fitting X-means across consecutive windows, executing Hungarian matching, and computing centroid displacement statistics. |
| `run_xmeans` | Executes X-means clustering over continuous feature matrices with dynamic $K \in [k_{\min}, k_{\max}]$ selection via BIC. |
| `map_new_clusters_to_old` | Solves minimum-weight bipartite matching between historical and current centroids via the Kuhn-Munkres algorithm. |
| `merge_clusters` | Utilities for consolidating sub-cluster partitions. |
| `compute_desc_stats_for_clusters` | Computes summary moments (mean, variance, skewness) per discovered sub-cluster. |

---

## API Specification

::: stride.xai.clustering
