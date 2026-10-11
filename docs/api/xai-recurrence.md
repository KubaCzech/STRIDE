# xAI: Recurring Concepts & Memory

The `stride.xai.recurrence` subpackage provides tools for archiving stream window representations, computing pairwise structural distance matrices, and discovering recurring concept regimes using HDBSCAN density clustering.

---

## Overview

Recurring concept detection addresses streams that alternate between distinct recurring distributions over time. By maintaining prototype vectors per window and clustering distance matrices with HDBSCAN, STRIDE recognizes when a previously seen concept re-emerges.

```python
from stride.xai.recurrence import (
    FullWindowStorage,
    cluster_windows,
    get_drift_from_clusters,
)
```

### Key Components

| Class / Function | Description |
| :--- | :--- |
| `FullWindowStorage` | Archives window snapshots (features, labels, class prototypes, fitted explainers) and computes pairwise inter-window distance matrices. |
| `cluster_windows` | Clusters precomputed window distance matrices using HDBSCAN with optional noise outlier reassignment (`fix_outliers=True`). |
| `get_drift_from_clusters` | Identifies concept transition timestamps where the assigned active concept regime ID changes. |

---

## API Specification

::: stride.xai.recurrence
