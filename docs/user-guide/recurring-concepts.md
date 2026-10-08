# Recurring Concepts & Prototypes

Streaming data often exhibits cyclical or recurring concepts (e.g. seasonal regimes, recurring fraud patterns). STRIDE provides prototype-based concept memory to recognize previously encountered states.

---

## Window Storage & Pairwise Distances

`FullWindowStorage` archives window snapshots (data points, labels, prototypes, and tree explainers) throughout the entire stream:

```python
from stride.xai.recurrence import FullWindowStorage

storage = FullWindowStorage()

# In stream loop:
storage.store_window(
    iteration=t,
    x=X_batch,
    y=y_batch,
    prototypes=prototypes_dict,
    explainer=explainer_model,
    drift=is_drift,
)

# Compute pairwise distance matrix across all iterations
dist_matrix = storage.compute_distance_matrix(measure="minimal_distance")
```

---

## Concept Clustering with HDBSCAN

By applying HDBSCAN to the precomputed distance matrix, STRIDE groups recurring windows into distinct historical concept regimes:

```python
from stride.xai.recurrence import cluster_windows, get_drift_from_clusters

labels = cluster_windows(dist_matrix, fix_outliers=True)
drift_points = get_drift_from_clusters(labels)

print("Discovered concept clusters:", set(labels))
print("Identified recurring transitions at:", drift_points)
```
