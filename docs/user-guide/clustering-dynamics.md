# Clustering Dynamics

Clustering dynamics track sub-population shifts and multimodal deformations across stream windows.

---

## Dynamic Clustering with X-Means

STRIDE uses **X-Means** clustering to estimate the optimal number of clusters $K$ dynamically by repeatedly splitting clusters and evaluating the Bayesian Information Criterion (BIC):

```python
from stride.xai.clustering import ClusterBasedDriftDetector

detector = ClusterBasedDriftDetector(
    X_before=X_ref,
    X_after=X_det,
    k_min=2,
    k_max=10,
    random_state=42,
)

results = detector.detect()
print("Pre-drift Clusters:", len(results["centers_before"]))
print("Post-drift Clusters:", len(results["centers_after"]))
```

---

## Hungarian Centroid Assignment

To quantify how cluster centroids migrated, STRIDE solves the optimal bipartite matching problem using the **Hungarian algorithm**:

$$\min_{\pi} \sum_{i=1}^K \| c_{before, i} - c_{after, \pi(i)} \|_2$$

This yields displacement vectors for each sub-cluster and flags emerging or disappearing clusters.
