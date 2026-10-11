# Clustering Dynamics

Clustering dynamics track sub-population migrations, multimodal deformations, and structural cluster topology changes between consecutive stream windows without requiring target labels.

---

## Overview

In complex streaming environments, distribution shifts are rarely uniform across the entire feature space. Sub-populations often move at different speeds, new modes emerge (cluster birth), or existing modes coalesce (cluster merging).

Standard drift detectors that aggregate global statistics across the entire dataset can dilute or overlook localized sub-population shifts. STRIDE introduces unsupervised clustering dynamics to:

1. Discover the natural modality $K$ dynamically in each window.
2. Formulate optimal one-to-one correspondences between historical and current clusters.
3. Compute precise displacement vectors and velocities for each sub-population.

---

## Dynamic Clustering with X-Means

Standard K-means requires a fixed, user-specified number of clusters $K$. In streaming data where the number of underlying concepts can vary over time, a static $K$ is insufficient.

STRIDE incorporates **X-Means** (*Pelleg & Moore, 2000*), which dynamically estimates $K \in [k_{\min}, k_{\max}]$:

1. **Initial Partition**: Runs standard K-means with $k_{\min}$.
2. **Recursive Splitting**: For each cluster, attempts a local 2-means split along its principal direction.
3. **Model Selection via BIC**: Evaluates whether the split improves the Bayesian Information Criterion (BIC):
   $$\text{BIC}(M) = \hat{l}(D) - \frac{p}{2} \ln(R)$$
   where $\hat{l}(D)$ is the log-likelihood of data points $D$, $p$ is the number of free parameters, and $R$ is the number of instances in the cluster.
4. If $\text{BIC}(\text{split}) > \text{BIC}(\text{parent})$, the split is accepted. This continues until no splits improve the score or $k_{\max}$ is reached.

---

## Hungarian Centroid Assignment

Between consecutive windows $W_{\text{before}}$ and $W_{\text{after}}$, cluster labels assigned by clustering algorithms are arbitrary (label permutation problem). A cluster labeled 0 in $W_{\text{before}}$ may correspond to cluster 2 in $W_{\text{after}}$.

To establish unambiguous correspondences between historical and current clusters, STRIDE constructs a pairwise Euclidean distance cost matrix:

$$C_{i, j} = \left\| c_{\text{before}, i} - c_{\text{after}, j} \right\|_2$$

and solves the optimal bipartite matching problem using the **Hungarian algorithm** (Kuhn-Munkres):

$$\min_{\pi \in \Pi} \sum_{i=1}^K \left\| c_{\text{before}, i} - c_{\text{after}, \pi(i)} \right\|_2$$

where $\Pi$ denotes the set of all bijective permutations, and $K = \min(K_{\text{before}}, K_{\text{after}})$.

### Drift Metrics Derived from Matching

* **Centroid Displacement Vector**: $\vec{d}_i = c_{\text{after}, \pi(i)} - c_{\text{before}, i}$, capturing both the magnitude and spatial direction of movement.
* **Cluster Velocity**: Mean displacement divided by the inter-window time step.
* **Cluster Birth & Death**: When $K_{\text{after}} > K_{\text{before}}$, unmatched centroids in $W_{\text{after}}$ represent newly emerging modalities. When $K_{\text{after}} < K_{\text{before}}$, unmatched historical centroids represent extinct modalities.

---

## Python API Usage

The clustering dynamics workflow is implemented in `ClusterBasedDriftDetector`:

```python
from stride.datasets import RbfDriftDataset
from stride.xai.clustering import ClusterBasedDriftDetector

# 1. Generate RBF dataset with moving Gaussian cluster centroids
dataset = RbfDriftDataset()
X, y = dataset.generate(
    n_samples_before=1000,
    n_samples_after=1000,
    n_features=4,
    drift_width=50,
    random_seed=42,
)

X_ref = X.iloc[:1000]
X_det = X.iloc[1000:]

# 2. Initialize detector with bounds for cluster discovery
detector = ClusterBasedDriftDetector(
    X_before=X_ref,
    X_after=X_det,
    k_min=2,
    k_max=8,
    random_state=42,
)

# 3. Detect clustering dynamics
results = detector.detect()

print(f"Discovered clusters: {len(results['centers_before'])} (ref) -> {len(results['centers_after'])} (det)")
print(f"Mean centroid displacement: {results['mean_distance']:.4f}")

# Inspect individual cluster displacements
for idx, (dist, match_idx) in enumerate(zip(results["distances"], results["matches"])):
    print(f"  Cluster {idx} -> Matched to Cluster {match_idx} | Displacement: {dist:.4f}")
```

### Return Dictionary Structure

| Key | Type | Description |
| :--- | :--- | :--- |
| `centers_before` | `np.ndarray` | Array of cluster centroids extracted from reference window $W_{\text{ref}}$. |
| `centers_after` | `np.ndarray` | Array of cluster centroids extracted from detection window $W_{\text{det}}$. |
| `matches` | `list[int]` | Optimal Hungarian assignment indices mapping reference clusters to detection clusters. |
| `distances` | `list[float]` | Euclidean distances between matched centroid pairs. |
| `mean_distance` | `float` | Average centroid displacement across matched pairs. |
| `max_distance` | `float` | Maximum individual displacement among matched pairs. |
