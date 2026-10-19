# Clustering Dynamics

Clustering dynamics track sub-population migrations, multimodal deformations, and structural cluster topology changes between two stream windows. The analysis is class-conditional: clusters are discovered separately within each class, so the detector needs the class labels of both windows.

---

## Overview

In complex streaming environments, distribution shifts are rarely uniform across the entire feature space. Sub-populations often move at different speeds, new modes emerge (cluster birth), or existing modes vanish (cluster death).

Standard drift detectors that aggregate global statistics across the entire dataset can dilute or overlook localized sub-population shifts. `ClusterBasedDriftDetector`:

1. Discovers the number of clusters $K$ of each class in the reference window with X-Means.
2. Relates the detection window to these reference clusters and identifies clusters that appeared or disappeared.
3. Measures centroid displacement, changes of per-cluster statistics, and changes of cluster spread.
4. Decides whether the observed change exceeds what random sampling would produce, using a permutation test.

---

## Dynamic Clustering with X-Means

Standard K-means requires a fixed, user-specified number of clusters $K$. In streaming data where the number of underlying concepts can vary over time, a static $K$ is insufficient.

STRIDE uses **X-Means** (*Pelleg & Moore, 2000*), which estimates $K \in [k_{\text{init}}, k_{\max}]$:

1. **Initial Partition**: Runs K-means with $k_{\text{init}}$ centroids (k-means++ initialisation).
2. **Local Splitting**: Splits each cluster into two with a local 2-means run.
3. **Model Selection via BIC**: Keeps a split only if it improves the Bayesian Information Criterion of a spherical Gaussian model:
   $$\text{BIC}(M) = \hat{l}(D) - \frac{p}{2} \ln(R)$$
   where $\hat{l}(D)$ is the log-likelihood of the data $D$, $p$ is the number of free parameters, and $R$ is the number of points.
4. Repeats until no split improves the score or $k_{\max}$ is reached.

Features are standardised with the statistics of the reference window before clustering. With `xmeans_repeat > 1`, X-Means is run several times with different seeds and the partition with the highest BIC is kept.

---

## Relating the Detection Window to the Reference Clusters

Two independent clusterings of the same distribution generally split it differently, especially when the data has no pronounced cluster structure. Comparing two such partitions confuses a different split with drift. By default (`matching="reference"`), STRIDE therefore compares both windows on **the same partition**:

* **Tracked clusters.** Every sample of the detection window is assigned to the nearest reference centroid of its class. A K-means partition is exactly the Voronoi partition of its centroids, so both windows are split by the same cell boundaries, and any difference between them is either sampling noise or genuine change.
* **Appeared clusters (cluster birth).** A detection-window sample is *novel* if it lies farther from its nearest reference centroid than the `novelty_quantile` (default 95%) of that reference cluster's own samples. The detection window is also clustered on its own; a cluster whose members are mostly novel covers a region the reference model does not explain and is reported as appeared.
* **Disappeared clusters (cluster death).** A reference cluster disappears when it receives significantly fewer detection-window samples than its reference share (one-sided binomial test) and less than 20% of the expected count. Its remaining samples are reassigned to the nearest surviving cluster.

The previous approach, `matching="independent"`, clusters both windows separately and matches centroids with the Hungarian algorithm on the cost matrix $C_{i,j} = \lVert c_{\text{before},i} - c_{\text{after},j} \rVert_2$; centroid pairs farther apart than `thr_centroid_disappear` count as a disappearance plus an appearance.

---

## Drift Criteria and Score

For every class the detector evaluates four criteria:

| Criterion | Evaluated on | Triggered when |
| :--- | :--- | :--- |
| Number of clusters | each class | at least `thr_clusters` clusters appeared or disappeared |
| Descriptive statistics | each feature of each tracked cluster | mean, median or standard deviation changed by more than `thr_desc_stats` (in standard deviations of the feature) |
| Centroid shift | each tracked cluster | $\lVert c_{\text{after}} - c_{\text{before}} \rVert_2$ exceeds `thr_centroid_shift` $\cdot \sqrt{d}$ |
| Dispersion | each tracked cluster | the mean sample-to-centroid distance changed by more than `thr_avg_distance_to_center_change` (relative) |

Appeared and disappeared clusters only count towards the first criterion, so a single structural change is not counted several times. Optionally (`alpha`), the last three criteria additionally require a two-sample test that accounts for the cluster size to be significant; this removes spurious flags in small clusters at the cost of some sensitivity.

The **drift strength** is the weighted share of triggered checks,

$$S = \sum_{c=1}^{4} w_c \, \frac{\#\text{triggered}_c}{\#\text{evaluated}_c},$$

with weights `weights` renormalised over criteria that had at least one evaluation (exposed as `effective_weights`).

### Permutation Test

The strength is a descriptive quantity: its value without drift depends on the data (number of features, cluster structure, window size). The decision is therefore made by a **permutation test**. The samples of both windows are pooled within each class and randomly re-split into two windows of the original sizes `n_permutations` times (default 50); the strength computed on each split forms the distribution of $S$ under the hypothesis of no drift. The p-value is

$$p = \frac{1 + \#\{b : S_b \geq S_{\text{observed}}\}}{1 + B},$$

and drift is reported when $p <$ `permutation_alpha` (default 0.05). This bounds the false-alarm rate by `permutation_alpha` for exchangeable windows, whatever the dataset. The permutations can run in parallel (`n_jobs`) without changing the result. With `n_permutations=0`, drift is instead reported when $S$ exceeds `decision_thr`.

---

## Python API Usage

```python
from stride.datasets import RBFDriftDataset
from stride.xai.clustering import ClusterBasedDriftDetector

# 1. Generate an RBF stream whose class-to-cluster assignment is swapped at the drift point
dataset = RBFDriftDataset()
X, y = dataset.generate(n_samples_before=1000, n_samples_after=1000, random_seed=42)

X_ref, y_ref = X.iloc[:1000], y.iloc[:1000]
X_det, y_det = X.iloc[1000:].reset_index(drop=True), y.iloc[1000:].reset_index(drop=True)

# 2. Initialize the detector; X-Means explores between k_init and k_max clusters per class
detector = ClusterBasedDriftDetector(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    k_init=2,
    k_max=8,
    random_state=42,
)

# 3. Detect clustering dynamics
drift_flag, details = detector.detect()

print(f"Drift detected: {drift_flag} (strength {detector.strength_of_drift:.3f}, p-value {detector.p_value:.3f})")
print(f"Clusters: {detector.number_of_clusters_old} (ref) -> {detector.number_of_clusters_new} (det)")

# Inspect individual clusters: tracked clusters carry their displacement,
# the others are reported as "appeared" or "disappeared".
for cluster_id, shift in sorted(detector.cluster_shifts.items()):
    if isinstance(shift, dict):
        print(f"  Cluster {cluster_id}: displacement {shift['euclidean_distance']:.4f}")
    else:
        print(f"  Cluster {cluster_id}: {shift}")
```

### Results

`detect()` returns `(drift_flag, details)`; further results are stored on the detector.

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `drift_flag` | `bool` | Whether drift was detected. |
| `strength_of_drift` | `float` | Weighted share of triggered checks across the four criteria, in $[0, 1]$. |
| `p_value` | `float \| None` | Permutation p-value of the strength (`None` when `n_permutations=0`). |
| `null_strengths` | `np.ndarray \| None` | Strengths obtained on the permuted splits. |
| `drift_details` | `dict` | Per-class breakdown: `nr_of_clusters`, `centroid_shift`, `desc_stats_changes`, `avg_distance_to_center`. |
| `cluster_labels_old`, `cluster_labels_new` | `np.ndarray` | Global cluster ID of every sample. Clusters of the reference window are numbered `0 … n-1` (grouped by class) and keep their ID in the detection window; clusters found only in the detection window are numbered from `n`. |
| `centers_old`, `centers_new` | `dict[int, np.ndarray \| None]` | Centroid per cluster ID (`None` if the cluster is absent from that window). |
| `cluster_shifts` | `dict[int, dict \| str]` | Displacement (`euclidean_distance`, `distance_per_feature`) or `"appeared"` / `"disappeared"`. |
| `stats_shifts` | `dict` | Change of per-cluster mean, median and standard deviation, in standard deviations of each feature. |
| `avg_distance_old`, `avg_distance_new`, `avg_distance_shift` | `dict` | Mean sample-to-centroid distance per cluster and its relative change. |
| `effective_weights` | `np.ndarray` | Criterion weights used for the score, renormalised over criteria with at least one evaluation. |

### Main Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `k_init`, `k_max` | `2`, `10` | Range of the number of clusters explored by X-Means per class. |
| `matching` | `"reference"` | `"reference"` (assign to reference clusters) or `"independent"` (Hungarian matching of two clusterings). |
| `n_permutations` | `50` | Number of permutations of the test; `0` uses `decision_thr` instead. |
| `permutation_alpha` | `0.05` | Significance level of the permutation test. |
| `alpha` | `None` | Optional per-check significance level for centroid, statistic and dispersion changes. |
| `n_jobs` | `None` | Worker processes for the permutation test. |
| `random_state` | `42` | Seed for X-Means and the permutations; results are reproducible. |
