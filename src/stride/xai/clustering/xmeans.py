"""X-Means clustering primitives used by ClusterBasedDriftDetector."""

import random
from typing import Sequence

import numpy as np
from stride.exceptions import OptionalDependencyError


def reshape_clusters(clusters: Sequence[Sequence[int]]) -> np.ndarray:
    """Convert a list of clusters (each a list of sample indices) into a flat label array.

    Parameters
    ----------
    clusters : Sequence[Sequence[int]]
        List of lists where each sublist contains the indices of samples belonging to
        that cluster, e.g. ``[[0, 2, 3], [1, 4]]``.

    Returns
    -------
    np.ndarray
        Flat array of integer cluster-label assignments, e.g. ``array([0, 1, 0, 0, 1])``.
    """
    n_samples = sum(len(c) for c in clusters)
    reshaped = np.empty(n_samples, dtype=int)
    for idx, cluster in enumerate(clusters):
        for sample_index in cluster:
            reshaped[sample_index] = idx
    return reshaped


def run_xmeans(
    X: np.ndarray,
    k_init: int,
    k_max: int,
    random_state: int | None = 42,
    repeat: int = 1,
) -> tuple[np.ndarray, list[list[int]]]:
    """Run the X-Means clustering algorithm on *X*.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (rows are samples, columns are features).
    k_init : int
        Minimum number of clusters passed to the k-means++ initialiser.
    k_max : int
        Maximum number of clusters allowed by X-Means.
    random_state : int | None, default=42
        Seed passed to pyclustering's k-means++ initialiser and to X-Means itself,
        so that every cluster split is reproducible.  When ``None`` the result is
        non-deterministic.
    repeat : int, default=1
        Number of independent X-Means runs, seeded with ``random_state``,
        ``random_state + 1``, ...; the result with the highest BIC (see
        :func:`bic_score`) is returned.  Higher values make the discovered
        structure less dependent on initialisation.

    Returns
    -------
    centers : np.ndarray
        2-D array of cluster-centroid coordinates.
    clusters : list[list[int]]
        List of lists where each sublist contains the sample indices belonging to
        the corresponding centroid.  Empty input yields no clusters, and input with
        fewer than ``k_init`` samples yields a single cluster.

    Notes
    -----
    pyclustering re-seeds Python's ``random`` module inside every k-means++ call
    (``random.seed(random_state)``), including the ones X-Means performs when
    splitting a cluster.  Seeding the global RNG beforehand is therefore not
    enough: ``random_state`` must be forwarded to ``xmeans`` explicitly.  Because
    this re-seeding touches the global ``random`` module, its state is saved and
    restored around the call so that callers' random streams are left untouched.

    pyclustering's own ``repeat`` argument is not used: it re-seeds every restart
    with the same ``random_state``, so all restarts are identical.
    """
    try:
        from pyclustering.cluster.xmeans import kmeans_plusplus_initializer, xmeans
    except ImportError as err:
        raise OptionalDependencyError(
            package_name="pyclustering",
            feature_name="X-Means clustering",
            extra_name="clustering",
        ) from err

    X = np.asarray(X)
    n_samples = len(X)
    if n_samples == 0:
        return np.empty((0, X.shape[1])), []
    if n_samples < k_init:
        # k-means++ needs at least k_init points; such a small group is a single cluster.
        return X.mean(axis=0, keepdims=True), [list(range(n_samples))]

    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")

    best: tuple[float, np.ndarray, list[list[int]]] | None = None
    rng_state = random.getstate()
    try:
        for r in range(repeat):
            seed = None if random_state is None else random_state + r
            init_centers = kmeans_plusplus_initializer(X, k_init, random_state=seed).initialize()

            # ccore=False: the bundled C++ library is built for x86_64 only and fails to load on arm64.
            xm = xmeans(X, init_centers, kmax=k_max, ccore=False, random_state=seed)
            xm.process()
            centers, clusters = np.array(xm.get_centers()), xm.get_clusters()
            if repeat == 1:
                return centers, clusters

            score = bic_score(X, centers, clusters)
            if best is None or score > best[0]:
                best = (score, centers, clusters)
    finally:
        random.setstate(rng_state)

    return best[1], best[2]


def bic_score(X: np.ndarray, centers: np.ndarray, clusters: Sequence[Sequence[int]]) -> float:
    r"""Bayesian Information Criterion of a hard clustering (higher is better).

    Uses the identical spherical Gaussian model of Pelleg & Moore (2000), with the
    per-dimension variance estimated from the pooled within-cluster scatter:

    .. math::

        \hat\sigma^2 = \frac{1}{d (N - K)} \sum_j \sum_{x \in C_j} \lVert x - c_j \rVert^2, \qquad
        \mathrm{BIC} = \hat\ell - \frac{p}{2} \log N, \quad p = (K - 1) + K d + 1

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(N, d)``.
    centers : np.ndarray
        Cluster centroids of shape ``(K, d)``.
    clusters : Sequence[Sequence[int]]
        Sample indices of each cluster, aligned with *centers*.

    Returns
    -------
    float
        BIC value; ``-inf`` when the variance cannot be estimated (``N <= K``)
        and ``+inf`` when every cluster has zero spread.
    """
    X = np.asarray(X)
    n, d = X.shape
    k = len(clusters)
    if n <= k:
        return float("-inf")

    sse = sum(float(np.sum((X[idx] - centers[j]) ** 2)) for j, idx in enumerate(clusters))
    if sse <= 0:
        return float("inf")
    sigma2 = sse / (d * (n - k))

    sizes = np.array([len(idx) for idx in clusters], dtype=float)
    sizes = sizes[sizes > 0]
    log_likelihood = float(np.sum(sizes * np.log(sizes / n))) - n * d / 2 * np.log(2 * np.pi * sigma2) - d * (n - k) / 2
    n_params = (k - 1) + k * d + 1
    return log_likelihood - n_params / 2 * np.log(n)
