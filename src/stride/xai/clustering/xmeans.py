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

    rng_state = random.getstate()
    try:
        init_centers = kmeans_plusplus_initializer(X, k_init, random_state=random_state).initialize()

        # ccore=False: the bundled C++ library is built for x86_64 only and fails to load on arm64.
        xm = xmeans(X, init_centers, kmax=k_max, ccore=False, random_state=random_state)
        xm.process()
    finally:
        random.setstate(rng_state)

    centers = np.array(xm.get_centers())
    clusters = xm.get_clusters()
    return centers, clusters
