"""X-Means clustering primitives used by ClusterBasedDriftDetector."""

import random
from typing import Sequence

import numpy as np
from pyclustering.cluster.xmeans import kmeans_plusplus_initializer, xmeans  # type: ignore


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
        Seed for NumPy and the Python ``random`` module to ensure reproducibility.
        When ``None`` the global random state is used unchanged.

    Returns
    -------
    centers : np.ndarray
        2-D array of cluster-centroid coordinates.
    clusters : list[list[int]]
        List of lists where each sublist contains the sample indices belonging to
        the corresponding centroid.

    Notes
    -----
    ``pyclustering`` relies on the global random state.  Setting ``random_state``
    here seeds both ``numpy.random`` and the built-in ``random`` module, which is
    sufficient for pyclustering's internal sampling.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Try to pass random_state if the installed pyclustering version supports it
    try:
        init_centers = kmeans_plusplus_initializer(X, k_init, random_state=random_state).initialize()
    except TypeError:
        init_centers = kmeans_plusplus_initializer(X, k_init).initialize()

    xm = xmeans(X, init_centers, kmax=k_max, ccore=False)
    xm.process()
    centers = np.array(xm.get_centers())
    clusters = xm.get_clusters()
    return centers, clusters
