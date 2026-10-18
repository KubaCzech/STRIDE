"""Hungarian-algorithm-based cluster matching utilities.

These functions solve the correspondence problem that arises when X-Means is
applied independently to two data blocks: the resulting cluster IDs are
arbitrary and must be aligned before any shift comparison can be made.
"""

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .xmeans import reshape_clusters


def map_new_clusters_to_old(
    centers_old: np.ndarray,
    centers_new: np.ndarray,
    thr_centroid_disappear: float,
) -> tuple[dict[int, int], list[int], list[int]]:
    """Match new cluster centroids to old ones using the Hungarian algorithm.

    Clusters whose centroid moved further than *thr_centroid_disappear* are
    treated as disappeared/appeared rather than shifted.

    Parameters
    ----------
    centers_old : np.ndarray
        Array of centroid coordinates for the *old* data block (shape ``(k_old, d)``).
    centers_new : np.ndarray
        Array of centroid coordinates for the *new* data block (shape ``(k_new, d)``).
    thr_centroid_disappear : float
        Euclidean-distance threshold above which a centroid pair is considered
        a disappearance + appearance rather than a migration.

    Returns
    -------
    mapping : dict[int, int]
        Maps each **new** cluster index to its matched **old** cluster index.
    disappeared : list[int]
        Old cluster indices that have no close match in the new block.
    appeared : list[int]
        New cluster indices that have no close match in the old block.
    """
    n_old = len(centers_old)
    n_new = len(centers_new)

    dist = cdist(centers_old, centers_new, metric="euclidean")
    dist[dist > thr_centroid_disappear] = 1e10

    size = max(n_old, n_new)
    padded = np.full((size, size), 1e9)
    padded[:n_old, :n_new] = dist

    row_ind, col_ind = linear_sum_assignment(padded)

    mapping: dict[int, int] = {}
    disappeared: list[int] = []
    appeared: list[int] = []

    for r, c in zip(row_ind, col_ind):
        if r < n_old and c < n_new:
            if dist[r, c] < 1e10:
                mapping[c] = r
            else:
                disappeared.append(r)
                appeared.append(c)
        elif r < n_old and c >= n_new:
            disappeared.append(r)
        elif r >= n_old and c < n_new:
            appeared.append(c)

    return mapping, disappeared, appeared


def transform_labels_with_mapping(labels: np.ndarray, mapp: dict[int, int | str]) -> np.ndarray:
    """Apply a cluster-ID mapping to a flat label array.

    Parameters
    ----------
    labels : np.ndarray
        Flat array of local cluster IDs (as produced by :func:`reshape_clusters`).
    mapp : dict[int, int | str]
        Mapping from local cluster ID to global ID or sentinel string ``"new_<id>"``.

    Returns
    -------
    np.ndarray
        Label array with the mapping applied (dtype ``object`` to accommodate
        mixed int / string sentinel values).
    """
    transformed = np.empty(len(labels), dtype=object)
    for idx, cluster_id in enumerate(labels):
        transformed[idx] = mapp[cluster_id]
    return transformed


def get_final_labels(
    transformed_labels: list[list[int | str]],
    maps: list[tuple[dict[int, int | str], list[int], list[int]]],
) -> list:
    """Assign globally unique integer labels across all classes.

    Resolves the sentinel ``"new_<id>"`` strings produced for appeared clusters
    into proper non-colliding integers.

    Parameters
    ----------
    transformed_labels : list[list[int | str]]
        Per-class lists of mapped cluster labels (may contain ``"new_<id>"`` strings).
    maps : list[tuple[dict, list, list]]
        Per-class tuples of ``(mapping, disappeared, appeared)`` as returned by
        :func:`map_new_clusters_to_old`.

    Returns
    -------
    list[list[int]]
        Same nested structure with all labels converted to non-colliding integers.
    """
    class_counter = 0

    for labels, mapp in zip(transformed_labels, maps):
        m, d, a = mapp
        for idx, lbl in enumerate(labels):
            if isinstance(lbl, (int, np.int64)):
                labels[idx] = labels[idx] + class_counter
        class_counter += len(m) + len(d) - len(a)

    for labels, mapp in zip(transformed_labels, maps):
        m, d, a = mapp
        for idx, lbl in enumerate(labels):
            if type(lbl) is str:
                labels[idx] = class_counter + a.index(int(lbl[4:]))
        class_counter += len(a)

    return transformed_labels


def merge_clusters(
    clusters: list[list[list[int]]],
    y: np.ndarray,
    classes: Sequence,
    maps: list[tuple[dict[int, int | str], list[int], list[int]]] | None = None,
) -> np.ndarray:
    """Flatten per-class X-Means results into a single global label array.

    Parameters
    ----------
    clusters : list[list[list[int]]]
        3-D list: outer = classes, middle = clusters, inner = sample indices.
    y : np.ndarray
        Class-label vector aligned with the original sample order.
    classes : Sequence
        Class labels in the same order as the outer level of *clusters* (and of
        *maps*).  Classes absent from *y* must have an empty cluster list.
    maps : list[tuple[dict, list, list]] | None
        Per-class matching tuples.  When ``None`` (old block), the identity
        mapping is used — no renaming needed.

    Returns
    -------
    np.ndarray
        Integer label array of shape ``(n_samples,)`` with globally unique IDs.
    """
    if len(classes) != len(clusters):
        raise ValueError(f"Got {len(classes)} classes but cluster results for {len(clusters)} classes.")

    y = np.asarray(y)
    final_labels = np.zeros(len(y))
    transformed_labels = []

    if maps is None:
        maps = [({j: j for j in range(len(clusters[i]))}, [], []) for i in range(len(clusters))]

    for mapp, klass in zip(maps, clusters):
        mapping, _, _ = mapp
        cluster_labels = reshape_clusters(klass)
        new_ids = set(np.unique(cluster_labels)).difference(set(mapping.keys()))
        for local_id in new_ids:
            mapping[local_id] = f"new_{local_id}"
        transformed_labels.append(transform_labels_with_mapping(cluster_labels, mapping))

    transformed_labels = get_final_labels(transformed_labels, maps)

    for klass, labels in zip(classes, transformed_labels):
        final_labels[y == klass] = np.array(labels)

    return final_labels.astype(int)


def assign_to_reference(
    X_old: np.ndarray,
    centers_old: np.ndarray,
    clusters_old: list[list[int]],
    X_new: np.ndarray,
    clusters_new: list[list[int]],
    novelty_quantile: float = 0.95,
    alpha: float = 0.01,
    min_share_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Label the new block with the clusters of the reference (old) block.

    Instead of matching two independent clusterings, new samples are assigned to
    the nearest *old* centroid, so both blocks are split by the same Voronoi
    cells (the cells of a k-means/X-Means partition).  Differences between the
    blocks then reflect sampling noise or genuine change rather than two
    different, equally valid partitions of the same data.

    Structural changes are detected relative to the reference model:

    * **appeared** — a cluster of the new block's own clustering (*clusters_new*)
      whose members are mostly *novel*, i.e. farther from their nearest old
      centroid than ``novelty_quantile`` of that old cluster's own samples;
    * **disappeared** — an old cluster that receives significantly fewer new
      samples than its reference share (one-sided binomial test at *alpha*) and
      less than ``min_share_ratio`` of the expected count.  Its new samples are
      reassigned to the nearest remaining old cluster.

    Parameters
    ----------
    X_old : np.ndarray
        Reference samples of one class, shape ``(n_old, d)``.
    centers_old : np.ndarray
        X-Means centroids of *X_old*, shape ``(k, d)``.
    clusters_old : list[list[int]]
        X-Means clusters of *X_old* (sample indices per centroid).
    X_new : np.ndarray
        New samples of the same class, shape ``(n_new, d)``.
    clusters_new : list[list[int]]
        Independent X-Means clusters of *X_new*, used only to find appeared clusters.
    novelty_quantile : float, default=0.95
        Quantile of the old within-cluster distances defining each cluster's radius.
    alpha : float, default=0.01
        Significance level of the disappearance test.
    min_share_ratio : float, default=0.2
        A cluster can only disappear if it keeps less than this fraction of its
        expected number of new samples.

    Returns
    -------
    labels_old : np.ndarray
        Local cluster label per old sample (``0 .. k-1``).
    labels_new : np.ndarray
        Local cluster label per new sample: an old cluster ID, or ``k + i`` for
        the *i*-th appeared cluster.
    appeared : list[int]
        Local IDs of appeared clusters.
    disappeared : list[int]
        Local IDs of disappeared old clusters.
    """
    from scipy.stats import binomtest

    k = len(centers_old)
    n_old, n_new = len(X_old), len(X_new)
    labels_old = reshape_clusters(clusters_old) if n_old else np.empty(0, dtype=int)

    if n_new == 0:
        return labels_old, np.empty(0, dtype=int), [], list(range(k))
    if k == 0:
        labels_new = reshape_clusters(clusters_new)
        return labels_old, labels_new, sorted(set(labels_new.tolist())), []

    radius = np.array(
        [np.quantile(np.linalg.norm(X_old[labels_old == j] - centers_old[j], axis=1), novelty_quantile) for j in range(k)]
    )
    dist = cdist(X_new, centers_old)
    nearest = dist.argmin(axis=1)
    novel = dist[np.arange(n_new), nearest] > radius[nearest]

    labels_new = nearest.copy()
    appeared = []
    for members in clusters_new:
        members = np.asarray(members)
        if len(members) and novel[members].mean() > 0.5:
            appeared.append(k + len(appeared))
            labels_new[members] = appeared[-1]

    shares = np.bincount(labels_old, minlength=k) / n_old
    counts = np.bincount(labels_new[labels_new < k], minlength=k)
    disappeared = [
        j
        for j in range(k)
        if counts[j] < min_share_ratio * shares[j] * n_new
        and binomtest(int(counts[j]), n_new, shares[j], alternative="less").pvalue < alpha
    ]

    if disappeared:
        orphans = np.isin(labels_new, disappeared)
        remaining = [j for j in range(k) if j not in disappeared]
        if remaining:
            labels_new[orphans] = np.asarray(remaining)[dist[np.ix_(orphans, remaining)].argmin(axis=1)]
        elif orphans.any():
            appeared.append(k + len(appeared))
            labels_new[orphans] = appeared[-1]

    return labels_old, labels_new, appeared, disappeared
