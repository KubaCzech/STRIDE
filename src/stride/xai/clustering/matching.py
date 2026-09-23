"""Hungarian-algorithm-based cluster matching utilities.

These functions solve the correspondence problem that arises when X-Means is
applied independently to two data blocks: the resulting cluster IDs are
arbitrary and must be aligned before any shift comparison can be made.
"""

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
    maps: list[tuple[dict[int, int | str], list[int], list[int]]] | None = None,
) -> np.ndarray:
    """Flatten per-class X-Means results into a single global label array.

    Parameters
    ----------
    clusters : list[list[list[int]]]
        3-D list: outer = classes, middle = clusters, inner = sample indices.
    y : np.ndarray
        Class-label vector aligned with the original sample order.
    maps : list[tuple[dict, list, list]] | None
        Per-class matching tuples.  When ``None`` (old block), the identity
        mapping is used — no renaming needed.

    Returns
    -------
    np.ndarray
        Integer label array of shape ``(n_samples,)`` with globally unique IDs.
    """
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

    classes = sorted(set(y))
    for klass, labels in zip(classes, transformed_labels):
        final_labels[np.array(y) == klass] = np.array(labels)

    return final_labels.astype(int)
