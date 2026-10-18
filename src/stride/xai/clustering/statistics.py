"""Descriptive-statistics computation and shift assessment for clusters.

These functions compute per-cluster statistics for two data blocks and
evaluate whether the observed shifts exceed configurable thresholds.
"""

import numpy as np
import pandas as pd


def compute_desc_stats_for_clusters(
    X_old: np.ndarray,
    X_new: np.ndarray,
    cluster_labels_old: np.ndarray,
    cluster_labels_new: np.ndarray,
    columns: pd.Index,
) -> pd.DataFrame:
    """Compute descriptive statistics per cluster for both data blocks.

    Parameters
    ----------
    X_old : np.ndarray
        Feature matrix of the old data block (scaled, shape ``(n_old, d)``).
    X_new : np.ndarray
        Feature matrix of the new data block (scaled, shape ``(n_new, d)``).
    cluster_labels_old : np.ndarray
        Global cluster label per sample in the old block.
    cluster_labels_new : np.ndarray
        Global cluster label per sample in the new block.
    columns : pd.Index
        Feature names used as column labels in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        3-level MultiIndex column DataFrame ``(block, feature, statistic)``
        where *block* is ``"before"`` or ``"after"``.
    """

    def _compute(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=columns)
        df["cluster"] = labels
        features = df.columns[:-1]
        clusters = sorted(df["cluster"].unique())

        records = []
        for cluster in clusters:
            cdf = df[df["cluster"] == cluster]
            stats: dict = {}
            for f in features:
                stats[(f, "median")] = cdf[f].median()
                stats[(f, "mean")] = cdf[f].mean()
                stats[(f, "std")] = cdf[f].std()
            stats[("cluster", "id")] = cluster
            records.append(stats)

        stats_df = pd.DataFrame(records)
        stats_df.set_index([("cluster", "id")], inplace=True)
        stats_df.sort_index(inplace=True)
        return stats_df

    stats_old = _compute(X_old, cluster_labels_old)
    stats_new = _compute(X_new, cluster_labels_new)

    stats_old.columns = pd.MultiIndex.from_tuples([("before", f, s) for f, s in stats_old.columns])
    stats_new.columns = pd.MultiIndex.from_tuples([("after", f, s) for f, s in stats_new.columns])

    combined = pd.concat([stats_old, stats_new], axis=1)
    combined.fillna(np.nan, inplace=True)
    combined.columns = combined.columns.set_levels(
        pd.CategoricalIndex(combined.columns.levels[0], categories=["before", "after"], ordered=True),
        level=0,
    )
    return combined.sort_index(axis=1)


def compare_desc_stats_for_clusters(
    stats_combined: pd.DataFrame,
) -> dict[int, dict[str, dict[str, float]]]:
    """Compute relative per-statistic shifts between data blocks for each cluster.

    Parameters
    ----------
    stats_combined : pd.DataFrame
        Output of :func:`compute_desc_stats_for_clusters`.

    Returns
    -------
    dict
        Nested ``{cluster: {feature: {statistic: change}}}`` mapping, where *change*
        is ``after - before``.  The detector compares standardised features, so the
        change is expressed in standard deviations of the feature.  A relative change
        would be unstable here: standardised means and medians are often close to 0.
        ``NaN`` marks statistics unavailable in either block.
    """
    details: dict = {}

    for cluster in stats_combined.index:
        row_df = stats_combined.loc[[cluster]]
        details[cluster] = {}
        features = row_df.columns.levels[1]

        for feature in features:
            details[cluster][feature] = {}
            stats_available = row_df.columns.levels[2]

            for stat in stats_available:
                col_old = ("before", feature, stat)
                col_new = ("after", feature, stat)

                if col_old not in row_df.columns or col_new not in row_df.columns:
                    details[cluster][feature][stat] = np.nan
                    continue

                old_val = row_df[col_old].iloc[0]
                new_val = row_df[col_new].iloc[0]

                if np.isnan(old_val) or np.isnan(new_val):
                    details[cluster][feature][stat] = np.nan
                    continue

                details[cluster][feature][stat] = new_val - old_val

    return details


def assess_statistics_shifts(
    stats_shifts: dict[int, dict[str, dict[str, float]]],
    thr_desc_stats: float,
) -> dict[int, dict[str, dict[str, bool]]]:
    """Convert numeric stat-shift values to boolean drift flags.

    Parameters
    ----------
    stats_shifts : dict
        Output of :func:`compare_desc_stats_for_clusters`.
    thr_desc_stats : float
        Absolute-change threshold above which a statistic is flagged as drifted.

    Returns
    -------
    dict
        Same nested structure with ``float`` values replaced by ``bool``.  NaN
        values (statistic unavailable in either block) are omitted, so they count
        neither as drift nor as stability.
    """
    return {
        cl: {
            f: {s: bool(abs(v) > thr_desc_stats) for s, v in stats.items() if not np.isnan(v)} for f, stats in features.items()
        }
        for cl, features in stats_shifts.items()
    }


def calculate_centroid_shifts(
    X_old: np.ndarray,
    X_new: np.ndarray,
    cluster_labels_old: np.ndarray,
    cluster_labels_new: np.ndarray,
) -> tuple[dict[int, np.ndarray | None], dict[int, np.ndarray | None], dict[int, dict | str]]:
    """Compute per-cluster centroid positions and Euclidean shift vectors.

    Parameters
    ----------
    X_old : np.ndarray
        Feature matrix of the old data block.
    X_new : np.ndarray
        Feature matrix of the new data block.
    cluster_labels_old : np.ndarray
        Global cluster label per sample in the old block.
    cluster_labels_new : np.ndarray
        Global cluster label per sample in the new block.

    Returns
    -------
    centers_old : dict[int, np.ndarray | None]
        Centroid per cluster ID for the old block; ``None`` for appeared clusters.
    centers_new : dict[int, np.ndarray | None]
        Centroid per cluster ID for the new block; ``None`` for disappeared clusters.
    shifts : dict[int, dict | str]
        Per-cluster shift info: ``{"distance_per_feature": ..., "euclidean_distance": ...}``
        or the strings ``"appeared"`` / ``"disappeared"``.
    """

    def _centers(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
        return {lbl: np.mean(X[labels == lbl], axis=0) for lbl in np.unique(labels)}

    centers_old = _centers(X_old, cluster_labels_old)
    centers_old.update({i: None for i in set(cluster_labels_new) - set(cluster_labels_old)})

    centers_new = _centers(X_new, cluster_labels_new)
    centers_new.update({i: None for i in set(cluster_labels_old) - set(cluster_labels_new)})

    shifts: dict = {}
    for i in centers_old:
        c_old = centers_old[i]
        c_new = centers_new[i]
        if c_old is None:
            shifts[i] = "appeared"
        elif c_new is None:
            shifts[i] = "disappeared"
        else:
            delta = c_new - c_old
            shifts[i] = {"distance_per_feature": delta, "euclidean_distance": float(np.linalg.norm(delta))}

    return centers_old, centers_new, shifts


def calculate_avg_distances(
    X_old: np.ndarray,
    X_new: np.ndarray,
    centers_old: dict[int, np.ndarray | None],
    centers_new: dict[int, np.ndarray | None],
    cluster_labels_old: np.ndarray,
    cluster_labels_new: np.ndarray,
) -> tuple[dict[int, float | None], dict[int, float | None], dict[int, float | None]]:
    """Compute the mean sample-to-centroid distance per cluster in both blocks.

    Parameters
    ----------
    X_old : np.ndarray
        Feature matrix of the old data block.
    X_new : np.ndarray
        Feature matrix of the new data block.
    centers_old : dict[int, np.ndarray | None]
        Old cluster centroids (``None`` for appeared clusters).
    centers_new : dict[int, np.ndarray | None]
        New cluster centroids (``None`` for disappeared clusters).
    cluster_labels_old : np.ndarray
        Global cluster label per sample in the old block.
    cluster_labels_new : np.ndarray
        Global cluster label per sample in the new block.

    Returns
    -------
    avg_old : dict[int, float | None]
        Mean distance per cluster in the old block (``None`` for appeared clusters).
    avg_new : dict[int, float | None]
        Mean distance per cluster in the new block (``None`` for disappeared clusters).
    shift : dict[int, float | None]
        Relative change ``(avg_new - avg_old) / avg_old`` per cluster, or ``None``
        when the cluster is absent from either block.  For clusters with zero
        spread in the old block the absolute change is returned instead.
    """

    def _mean_dist(X: np.ndarray, centers: dict, labels: np.ndarray) -> dict[int, float | None]:
        result: dict = {}
        for cid, center in centers.items():
            if center is None:
                result[cid] = None
            else:
                pts = X[labels == cid]
                result[cid] = float(np.linalg.norm(pts - center, axis=1).mean())
        return result

    eps = 1e-10
    avg_old = _mean_dist(X_old, centers_old, cluster_labels_old)
    avg_new = _mean_dist(X_new, centers_new, cluster_labels_new)

    # A single-sample (or fully degenerate) cluster has zero spread; fall back to the
    # absolute change rather than dividing by zero.
    shift = {
        i: (
            (avg_new[i] - avg_old[i]) / (avg_old[i] if avg_old[i] >= eps else 1.0)
            if avg_old[i] is not None and avg_new[i] is not None
            else None
        )
        for i in avg_old
    }
    return avg_old, avg_new, shift


def calculate_avg_distance_from_centroid(
    X_old: np.ndarray,
    X_new: np.ndarray,
    centers_old: dict[int, np.ndarray | None],
    centers_new: dict[int, np.ndarray | None],
    cluster_labels_old: np.ndarray,
    cluster_labels_new: np.ndarray,
) -> dict[int, float | None]:
    """Relative change in mean sample-to-centroid distance per cluster.

    Shortcut for the third element of :func:`calculate_avg_distances`; see there
    for parameters.

    Returns
    -------
    dict[int, float | None]
        Relative shift ``(mean_new - mean_old) / mean_old`` per cluster, or
        ``None`` when either centroid is absent.
    """
    return calculate_avg_distances(X_old, X_new, centers_old, centers_new, cluster_labels_old, cluster_labels_new)[2]
