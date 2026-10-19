"""Descriptive-statistics computation and shift assessment for clusters.

These functions compute per-cluster statistics for two data blocks and
evaluate whether the observed shifts exceed configurable thresholds.
"""

from typing import Iterable, Sequence

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

    Args:
        X_old: Feature matrix of the old data block (scaled, shape `(n_old, d)`).
        X_new: Feature matrix of the new data block (scaled, shape `(n_new, d)`).
        cluster_labels_old: Global cluster label per sample in the old block.
        cluster_labels_new: Global cluster label per sample in the new block.
        columns: Feature names used as column labels in the returned DataFrame.

    Returns:
        3-level MultiIndex column DataFrame `(block, feature, statistic)`
        where *block* is `"before"` or `"after"`.
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

    Args:
        stats_combined: Output of `compute_desc_stats_for_clusters`.

    Returns:
        Nested `{cluster: {feature: {statistic: change}}}` mapping, where *change*
        is `after - before`. The detector compares standardised features, so the
        change is expressed in standard deviations of the feature.
        `NaN` marks statistics unavailable in either block.
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

    Args:
        stats_shifts: Output of `compare_desc_stats_for_clusters`.
        thr_desc_stats: Absolute-change threshold above which a statistic is flagged as drifted.

    Returns:
        Same nested structure with `float` values replaced by `bool`.
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

    Args:
        X_old: Feature matrix of the old data block.
        X_new: Feature matrix of the new data block.
        cluster_labels_old: Global cluster label per sample in the old block.
        cluster_labels_new: Global cluster label per sample in the new block.

    Returns:
        Tuple of `(centers_old, centers_new, shifts)` where:
            - `centers_old`: Centroid per cluster ID for old block (`None` for appeared clusters).
            - `centers_new`: Centroid per cluster ID for new block (`None` for disappeared clusters).
            - `shifts`: Per-cluster shift info dictionary or string flag (`"appeared"` / `"disappeared"`).
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

    Args:
        X_old: Feature matrix of the old data block.
        X_new: Feature matrix of the new data block.
        centers_old: Old cluster centroids (`None` for appeared clusters).
        centers_new: New cluster centroids (`None` for disappeared clusters).
        cluster_labels_old: Global cluster label per sample in the old block.
        cluster_labels_new: Global cluster label per sample in the new block.

    Returns:
        Tuple of `(avg_old, avg_new, shift)`:
            - `avg_old`: Mean distance per cluster in old block (`None` for appeared clusters).
            - `avg_new`: Mean distance per cluster in new block (`None` for disappeared clusters).
            - `shift`: Relative change `(avg_new - avg_old) / avg_old` per cluster, or `None`.
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

    Shortcut for the third element of `calculate_avg_distances`.

    Returns:
        Relative shift `(mean_new - mean_old) / mean_old` per cluster, or
        `None` when either centroid is absent.
    """
    return calculate_avg_distances(X_old, X_new, centers_old, centers_new, cluster_labels_old, cluster_labels_new)[2]


def calculate_shift_significance(
    X_old: np.ndarray,
    X_new: np.ndarray,
    cluster_labels_old: np.ndarray,
    cluster_labels_new: np.ndarray,
    clusters: Iterable[int],
    columns: Sequence[str],
) -> dict[int, dict]:
    r"""Two-sample test statistics for the per-cluster shifts.

    The thresholds on centroid shift, descriptive statistics and dispersion are
    in absolute units, but the noise of these estimates shrinks with the cluster
    size: a 0.2-sigma change of the mean is noise for 10 samples and a clear
    signal for 500. This function provides the matching standard-error-scaled
    statistics so that a change can be required to be both large *and*
    statistically significant.

    For a cluster with samples $A$ (old) and $B$ (new):

    * **centroid** — $\chi^2 = \sum_f \Delta_f^2 / (s_{A,f}^2/n_A + s_{B,f}^2/n_B)$
      with $d$ degrees of freedom; its upper-tail p-value is returned;
    * **mean / median / std** — Welch-type z-scores using the standard errors
      $s/\sqrt{n}$, $1.2533\,s/\sqrt{n}$ and $s/\sqrt{2(n-1)}$;
    * **dispersion** — Welch z-score of the mean sample-to-centroid distance.

    Statistics are `NaN` when a cluster has fewer than two samples in a block.

    Args:
        X_old: Feature matrix of the old data block.
        X_new: Feature matrix of the new data block.
        cluster_labels_old: Global cluster label per sample in the old block.
        cluster_labels_new: Global cluster label per sample in the new block.
        clusters: Cluster IDs present in both blocks.
        columns: Feature names, used as keys of `stats_z`.

    Returns:
        Nested dictionary of `{cluster: {"centroid_p": float, "stats_z": {feature: {stat: float}}, "dispersion_z": float}}`.
    """
    from scipy.stats import chi2

    def _welch_z(diff, var_a, var_b):
        se = np.sqrt(var_a + var_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(se > 0, diff / np.where(se > 0, se, 1.0), np.where(diff == 0, 0.0, np.inf) * np.sign(diff))

    result: dict = {}
    for k in clusters:
        A = X_old[cluster_labels_old == k]
        B = X_new[cluster_labels_new == k]
        n_a, n_b = len(A), len(B)
        if n_a < 2 or n_b < 2:
            nan_stats = {f: {"mean": np.nan, "median": np.nan, "std": np.nan} for f in columns}
            result[k] = {"centroid_p": np.nan, "stats_z": nan_stats, "dispersion_z": np.nan}
            continue

        s_a, s_b = A.std(axis=0, ddof=1), B.std(axis=0, ddof=1)
        mean_diff = B.mean(axis=0) - A.mean(axis=0)
        var_mean_a, var_mean_b = s_a**2 / n_a, s_b**2 / n_b

        z_mean = _welch_z(mean_diff, var_mean_a, var_mean_b)
        z_median = _welch_z(np.median(B, axis=0) - np.median(A, axis=0), 1.2533**2 * var_mean_a, 1.2533**2 * var_mean_b)
        z_std = _welch_z(s_b - s_a, s_a**2 / (2 * (n_a - 1)), s_b**2 / (2 * (n_b - 1)))

        statistic = float(np.sum(np.where(np.isfinite(z_mean), z_mean, 0.0) ** 2))
        centroid_p = 0.0 if np.isinf(z_mean).any() else float(chi2.sf(statistic, df=A.shape[1]))

        d_a = np.linalg.norm(A - A.mean(axis=0), axis=1)
        d_b = np.linalg.norm(B - B.mean(axis=0), axis=1)
        z_disp = float(_welch_z(d_b.mean() - d_a.mean(), d_a.var(ddof=1) / n_a, d_b.var(ddof=1) / n_b))

        result[k] = {
            "centroid_p": centroid_p,
            "stats_z": {
                f: {"mean": float(z_mean[i]), "median": float(z_median[i]), "std": float(z_std[i])}
                for i, f in enumerate(columns)
            },
            "dispersion_z": z_disp,
        }
    return result
