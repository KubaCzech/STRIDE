"""Cluster-based concept drift detector (orchestrator).

This module contains only the high-level :class:`ClusterBasedDriftDetector`
class.  The algorithmic building blocks it relies on are:

* :mod:`.xmeans`     — X-Means clustering runner & label reshape
* :mod:`.matching`   — Hungarian-algorithm cluster matching
* :mod:`.statistics` — descriptive statistics & shift assessment
"""

import warnings
from typing import Sequence, Union

import numpy as np
import pandas as pd

from stride.common import DataScaler, ScalingType

from .matching import map_new_clusters_to_old, merge_clusters
from .statistics import (
    assess_statistics_shifts,
    calculate_avg_distance_from_centroid,
    calculate_centroid_shifts,
    compare_desc_stats_for_clusters,
    compute_desc_stats_for_clusters,
)
from .xmeans import run_xmeans

if not hasattr(np, "warnings"):
    np.warnings = warnings


class ClusterBasedDriftDetector:
    """Cluster-based data drift detector using X-Means clustering.

    Clustering is performed independently within each class label
    (per class). The resulting clusters are then remapped and merged
    into a global cluster labeling space to allow cross-class
    comparison and unified drift assessment.

    Drift is detected based on four complementary criteria:

    1. Changes in the number of clusters within a class.
    2. Changes in descriptive statistics computed for each cluster.
    3. Shifts of cluster centroids between the old and new datasets.
    4. Changes in average distance of samples to their cluster centroids.

    Clusters from the new dataset are matched to clusters from the old dataset
    using the Hungarian algorithm applied to pairwise centroid distances.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix of the old dataset.
    y_before : pd.Series or np.ndarray
        Class labels of the old dataset.
    X_after : pd.DataFrame
        Feature matrix of the new dataset.
    y_after : pd.Series or np.ndarray
        Class labels of the new dataset.
    k_init : int, default=2
        Minimal number of clusters passed to the k-means++ initialiser.
    k_max : int, default=10
        Maximal number of clusters allowed per X-Means run.
    thr_clusters : int, default=1
        Minimal difference in cluster count to acknowledge drift.
    thr_centroid_shift : float, default=0.15
        Minimal centroid shift (Euclidean, pre-dimensionality scaling) to
        acknowledge drift.
    thr_centroid_disappear : float, default=0.5
        Euclidean distance above which a centroid pair is considered a
        disappearance + appearance rather than a migration.
    thr_desc_stats : float, default=0.2
        Minimal relative change in a descriptive statistic to acknowledge drift.
    thr_avg_distance_to_center_change : float, default=0.1
        Minimal relative change in average sample-to-centroid distance to
        acknowledge drift.
    decision_thr : float, default=0.5
        Weighted-average threshold above which overall drift is flagged.
    weights : Sequence[float], default=[0.4, 0.25, 0.25, 0.1]
        Weights for the four drift criteria (automatically normalised to sum 1).
        Length must be exactly 4.
    random_state : int, default=42
        Seed for reproducibility of X-Means clustering.

    Attributes
    ----------
    drift_flag : bool
        Whether drift was detected.
    strength_of_drift : float
        Weighted-average strength of detected drift (0–1).
    drift_details : dict | None
        Detailed per-class breakdown of which criteria fired.
    """

    X_old: Union[np.ndarray, pd.DataFrame]
    y_old: np.ndarray
    X_new: Union[np.ndarray, pd.DataFrame]
    y_new: np.ndarray
    X_old_unscaled: np.ndarray | None
    X_new_unscaled: np.ndarray | None

    k_init: int
    k_max: int

    thr_clusters: int
    thr_centroid_shift: float
    thr_centroid_disappear: float
    thr_desc_stats: float
    thr_avg_distance_to_center_change: float

    decision_thr: float
    weights: Sequence[float]

    centers_old: dict[int, np.ndarray | None] | None
    centers_new: dict[int, np.ndarray | None] | None

    cluster_labels_old: np.ndarray | None
    cluster_labels_new: np.ndarray | None

    stats_combined: pd.DataFrame | None
    stats_shifts: pd.DataFrame | None
    cluster_shifts: dict[int, dict | str] | None

    number_of_clusters_old: int | None
    number_of_clusters_new: int | None

    drift_flag: bool
    strength_of_drift: float
    drift_details: dict | None

    random_state: int

    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: Union[np.ndarray, pd.Series],
        X_after: pd.DataFrame,
        y_after: Union[np.ndarray, pd.Series],
        k_init: int = 2,
        k_max: int = 10,
        thr_clusters: int = 1,
        thr_centroid_shift: float = 0.15,
        thr_centroid_disappear: float = 0.5,
        thr_desc_stats: float = 0.2,
        thr_avg_distance_to_center_change: float = 0.1,
        decision_thr: float = 0.5,
        weights: Sequence[float] = [0.4, 0.25, 0.25, 0.1],
        random_state: int = 42,
    ) -> None:
        """Validate inputs, scale feature matrices, and store hyperparameters."""

        X_before, X_after = self._scale_data(X_before, X_after)

        assert sorted(X_after.columns.values) == sorted(X_before.columns.values)
        self.columns = X_before.columns

        if hasattr(X_before, "values"):
            X_before = X_before.values
        if hasattr(y_before, "values"):
            y_before = y_before.values
        if hasattr(X_after, "values"):
            X_after = X_after.values
        if hasattr(y_after, "values"):
            y_after = y_after.values

        assert thr_centroid_disappear > thr_centroid_shift, "thr_centroid_disappear must be greater than thr_centroid_shift"

        self.X_old = X_before
        self.y_old = y_before
        self.X_new = X_after
        self.y_new = y_after

        self.k_init = k_init
        self.k_max = k_max

        n_features = self.X_old.shape[1]
        self.thr_clusters = thr_clusters
        self.thr_centroid_shift = thr_centroid_shift * np.sqrt(n_features)
        self.thr_centroid_disappear = thr_centroid_disappear * np.sqrt(n_features)
        self.thr_desc_stats = thr_desc_stats
        self.thr_avg_distance_to_center_change = thr_avg_distance_to_center_change

        self.decision_thr = decision_thr

        assert len(weights) == 4, "Number of weights must be 4"
        self.weights = np.array(weights) / np.sum(weights)

        self.centers_old = None
        self.centers_new = None
        self.cluster_labels_old = None
        self.cluster_labels_new = None

        self.stats_combined = None
        self.stats_shifts = None
        self.cluster_shifts = None

        self.number_of_clusters_old = None
        self.number_of_clusters_new = None

        self.drift_flag = False
        self.drift_details = None
        self.strength_of_drift = 0.0

        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> tuple[bool, dict[int, dict]]:
        """Run drift detection across all classes.

        Returns
        -------
        drift_flag : bool
            Whether drift was detected.
        details : dict[int, dict]
            Per-class drift breakdown keyed by class label.
        """
        classes = set(self.y_old).union(set(self.y_new))

        clusters_all: dict[str, list] = {"old": [], "new": []}
        details: dict = {cl: {} for cl in classes}
        maps = []

        # --- Step 1: cluster each class, build cross-block mapping ---
        for cl in classes:
            X_old_cl = self.X_old[np.array(self.y_old) == cl]
            X_new_cl = self.X_new[np.array(self.y_new) == cl]

            centers_old_cl, clusters_old = run_xmeans(X_old_cl, self.k_init, self.k_max, self.random_state)
            centers_new_cl, clusters_new = run_xmeans(X_new_cl, self.k_init, self.k_max, self.random_state)

            mapp = map_new_clusters_to_old(centers_old_cl, centers_new_cl, self.thr_centroid_disappear)

            clusters_all["old"].append(clusters_old)
            clusters_all["new"].append(clusters_new)
            maps.append(mapp)

        self.cluster_labels_old = merge_clusters(clusters_all["old"], self.y_old)
        self.cluster_labels_new = merge_clusters(clusters_all["new"], self.y_new, maps=maps)

        self.number_of_clusters_old = len(set(self.cluster_labels_old))
        self.number_of_clusters_new = len(set(self.cluster_labels_new))

        # --- Step 2: centroid shifts ---
        self.centers_old, self.centers_new, self.cluster_shifts = calculate_centroid_shifts(
            self.X_old, self.X_new, self.cluster_labels_old, self.cluster_labels_new
        )

        # --- Step 3: descriptive statistics ---
        self.stats_combined = compute_desc_stats_for_clusters(
            self.X_old, self.X_new, self.cluster_labels_old, self.cluster_labels_new, self.columns
        )
        self.stats_shifts = compare_desc_stats_for_clusters(self.stats_combined)
        details_stats = assess_statistics_shifts(self.stats_shifts, self.thr_desc_stats)

        # --- Step 4: average distance to centroid ---
        avg_distance_shift = calculate_avg_distance_from_centroid(
            self.X_old,
            self.X_new,
            self.centers_old,
            self.centers_new,
            self.cluster_labels_old,
            self.cluster_labels_new,
        )

        # --- Assemble per-class detail dict ---
        for cl in classes:
            mask_old = self.y_old == cl
            mask_new = self.y_new == cl
            idx = set(self.cluster_labels_old[mask_old]).union(set(self.cluster_labels_new[mask_new]))

            details[cl]["nr_of_clusters"] = (
                len(set(self.cluster_labels_new[mask_new]) ^ set(self.cluster_labels_old[mask_old])) >= self.thr_clusters
            )

            details[cl]["centroid_shift"] = {
                k: (
                    bool(self.cluster_shifts[k]["euclidean_distance"] > self.thr_centroid_shift)
                    if isinstance(self.cluster_shifts[k], dict)
                    else True
                )
                for k in idx
            }

            details[cl]["desc_stats_changes"] = {k: details_stats[k] for k in idx if k in details_stats}

            details[cl]["avg_distance_to_center"] = {
                k: (
                    bool(abs(avg_distance_shift[k]) > self.thr_avg_distance_to_center_change)
                    if avg_distance_shift[k] is not None
                    else True
                )
                for k in idx
            }

        self.drift_details = details
        self._generate_drift_flag()
        return self.drift_flag, self.drift_details

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale_data(
        self,
        X_before: pd.DataFrame | np.ndarray,
        X_after: pd.DataFrame | np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Jointly standardise both data blocks.

        Parameters
        ----------
        X_before : pd.DataFrame | np.ndarray
        X_after : pd.DataFrame | np.ndarray

        Returns
        -------
        X_before_scaled, X_after_scaled : pd.DataFrame
        """
        X_old_raw = X_before.values.copy() if hasattr(X_before, "values") else X_before.copy()
        X_new_raw = X_after.values.copy() if hasattr(X_after, "values") else X_after.copy()
        self.X_old_unscaled = X_old_raw
        self.X_new_unscaled = X_new_raw

        ds = DataScaler(ScalingType.Standard)
        X_before = ds.fit_transform(X_before.copy(), return_df=True)
        X_after = ds.transform(X_after.copy(), return_df=True)
        return X_before, X_after

    def _generate_drift_flag(self) -> None:
        """Compute the weighted-average drift score and set :attr:`drift_flag`."""
        eps = 1e-10
        classes = set(self.y_old).union(set(self.y_new))

        true_counts = np.array(
            [
                sum(self.drift_details[cl]["nr_of_clusters"] is True for cl in classes),
                sum(
                    stat is True
                    for klass in self.drift_details.values()
                    for cluster in klass["desc_stats_changes"].values()
                    for feature in cluster.values()
                    for stat in feature.values()
                ),
                sum(
                    self.drift_details[cl]["centroid_shift"][lbl] is True
                    for cl in classes
                    for lbl in self.drift_details[cl]["centroid_shift"]
                ),
                sum(
                    self.drift_details[cl]["avg_distance_to_center"][lbl] is True
                    for cl in classes
                    for lbl in self.drift_details[cl]["avg_distance_to_center"]
                ),
            ]
        )

        false_counts = np.array(
            [
                sum(self.drift_details[cl]["nr_of_clusters"] is False for cl in classes),
                sum(
                    stat is False
                    for klass in self.drift_details.values()
                    for cluster in klass["desc_stats_changes"].values()
                    for feature in cluster.values()
                    for stat in feature.values()
                ),
                sum(
                    self.drift_details[cl]["centroid_shift"][lbl] is False
                    for cl in classes
                    for lbl in self.drift_details[cl]["centroid_shift"]
                ),
                sum(
                    self.drift_details[cl]["avg_distance_to_center"][lbl] is False
                    for cl in classes
                    for lbl in self.drift_details[cl]["avg_distance_to_center"]
                ),
            ]
        )

        self.strength_of_drift = float((true_counts / (true_counts + false_counts + eps)) @ self.weights)
        self.drift_flag = bool(self.strength_of_drift > self.decision_thr)
