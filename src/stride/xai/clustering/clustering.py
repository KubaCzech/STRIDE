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
    calculate_avg_distances,
    calculate_centroid_shifts,
    compare_desc_stats_for_clusters,
    compute_desc_stats_for_clusters,
)
from .xmeans import run_xmeans

if not hasattr(np, "warnings"):
    np.warnings = warnings


class ClusterBasedDriftDetector:
    r"""Cluster-based concept and data drift detector using X-Means clustering.

    Performs dynamic X-Means clustering independently within each class label.
    Remaps and matches clusters across stream windows using the Hungarian optimal
    bipartite assignment algorithm. Drift is evaluated across four criteria:
    1. Changes in the number of clusters within a class (clusters that appeared or
       disappeared).
    2. Changes in descriptive statistics computed for each cluster.
    3. Spatial shifts of cluster centroids ($\|c_{\text{before}} - c_{\text{after}}\|_2$).
    4. Changes in average sample-to-centroid dispersion.
    Criteria 2-4 are evaluated only for clusters matched across both blocks.

    Attributes:
        drift_flag: Whether aggregate drift score exceeded `decision_thr`.
        effective_weights: Criterion weights actually used for the score, i.e. `weights`
            renormalised over criteria that had at least one evaluation.
        strength_of_drift: Weighted-average strength of detected drift in $[0.0, 1.0]$.
        drift_details: Detailed breakdown of per-class criteria triggers.

    Examples:
        >>> detector = ClusterBasedDriftDetector(X_ref, y_ref, X_det, y_det)
        >>> flag, details = detector.detect()
    """

    X_old: np.ndarray | pd.DataFrame
    y_old: np.ndarray
    X_new: np.ndarray | pd.DataFrame
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
    effective_weights: np.ndarray | None

    centers_old: dict[int, np.ndarray | None] | None
    centers_new: dict[int, np.ndarray | None] | None

    cluster_labels_old: np.ndarray | None
    cluster_labels_new: np.ndarray | None

    stats_combined: pd.DataFrame | None
    stats_shifts: pd.DataFrame | None
    cluster_shifts: dict[int, dict | str] | None
    avg_distance_old: dict[int, float | None] | None
    avg_distance_new: dict[int, float | None] | None
    avg_distance_shift: dict[int, float | None] | None

    number_of_clusters_old: int | None
    number_of_clusters_new: int | None

    drift_flag: bool
    strength_of_drift: float
    drift_details: dict | None

    random_state: int
    classes: list | None

    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: np.ndarray | pd.Series,
        X_after: pd.DataFrame,
        y_after: np.ndarray | pd.Series,
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
        """Initialize the cluster-based drift detector.

        Args:
            X_before: Feature matrix of the reference data block.
            y_before: Class labels of the reference data block.
            X_after: Feature matrix of the detection data block.
            y_after: Class labels of the detection data block.
            k_init: Initial number of clusters for X-Means initialization.
            k_max: Maximum allowable clusters per class.
            thr_clusters: Cluster count difference threshold to flag drift.
            thr_centroid_shift: Normalized Euclidean centroid displacement threshold.
            thr_centroid_disappear: Distance threshold separating migration from cluster disappearance.
            thr_desc_stats: Threshold on the absolute change of per-cluster mean, median and
                standard deviation, in standard deviations of the (standardised) feature.
            thr_avg_distance_to_center_change: Threshold on the relative change of the mean
                sample-to-centroid distance.
            decision_thr: Weighted score threshold required to declare global drift.
            weights: 4-element sequence specifying relative weights for each criterion.
            random_state: Random state for deterministic clustering.
        """

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
        self.effective_weights = None

        self.centers_old = None
        self.centers_new = None
        self.cluster_labels_old = None
        self.cluster_labels_new = None

        self.stats_combined = None
        self.stats_shifts = None
        self.cluster_shifts = None
        self.avg_distance_old = None
        self.avg_distance_new = None
        self.avg_distance_shift = None

        self.number_of_clusters_old = None
        self.number_of_clusters_new = None

        self.drift_flag = False
        self.drift_details = None
        self.strength_of_drift = 0.0

        self.random_state = random_state
        self.classes = None

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
        # A single, fixed class order: the per-class results below are matched to
        # samples by position, so every consumer must iterate classes identically.
        classes = sorted(set(self.y_old).union(set(self.y_new)))
        self.classes = classes

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

        self.cluster_labels_old = merge_clusters(clusters_all["old"], self.y_old, classes)
        self.cluster_labels_new = merge_clusters(clusters_all["new"], self.y_new, classes, maps=maps)

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
        self.avg_distance_old, self.avg_distance_new, self.avg_distance_shift = calculate_avg_distances(
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
            # Appeared/disappeared clusters are already captured by the cluster-count
            # criterion; the per-cluster criteria only compare clusters present in both
            # blocks, so a single structural change is not counted several times.
            matched = {k for k in idx if isinstance(self.cluster_shifts[k], dict)}

            details[cl]["nr_of_clusters"] = (
                len(set(self.cluster_labels_new[mask_new]) ^ set(self.cluster_labels_old[mask_old])) >= self.thr_clusters
            )

            details[cl]["centroid_shift"] = {
                k: bool(self.cluster_shifts[k]["euclidean_distance"] > self.thr_centroid_shift) for k in matched
            }

            details[cl]["desc_stats_changes"] = {k: details_stats[k] for k in matched if k in details_stats}

            details[cl]["avg_distance_to_center"] = {
                k: bool(abs(self.avg_distance_shift[k]) > self.thr_avg_distance_to_center_change) for k in matched
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

        Args:
            X_before: Reference feature data block.
            X_after: Detection feature data block.

        Returns:
            Tuple of `(X_before_scaled, X_after_scaled)` standardized DataFrames.
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
        """Compute the weighted-average drift score and set :attr:`drift_flag`.

        Each criterion contributes the share of its evaluations that flagged drift.
        Criteria with no evaluations at all (e.g. no cluster could be matched across
        blocks, so there is no centroid shift to measure) are left out and the
        remaining weights are renormalised; they are stored in :attr:`effective_weights`.
        """
        classes = self.classes

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

        evaluated = true_counts + false_counts
        weights = np.where(evaluated > 0, self.weights, 0.0)
        self.effective_weights = weights / weights.sum() if weights.sum() > 0 else weights
        self.strength_of_drift = float((true_counts / np.maximum(evaluated, 1)) @ self.effective_weights)
        self.drift_flag = bool(self.strength_of_drift > self.decision_thr)
