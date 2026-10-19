"""Cluster-based concept drift detector (orchestrator).

This module contains only the high-level :class:`ClusterBasedDriftDetector`
class.  The algorithmic building blocks it relies on are:

* :mod:`.xmeans`     — X-Means clustering runner & label reshape
* :mod:`.matching`   — Hungarian-algorithm cluster matching
* :mod:`.statistics` — descriptive statistics & shift assessment
"""

import copy
import warnings
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from stride.common import DataScaler, ScalingType

from .matching import assign_to_reference, map_new_clusters_to_old, merge_clusters
from .statistics import (
    assess_statistics_shifts,
    calculate_avg_distances,
    calculate_shift_significance,
    calculate_centroid_shifts,
    compare_desc_stats_for_clusters,
    compute_desc_stats_for_clusters,
)
from .xmeans import run_xmeans

if not hasattr(np, "warnings"):
    np.warnings = warnings


class ClusterBasedDriftDetector:
    r"""Cluster-based concept and data drift detector using X-Means clustering.

    Within each class, X-Means discovers the cluster structure of the reference
    block. By default (``matching="reference"``) the detection block is assigned to
    the nearest reference centroid, so both blocks are compared on the same
    partition; clusters that appear (regions not covered by the reference model) or
    disappear (reference clusters that lose their mass) are detected relative to
    that model. Drift is evaluated across four criteria:

    1. Changes in the number of clusters within a class (clusters that appeared or
       disappeared).
    2. Changes in descriptive statistics computed for each cluster.
    3. Spatial shifts of cluster centroids ($\|c_{\text{before}} - c_{\text{after}}\|_2$).
    4. Changes in average sample-to-centroid dispersion.

    Criteria 2-4 are evaluated only for clusters present in both blocks. Their
    weighted share of triggered checks is the drift strength. The decision is made
    by a permutation test: blocks are re-split at random within each class and the
    strength is recomputed, giving a p-value that controls the false-alarm rate.

    Attributes:
        drift_flag: Whether drift was detected (permutation p-value below
            `permutation_alpha`, or strength above `decision_thr` without permutations).
        strength_of_drift: Weighted share of triggered criteria in $[0.0, 1.0]$.
        p_value: Permutation p-value of `strength_of_drift` (``None`` without permutations).
        null_strengths: Drift strengths obtained under the permutations.
        effective_weights: Criterion weights actually used for the score, i.e. `weights`
            renormalised over criteria that had at least one evaluation.
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

    matching: str
    alpha: float | None
    n_permutations: int
    permutation_alpha: float
    xmeans_repeat: int
    novelty_quantile: float
    n_jobs: int | None
    p_value: float | None
    null_strengths: np.ndarray | None
    shift_significance: dict[int, dict] | None

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
        matching: str = "reference",
        alpha: float | None = None,
        n_permutations: int = 50,
        permutation_alpha: float = 0.05,
        xmeans_repeat: int = 1,
        novelty_quantile: float = 0.95,
        n_jobs: int | None = None,
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
            decision_thr: Weighted score threshold required to declare global drift when
                ``n_permutations`` is 0. With permutations (the default) the decision is
                made by the permutation test and this threshold is not used.
            weights: 4-element sequence specifying relative weights for each criterion.
            random_state: Random state for deterministic clustering and permutations.
            matching: How the new block's clusters are related to the reference block.
                ``"independent"`` clusters both blocks separately and matches centroids
                with the Hungarian algorithm (centroids farther apart than
                `thr_centroid_disappear` count as disappeared + appeared).
                ``"reference"`` assigns new samples to the nearest reference centroid, so
                both blocks are compared on the same partition, and detects
                appeared/disappeared clusters relative to the reference model (see
                :func:`~stride.xai.clustering.matching.assign_to_reference`).
            alpha: Per-test significance level. When set, a centroid shift, statistic
                change or dispersion change is flagged only if it also passes a
                two-sample test that accounts for the cluster size (see
                :func:`~stride.xai.clustering.statistics.calculate_shift_significance`).
                ``None`` uses the thresholds alone. In ``"reference"`` matching it is
                also the level of the cluster-disappearance test.
            n_permutations: Number of permutations for calibrating the decision. When
                positive, samples are shuffled between the blocks within each class,
                the drift strength is recomputed for every permutation, and
                `drift_flag` is set when the permutation p-value is below
                `permutation_alpha` (instead of comparing the strength with
                `decision_thr`).
            permutation_alpha: Significance level of the permutation test.
            xmeans_repeat: Number of X-Means runs per clustering; the run with the best
                BIC is kept.
            novelty_quantile: In ``"reference"`` matching, quantile of reference
                within-cluster distances beyond which a new sample counts as novel.
            n_jobs: Number of worker processes for the permutation test (joblib
                semantics; ``None`` or 1 runs sequentially). The result does not
                depend on this value. Each worker imports ``stride`` anew, which
                takes a few seconds, so parallelism pays off for long series of
                detections in batch jobs rather than for single interactive calls.
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

        if matching not in ("independent", "reference"):
            raise ValueError(f"matching must be 'independent' or 'reference', got {matching!r}")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1) or None, got {alpha}")
        if n_permutations < 0:
            raise ValueError(f"n_permutations must be non-negative, got {n_permutations}")
        if not 0 < permutation_alpha < 1:
            raise ValueError(f"permutation_alpha must be in (0, 1), got {permutation_alpha}")
        if not 0 < novelty_quantile < 1:
            raise ValueError(f"novelty_quantile must be in (0, 1), got {novelty_quantile}")

        self.random_state = random_state
        self.classes = None
        self.matching = matching
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.permutation_alpha = permutation_alpha
        self.xmeans_repeat = xmeans_repeat
        self.novelty_quantile = novelty_quantile
        self.n_jobs = n_jobs
        self.p_value = None
        self.null_strengths = None
        self.shift_significance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> tuple[bool, dict[int, dict]]:
        """Run drift detection across all classes.

        Returns:
            Tuple of `(drift_flag, details)` where:
                - `drift_flag`: Whether drift was detected.
                - `details`: Per-class drift breakdown keyed by class label.
        """
        self._run()

        if self.n_permutations > 0:
            self.null_strengths = self._permutation_strengths()
            # Small tolerance so that permutations tying with the observed value count as
            # at least as extreme despite floating-point noise.
            n_extreme = int(np.sum(self.null_strengths >= self.strength_of_drift - 1e-12))
            self.p_value = (1 + n_extreme) / (1 + self.n_permutations)
            self.drift_flag = bool(self.p_value < self.permutation_alpha)
        else:
            self.drift_flag = bool(self.strength_of_drift > self.decision_thr)

        return self.drift_flag, self.drift_details

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Cluster both blocks, evaluate all criteria and compute the drift strength."""
        # A single, fixed class order: the per-class results below are matched to
        # samples by position, so every consumer must iterate classes identically.
        classes = sorted(set(self.y_old).union(set(self.y_new)))
        self.classes = classes

        if self.matching == "reference":
            self.cluster_labels_old, self.cluster_labels_new = self._cluster_reference(classes)
        else:
            self.cluster_labels_old, self.cluster_labels_new = self._cluster_independent(classes)

        self.number_of_clusters_old = len(set(self.cluster_labels_old))
        self.number_of_clusters_new = len(set(self.cluster_labels_new))

        # --- Centroid shifts ---
        self.centers_old, self.centers_new, self.cluster_shifts = calculate_centroid_shifts(
            self.X_old, self.X_new, self.cluster_labels_old, self.cluster_labels_new
        )

        # --- Descriptive statistics ---
        self.stats_combined = compute_desc_stats_for_clusters(
            self.X_old, self.X_new, self.cluster_labels_old, self.cluster_labels_new, self.columns
        )
        self.stats_shifts = compare_desc_stats_for_clusters(self.stats_combined)
        details_stats = assess_statistics_shifts(self.stats_shifts, self.thr_desc_stats)

        # --- Average distance to centroid ---
        self.avg_distance_old, self.avg_distance_new, self.avg_distance_shift = calculate_avg_distances(
            self.X_old,
            self.X_new,
            self.centers_old,
            self.centers_new,
            self.cluster_labels_old,
            self.cluster_labels_new,
        )

        matched_all = [k for k, v in self.cluster_shifts.items() if isinstance(v, dict)]
        significance = None
        if self.alpha is not None:
            significance = calculate_shift_significance(
                self.X_old, self.X_new, self.cluster_labels_old, self.cluster_labels_new, matched_all, self.columns
            )
            z_crit = float(norm.isf(self.alpha / 2))
        self.shift_significance = significance

        def _significant_z(z: float) -> bool:
            return significance is None or bool(np.isfinite(z) and abs(z) > z_crit) or bool(np.isinf(z))

        # --- Assemble per-class detail dict ---
        details: dict = {cl: {} for cl in classes}
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
                k: bool(self.cluster_shifts[k]["euclidean_distance"] > self.thr_centroid_shift)
                and (significance is None or bool(significance[k]["centroid_p"] < self.alpha))
                for k in matched
            }

            details[cl]["desc_stats_changes"] = {
                k: {
                    f: {
                        st: flag and (significance is None or _significant_z(significance[k]["stats_z"][f][st]))
                        for st, flag in stats.items()
                    }
                    for f, stats in details_stats[k].items()
                }
                for k in matched
                if k in details_stats
            }

            details[cl]["avg_distance_to_center"] = {
                k: bool(abs(self.avg_distance_shift[k]) > self.thr_avg_distance_to_center_change)
                and (significance is None or _significant_z(significance[k]["dispersion_z"]))
                for k in matched
            }

        self.drift_details = details
        self._compute_strength()

    def _cluster_independent(self, classes: list) -> tuple[np.ndarray, np.ndarray]:
        """Cluster both blocks separately and match centroids with the Hungarian algorithm."""
        clusters_all: dict[str, list] = {"old": [], "new": []}
        maps = []
        for cl in classes:
            X_old_cl = self.X_old[np.array(self.y_old) == cl]
            X_new_cl = self.X_new[np.array(self.y_new) == cl]

            centers_old_cl, clusters_old = run_xmeans(X_old_cl, self.k_init, self.k_max, self.random_state, self.xmeans_repeat)
            centers_new_cl, clusters_new = run_xmeans(X_new_cl, self.k_init, self.k_max, self.random_state, self.xmeans_repeat)

            maps.append(map_new_clusters_to_old(centers_old_cl, centers_new_cl, self.thr_centroid_disappear))
            clusters_all["old"].append(clusters_old)
            clusters_all["new"].append(clusters_new)

        labels_old = merge_clusters(clusters_all["old"], self.y_old, classes)
        labels_new = merge_clusters(clusters_all["new"], self.y_new, classes, maps=maps)
        return labels_old, labels_new

    def _cluster_reference(self, classes: list) -> tuple[np.ndarray, np.ndarray]:
        """Cluster the reference block and assign the new block to its clusters."""
        labels_old = np.zeros(len(self.y_old), dtype=int)
        labels_new = np.zeros(len(self.y_new), dtype=int)
        offset = 0
        for cl in classes:
            mask_old = np.asarray(self.y_old) == cl
            mask_new = np.asarray(self.y_new) == cl
            X_old_cl, X_new_cl = self.X_old[mask_old], self.X_new[mask_new]

            centers_old_cl, clusters_old = run_xmeans(X_old_cl, self.k_init, self.k_max, self.random_state, self.xmeans_repeat)
            _, clusters_new = run_xmeans(X_new_cl, self.k_init, self.k_max, self.random_state, self.xmeans_repeat)

            local_old, local_new, appeared, _ = assign_to_reference(
                X_old_cl,
                centers_old_cl,
                clusters_old,
                X_new_cl,
                clusters_new,
                novelty_quantile=self.novelty_quantile,
                alpha=self.alpha if self.alpha is not None else 0.01,
            )
            labels_old[mask_old] = local_old + offset
            labels_new[mask_new] = local_new + offset
            offset += len(centers_old_cl) + len(appeared)
        return labels_old, labels_new

    def _permutation_strengths(self) -> np.ndarray:
        """Drift strength under random re-splits of the pooled data within each class."""
        rng = np.random.default_rng(self.random_state)
        y_old, y_new = np.asarray(self.y_old), np.asarray(self.y_new)

        # All splits are drawn up front, so the result does not depend on n_jobs.
        splits = []
        for _ in range(self.n_permutations):
            X_old_p = np.empty_like(self.X_old)
            X_new_p = np.empty_like(self.X_new)
            for cl in self.classes:
                mask_old, mask_new = y_old == cl, y_new == cl
                pooled = np.vstack([self.X_old[mask_old], self.X_new[mask_new]])
                order = rng.permutation(len(pooled))
                n_old_cl = int(mask_old.sum())
                X_old_p[mask_old] = pooled[order[:n_old_cl]]
                X_new_p[mask_new] = pooled[order[n_old_cl:]]
            splits.append((X_old_p, X_new_p))

        template = self._permutation_template()
        if self.n_jobs is None or self.n_jobs == 1:
            return np.array([_permuted_strength(template, X_old_p, X_new_p) for X_old_p, X_new_p in splits])

        from joblib import Parallel, delayed

        strengths = Parallel(n_jobs=self.n_jobs)(
            delayed(_permuted_strength)(template, X_old_p, X_new_p) for X_old_p, X_new_p in splits
        )
        return np.array(strengths)

    def _permutation_template(self) -> "ClusterBasedDriftDetector":
        """Copy of the detector without results, cheap to send to worker processes."""
        template = copy.copy(self)
        for attr in (
            "stats_combined",
            "stats_shifts",
            "cluster_shifts",
            "drift_details",
            "shift_significance",
            "X_old_unscaled",
            "X_new_unscaled",
            "null_strengths",
        ):
            setattr(template, attr, None)
        return template

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

    def _compute_strength(self) -> None:
        """Compute the weighted-average drift score :attr:`strength_of_drift`.

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


def _permuted_strength(template: ClusterBasedDriftDetector, X_old: np.ndarray, X_new: np.ndarray) -> float:
    """Drift strength of *template*'s configuration on one permuted split of the data."""
    clone = copy.copy(template)
    clone.X_old, clone.X_new = X_old, X_new
    clone._run()
    return clone.strength_of_drift
