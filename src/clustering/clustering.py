import warnings
import numpy as np
import pandas as pd

from typing import Optional, Union, Sequence
from src.common import DataScaler, ScalingType

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer  # type: ignore

if not hasattr(np, "warnings"):
    np.warnings = warnings


class ClusterBasedDriftDetector:
    """
    Cluster-based data drift detector using X-Means clustering.

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
    y_before : pd.Series or pd.ndarray
        Class labels of the old dataset.
    X_after : pd.DataFrame
        Feature matrix of the new dataset.
    y_after : pd.Series or np.ndarray
        Class labels of the new dataset.

    k_init : int, default=2
        Minimal number of clusters that must be present in each iteration of X-means.
    k_max : int, default=10
        Maximal number of clusters that can be present in each iteration of X-means.

    thr_clusters : int, default = 1
        Minimal difference in number of clusters between data blocks to acknowledge
        the drift.
    thr_centroid_shift : float, default = 0.15
        Minimal difference in shift between clusters to acknowledge the drift.
    thr_centroid_disappear : float, default = 0.5
        Measure how much the cluster must be shifted (in case of euclidean distance) so
        it does not appear that the cluster was shifted, but instead that the old cluster
        disappeared and a new one appeared.
    thr_desc_stats : float, default = 0.2
        Minimal relative change between corresponding clusters' descriptive statistics
        to acknowledge the drift.
    thr_avg_distance_to_center_change : float, default = 0.1
        Minimal change of average euclidean distance between data points and center of
        cluster to acknowledge the drift.

    decision_thr : float, default=0.5
        Weighted average threshold above which drift is considered to occur.

    weights : Sequence[float], default=[0.4, 0.25, 0.25, 0.1]
        Values of weights used for calculating weighted average. Normalized, so that the sum
        of all is equal 1.0. Size of weights must be equal to 4.

    random_state : int, default = 42
        Value of numpy random state to ensure determinism.


    Attributes
    ----------
    X_old : np.ndarray | pd.DataFrame
        Feature matrix of the old dataset.
    y_old : np.ndarray
        Class labels of the old dataset.
    X_new : np.ndarray | pd.DataFrame
        Feature matrix of the new dataset.
    y_new : np.ndarray
        Class labels of the new dataset.
    X_old_unscaled : Optional[np.ndarray]
        Unscaled feature matrix of the old dataset.
    X_new_unscaled : Optional[np.ndarray]
        Unscaled feature matrix of the new dataset.

    k_init : int
        Minimal number of clusters that must be present in each iteration of X-means.
    k_max : int
        Maximal number of clusters that can be present in each iteration of X-means.

    thr_clusters : int
        Minimal difference in number of clusters between data blocks to acknowledge
        the drift.
    thr_centroid_shift : float
        Minimal difference in shift between clusters to acknowledge the drift.
    thr_centroid_disappear : float
        Measure of how much the cluster must be shifted so it does not appear that the
        cluster was shifted, but instead that the old cluster disappeared and a new one
        appeared.
    thr_desc_stats : float
        Minimal relative change between corresponding clusters' descriptive statistics
        to acknowledge the drift.
    thr_avg_distance_to_center_change : float
        Minimal change of average euclidean distance between data points and center of
        cluster to acknowledge the drift.

    decision_thr : float
        Weighted average threshold above which drift is considered to occur.

    weights : Sequence[float]
        Values of weights used for calculating weighted average.

    centers_old : dict | None
        Cluster centroids coordinates for the old dataset indexed by global cluster id.
    centers_new : dict | None
        Cluster centroids coordinates for the new dataset indexed by global cluster id.

    cluster_labels_old : np.ndarray | None
        Global cluster labels assigned to samples in the old dataset.
    cluster_labels_new : np.ndarray | None
        Global cluster labels assigned to samples in the new dataset.

    stats_combined : pd.DataFrame | None
        Descriptive statistics for each cluster between data blocks.
    stats_shifts : pd.DataFrame | None
        Relative changes of descriptive statistics between corresponding clusters.
    cluster_shifts : dict | None
        Euclidean distances between matched old and new cluster centroids.

    number_of_clusters_old : int | None
        Total number of clusters detected in the old dataset.
    number_of_clusters_new : int | None
        Total number of clusters detected in the new dataset.

    drift_flag : bool
        Indicates whether drift was detected.
    strength_of_drift : float
        Strength of detected drift as a weighted average of individual criteria.

    drift_details : dict | None
        Detailed drift information reported per class label.

    random_state : int
        Value of numpy random state to ensure determinism.

    Notes
    -----
    This detector assumes that class labels are available for both datasets
    and that drift should be analyzed independently within each class.

    The X-Means algorithm is used to automatically determine the number of
    clusters within a predefined range.
    """

    X_old: Union[np.ndarray, pd.DataFrame]
    y_old: np.ndarray
    X_new: Union[np.ndarray, pd.DataFrame]
    y_new: np.ndarray
    X_old_unscaled: Optional[np.ndarray]
    X_new_unscaled: Optional[np.ndarray]

    k_init: int
    k_max: int

    thr_clusters: int
    thr_centroid_shift: float
    thr_centroid_disappear: float
    thr_desc_stats: float
    thr_avg_distance_to_center_change: float

    decision_thr: float

    weights: Sequence[float]

    centers_old: Optional[dict[int, Optional[np.ndarray]]]
    centers_new: Optional[dict[int, Optional[np.ndarray]]]

    cluster_labels_old: Optional[np.ndarray]
    cluster_labels_new: Optional[np.ndarray]

    stats_combined: Optional[pd.DataFrame]
    stats_shifts: Optional[pd.DataFrame]
    cluster_shifts: Optional[dict[int, dict | str]]

    number_of_clusters_old: Optional[int]
    number_of_clusters_new: Optional[int]

    drift_flag: bool
    strength_of_drift: float

    drift_details: dict

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
        """
        Initializes the detector by validating inputs, scaling feature space,
        and setting drift detection thresholds.

        Input feature matrices are scaled jointly to ensure comparability
        between the before and after windows. Thresholds related to centroid
        distances are automatically adjusted for data dimensionality.

        No drift detection is performed during initialization.
        """

        X_before, X_after = self.scale_data(X_before, X_after)

        assert sorted(X_after.columns.values) == sorted(X_before.columns.values)
        self.columns = X_before.columns

        # Handle data conversion
        if hasattr(X_before, "values"):
            X_before = X_before.values
        if hasattr(y_before, "values"):
            y_before = y_before.values
        if hasattr(X_after, "values"):
            X_after = X_after.values
        if hasattr(y_after, "values"):
            y_after = y_after.values

        assert thr_centroid_disappear > thr_centroid_shift, (
            "Threshold of cluster to disappear must be higher than maximum allowed shift of centroids between data blocks"
        )

        self.X_old = X_before
        self.y_old = y_before
        self.X_new = X_after
        self.y_new = y_after

        self.k_init = k_init
        self.k_max = k_max

        self.thr_clusters = thr_clusters
        self.thr_centroid_shift = thr_centroid_shift * np.sqrt(self.X_old.shape[1])
        self.thr_centroid_disappear = thr_centroid_disappear * np.sqrt(self.X_old.shape[1])
        self.thr_desc_stats = thr_desc_stats
        self.thr_avg_distance_to_center_change = thr_avg_distance_to_center_change

        self.decision_thr = decision_thr

        assert len(weights) == 4, "Number of weights must be 4"
        self.weights = np.array(weights) / np.sum(weights)

        self.centers_old, self.centers_new = (None, None)
        self.cluster_labels_old, self.cluster_labels_new = (None, None)

        self.stats_combined = None
        self.stats_shifts = None
        self.cluster_shifts = None

        self.number_of_clusters_old, self.number_of_clusters_new = (None, None)

        self.drift_flag = False
        self.drift_details = None

        self.random_state = random_state

    def scale_data(
        self, X_before: Union[pd.DataFrame, np.ndarray], X_after: Union[pd.DataFrame, np.ndarray]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale data using Standardization to ensure comparability between
        before and after data blocks.

        Parameters
        ----------
        X_before : pd.DataFrame | np.ndarray
            Feature values for first (old) data block.
        X_after : pd.DataFrame | np.ndarray
            Feature values for second (new) data block.

        Returns
        -------
        X_before_scaled : pd.DataFrame
            Scaled feature values for first (old) data block.
        X_after_scaled : pd.DataFrame
            Scaled feature values for second (new) data block.
        """

        X_old_unscaled = X_before.copy()
        X_new_unscaled = X_after.copy()

        if hasattr(X_old_unscaled, "values"):
            X_old_unscaled = X_old_unscaled.values
        if hasattr(X_new_unscaled, "values"):
            X_new_unscaled = X_new_unscaled.values

        self.X_old_unscaled = X_old_unscaled
        self.X_new_unscaled = X_new_unscaled

        ds = DataScaler(ScalingType.Standard)
        X_before = ds.fit_transform(X_before.copy(), return_df=True)
        X_after = ds.transform(X_after.copy(), return_df=True)

        return X_before, X_after

    def _reshape_clusters(self, clusters: Sequence[Sequence[int]]) -> np.ndarray:
        """
        Convert a list of clusters (each a list of indices) into a flat array
        of cluster labels.

        Parameters
        ----------
        clusters : Sequence[Sequence[int]]
            List of lists where each sublist consists of indexes of data forming a cluster, e. g. [[0, 2, 3], [1, 4]].

        Returns
        -------
        reshaped_clusters : np.ndarray
            Flat array of cluster labels assignment, e. g. np.array([0, 1, 0, 0, 1]).
        """
        n_samples = sum(len(c) for c in clusters)
        reshaped_clusters = np.empty(n_samples, dtype=int)
        for idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                reshaped_clusters[sample_index] = idx
        return reshaped_clusters

    def _transform_labels_with_mapping(self, labels: np.ndarray, mapp: dict[int, int | str]) -> np.ndarray:
        """
        Transform cluster labels using a given mapping (i.e. when we use X-means
        twice centers before and after may have different class labels; this
        function handles the problem).

        Parameters
        ----------
        labels : np.ndarray
            Array of class labels to be mapped.
        mapp : dict[int, int | str]
            Dictionary with mappings.

        Returns
        -------
        transformed_labels : np.ndarray
            Array with labels after transformation with mapping.
        """
        n_samples = len(labels)
        transformed_labels = np.empty(n_samples, dtype=object)
        for idx, cluster_number in enumerate(labels):
            transformed_labels[idx] = mapp[cluster_number]
        return transformed_labels

    def _get_final_labels(
        self, transformed_labels: list[list[int | str]], maps: list[tuple[dict[int, int | str], list[int], list[int]]]
    ) -> list:
        """
        Assign labels in such a way, that they are not repeated.
        Handles also the case of disappearing and appearing labels.
        Ensure cluster labels are unique across all classes and handle
        disappearing/appearing clusters.

        Parameters
        ----------
        transformed_labels : list
            List of lists, where each sublist corresponds to cluster labels
            belonging to the subclass.

        maps : tuple(dict, list, list)
            Tuple with mapping of clusters (before/after), list of clusters
            that disappeared and list of clusters that appeared.

        Returns
        -------
        transformed_labels : list
            List of lists, where each sublist corresponds to cluster labels
            after transformation.
        """
        class_counter = 0

        for labels, mapp in zip(transformed_labels, maps):
            m, d, a = mapp
            for idx, l in enumerate(labels):
                if isinstance(l, np.int64) or isinstance(l, int):
                    labels[idx] = labels[idx] + class_counter
            class_counter += len(m) + len(d) - len(a)  # clusters + disappeared clusters - new clusters

        for labels, mapp in zip(transformed_labels, maps):
            m, d, a = mapp
            for idx, l in enumerate(labels):
                if type(l) is str:
                    labels[idx] = class_counter + a.index(int(l[4:]))
            class_counter += len(a)  # new clusters
        return transformed_labels

    def _merge_clusters(
        self,
        clusters: list[list[list[int]]],
        y: np.ndarray,
        maps: Optional[list[tuple[dict[int, int | str], list[int], list[int]]]] = None,
    ) -> np.ndarray:
        """
        Transform list of lists into proper cluster labels (each sublist represents assignments
        produced by X-Means). Flattens class-wise clusters into a unified cluster labeling space.

        Parameters
        ----------
        clusters : list
            3D list of cluster assignments produced by X-means (in form of list of lists with
            indexes) for each class

        y : np.ndarray
            class labels (≠ cluster labels)

        maps : tuple(dict, list, list) or None
            Tuple consisting of map (before/after), list of disappeared clusters, list of appeared
            clusters. If it is None, then the mapping won't be performed because it is not needed
            (in other words, there is no sense to map box A to box A).

        Returns
        -------
        final_labels : list
            Cluster assignments in form of flat list after all necessary transformations.
        """
        final_labels = np.zeros(len(y))
        transformed_labels = []
        if maps is None:
            maps = [({j: j for j in range(len(clusters[i]))}, [], []) for i in range(len(clusters))]

        for mapp, klass in zip(maps, clusters):
            mapping, _, _ = mapp
            cluster_labels = self._reshape_clusters(klass)
            unique_local_clusters = set(np.unique(cluster_labels))

            new_cluster_ids = unique_local_clusters.difference(set(mapping.keys()))  # appeared clusters
            for local_cluster_id in new_cluster_ids:
                mapping[local_cluster_id] = f"new_{local_cluster_id}"
            transformed_labels.append(self._transform_labels_with_mapping(cluster_labels, mapping))

        transformed_labels = self._get_final_labels(transformed_labels, maps)

        classes = sorted(list(set(y)))
        for klass, labels in zip(classes, transformed_labels):
            final_labels[np.array(y) == klass] = np.array(labels)

        return final_labels.astype(int)

    def _map_new_clusters_to_old(self) -> tuple[dict[int, int], list[int], list[int]]:
        """
        Map new clusters' assignment to old clusters' assignment based on centroid distances
        using the Hungarian algorithm (it may happen that clusters before and after have the
        same center coordinates but different labels; this function handles the problem by
        creating mapping).

        Returns
        -------
        mapping : dict
            Mapping created by running Hungarian algorithm.
        disappeared : list
            List of IDs of clusters that disappeared (those that were in first data block,
            but somehow were not present in second data block).
        appeared : list
            List of IDs of clusters that appeared (those that were not in first data block,
            but somehow were present in second data block)
        """

        n_old = len(self.centers_old)
        n_new = len(self.centers_new)

        dist = cdist(self.centers_old, self.centers_new, metric="euclidean")
        dist[dist > self.thr_centroid_disappear] = 1e10  # cluster moved above the threshold

        # padding if number of clusters changed
        size = max(n_old, n_new)
        padded = np.full((size, size), 1e9)
        padded[:n_old, :n_new] = dist

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(padded)

        mapping = {}  # new_cluster -> old_cluster
        disappeared = []  # cluster existed before, no match after
        appeared = []  # cluster new

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

    def _xmeans(self, X: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
        """
        Perform X-Means clustering on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to cluster (rows are samples, columns are features).

        Returns
        -------
        centers : np.ndarray
            Array of arrays that contain coordinates of each cluster.
        clusters : list
            List of lists where each sublist contains indices of data points belonging to
            cluster at that position.

        Notes
        -----
        kmeans_plusplus_initializer from pyclustering can take random_state but it's not standard.
        However, looking at the library, it usually uses random module. But if the user says
        "ensure deterministic", passing random_state implies we should manage seeds. The standard
        pyclustering kmeans_plusplus_initializer might access global random state. But let's check
        if it accepts a random_state argument in initializer. Assuming standard signature:
        kmeans_plusplus_initializer(data, amount_centers, candidate_centers=None, random_state=None)
        If not, we rely on global seed which we can set here if self.random_state is set.

        """
        if self.random_state is not None:
            import random

            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Try to pass random_state if supported
        try:
            init_centers = kmeans_plusplus_initializer(X, self.k_init, random_state=self.random_state).initialize()
        except TypeError:
            # Fallback if random_state is not supported by this version of pyclustering
            init_centers = kmeans_plusplus_initializer(X, self.k_init).initialize()

        # xmeans also uses random splitting sometimes in various implementations, typically relies on global random
        # setting global seeds above handles it if pyclustering relies on random/np.random

        xm = xmeans(X, init_centers, kmax=self.k_max, ccore=False)
        xm.process()
        centers = np.array(xm.get_centers())
        clusters = xm.get_clusters()
        return centers, clusters

    def _detect(
        self, X_old: np.ndarray, X_new: np.ndarray
    ) -> tuple[list[list[int]], list[list[int]], tuple[dict[int, int], list[int], list[int]]]:
        """
        Detect drift between the same class for two data blocks using X-Means clustering.

        Parameters
        ----------
        X_old : pd.DataFrame
            Feature values for first (old) data block.
        X_new : pd.DataFrame
            Feature values for second (new) data block.

        Returns
        -------
        clusters_old : list[list[int]]
            Indices of samples per cluster for old data.
        clusters_new : list[list[int]]
            Indices of samples per cluster for new data.
        mapp : tuple[dict[int, int], list[int], list[int]]
            Mapping of new cluster IDs to old ones, list of disappeared cluster IDs, list of appeared cluster IDs.
        """

        self.centers_old, clusters_old = self._xmeans(X_old)
        self.centers_new, clusters_new = self._xmeans(X_new)

        mapp = self._map_new_clusters_to_old()

        return clusters_old, clusters_new, mapp

    def detect(self) -> tuple[bool, dict[int, dict]]:
        """
        Detect drift between two data blocks using X-Means clustering for each class separately.
        Detects drift per class and returns flag and detailed explanation per class.

        Returns
        -------
        drift_flag : bool
            Flag that tells whether the drift occurred or not.
        details : dict
            Dictionary that explains why drift happened (each category marked as True).
        """

        classes = set(self.y_old).union(set(self.y_new))

        clusters_all = {"old": [], "new": []}
        details = {cl: {} for cl in classes}
        maps = []

        # 1. number of clusters
        for cl in classes:
            X_old_cl = self.X_old[np.array(self.y_old) == cl]
            X_new_cl = self.X_new[np.array(self.y_new) == cl]
            clusters_old, clusters_new, mapp = self._detect(X_old_cl, X_new_cl)

            clusters_all["old"].append(clusters_old)
            clusters_all["new"].append(clusters_new)
            maps.append(mapp)

        self.cluster_labels_old = self._merge_clusters(clusters_all["old"], self.y_old)
        self.cluster_labels_new = self._merge_clusters(clusters_all["new"], self.y_new, maps=maps)

        self.number_of_clusters_old = len(set(self.cluster_labels_old))
        self.number_of_clusters_new = len(set(self.cluster_labels_new))

        # 2. centroid shifts
        self.calculate_centroid_shifts()
        self.stats_combined = self.compute_desc_stats_for_clusters()

        # 3. desc stats change
        self.stats_shifts = self.compare_desc_stats_for_clusters(self.stats_combined)
        details_stats_shifts = self.assess_statistics_shifts(self.stats_shifts)

        # 4. Avg distance to center
        self.calculate_avg_distance_from_centroid()

        for cl in classes:
            mask_old = self.y_old == cl
            cl_old = set(self.cluster_labels_old[mask_old])

            mask_new = self.y_new == cl
            cl_new = set(self.cluster_labels_new[mask_new])

            details[cl]["nr_of_clusters"] = (
                len(set(self.cluster_labels_new[mask_new]) ^ set(self.cluster_labels_old[mask_old])) >= self.thr_clusters
            )

            idx = set(self.cluster_labels_old[self.y_old == cl]).union(set(self.cluster_labels_new[self.y_new == cl]))

            details[cl]["centroid_shift"] = {
                k: (
                    bool(self.cluster_shifts[k]["euclidean_distance"] > self.thr_centroid_shift)
                    if isinstance(self.cluster_shifts[k], dict)
                    else True
                )
                for k in idx
            }

            details[cl]["desc_stats_changes"] = {k: details_stats_shifts[k] for k in idx if k in details_stats_shifts}

            details[cl]["avg_distance_to_center"] = {
                k: (
                    bool(abs(self.avg_distance_shift[k]) > self.thr_avg_distance_to_center_change)
                    if self.avg_distance_shift[k] is not None
                    else True
                )
                for k in idx
            }

        self.drift_details = details
        self.generate_drift_flag()

        return self.drift_flag, self.drift_details

    def calculate_centroid_shifts(self) -> None:
        """
        Calculate Euclidean centroid shifts between corresponding clusters across data blocks.
        """

        def calculate_cluster_centers(X, labels):
            unique_labels = np.unique(labels)
            centers = {}
            for label in unique_labels:
                cluster_points = X[labels == label]
                center = np.mean(cluster_points, axis=0)
                centers[label] = center
            return centers

        self.centers_old = calculate_cluster_centers(self.X_old, self.cluster_labels_old)
        self.centers_old.update({i: None for i in set(self.cluster_labels_new) - set(self.cluster_labels_old)})

        self.centers_new = calculate_cluster_centers(self.X_new, self.cluster_labels_new)
        self.centers_new.update({i: None for i in set(self.cluster_labels_old) - set(self.cluster_labels_new)})

        shifts = {}
        for i in self.centers_old.keys():
            center_old = self.centers_old[i]
            center_new = self.centers_new[i]

            if center_old is None:
                shift = "appeared"
            elif center_new is None:
                shift = "disappeared"
            else:
                delta = center_new - center_old
                shift = {
                    "distance_per_feature": center_new - center_old,
                    "euclidean_distance": np.linalg.norm(delta),
                }
            shifts[i] = shift
        self.cluster_shifts = shifts

    def compute_desc_stats_for_clusters(self) -> pd.DataFrame:
        """
        Compute descriptive statistics (min, median, mean, max, std) for each cluster separately between
        data blocks. As a result, a pandas DataFrame with a 3-level MultiIndex on columns will be created
        and saved to self.stats_combined.

        Returns
        -------
        stats_combined : pd.DataFrame
            Pandas DataFrame with descriptive statistics.
        """

        def compute_cluster_stats(X, labels):
            df = pd.DataFrame(X, columns=self.columns)
            df["cluster"] = labels
            features = df.columns[:-1]
            clusters = sorted(df["cluster"].unique())

            records = []
            for cluster in clusters:
                cluster_df = df[df["cluster"] == cluster]
                stats_dict = {}
                for f in features:
                    # stats_dict[(f, 'min')] = cluster_df[f].min()
                    stats_dict[(f, "median")] = cluster_df[f].median()
                    stats_dict[(f, "mean")] = cluster_df[f].mean()
                    stats_dict[(f, "std")] = cluster_df[f].std()
                    # stats_dict[(f, 'max')] = cluster_df[f].max()
                stats_dict[("cluster", "id")] = cluster
                records.append(stats_dict)

            stats_df = pd.DataFrame(records)
            stats_df.set_index([("cluster", "id")], inplace=True)
            stats_df.sort_index(inplace=True)
            return stats_df

        stats_old = compute_cluster_stats(self.X_old, self.cluster_labels_old)
        stats_new = compute_cluster_stats(self.X_new, self.cluster_labels_new)

        stats_old.columns = pd.MultiIndex.from_tuples([("before", f, stat) for f, stat in stats_old.columns])
        stats_new.columns = pd.MultiIndex.from_tuples([("after", f, stat) for f, stat in stats_new.columns])

        stats_combined = pd.concat([stats_old, stats_new], axis=1)
        stats_combined.fillna(np.nan, inplace=True)

        stats_combined.columns = stats_combined.columns.set_levels(
            pd.CategoricalIndex(
                stats_combined.columns.levels[0],
                categories=["before", "after"],
                ordered=True,
            ),
            level=0,
        )
        stats_combined = stats_combined.sort_index(axis=1)
        return stats_combined

    def compare_desc_stats_for_clusters(self, stats_combined: pd.DataFrame) -> dict[int, dict[int, dict[str, float | str]]]:
        """
        Compare descriptive statistics and calculate the shift between them for corresponding clusters
        between data blocks.

        Parameters
        ----------
        stats_combined : pd.DataFrame
            Pandas DataFrame with a 3-level MultiIndex on columns with descriptive statistics
            of each feature between corresponding clusters and data blocks.

        Returns
        -------
        details : dict
            Relative changes between descriptive statistics across data blocks.
        """
        eps = 1e-10
        details = {}

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

                    old_value = row_df[col_old].iloc[0]
                    new_value = row_df[col_new].iloc[0]

                    if np.isnan(old_value) or np.isnan(new_value):
                        details[cluster][feature][stat] = np.nan
                        continue

                    denom = abs(old_value)
                    if denom < eps:
                        denom = 1.0

                    change = (new_value - old_value) / denom
                    details[cluster][feature][stat] = change
        return details

    def assess_statistics_shifts(
        self, stats_shifts: dict[int, dict[int, dict[int, float | str]]]
    ) -> dict[int, dict[int, dict[int, bool]]]:
        """
        Evaluate whether statistical shifts exceed a predefined threshold.

        Parameters
        ----------
        stats_shifts : dict
            Nested dictionary of statistical shift values in the form
            {class: {feature: {statistic: value | NaN}}}.

        Returns
        -------
        dict
            Nested dictionary with the same structure as stats_shifts, where each
            statistic is mapped to:
            - True if shift exceeds the threshold or the value was NaN,
            - False if not.
        """
        return {
            cl: {
                f: {s: bool(abs(v) > self.thr_desc_stats) if not np.isnan(v) else True for s, v in stats.items()}
                for f, stats in features.items()
            }
            for cl, features in stats_shifts.items()
        }

    def calculate_avg_distance_from_centroid(self) -> None:
        """
        Compute average Euclidean distance of samples to their respective cluster centroids
        for both data blocks and calculate the relative shift between them.
        """
        assert self.centers_old is not None
        assert self.centers_new is not None

        def calculate_avg_distance(X, centers, labels):
            mean_distances = {}
            for cluster_id, center in centers.items():
                if center is None:
                    mean_distances[cluster_id] = None
                else:
                    mask = labels == cluster_id
                    points = X[mask]

                    distances = np.linalg.norm(points - center, axis=1)
                    mean_distances[cluster_id] = distances.mean()

            return mean_distances

        self.avg_distance_old = calculate_avg_distance(self.X_old, self.centers_old, self.cluster_labels_old)
        self.avg_distance_new = calculate_avg_distance(self.X_new, self.centers_new, self.cluster_labels_new)

        self.avg_distance_shift = {
            i: (
                (self.avg_distance_new[i] - self.avg_distance_old[i]) / self.avg_distance_old[i]
                if self.avg_distance_old[i] is not None and self.avg_distance_new[i] is not None
                else None
            )
            for i in self.avg_distance_old.keys()
        }

    def generate_drift_flag(self) -> None:
        """
        Generate the final drift flag based on weighted average of individual criteria.
        Uses thresholds and weights provided at initialization.
        """
        eps = 1e-10
        true_counts = np.array(
            [
                sum(self.drift_details[cl]["nr_of_clusters"] is True for cl in set(self.y_old).union(set(self.y_new))),
                sum(
                    stat is True
                    for klass in self.drift_details.values()
                    for cluster in klass["desc_stats_changes"].values()
                    for feature in cluster.values()
                    for stat in feature.values()
                ),
                sum(
                    [
                        self.drift_details[cl]["centroid_shift"][label] is True
                        for cl in set(self.y_old).union(set(self.y_new))
                        for label in self.drift_details[cl]["centroid_shift"].keys()
                    ]
                ),
                sum(
                    [
                        self.drift_details[cl]["avg_distance_to_center"][label] is True
                        for cl in set(self.y_old).union(set(self.y_new))
                        for label in self.drift_details[cl]["avg_distance_to_center"].keys()
                    ]
                ),
            ]
        )

        false_counts = np.array(
            [
                sum(self.drift_details[cl]["nr_of_clusters"] is False for cl in set(self.y_old).union(set(self.y_new))),
                sum(
                    stat is False
                    for klass in self.drift_details.values()
                    for cluster in klass["desc_stats_changes"].values()
                    for feature in cluster.values()
                    for stat in feature.values()
                ),
                sum(
                    [
                        self.drift_details[cl]["centroid_shift"][label] is False
                        for cl in set(self.y_old).union(set(self.y_new))
                        for label in self.drift_details[cl]["centroid_shift"].keys()
                    ]
                ),
                sum(
                    [
                        self.drift_details[cl]["avg_distance_to_center"][label] is False
                        for cl in set(self.y_old).union(set(self.y_new))
                        for label in self.drift_details[cl]["avg_distance_to_center"].keys()
                    ]
                ),
            ]
        )

        self.strength_of_drift = (true_counts / (true_counts + false_counts + eps)) @ self.weights
        self.drift_flag = bool(self.strength_of_drift > self.decision_thr)
