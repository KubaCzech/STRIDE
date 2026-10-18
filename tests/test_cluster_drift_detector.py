"""Tests for class handling and drift scoring in the cluster-based drift detector."""

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from stride.xai.clustering.matching import merge_clusters
from stride.xai.clustering.statistics import (
    assess_statistics_shifts,
    calculate_avg_distance_from_centroid,
    calculate_centroid_shifts,
    compare_desc_stats_for_clusters,
    compute_desc_stats_for_clusters,
)

HAS_PYCLUSTERING = importlib.util.find_spec("pyclustering") is not None
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def _make_blobs(centers_per_class: dict, n_per_blob: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Well-separated 2-D Gaussian blobs, shuffled so class samples are interleaved."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for klass, centers in centers_per_class.items():
        for center in centers:
            X.append(rng.normal(center, 0.2, size=(n_per_blob, 2)))
            y.extend([klass] * n_per_blob)
    X = np.vstack(X)
    y = np.array(y)
    order = rng.permutation(len(y))
    return pd.DataFrame(X[order], columns=["x0", "x1"]), y[order]


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two label arrays describe the same partition up to renaming."""
    return len(set(zip(a, b))) == len(set(a)) == len(set(b))


def _assert_labels_follow_class_clustering(test: unittest.TestCase, detector) -> None:
    """Cluster labels must be derived from each class's own samples.

    The reference block is always partitioned exactly as X-Means partitions that
    class alone.  The new block is, in ``"independent"`` matching, likewise its own
    X-Means partition; in ``"reference"`` matching every new sample that is not in an
    appeared cluster carries the nearest reference cluster *of its own class*.
    """
    from stride.xai.clustering.xmeans import reshape_clusters, run_xmeans

    blocks = [(detector.X_old, detector.y_old, detector.cluster_labels_old, True)]
    blocks.append((detector.X_new, detector.y_new, detector.cluster_labels_new, detector.matching == "independent"))
    for X, y, labels, own_clustering in blocks:
        for klass in np.unique(y):
            mask = y == klass
            if own_clustering:
                _, clusters = run_xmeans(X[mask], detector.k_init, detector.k_max, detector.random_state)
                test.assertTrue(
                    _same_partition(labels[mask], reshape_clusters(clusters)),
                    f"cluster labels of class {klass!r} do not match its own clustering",
                )
                continue

            old_ids = set(detector.cluster_labels_old[detector.y_old == klass])
            tracked = [k for k in sorted(old_ids) if k in set(labels[mask])]
            in_tracked = np.isin(labels[mask], tracked)
            if not tracked:
                continue
            centers = np.array([detector.X_old[detector.cluster_labels_old == k].mean(axis=0) for k in tracked])
            nearest = np.asarray(tracked)[np.linalg.norm(X[mask][:, None] - centers[None], axis=2).argmin(axis=1)]
            test.assertTrue(
                np.array_equal(labels[mask][in_tracked], nearest[in_tracked]),
                f"new samples of class {klass!r} are not assigned to their nearest reference cluster",
            )


class TestMergeClusters(unittest.TestCase):
    def test_follows_given_class_order(self):
        y = np.array(["a", "b", "a", "b", "b"])
        # outer order: "b" first, then "a" (not sorted)
        clusters = [[[0, 2], [1]], [[0], [1]]]  # class "b": 2 clusters, class "a": 2 clusters

        labels = merge_clusters(clusters, y, classes=["b", "a"])

        np.testing.assert_array_equal(labels, [2, 0, 3, 1, 0])

    def test_class_absent_from_block(self):
        y = np.array([0, 0, 0])
        clusters = [[[0, 1], [2]], []]  # class 1 has no samples in this block

        labels = merge_clusters(clusters, y, classes=[0, 1])

        np.testing.assert_array_equal(labels, [0, 0, 1])

    def test_rejects_mismatched_classes(self):
        with self.assertRaises(ValueError):
            merge_clusters([[[0]]], np.array([0]), classes=[0, 1])


class TestAvgDistanceFromCentroid(unittest.TestCase):
    def test_single_sample_cluster_in_old_block(self):
        X_old = np.array([[0.0, 0.0]])
        X_new = np.array([[0.0, 0.0], [1.0, 0.0]])
        labels_old = np.array([0])
        labels_new = np.array([0, 0])
        centers_old, centers_new, _ = calculate_centroid_shifts(X_old, X_new, labels_old, labels_new)

        shifts = calculate_avg_distance_from_centroid(X_old, X_new, centers_old, centers_new, labels_old, labels_new)

        self.assertAlmostEqual(shifts[0], 0.5)


class TestDescriptiveStatisticsShifts(unittest.TestCase):
    def test_change_is_absolute_difference_on_standardised_data(self):
        # Features are standardised, so a mean moving from 0.01 to 0.11 is a 0.1-sigma
        # change; a relative change would report it as +1000%.
        X_old = np.array([[0.0], [0.02]])
        X_new = np.array([[0.1], [0.12]])
        labels = np.array([0, 0])
        stats = compute_desc_stats_for_clusters(X_old, X_new, labels, labels, pd.Index(["f"]))

        shifts = compare_desc_stats_for_clusters(stats)

        self.assertAlmostEqual(shifts[0]["f"]["mean"], 0.1)
        self.assertAlmostEqual(shifts[0]["f"]["median"], 0.1)
        self.assertAlmostEqual(shifts[0]["f"]["std"], 0.0)

    def test_missing_values_are_not_counted_as_drift(self):
        assessed = assess_statistics_shifts({0: {"f": {"mean": 0.5, "median": 0.0, "std": np.nan}}}, thr_desc_stats=0.2)

        self.assertEqual(assessed, {0: {"f": {"mean": True, "median": False}}})


@unittest.skipUnless(HAS_PYCLUSTERING, "pyclustering not installed")
class TestClusterBasedDriftDetectorScoring(unittest.TestCase):
    def _detector(self, centers_old: dict, centers_new: dict):
        from stride.xai.clustering import ClusterBasedDriftDetector

        X_old, y_old = _make_blobs(centers_old, n_per_blob=60, seed=0)
        X_new, y_new = _make_blobs(centers_new, n_per_blob=60, seed=1)
        detector = ClusterBasedDriftDetector(X_old, y_old, X_new, y_new)
        detector.detect()
        return detector

    def test_exposes_average_distances_used_by_dashboard(self):
        centers = {0: TestClusterBasedDriftDetectorClasses.CENTERS_A, 1: TestClusterBasedDriftDetectorClasses.CENTERS_B}
        detector = self._detector(centers, centers)

        all_clusters = set(detector.cluster_shifts)
        self.assertEqual(set(detector.avg_distance_old), all_clusters)
        self.assertEqual(set(detector.avg_distance_new), all_clusters)
        self.assertEqual(set(detector.avg_distance_shift), all_clusters)
        for k, shift in detector.avg_distance_shift.items():
            old, new = detector.avg_distance_old[k], detector.avg_distance_new[k]
            self.assertAlmostEqual(shift, (new - old) / old)

    def test_appeared_cluster_counts_only_towards_cluster_count(self):
        T = TestClusterBasedDriftDetectorClasses
        detector = self._detector(
            {0: T.CENTERS_A, 1: T.CENTERS_B},
            {0: T.CENTERS_A + [(0, 8)], 1: T.CENTERS_B},  # class 0 gains a far-away blob
        )

        appeared = {k for k, v in detector.cluster_shifts.items() if v == "appeared"}
        matched = {k for k, v in detector.cluster_shifts.items() if isinstance(v, dict)}
        self.assertTrue(matched)
        self.assertTrue(appeared)
        self.assertTrue(detector.drift_details[0]["nr_of_clusters"])
        for klass, details in detector.drift_details.items():
            for criterion in ("centroid_shift", "avg_distance_to_center", "desc_stats_changes"):
                with self.subTest(klass=klass, criterion=criterion):
                    self.assertLessEqual(set(details[criterion]), matched)

    def test_complete_structural_change_is_full_strength_drift(self):
        # Every cluster of every class moves far away, so no cluster can be matched and
        # only the cluster-count criterion has anything to evaluate. Criteria with no
        # evaluations must not dilute the score.
        T = TestClusterBasedDriftDetectorClasses
        far_a = [(x, y + 20) for x, y in T.CENTERS_A]
        far_b = [(x, y + 20) for x, y in T.CENTERS_B]
        detector = self._detector({0: T.CENTERS_A, 1: T.CENTERS_B}, {0: far_a, 1: far_b})

        self.assertFalse(any(isinstance(v, dict) for v in detector.cluster_shifts.values()))
        self.assertAlmostEqual(detector.strength_of_drift, 1.0, places=6)
        self.assertTrue(detector.drift_flag)

    def test_no_drift_on_identical_blocks(self):
        T = TestClusterBasedDriftDetectorClasses
        from stride.xai.clustering import ClusterBasedDriftDetector

        X, y = _make_blobs({0: T.CENTERS_A, 1: T.CENTERS_B}, n_per_blob=60, seed=0)
        detector = ClusterBasedDriftDetector(X, y, X.copy(), y.copy())
        flag, _ = detector.detect()

        self.assertFalse(flag)
        self.assertAlmostEqual(detector.strength_of_drift, 0.0, places=6)


@unittest.skipUnless(HAS_PYCLUSTERING, "pyclustering not installed")
class TestRunXMeansSmallInput(unittest.TestCase):
    def test_empty_input(self):
        from stride.xai.clustering.xmeans import run_xmeans

        centers, clusters = run_xmeans(np.empty((0, 3)), k_init=2, k_max=10)

        self.assertEqual(centers.shape, (0, 3))
        self.assertEqual(clusters, [])

    def test_fewer_samples_than_k_init(self):
        from stride.xai.clustering.xmeans import run_xmeans

        X = np.array([[1.0, 2.0, 3.0]])
        centers, clusters = run_xmeans(X, k_init=2, k_max=10)

        np.testing.assert_array_equal(centers, X)
        self.assertEqual(clusters, [[0]])


@unittest.skipUnless(HAS_PYCLUSTERING, "pyclustering not installed")
class TestClusterBasedDriftDetectorClasses(unittest.TestCase):
    CENTERS_A = [(0, 0), (0, 4)]
    CENTERS_B = [(8, 0), (8, 4), (8, 8)]

    def _detector(self, centers_old: dict, centers_new: dict, **kwargs):
        from stride.xai.clustering import ClusterBasedDriftDetector

        X_old, y_old = _make_blobs(centers_old, n_per_blob=60, seed=0)
        X_new, y_new = _make_blobs(centers_new, n_per_blob=60, seed=1)
        detector = ClusterBasedDriftDetector(X_old, y_old, X_new, y_new, **kwargs)
        detector.detect()
        return detector

    def test_labels_when_set_order_differs_from_sorted_order(self):
        # In CPython set({1, 8}) iterates as [8, 1], unlike sorted order [1, 8].
        centers = {1: self.CENTERS_A, 8: self.CENTERS_B}
        for matching in ("reference", "independent"):
            with self.subTest(matching=matching):
                detector = self._detector(centers, centers, matching=matching, n_permutations=0)

                _assert_labels_follow_class_clustering(self, detector)

    def test_string_labels_independent_of_hash_seed(self):
        # The iteration order of a set of strings depends on PYTHONHASHSEED, so the
        # check runs in fresh interpreters with several seeds.
        script = textwrap.dedent(
            f"""
            import sys, unittest
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            from test_cluster_drift_detector import (
                TestClusterBasedDriftDetectorClasses as T, _assert_labels_follow_class_clustering,
            )
            for centers in (
                {{"cat": T.CENTERS_A, "dog": T.CENTERS_B}},  # classes of different size
                {{"cat": T.CENTERS_A, "dog": T.CENTERS_B[:2]}},  # classes of equal size
            ):
                _assert_labels_follow_class_clustering(unittest.TestCase(), T()._detector(centers, centers))
            """
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC_DIR), env.get("PYTHONPATH")]))
        for seed in range(6):
            with self.subTest(PYTHONHASHSEED=seed):
                env["PYTHONHASHSEED"] = str(seed)
                result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_class_only_in_new_block(self):
        detector = self._detector({0: self.CENTERS_A}, {0: self.CENTERS_A, 1: self.CENTERS_B})

        _assert_labels_follow_class_clustering(self, detector)
        new_class_clusters = set(detector.cluster_labels_new[detector.y_new == 1])
        self.assertTrue(new_class_clusters)
        self.assertTrue(all(detector.cluster_shifts[c] == "appeared" for c in new_class_clusters))
        self.assertTrue(detector.drift_details[1]["nr_of_clusters"])

    def test_class_only_in_old_block(self):
        detector = self._detector({0: self.CENTERS_A, 1: self.CENTERS_B}, {0: self.CENTERS_A})

        _assert_labels_follow_class_clustering(self, detector)
        old_class_clusters = set(detector.cluster_labels_old[detector.y_old == 1])
        self.assertTrue(old_class_clusters)
        self.assertTrue(all(detector.cluster_shifts[c] == "disappeared" for c in old_class_clusters))
        self.assertTrue(detector.drift_details[1]["nr_of_clusters"])

    def test_class_with_single_sample(self):
        from stride.xai.clustering import ClusterBasedDriftDetector

        X_old, y_old = _make_blobs({0: self.CENTERS_A}, n_per_blob=60, seed=0)
        X_new, y_new = _make_blobs({0: self.CENTERS_A}, n_per_blob=60, seed=1)
        y_old[0] = 1
        y_new[0] = 1

        detector = ClusterBasedDriftDetector(X_old, y_old, X_new, y_new)
        detector.detect()

        self.assertIn(1, detector.drift_details)


if __name__ == "__main__":
    unittest.main()
