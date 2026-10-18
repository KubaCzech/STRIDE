"""Tests for class handling in the cluster-based drift detector."""

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
from stride.xai.clustering.statistics import calculate_avg_distance_from_centroid, calculate_centroid_shifts

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
    """Every class's global cluster labels must partition that class's own samples
    exactly as X-Means did when run on that class alone."""
    from stride.xai.clustering.xmeans import reshape_clusters, run_xmeans

    for X, y, labels in (
        (detector.X_old, detector.y_old, detector.cluster_labels_old),
        (detector.X_new, detector.y_new, detector.cluster_labels_new),
    ):
        for klass in np.unique(y):
            mask = y == klass
            _, clusters = run_xmeans(X[mask], detector.k_init, detector.k_max, detector.random_state)
            test.assertTrue(
                _same_partition(labels[mask], reshape_clusters(clusters)),
                f"cluster labels of class {klass!r} do not match its own clustering",
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

    def _detector(self, centers_old: dict, centers_new: dict):
        from stride.xai.clustering import ClusterBasedDriftDetector

        X_old, y_old = _make_blobs(centers_old, n_per_blob=60, seed=0)
        X_new, y_new = _make_blobs(centers_new, n_per_blob=60, seed=1)
        detector = ClusterBasedDriftDetector(X_old, y_old, X_new, y_new)
        detector.detect()
        return detector

    def test_labels_when_set_order_differs_from_sorted_order(self):
        # In CPython set({1, 8}) iterates as [8, 1], unlike sorted order [1, 8].
        centers = {1: self.CENTERS_A, 8: self.CENTERS_B}
        detector = self._detector(centers, centers)

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
