"""Tests for the reference matching, significance tests, permutation calibration and
X-Means restarts of the cluster-based drift detector."""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from stride.xai.clustering.statistics import calculate_shift_significance
from stride.xai.clustering.xmeans import bic_score

HAS_PYCLUSTERING = importlib.util.find_spec("pyclustering") is not None


def _blobs(centers, n_per_blob: int, seed: int, std: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(c, std, size=(n_per_blob, len(c))) for c in centers])


def _labels(n_per_blob: int, n_blobs: int) -> np.ndarray:
    return np.repeat(np.arange(n_blobs), n_per_blob)


class TestBicScore(unittest.TestCase):
    def test_peaks_at_true_number_of_clusters(self):
        centers = [(0, 0), (3, 0), (0, 3)]
        X = _blobs(centers, 100, seed=0)
        scores = {}
        for k, assignment in {
            1: np.zeros(300, dtype=int),
            3: _labels(100, 3),
            6: np.repeat(np.arange(6), 50),  # each true blob split in two
        }.items():
            clusters = [list(np.where(assignment == j)[0]) for j in range(k)]
            centroids = np.array([X[c].mean(axis=0) for c in clusters])
            scores[k] = bic_score(X, centroids, clusters)

        self.assertEqual(max(scores, key=scores.get), 3)


class TestShiftSignificance(unittest.TestCase):
    def setUp(self):
        self.columns = ["a", "b"]
        rng = np.random.default_rng(0)
        self.X_old = rng.normal(0, 1, size=(400, 2))
        self.labels = np.zeros(400, dtype=int)

    def test_same_distribution_is_not_significant(self):
        X_new = np.random.default_rng(1).normal(0, 1, size=(400, 2))

        sig = calculate_shift_significance(self.X_old, X_new, self.labels, self.labels, [0], self.columns)[0]

        self.assertGreater(sig["centroid_p"], 0.001)
        self.assertLess(abs(sig["stats_z"]["a"]["mean"]), 3.3)
        self.assertLess(abs(sig["dispersion_z"]), 3.3)

    def test_shifted_mean_is_significant(self):
        X_new = np.random.default_rng(1).normal(0, 1, size=(400, 2)) + [0.5, 0.0]

        sig = calculate_shift_significance(self.X_old, X_new, self.labels, self.labels, [0], self.columns)[0]

        self.assertLess(sig["centroid_p"], 1e-6)
        self.assertGreater(sig["stats_z"]["a"]["mean"], 5)
        self.assertLess(abs(sig["stats_z"]["b"]["mean"]), 3.3)

    def test_same_change_is_less_significant_in_smaller_cluster(self):
        small = calculate_shift_significance(
            self.X_old[:20], self.X_old[:20] + 0.3, self.labels[:20], self.labels[:20], [0], self.columns
        )[0]
        large = calculate_shift_significance(self.X_old, self.X_old + 0.3, self.labels, self.labels, [0], self.columns)[0]

        self.assertLess(abs(small["stats_z"]["a"]["mean"]), abs(large["stats_z"]["a"]["mean"]))

    def test_single_sample_cluster_gives_nan(self):
        sig = calculate_shift_significance(self.X_old[:1], self.X_old[:5], self.labels[:1], self.labels[:5], [0], self.columns)

        self.assertTrue(np.isnan(sig[0]["centroid_p"]))


@unittest.skipUnless(HAS_PYCLUSTERING, "pyclustering not installed")
class TestRunXMeansRepeat(unittest.TestCase):
    def setUp(self):
        X = np.random.default_rng(0).uniform(size=(1000, 2))
        self.X = (X - X.mean(axis=0)) / X.std(axis=0)

    def test_repeat_keeps_best_bic(self):
        from stride.xai.clustering.xmeans import run_xmeans

        single = bic_score(self.X, *run_xmeans(self.X, 2, 10, random_state=42, repeat=1))
        repeated = bic_score(self.X, *run_xmeans(self.X, 2, 10, random_state=42, repeat=5))

        self.assertGreaterEqual(repeated, single)

    def test_repeat_is_reproducible(self):
        from stride.xai.clustering.xmeans import run_xmeans

        a = run_xmeans(self.X, 2, 10, random_state=42, repeat=3)
        b = run_xmeans(self.X, 2, 10, random_state=42, repeat=3)

        np.testing.assert_array_equal(a[0], b[0])

    def test_rejects_non_positive_repeat(self):
        from stride.xai.clustering.xmeans import run_xmeans

        with self.assertRaises(ValueError):
            run_xmeans(self.X, 2, 10, repeat=0)


@unittest.skipUnless(HAS_PYCLUSTERING, "pyclustering not installed")
class TestAssignToReference(unittest.TestCase):
    CENTERS = [(0, 0), (4, 0), (0, 4)]

    def _assign(self, centers_new, n_new=100):
        from stride.xai.clustering.matching import assign_to_reference
        from stride.xai.clustering.xmeans import run_xmeans

        X_old = _blobs(self.CENTERS, 100, seed=0)
        X_new = _blobs(centers_new, n_new, seed=1)
        centers_old, clusters_old = run_xmeans(X_old, 2, 10)
        _, clusters_new = run_xmeans(X_new, 2, 10)
        return X_new, centers_old, assign_to_reference(X_old, centers_old, clusters_old, X_new, clusters_new)

    def test_same_distribution_has_no_structural_change(self):
        X_new, centers_old, (labels_old, labels_new, appeared, disappeared) = self._assign(self.CENTERS)

        self.assertEqual((appeared, disappeared), ([], []))
        nearest = np.linalg.norm(X_new[:, None] - centers_old[None], axis=2).argmin(axis=1)
        np.testing.assert_array_equal(labels_new, nearest)

    def test_new_blob_appears(self):
        _, centers_old, (_, labels_new, appeared, disappeared) = self._assign(self.CENTERS + [(10, 10)])

        self.assertEqual(len(appeared), 1)
        self.assertEqual(disappeared, [])
        np.testing.assert_array_equal(np.unique(labels_new[-100:]), appeared)

    def test_missing_blob_disappears(self):
        _, centers_old, (_, labels_new, appeared, disappeared) = self._assign(self.CENTERS[:2], n_new=150)

        self.assertEqual(appeared, [])
        self.assertEqual(len(disappeared), 1)
        self.assertNotIn(disappeared[0], set(labels_new))


@unittest.skipUnless(HAS_PYCLUSTERING, "pyclustering not installed")
class TestDetectorCalibration(unittest.TestCase):
    CENTERS_A = [(0, 0), (0, 4)]
    CENTERS_B = [(8, 0), (8, 4)]

    def _data(self, shift_b: float, seed_new: int = 1):
        def block(seed, shift):
            X = np.vstack([_blobs(self.CENTERS_A, 60, seed), _blobs(self.CENTERS_B, 60, seed + 100) + [0, shift]])
            return pd.DataFrame(X, columns=["x0", "x1"]), np.repeat([0, 1], 120)

        return (*block(0, 0.0), *block(seed_new, shift_b))

    def _detector(self, *data, **kwargs):
        from stride.xai.clustering import ClusterBasedDriftDetector

        detector = ClusterBasedDriftDetector(*data, **kwargs)
        detector.detect()
        return detector

    def test_no_permutations_leaves_p_value_unset(self):
        detector = self._detector(*self._data(0.0), n_permutations=0)

        self.assertIsNone(detector.p_value)
        self.assertIsNone(detector.null_strengths)

    def test_permutation_test_on_same_distribution(self):
        detector = self._detector(*self._data(0.0), matching="reference", alpha=0.01, n_permutations=30)

        self.assertEqual(len(detector.null_strengths), 30)
        self.assertGreater(detector.p_value, 0.05)
        self.assertFalse(detector.drift_flag)

    def test_permutation_test_detects_shift(self):
        detector = self._detector(*self._data(1.0), matching="reference", alpha=0.01, n_permutations=30)

        self.assertLess(detector.p_value, 0.05)
        self.assertTrue(detector.drift_flag)

    def test_reference_matching_has_no_structural_change_without_drift(self):
        detector = self._detector(*self._data(0.0), matching="reference")

        self.assertFalse(any(isinstance(v, str) for v in detector.cluster_shifts.values()))
        self.assertFalse(any(d["nr_of_clusters"] for d in detector.drift_details.values()))

    def test_invalid_parameters_raise(self):
        data = self._data(0.0)
        for kwargs in ({"matching": "nope"}, {"alpha": 1.5}, {"n_permutations": -1}, {"permutation_alpha": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                from stride.xai.clustering import ClusterBasedDriftDetector

                ClusterBasedDriftDetector(*data, **kwargs)


if __name__ == "__main__":
    unittest.main()
