"""Tests for the X-Means clustering wrapper."""

import importlib.util
import random
import unittest

import numpy as np

from stride.xai.clustering.xmeans import run_xmeans


def _cluster_signature(centers: np.ndarray, clusters: list[list[int]]) -> tuple:
    return tuple(sorted(tuple(sorted(c)) for c in clusters)), np.round(centers, 10).tobytes()


@unittest.skipUnless(importlib.util.find_spec("pyclustering"), "pyclustering not installed")
class TestRunXMeansReproducibility(unittest.TestCase):
    def setUp(self):
        # Structureless data: X-Means splits are sensitive to k-means++ seeding here.
        X = np.random.default_rng(5).uniform(size=(1000, 2))
        self.X = (X - X.mean(axis=0)) / X.std(axis=0)

    def test_same_random_state_gives_identical_clustering(self):
        signatures = {_cluster_signature(*run_xmeans(self.X, k_init=2, k_max=10, random_state=42)) for _ in range(10)}
        self.assertEqual(len(signatures), 1)

    def test_splits_happen_on_test_data(self):
        _, clusters = run_xmeans(self.X, k_init=2, k_max=10, random_state=42)
        self.assertGreater(len(clusters), 2)

    def test_global_random_state_is_not_modified(self):
        random.seed(0)
        np.random.seed(0)
        expected = (random.random(), np.random.random())

        random.seed(0)
        np.random.seed(0)
        run_xmeans(self.X, k_init=2, k_max=10, random_state=42)

        self.assertEqual((random.random(), np.random.random()), expected)


if __name__ == "__main__":
    unittest.main()
