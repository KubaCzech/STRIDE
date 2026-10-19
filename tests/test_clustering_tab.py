"""Tests for the table of changes per cluster in the dashboard clustering tab."""

import unittest
import warnings

import numpy as np
import pandas as pd

from dashboard.components.tabs.clustering_analysis import _findings_dataframe
from stride.xai.clustering import ClusterBasedDriftDetector


def _blobs(centers_per_class: dict, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.2, size=(60, 2)) for centers in centers_per_class.values() for c in centers])
    y = np.repeat(list(centers_per_class), [60 * len(c) for c in centers_per_class.values()])
    return pd.DataFrame(X, columns=["x0", "x1"]), y


class TestFindingsTable(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        centers = {0: [(0, 0), (0, 4)], 1: [(8, 0), (8, 4), (8, 8)]}
        X_old, y_old = _blobs(centers, seed=0)
        # Class 0 gains a blob high up on x1; class 1 does not change.
        X_new, y_new = _blobs({0: centers[0] + [(0, 8)], 1: centers[1]}, seed=1)
        # alpha: with 60 samples per blob the spread of an unchanged cluster varies by up to ~20%
        # between draws, so per-cluster changes must also be significant.
        self.detector = ClusterBasedDriftDetector(X_old, y_old, X_new, y_new, alpha=0.01, n_permutations=0, random_state=0)
        self.detector.detect()
        self.df = _findings_dataframe(self.detector)

    def test_one_row_per_cluster(self):
        self.assertEqual(sorted(self.df["Cluster"]), sorted(self.detector.cluster_shifts))

    def test_rows_are_grouped_by_class(self):
        self.assertEqual(list(self.df["Class"]), ["0"] * 3 + ["1"] * 3)

    def test_appeared_cluster_comes_first_and_is_located_by_its_features(self):
        first = self.df.iloc[0]

        self.assertEqual((first["Class"], first["Changes"]), ("0", "appeared"))
        self.assertEqual(first["Size"], "0 → 60")
        self.assertEqual(first["Share of Class"], "0% → 33%")
        # The new blob lies above the class on x1.
        self.assertEqual(first["Key Features"], "x1 +2.00σ")

    def test_unchanged_class_has_unchanged_clusters(self):
        class_1 = self.df[self.df["Class"] == "1"]

        self.assertEqual(len(class_1), 3)
        self.assertEqual(set(class_1["Changes"]), {"unchanged"})
        self.assertEqual(set(class_1["Key Features"]), {""})


if __name__ == "__main__":
    unittest.main()
