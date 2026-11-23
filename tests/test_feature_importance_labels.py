"""Tests that feature importance accepts non-numeric class labels."""

import unittest

import numpy as np

from stride.xai.importance import FeatureImportanceDriftAnalyzer


class TestStringLabels(unittest.TestCase):
    def test_concept_drift_with_string_labels(self):
        rng = np.random.default_rng(0)
        X_before = rng.normal(size=(120, 3))
        X_after = rng.normal(loc=[1.0, 0.0, 0.0], size=(120, 3))
        species = np.array(["Aedes_aegypti_male", "Culex_female", "Musca_domestica"])
        y_before = species[rng.integers(0, 3, size=120)]
        y_after = species[rng.integers(0, 3, size=120)]

        analyzer = FeatureImportanceDriftAnalyzer(
            X_before, y_before, X_after, y_after, feature_names=["f0", "f1", "f2"]
        )
        result = analyzer.compute_drift_importance(include_target=True)

        self.assertEqual(result["feature_names"], ["f0", "f1", "f2"])
        self.assertEqual(len(result["importance_mean"]), 3)


if __name__ == "__main__":
    unittest.main()
