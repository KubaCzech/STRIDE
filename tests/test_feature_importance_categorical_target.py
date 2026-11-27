"""Tests for categorical/string target labels and multiclass/single-class edge cases in feature importance."""

import unittest
import numpy as np
import pandas as pd

from stride.models import MLPModel, RandomForestModel
from stride.xai.importance import AbstainStrategy, FeatureImportanceDriftAnalyzer


class TestFeatureImportanceCategoricalTarget(unittest.TestCase):
    """Test suite for string target encoding and multiclass/single-class handling."""

    def setUp(self):
        """Build synthetic multi-class stream with string labels and class disappearance."""
        rng = np.random.RandomState(42)
        n = 90

        # Pre-drift: 3 string classes
        X_before = rng.normal(loc=0.0, scale=1.0, size=(n, 4))
        X_before[:30, 0] -= 2.0
        X_before[60:, 0] += 2.0
        y_before = np.array(["Aedes_aegypti_female"] * 30 + ["Aedes_albopictus_female"] * 30 + ["Aedes_albopictus_male"] * 30)

        # Post-drift: class disappearance (only 1 class remains, shifted on feature f0)
        X_after = rng.normal(loc=[3.0, 0.0, 0.0, 0.0], scale=1.0, size=(n, 4))
        y_after = np.array(["Aedes_albopictus_female"] * n)

        self.feature_names = ["f0", "f1", "f2", "f3"]
        self.X_before = pd.DataFrame(X_before, columns=self.feature_names)
        self.y_before = pd.Series(y_before, name="class_name")
        self.X_after = pd.DataFrame(X_after, columns=self.feature_names)
        self.y_after = pd.Series(y_after, name="class_name")

        self.analyzer = FeatureImportanceDriftAnalyzer(
            self.X_before,
            self.y_before,
            self.X_after,
            self.y_after,
            feature_names=self.feature_names,
        )

    def test_drift_importance_with_string_target_unconstrained(self):
        """Verify compute_drift_importance(include_target=True) works with string labels."""
        res = self.analyzer.compute_drift_importance(
            importance_method="permutation",
            include_target=True,
            model_class=RandomForestModel,
            model_params={"n_estimators": 10, "random_state": 42},
            abstain_strategy=None,
        )

        self.assertEqual(len(res["feature_names"]), 4)
        self.assertEqual(len(res["importance_mean"]), 4)
        self.assertGreater(res["accuracy"], 0.7)

    def test_drift_importance_with_string_target_and_abstention(self):
        """Verify confidence threshold abstention works with string target labels."""
        res = self.analyzer.compute_drift_importance(
            importance_method="permutation",
            include_target=True,
            model_class=MLPModel,
            model_params={"max_iter": 50, "random_state": 42},
            abstain_strategy=AbstainStrategy.CONFIDENCE_THRESHOLD,
            tau=0.15,
            min_samples=10,
        )

        self.assertEqual(len(res["importance_mean"]), 4)
        self.assertIn("drifting_mask", res)

    def test_predictive_importance_shift_multiclass_and_single_class(self):
        """Verify predictive shift handles multiclass string labels and single-class post-drift."""
        res = self.analyzer.compute_predictive_importance_shift(
            importance_method="permutation",
            model_class=RandomForestModel,
            model_params={"n_estimators": 10, "random_state": 42},
        )

        self.assertEqual(len(res["fi_before"]["importances_mean"]), 4)
        self.assertEqual(len(res["fi_after"]["importances_mean"]), 4)
        self.assertEqual(res["accuracy_after"], 1.0)


if __name__ == "__main__":
    unittest.main()
