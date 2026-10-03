"""Tests for selective discriminator abstention and conformal localization in feature importance."""

import unittest
import warnings
import numpy as np

from stride.models import MLPModel, RandomForestModel
from stride.xai.importance import AbstainStrategy, FeatureImportanceDriftAnalyzer


class TestFeatureImportanceAbstention(unittest.TestCase):
    """Test suite for discriminator abstention and conformal localization."""

    def setUp(self):
        """Set up controlled localized drift benchmark dataset."""
        np.random.seed(42)
        n_invariant = 200
        n_drifting = 40

        # Features: [f0 (drifting in locus), f1, f2, f3 (pure noise)]
        # Invariant cluster (83% of data): identical N(0, 1) in both windows
        X0_inv = np.random.normal(loc=0.0, scale=1.0, size=(n_invariant, 4))
        X1_inv = np.random.normal(loc=0.0, scale=1.0, size=(n_invariant, 4))

        # Localized drift cluster (17% of data): distinct region where f0 shifts dramatically
        X0_drift = np.random.normal(loc=[5.0, 5.0, 0.0, 0.0], scale=0.5, size=(n_drifting, 4))
        X1_drift = np.random.normal(loc=[10.0, 5.0, 0.0, 0.0], scale=0.5, size=(n_drifting, 4))

        self.X_before = np.vstack([X0_inv, X0_drift])
        self.X_after = np.vstack([X1_inv, X1_drift])

        self.y_before = np.random.randint(0, 2, size=len(self.X_before))
        self.y_after = np.random.randint(0, 2, size=len(self.X_after))

        self.feature_names = ["f_drift", "f_cluster", "f_noise1", "f_noise2"]
        self.analyzer = FeatureImportanceDriftAnalyzer(
            self.X_before,
            self.y_before,
            self.X_after,
            self.y_after,
            feature_names=self.feature_names,
        )

    def test_backwards_compatibility_none_strategy(self):
        """Verify that abstain_strategy=None retains 100% backward compatibility and valid schema."""
        res = self.analyzer.compute_drift_importance(
            importance_method="permutation",
            include_target=False,
            model_class=MLPModel,
            model_params={"max_iter": 50, "random_state": 42},
            abstain_strategy=None,
        )

        self.assertIn("accuracy", res)
        self.assertIn("drift_coverage", res)
        self.assertIn("selective_accuracy", res)
        self.assertIn("rejection_rate", res)
        self.assertIn("drifting_mask", res)
        self.assertIn("drift_locus_fallback", res)

        self.assertEqual(res["drift_coverage"], 1.0)
        self.assertEqual(res["rejection_rate"], 0.0)
        self.assertEqual(res["selective_accuracy"], res["accuracy"])
        self.assertFalse(res["drift_locus_fallback"])
        self.assertTrue(np.all(res["drifting_mask"]))

    def test_confidence_threshold_localization(self):
        """Verify that confidence margin abstention isolates drift locus and boosts selective accuracy."""
        # Baseline unconstrained
        baseline = self.analyzer.compute_drift_importance(
            importance_method="permutation",
            include_target=False,
            model_class=RandomForestModel,
            model_params={"max_depth": 4, "n_estimators": 20, "random_state": 42},
            abstain_strategy=None,
        )

        # Abstaining discriminator with margin tau=0.15
        localized = self.analyzer.compute_drift_importance(
            importance_method="permutation",
            include_target=False,
            model_class=RandomForestModel,
            model_params={"max_depth": 4, "n_estimators": 20, "random_state": 42},
            abstain_strategy=AbstainStrategy.CONFIDENCE_THRESHOLD,
            tau=0.15,
            min_samples=10,
        )

        # 1. Coverage should be localized (< 0.40 since invariant region is ~83%)
        self.assertLess(localized["drift_coverage"], 0.45)
        self.assertGreater(localized["drift_coverage"], 0.05)
        self.assertAlmostEqual(localized["rejection_rate"], 1.0 - localized["drift_coverage"])

        # 2. Selective accuracy on drift locus should be high (> 80%) and exceed baseline
        self.assertGreater(localized["selective_accuracy"], 0.80)
        self.assertGreaterEqual(localized["selective_accuracy"], baseline["accuracy"])

        # 3. f_drift (index 0) should have the highest importance score
        mean_imp = localized["importance_mean"]
        self.assertEqual(np.argmax(mean_imp), 0)

    def test_conformal_localization(self):
        """Verify that conformal p-value hypothesis testing identifies drifting points with statistical significance."""
        localized = self.analyzer.compute_drift_importance(
            importance_method="permutation",
            include_target=False,
            model_class=RandomForestModel,
            model_params={"n_estimators": 10, "random_state": 42},
            abstain_strategy=AbstainStrategy.CONFORMAL,
            alpha=0.05,
            n_bootstraps=15,
            min_samples=10,
            random_state=42,
        )

        self.assertLess(localized["drift_coverage"], 0.50)
        self.assertGreater(localized["drift_coverage"], 0.02)
        self.assertGreater(localized["selective_accuracy"], 0.75)
        self.assertEqual(np.argmax(localized["importance_mean"]), 0)

    def test_small_sample_fallback(self):
        """Verify that locus with fewer than min_samples triggers graceful fallback with warning."""
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            res = self.analyzer.compute_drift_importance(
                importance_method="permutation",
                include_target=False,
                model_class=MLPModel,
                model_params={"max_iter": 30, "random_state": 42},
                abstain_strategy=AbstainStrategy.CONFIDENCE_THRESHOLD,
                tau=0.49,  # Extremely strict threshold -> almost no samples
                min_samples=1000,  # Impossible to reach
            )

            # Warning should have been emitted
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught_warnings))
            self.assertTrue(res["drift_locus_fallback"])
            self.assertIn("f_drift", res["feature_names"])
            self.assertEqual(len(res["importance_mean"]), 4)

    def test_global_drift_coverage(self):
        """Verify that global drift yields near 100% coverage."""
        np.random.seed(42)
        X0_all = np.random.normal(0, 1, size=(80, 2))
        X1_all = np.random.normal(5, 1, size=(80, 2))  # Complete distribution shift

        analyzer_global = FeatureImportanceDriftAnalyzer(
            X0_all,
            np.zeros(len(X0_all)),
            X1_all,
            np.ones(len(X1_all)),
            feature_names=["x1", "x2"],
        )

        res = analyzer_global.compute_drift_importance(
            importance_method="permutation",
            include_target=False,
            model_class=RandomForestModel,
            model_params={"n_estimators": 10, "random_state": 42},
            abstain_strategy=AbstainStrategy.CONFIDENCE_THRESHOLD,
            tau=0.10,
        )

        self.assertGreater(res["drift_coverage"], 0.85)
        self.assertFalse(res["drift_locus_fallback"])

    def test_invalid_abstain_strategy_raises_value_error(self):
        """Verify that unsupported abstain strategy raises ValueError with informative message."""
        with self.assertRaises(ValueError) as ctx:
            self.analyzer.compute_drift_importance(abstain_strategy="invalid_strategy")
        self.assertIn("Supported strategies are", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
