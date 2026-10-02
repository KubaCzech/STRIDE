import unittest
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from river.drift import ADWIN

from stride.datasets import (
    HyperplaneDriftDataset,
    LinearWeightInversionDriftDataset,
    RBFDriftDataset,
    RandomTreeMultiWindowDataset,
    SeaDriftDataset,
)
from stride.drift import BinaryErrorDriftDescriptor


class TestADWINDriftDetection(unittest.TestCase):
    def setUp(self):
        self.random_state = 42

    def _run_prequential_evaluation(self, X, y, detector, burn_in=500):
        """Runs test-then-train evaluation and returns drift descriptions."""
        from sklearn.naive_bayes import GaussianNB

        model = GaussianNB()
        # Initialize model
        classes = np.unique(y)
        model.partial_fit(X[0].reshape(1, -1), [y[0]], classes=classes)

        drift_descriptor = BinaryErrorDriftDescriptor(
            ddm=detector,
            lookback_method="none",
            lookforward_method="none",
            rate_calculation_sample_size=100,
        )

        drifts = []

        for i in range(1, len(X)):
            x_i = X[i].reshape(1, -1)
            y_true = y[i]

            y_pred = model.predict(x_i)[0]
            error = int(y_pred != y_true)

            if i > burn_in:
                drift_descriptor.update(error)

                if drift_descriptor.drift_detected:
                    drift = drift_descriptor.last_detected_drift
                    drift.detected_at = i
                    drifts.append(drift)

            model.partial_fit(x_i, [y_true])

        return drift_descriptor.post_process_drift_ends(drifts)

    def test_sea_drift(self):
        dataset = SeaDriftDataset()
        X, y = dataset.generate(n_samples_before=1000, n_samples_after=1000, random_seed=self.random_state)

        adwin = ADWIN(delta=0.1)
        drifts = self._run_prequential_evaluation(
            X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y, adwin
        )

        self.assertGreaterEqual(len(drifts), 1)
        # Check if first drift is around index 1000
        detected_idx = [d.detected_at for d in drifts]
        self.assertTrue(
            any(900 <= idx <= 1900 for idx in detected_idx),
            f"Drift detected at {detected_idx} which is outside expected range",
        )

    def test_hyperplane_drift(self):
        dataset = HyperplaneDriftDataset()
        X, y = dataset.generate(n_samples_before=1000, n_samples_after=1000, drift_width=200, random_seed=self.random_state)

        adwin = ADWIN(delta=0.1)
        drifts = self._run_prequential_evaluation(
            X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y, adwin
        )

        self.assertGreaterEqual(len(drifts), 1)
        detected_idx = [d.detected_at for d in drifts]
        self.assertTrue(
            any(900 <= idx <= 1900 for idx in detected_idx),
            f"Drift detected at {detected_idx} which is outside expected range",
        )

    def test_linear_weight_inversion_drift(self):
        dataset = LinearWeightInversionDriftDataset()
        X, y = dataset.generate(n_samples_before=1000, n_samples_after=1000, random_seed=self.random_state)

        adwin = ADWIN(delta=0.1)
        drifts = self._run_prequential_evaluation(
            X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y, adwin
        )

        self.assertGreaterEqual(len(drifts), 1)
        detected_idx = [d.detected_at for d in drifts]
        self.assertTrue(
            any(900 <= idx <= 1900 for idx in detected_idx),
            f"Drift detected at {detected_idx} which is outside expected range",
        )

    def test_rbf_drift(self):
        dataset = RBFDriftDataset()
        X, y = dataset.generate(n_samples_before=1000, n_samples_after=1000, random_seed=self.random_state)

        adwin = ADWIN(delta=0.1)  # RBF might need very sensitive delta for quick SGD
        drifts = self._run_prequential_evaluation(
            X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y, adwin
        )

        self.assertGreaterEqual(len(drifts), 1)
        detected_idx = [d.detected_at for d in drifts]
        self.assertTrue(
            any(900 <= idx <= 1900 for idx in detected_idx),
            f"Drift detected at {detected_idx} which is outside expected range",
        )

    def test_random_tree_multi_window_drift(self):
        dataset = RandomTreeMultiWindowDataset()
        X, y = dataset.generate(window_length=1000, num_windows=3, random_seed=self.random_state)

        adwin = ADWIN(delta=0.1)  # Make it very sensitive for this test
        drifts = self._run_prequential_evaluation(
            X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y, adwin
        )

        self.assertGreaterEqual(len(drifts), 1)

        # Check if ADWIN detects multiple drifts
        self.assertTrue(any(d.detected_at > 900 for d in drifts))


if __name__ == "__main__":
    unittest.main()
