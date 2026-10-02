import unittest
import numpy as np
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
from stride.drift.binary_descriptor import DriftDescription


class TestADWINSyntheticIntegration(unittest.TestCase):
    """
    Integration tests verifying end-to-end prequential evaluation pipeline
    with ADWIN across STRIDE's 5 synthetic benchmark stream generators.

    These tests validate pipeline execution integrity, error stream processing,
    and DriftDescription structure rather than brittle statistical convergence bounds.
    """

    def setUp(self):
        self.random_state = 42

    def _run_prequential_pipeline(self, X, y, detector):
        """Runs test-then-train streaming evaluation and verifies pipeline invariants."""
        # Convert DataFrame/Series to numpy if necessary
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        classes = np.unique(y_arr)
        model = SGDClassifier(loss="log_loss", random_state=self.random_state)
        model.partial_fit(X_arr[0].reshape(1, -1), [y_arr[0]], classes=classes)

        drift_descriptor = BinaryErrorDriftDescriptor(
            ddm=detector,
            lookback_method="gradient",
            lookforward_method="peak",
            rate_calculation_sample_size=50,
        )

        drifts = []

        for i in range(1, len(X_arr)):
            x_i = X_arr[i].reshape(1, -1)
            y_true = y_arr[i]

            y_pred = model.predict(x_i)[0]
            error = int(y_pred != y_true)

            drift_descriptor.update(error)

            if drift_descriptor.drift_detected:
                drift = drift_descriptor.last_detected_drift
                drift.detected_at = i
                drifts.append(drift)

            model.partial_fit(x_i, [y_true])

        processed_drifts = drift_descriptor.post_process_drift_ends(drifts)

        # Invariant 1: Error history length must equal total streaming steps evaluated
        self.assertEqual(len(drift_descriptor.complete_error_history), len(X_arr) - 1)

        # Invariant 2: Any detected drifts must adhere to valid structural properties
        for d in processed_drifts:
            self.assertIsInstance(d, DriftDescription)
            self.assertIsNotNone(d.detected_at)
            self.assertIsNotNone(d.drift_start_index)
            self.assertLessEqual(d.drift_start_index, d.detected_at)

        return processed_drifts

    def test_sea_drift_pipeline(self):
        """Test pipeline integration with SeaDriftDataset."""
        dataset = SeaDriftDataset()
        X, y = dataset.generate(n_samples_before=200, n_samples_after=200, random_seed=self.random_state)

        adwin = ADWIN(delta=0.01)
        drifts = self._run_prequential_pipeline(X, y, adwin)
        self.assertIsInstance(drifts, list)

    def test_hyperplane_drift_pipeline(self):
        """Test pipeline integration with HyperplaneDriftDataset."""
        dataset = HyperplaneDriftDataset()
        X, y = dataset.generate(n_samples_before=200, n_samples_after=200, drift_width=50, random_seed=self.random_state)

        adwin = ADWIN(delta=0.01)
        drifts = self._run_prequential_pipeline(X, y, adwin)
        self.assertIsInstance(drifts, list)

    def test_linear_weight_inversion_pipeline(self):
        """Test pipeline integration with LinearWeightInversionDriftDataset."""
        dataset = LinearWeightInversionDriftDataset()
        X, y = dataset.generate(n_samples_before=200, n_samples_after=200, random_seed=self.random_state)

        adwin = ADWIN(delta=0.01)
        drifts = self._run_prequential_pipeline(X, y, adwin)
        self.assertIsInstance(drifts, list)

    def test_rbf_drift_pipeline(self):
        """Test pipeline integration with RBFDriftDataset."""
        dataset = RBFDriftDataset()
        X, y = dataset.generate(n_samples_before=200, n_samples_after=200, random_seed=self.random_state)

        adwin = ADWIN(delta=0.01)
        drifts = self._run_prequential_pipeline(X, y, adwin)
        self.assertIsInstance(drifts, list)

    def test_random_tree_multi_window_pipeline(self):
        """Test pipeline integration with RandomTreeMultiWindowDataset."""
        dataset = RandomTreeMultiWindowDataset()
        X, y = dataset.generate(window_length=200, num_windows=2, random_seed=self.random_state)

        adwin = ADWIN(delta=0.01)
        drifts = self._run_prequential_pipeline(X, y, adwin)
        self.assertIsInstance(drifts, list)


if __name__ == "__main__":
    unittest.main()
