import unittest
import sys
import os
import pandas as pd


# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from datasets.hyperplane_drift import HyperplaneDriftDataset  # noqa: E402
from datasets.linear_weight_inversion_drift import LinearWeightInversionDriftDataset  # noqa: E402
from datasets.rbf_drift import RBFDriftDataset  # noqa: E402
from datasets.sea_drift import SeaDriftDataset  # noqa: E402


class TestSyntheticDatasets(unittest.TestCase):
    def setUp(self):
        self.n_before = 500
        self.n_after = 500
        self.total_samples = self.n_before + self.n_after

    def _verify_shape(self, X, y, expected_features):
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape, (self.total_samples, expected_features))
        self.assertEqual(y.shape, (self.total_samples,))

    def test_sea_drift_generation(self):
        """Test SEA Drift dataset generation."""
        ds = SeaDriftDataset()
        X, y = ds.generate(n_samples_before=self.n_before, n_samples_after=self.n_after, drift_width=200)
        self._verify_shape(X, y, 3)

    def test_rbf_drift_generation(self):
        """Test RBF Drift dataset generation."""
        ds = RBFDriftDataset()
        X, y = ds.generate(n_samples_before=self.n_before, n_samples_after=self.n_after, drift_width=200)
        self._verify_shape(X, y, 4)  # Default n_features=4

    def test_linear_weight_inversion_drift_generation(self):
        """Test Linear Weight Inversion Drift dataset generation."""
        ds = LinearWeightInversionDriftDataset()
        n_features = 11
        X, y = ds.generate(
            n_samples_before=self.n_before, n_samples_after=self.n_after, n_features=n_features, drift_width=200
        )
        self._verify_shape(X, y, n_features)

    def test_hyperplane_drift_generation(self):
        """Test Hyperplane Drift dataset generation."""
        ds = HyperplaneDriftDataset()
        n_features = 5
        X, y = ds.generate(
            n_samples_before=self.n_before,
            n_samples_after=self.n_after,
            n_features=n_features,
            n_drift_features=2,
            drift_width=200,
        )
        self._verify_shape(X, y, n_features)

    def test_drift_width_parameter_overflow(self):
        """
        Test that generating data with various drift widths does not cause
        OverflowError (specifically checks the sigmoid fix/workaround).
        """
        datasets = [
            (SeaDriftDataset(), {"n_features": 3}),
            (RBFDriftDataset(), {}),
            (LinearWeightInversionDriftDataset(), {"n_features": 11}),
            (HyperplaneDriftDataset(), {"n_features": 5, "n_drift_features": 2}),
        ]

        widths_to_test = [1, 50, 500, 5000]

        for ds, kwargs in datasets:
            with self.subTest(dataset=ds.name):
                for width in widths_to_test:
                    try:
                        ds.generate(n_samples_before=100, n_samples_after=100, drift_width=width, **kwargs)
                    except OverflowError:
                        self.fail(f"{ds.name} raised OverflowError with drift_width={width}")
                    except Exception as e:
                        self.fail(f"{ds.name} raised unexpected exception {e} with drift_width={width}")

    def test_reproducibility(self):
        """Test that random_seed ensures reproducibility."""
        datasets = [
            (SeaDriftDataset(), {"n_features": 3}),
            (RBFDriftDataset(), {}),
            (LinearWeightInversionDriftDataset(), {"n_features": 11}),
            (HyperplaneDriftDataset(), {"n_features": 5, "n_drift_features": 2}),
        ]

        seed = 42
        for ds, kwargs in datasets:
            with self.subTest(dataset=ds.name):
                X1, y1 = ds.generate(n_samples_before=100, n_samples_after=100, drift_width=100, random_seed=seed, **kwargs)
                X2, y2 = ds.generate(n_samples_before=100, n_samples_after=100, drift_width=100, random_seed=seed, **kwargs)

                pd.testing.assert_frame_equal(X1, X2)
                pd.testing.assert_series_equal(y1, y2)

    def test_input_validation(self):
        """Test invalid input parameters."""
        ds = LinearWeightInversionDriftDataset()
        with self.assertRaises(ValueError):
            ds.generate(n_features=5, n_drift_features=6)  # Drift > Total

        ds = HyperplaneDriftDataset()
        with self.assertRaises(ValueError):
            ds.generate(n_features=5, n_drift_features=6)


if __name__ == "__main__":
    unittest.main()
