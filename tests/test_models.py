import unittest
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from stride.models import MODELS, MLPModel


class TestModels(unittest.TestCase):
    def setUp(self):
        # Generate random data for testing models
        np.random.seed(42)  # For reproducibility
        self.X = np.random.rand(100, 5)
        self.y = np.random.randint(0, 2, 100)

    def test_fit_score(self):
        """Test that all models can fit and score."""
        for name, model_class in MODELS.items():
            with self.subTest(model=name):
                model = model_class()
                try:
                    model.fit(self.X, self.y)
                    score = model.score(self.X, self.y)
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)
                except Exception as e:
                    self.fail(f"Model {name} failed with error: {e}")

    def test_mlp_standardization_enabled_and_disabled(self):
        """Verify MLPModel wraps MLPClassifier in StandardScaler Pipeline by default and supports disabling."""
        rng = np.random.RandomState(42)
        n = 200
        # Feature 0 is pure noise at large scale (1e3); Feature 1 is informative at small scale (1e-2)
        noise_large = rng.normal(loc=0.0, scale=1000.0, size=n)
        signal_small = rng.normal(loc=0.0, scale=0.01, size=n)
        y = (signal_small > 0.0).astype(int)
        X_multi_scale = np.column_stack([noise_large, signal_small])

        scaled_mlp = MLPModel(hidden_layer_sizes=(16,), max_iter=300, random_state=42, scale_features=True)
        scaled_mlp.fit(X_multi_scale, y)
        self.assertIsInstance(scaled_mlp.model_, Pipeline)
        self.assertGreater(scaled_mlp.score(X_multi_scale, y), 0.90)

        unscaled_mlp = MLPModel(hidden_layer_sizes=(16,), max_iter=300, random_state=42, scale_features=False)
        unscaled_mlp.fit(X_multi_scale, y)
        self.assertIsInstance(unscaled_mlp.model_, MLPClassifier)
        self.assertLessEqual(unscaled_mlp.score(X_multi_scale, y), 1.0)


if __name__ == "__main__":
    unittest.main()
