import unittest
import sys
import os
import numpy as np


# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import MODELS  # noqa: E402


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
                # print(f"Testing {name}...")
                model = model_class()
                try:
                    model.fit(self.X, self.y)
                    score = model.score(self.X, self.y)
                    # print(f"{name} score: {score}")
                    self.assertGreaterEqual(score, 0.0)
                    self.assertLessEqual(score, 1.0)
                except Exception as e:
                    self.fail(f"Model {name} failed with error: {e}")


if __name__ == "__main__":
    unittest.main()
