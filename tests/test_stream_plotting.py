import unittest
import matplotlib
from matplotlib.figure import Figure
import numpy as np

from stride.plotting.stream import visualize_data_stream


class TestStreamPlotting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 4
        self.feature_names = [f"f{i}" for i in range(self.n_features)]
        self.X = np.random.randn(self.n_samples, self.n_features)

    def test_visualize_data_stream_binary(self):
        """Test visualize_data_stream with binary targets (<= 2 classes)."""
        y = np.random.choice([0, 1], size=self.n_samples)
        figs = visualize_data_stream(
            self.X,
            y,
            window_before_start=0,
            window_after_start=100,
            window_length=50,
            feature_names=self.feature_names,
        )
        self.assertEqual(len(figs), 3)
        for fig in figs:
            self.assertIsInstance(fig, Figure)
            matplotlib.pyplot.close(fig)

    def test_visualize_data_stream_multiclass(self):
        """Test visualize_data_stream with multiclass targets (> 2 classes, e.g. RBF multi-window)."""
        y = np.random.choice([0, 1, 2], size=self.n_samples)
        figs = visualize_data_stream(
            self.X,
            y,
            window_before_start=0,
            window_after_start=100,
            window_length=50,
            feature_names=self.feature_names,
        )
        self.assertEqual(len(figs), 3)
        for fig in figs:
            self.assertIsInstance(fig, Figure)
            matplotlib.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main()
