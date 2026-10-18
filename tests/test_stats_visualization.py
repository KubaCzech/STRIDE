"""Regression tests for the feature distribution plots."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stride.xai.stats.visualization import PlotOptions, plot_boxplot, plot_ecdf, plot_histogram, plot_kde, plot_qq


class TestDistributionPlots(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X_before = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
        self.X_after = pd.DataFrame({"a": rng.normal(1, 1, 200), "b": rng.normal(0, 1, 200)})
        self.y = np.repeat([0, 1], 100)

    def tearDown(self):
        plt.close("all")

    def test_text_class_labels(self):
        y = np.where(self.y == 0, "neg", "pos")

        fig = plot_ecdf(self.X_before, y, self.X_after, y)

        self.assertEqual({ax.get_title() for ax in fig.axes if ax.get_title()}, {"Class neg", "Class pos"})

    def test_kde_with_constant_feature(self):
        X_before = self.X_before.assign(b=1.0)

        plot_kde(X_before, self.y, self.X_after, self.y)

    def test_qq_pairs_quantiles_of_both_blocks(self):
        fig = plot_qq(self.X_before[["a"]], self.y, self.X_after[["a"]], self.y)

        points = fig.axes[0].collections[0].get_offsets()
        self.assertEqual(len(points), 99)
        # The second block is shifted by +1, so its quantiles lie about 1 above the first block's.
        self.assertAlmostEqual(float(np.median(points[:, 1] - points[:, 0])), 1.0, delta=0.3)

    def test_all_plots_render(self):
        for plot, kwargs in (
            (plot_boxplot, {"show_": PlotOptions.Both}),
            (plot_histogram, {}),
            (plot_kde, {}),
            (plot_ecdf, {}),
            (plot_qq, {}),
        ):
            with self.subTest(plot=plot.__name__):
                plot(self.X_before, self.y, self.X_after, self.y, **kwargs)


if __name__ == "__main__":
    unittest.main()
