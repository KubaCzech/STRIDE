"""Regression tests for the cluster drift plots."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stride.xai.clustering.visualization import (
    cluster_color,
    colors,
    plot_centers_shift,
    plot_clusters_by_class,
    plot_drift_clustered,
)


class TestClusterPlots(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X_before = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
        self.X_after = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])

    def tearDown(self):
        plt.close("all")

    def test_more_clusters_than_palette_colours(self):
        n_clusters = len(colors) + 10
        labels_old = np.arange(120) % n_clusters
        labels_new = (np.arange(120) + 5) % (n_clusters + 3)
        y = np.zeros(120, dtype=int)

        for plot in (plot_centers_shift, plot_drift_clustered, plot_clusters_by_class):
            with self.subTest(plot=plot.__name__):
                plot(self.X_before, self.X_after, y, y, labels_old, labels_new)

    def test_string_class_labels(self):
        y = np.where(np.arange(120) % 2 == 0, "pos", "neg")
        labels = np.arange(120) % 4

        fig = plot_clusters_by_class(self.X_before, self.X_after, y, y, labels, labels)

        titles = {ax.get_title() for ax in fig.axes}
        self.assertIn("Before Drift – Class neg", titles)
        self.assertIn("After Drift – Class pos", titles)

    def test_cluster_colour_is_stable_and_cycles(self):
        self.assertEqual(cluster_color(3), colors[3])
        self.assertEqual(cluster_color(np.int64(len(colors) + 3)), colors[3])


if __name__ == "__main__":
    unittest.main()
