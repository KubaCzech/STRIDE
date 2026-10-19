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
    plot_clustering_heatmap,
    plot_clusters_by_class,
    plot_drift_clustered,
    plot_feature_shift_heatmap,
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

    def test_axes_are_named_after_the_features_of_2d_data(self):
        X_before, X_after = self.X_before[["a", "b"]], self.X_after[["a", "b"]]
        labels, y = np.arange(120) % 3, np.zeros(120, dtype=int)

        for plot in (plot_centers_shift, plot_drift_clustered, plot_clusters_by_class):
            with self.subTest(plot=plot.__name__):
                fig = plot(X_before, X_after, y, y, labels, labels)
                self.assertEqual({(ax.get_xlabel(), ax.get_ylabel()) for ax in fig.axes}, {("a", "b")})

    def test_axes_name_principal_components_of_reduced_data(self):
        labels, y = np.arange(120) % 3, np.zeros(120, dtype=int)

        fig = plot_drift_clustered(self.X_before, self.X_after, y, y, labels, labels)

        self.assertEqual(fig.axes[0].get_xlabel(), "Component 1")
        self.assertEqual(fig.axes[0].get_ylabel(), "Component 2")


class TestShiftHeatmaps(unittest.TestCase):
    def setUp(self):
        self.stats_shifts = {
            0: {"a": {"mean": 0.9, "median": 0.8, "std": 0.05}, "b": {"mean": -0.4, "median": float("nan"), "std": 0.0}},
            3: {"a": {"mean": 0.01, "median": 0.02, "std": -0.3}, "b": {"mean": 0.1, "median": 0.1, "std": 0.1}},
        }

    def tearDown(self):
        plt.close("all")

    def test_clustering_heatmap_marks_shifts_above_threshold_as_drift(self):
        fig = plot_clustering_heatmap(self.stats_shifts, threshold=0.2)

        legend = {
            t.get_text(): p.get_facecolor()
            for t, p in zip(fig.axes[0].get_legend().get_texts(), fig.axes[0].get_legend().get_patches())
        }
        self.assertIn("|value| > 0.2 (drift)", legend)
        # Drift is drawn in red, values within the threshold in a neutral colour.
        self.assertGreater(legend["|value| > 0.2 (drift)"][0], legend["|value| > 0.2 (drift)"][1])
        self.assertLess(legend["Within threshold"][0], 0.9)

    def test_feature_shift_heatmap_shows_signed_shift_per_feature_and_cluster(self):
        fig = plot_feature_shift_heatmap(self.stats_shifts, threshold=0.2, cluster_names={0: "0 (class A)", 3: "3 (class B)"})

        ax = fig.axes[0]
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["0 (class A)", "3 (class B)"])
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["a", "b"])
        annotations = [t.get_text() for t in ax.texts]
        self.assertEqual(annotations, ["+0.90", "+0.01", "−0.40", "+0.10"])

    def test_feature_shift_heatmap_of_another_statistic_leaves_missing_values_empty(self):
        fig = plot_feature_shift_heatmap(self.stats_shifts, threshold=0.2, statistic="median")

        self.assertEqual([t.get_text() for t in fig.axes[0].texts], ["+0.80", "+0.02", "+0.10"])


if __name__ == "__main__":
    unittest.main()
