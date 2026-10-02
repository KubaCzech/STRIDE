"""Tests for verifying base minimal installation and optional dependency decoupling."""

import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from stride.common import DataDimensionsReducer, ReducerType
from stride.datasets.sea_drift import SeaDriftDataset
from stride.drift import BinaryErrorDriftDescriptor
from stride.exceptions import OptionalDependencyError
from stride.models import MLPModel, RandomForestModel
from stride.plotting.stream import plot_feature_space, visualize_data_stream
from stride.xai.boundary.analysis import DecisionBoundaryDriftAnalyzer
from stride.xai.boundary.ssnp import SSNP
from stride.xai.clustering.xmeans import run_xmeans
from stride.xai.importance.analysis import FeatureImportanceDriftAnalyzer
from stride.xai.importance.methods import calculate_feature_importance
from stride.xai.recurrence.methods import cluster_windows
from stride.xai.stats import (
    DescriptiveStatisticsDriftDetector,
    StatisticalTestType,
    StatisticalTestsDriftDetector,
    StatisticsType,
)


# Mapping of optional third-party modules to mock as absent
MOCKED_ABSENT_OPTIONAL_MODULES = {
    "shap": None,
    "lime": None,
    "lime.lime_tabular": None,
    "umap": None,
    "pyclustering": None,
    "pyclustering.cluster.xmeans": None,
    "hdbscan": None,
    "tensorflow": None,
    "matplotlib": None,
    "matplotlib.pyplot": None,
}


class TestMinimalInstallImport(unittest.TestCase):
    """Verify that importing and using core functionality succeeds in minimal base environment."""

    def test_top_level_stride_import_without_optional_dependencies(self):
        """Verify that 'import stride' succeeds when all optional libraries are uninstalled."""
        with patch.dict("sys.modules", MOCKED_ABSENT_OPTIONAL_MODULES):
            import stride

            self.assertIsNotNone(stride.__version__)
            self.assertTrue(hasattr(stride, "BinaryErrorDriftDescriptor"))
            self.assertTrue(hasattr(stride, "RandomForestModel"))
            self.assertTrue(hasattr(stride, "MLPModel"))
            self.assertTrue(hasattr(stride, "DescriptiveStatisticsDriftDetector"))
            self.assertTrue(hasattr(stride, "DecisionBoundaryDriftAnalyzer"))
            self.assertTrue(hasattr(stride, "FeatureImportanceDriftAnalyzer"))

    def test_core_algorithms_execute_without_optional_dependencies(self):
        """Verify that core models, datasets, statistics, and descriptors execute cleanly."""
        np.random.seed(42)
        X = np.random.rand(60, 4)
        y = np.random.randint(0, 2, 60)

        # 1. Core models
        rf = RandomForestModel(n_estimators=5, random_state=42)
        rf.fit(X, y)
        self.assertGreaterEqual(rf.score(X, y), 0.0)

        mlp = MLPModel(max_iter=20, random_state=42)
        mlp.fit(X, y)
        self.assertGreaterEqual(mlp.score(X, y), 0.0)

        # 2. Core dataset generation (powered by promoted river engine)
        sea = SeaDriftDataset()
        X_df, y_s = sea.generate(n_samples_before=50, n_samples_after=50, drift_width=20)
        self.assertEqual(len(X_df), 100)
        self.assertEqual(len(y_s), 100)

        # 3. Core drift descriptor
        descriptor = BinaryErrorDriftDescriptor()
        self.assertIsNotNone(descriptor)

        # 4. Core statistical tests
        df_before = pd.DataFrame(X[:30], columns=[f"f{i}" for i in range(4)])
        df_after = pd.DataFrame(X[30:], columns=[f"f{i}" for i in range(4)])
        y_before = y[:30]
        y_after = y[30:]

        stats_detector = DescriptiveStatisticsDriftDetector(df_before, y_before, df_after, y_after)
        drift_flag, details = stats_detector.detect(StatisticsType.Mean)
        self.assertIsInstance(drift_flag, (bool, np.bool_))

        test_detector = StatisticalTestsDriftDetector(df_before, y_before, df_after, y_after)
        drift_test_flag = test_detector.detect(StatisticalTestType.WassersteinDistance)
        self.assertIsInstance(drift_test_flag, (bool, np.bool_))

        # 5. Core dimensionality reduction (PCA)
        pca_reducer = DataDimensionsReducer(ReducerType.PCA, n_components=2)
        X_reduced = pca_reducer.fit_transform(X)
        self.assertEqual(X_reduced.shape, (60, 2))

        # 6. Core feature importance (Permutation / PFI)
        pfi_res = calculate_feature_importance(rf, X, y, method="permutation", n_repeats=2, random_state=42)
        self.assertEqual(pfi_res["method"], "PFI")
        self.assertEqual(len(pfi_res["importances_mean"]), 4)

        # 7. Core 2D decision boundary drift analyzer (uses DummyProjector, no TF required)
        analyzer_2d = DecisionBoundaryDriftAnalyzer(X[:30, :2], y[:30], X[30:, :2], y[30:], random_state=42)
        res_boundary = analyzer_2d.analyze()
        self.assertIn("pre", res_boundary)
        self.assertIn("post", res_boundary)
        self.assertTrue(res_boundary["is_2d"])


class TestOptionalDependencyGuardrails(unittest.TestCase):
    """Verify that attempting to invoke optional features raises actionable OptionalDependencyError."""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.rand(40, 4)
        self.y = np.random.randint(0, 2, 40)
        self.rf = RandomForestModel(n_estimators=5, random_state=42)
        self.rf.fit(self.X, self.y)

    def test_shap_missing_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"shap": None}):
            with self.assertRaises(OptionalDependencyError) as ctx:
                calculate_feature_importance(self.rf, self.X, self.y, method="shap")
            self.assertEqual(ctx.exception.package_name, "shap")
            self.assertEqual(ctx.exception.extra_name, "xai")
            self.assertIn("pip install stride-xai[xai]", str(ctx.exception))

    def test_lime_missing_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"lime": None, "lime.lime_tabular": None}):
            with self.assertRaises(OptionalDependencyError) as ctx:
                calculate_feature_importance(self.rf, self.X, self.y, method="lime")
            self.assertEqual(ctx.exception.package_name, "lime")
            self.assertEqual(ctx.exception.extra_name, "xai")
            self.assertIn("pip install stride-xai[xai]", str(ctx.exception))

    def test_feature_importance_analyzer_shap_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"shap": None}):
            analyzer = FeatureImportanceDriftAnalyzer(self.X[:20], self.y[:20], self.X[20:], self.y[20:])
            with self.assertRaises(OptionalDependencyError) as ctx:
                analyzer.compute_drift_importance(importance_method="shap")
            self.assertEqual(ctx.exception.package_name, "shap")
            self.assertIn("pip install stride-xai[xai]", str(ctx.exception))

    def test_umap_missing_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"umap": None}):
            with self.assertRaises(OptionalDependencyError) as ctx:
                DataDimensionsReducer(ReducerType.UMAP, n_components=2)
            self.assertEqual(ctx.exception.package_name, "umap-learn")
            self.assertEqual(ctx.exception.extra_name, "clustering")
            self.assertIn("pip install stride-xai[clustering]", str(ctx.exception))

    def test_xmeans_missing_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"pyclustering": None, "pyclustering.cluster.xmeans": None}):
            with self.assertRaises(OptionalDependencyError) as ctx:
                run_xmeans(self.X, k_init=2, k_max=4)
            self.assertEqual(ctx.exception.package_name, "pyclustering")
            self.assertEqual(ctx.exception.extra_name, "clustering")
            self.assertIn("pip install stride-xai[clustering]", str(ctx.exception))

    def test_hdbscan_missing_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"hdbscan": None}):
            matrix = pd.DataFrame(np.zeros((5, 5)))
            with self.assertRaises(OptionalDependencyError) as ctx:
                cluster_windows(matrix)
            self.assertEqual(ctx.exception.package_name, "hdbscan")
            self.assertEqual(ctx.exception.extra_name, "clustering")
            self.assertIn("pip install stride-xai[clustering]", str(ctx.exception))

    def test_tensorflow_ssnp_missing_raises_optional_dependency_error(self):
        with patch.dict("sys.modules", {"tensorflow": None}):
            # 1. Direct SSNP class instantiation
            with patch("stride.xai.boundary.ssnp._HAS_TF", False):
                with self.assertRaises(OptionalDependencyError) as ctx:
                    SSNP()
                self.assertEqual(ctx.exception.package_name, "tensorflow")
                self.assertEqual(ctx.exception.extra_name, "deeplearning")
                self.assertIn("pip install stride-xai[deeplearning]", str(ctx.exception))

            # 2. DecisionBoundaryDriftAnalyzer on high-dimensional (>2D) data
            analyzer_hd = DecisionBoundaryDriftAnalyzer(self.X[:20], self.y[:20], self.X[20:], self.y[20:], random_state=42)
            with self.assertRaises(OptionalDependencyError) as ctx:
                analyzer_hd.analyze()
            self.assertEqual(ctx.exception.package_name, "tensorflow")
            self.assertEqual(ctx.exception.extra_name, "deeplearning")
            self.assertIn("pip install stride-xai[deeplearning]", str(ctx.exception))

    def test_matplotlib_stream_plotting_missing_raises_optional_dependency_error(self):
        with patch("stride.plotting.stream._HAS_MATPLOTLIB", False):
            with self.assertRaises(OptionalDependencyError) as ctx:
                plot_feature_space(
                    n_features=2,
                    feature_names=["f1", "f2"],
                    X_before=self.X[:20, :2],
                    X_after=self.X[20:, :2],
                    y_before=self.y[:20],
                    y_after=self.y[20:],
                    class_colors={0: "red", 1: "blue"},
                )
            self.assertEqual(ctx.exception.package_name, "matplotlib")
            self.assertEqual(ctx.exception.extra_name, "vis")
            self.assertIn("pip install stride-xai[vis]", str(ctx.exception))

            with self.assertRaises(OptionalDependencyError) as ctx:
                visualize_data_stream(
                    X=self.X,
                    y=self.y,
                    window_before_start=0,
                    window_after_start=20,
                    window_length=20,
                    feature_names=["f1", "f2", "f3", "f4"],
                )
            self.assertEqual(ctx.exception.package_name, "matplotlib")
            self.assertEqual(ctx.exception.extra_name, "vis")
            self.assertIn("pip install stride-xai[vis]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
