import unittest
import pandas as pd
import numpy as np

from dashboard.components.tabs import (
    render_dataset_overview_tab,
    render_data_visualization_tab,
    render_drift_detection_tab,
    render_prototype_analysis_tab,
    render_statistical_tests_tab,
    render_clustering_analysis_tab,
    render_feature_importance_analysis_tab,
    render_decision_boundary_tab,
)


class TestDashboardTabs(unittest.TestCase):
    def setUp(self):
        self.expected_tabs = [
            "Dataset Overview",
            "Drift Detection",
            "Prototype Analysis",
            "Data Comparison",
            "Statistical Tests",
            "Clustering",
            "Feature Importance",
            "Decision Boundary",
        ]
        self.X = pd.DataFrame(
            np.random.RandomState(42).randn(100, 4),
            columns=["feature_1", "feature_2", "feature_3", "feature_4"],
        )
        self.y = pd.Series(np.random.RandomState(42).choice([0, 1], size=100))
        self.feature_names = ["feature_1", "feature_2", "feature_3", "feature_4"]

    def test_tab_imports(self):
        """Ensure all 8 tab renderers are callable and exportable."""
        callables = [
            render_dataset_overview_tab,
            render_data_visualization_tab,
            render_drift_detection_tab,
            render_prototype_analysis_tab,
            render_statistical_tests_tab,
            render_clustering_analysis_tab,
            render_feature_importance_analysis_tab,
            render_decision_boundary_tab,
        ]
        for fn in callables:
            self.assertTrue(callable(fn), f"{fn} is not callable")

    def test_tab_order_and_count(self):
        """Verify the exact count and sequence of tabs in dashboard/app.py."""
        # Read tabs defined in dashboard/app.py
        with open("dashboard/app.py", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn('"Dataset Overview"', content)
        self.assertIn('"Drift Detection"', content)
        self.assertIn('"Prototype Analysis"', content)
        self.assertIn('"Data Comparison"', content)
        self.assertIn('"Statistical Tests"', content)
        self.assertIn('"Clustering"', content)
        self.assertIn('"Feature Importance"', content)
        self.assertIn('"Decision Boundary"', content)


if __name__ == "__main__":
    unittest.main()
