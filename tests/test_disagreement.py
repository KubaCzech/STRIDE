import unittest

import numpy as np

from stride.xai.boundary.disagreement import compute_disagreement_analysis


class _Classifier:
    def __init__(self, predict):
        self._predict = predict

    def predict(self, X):
        return self._predict(np.asarray(X))


class TestDisagreementRules(unittest.TestCase):
    """The two classifiers disagree only in the band 0.4 < x0 < 0.6; x2 is noise in the real data."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.feature_names = ["x0", "x1", "x2"]
        self.clf_pre = _Classifier(lambda X: (X[:, 0] > 0.4).astype(int))
        self.clf_post = _Classifier(lambda X: (X[:, 0] > 0.6).astype(int))
        self.X_eval = rng.rand(500, 3)
        # Decoded grid: like a projection's decoder, it ties the noise feature x2 to the band,
        # so x2 separates the grid's disagreement in one split while x0 needs two.
        t = np.linspace(0, 1, 400)
        self.X_grid = np.c_[t, rng.rand(400), np.abs(t - 0.5)]
        self.disagree = (self.X_eval[:, 0] > 0.4) & (self.X_eval[:, 0] <= 0.6)

    def _analyze(self):
        return compute_disagreement_analysis(
            self.clf_pre,
            self.clf_post,
            X_eval_raw=self.X_eval,
            X_eval_scaled=self.X_eval,
            X_grid_high_scaled=self.X_grid,
            feature_names=self.feature_names,
        )

    def test_rules_ignore_features_that_only_the_grid_ties_to_disagreement(self):
        tree = self._analyze()["viz_tree"]

        self.assertEqual(tree.tree_.n_node_samples[0], len(self.X_eval))
        self.assertEqual(tree.feature_importances_[2], 0.0)
        self.assertGreater(tree.feature_importances_[0], 0.9)

    def test_rule_confidence_is_the_real_disagreement_rate_in_the_rule(self):
        res = self._analyze()
        leaves = res["viz_tree"].apply(self.X_eval)
        grid_leaves = res["viz_tree"].apply(self.X_grid)

        self.assertFalse(res["disagreement_table"].empty)
        for _, row in res["disagreement_table"].iterrows():
            in_leaf = leaves == row["Leaf_ID"]
            self.assertEqual(row["Drift Conf."], f"{self.disagree[in_leaf].mean():.1%}")
            self.assertEqual(row["Samples"], int(in_leaf.sum()))
            self.assertEqual(row["Visual Area"], f"{np.mean(grid_leaves == row['Leaf_ID']):.1%}")

    def test_disagreement_everywhere_is_one_rule_covering_all_points(self):
        res = compute_disagreement_analysis(
            _Classifier(lambda X: np.zeros(len(X), dtype=int)),
            _Classifier(lambda X: np.ones(len(X), dtype=int)),
            X_eval_raw=self.X_eval,
            X_eval_scaled=self.X_eval,
            X_grid_high_scaled=self.X_grid,
            feature_names=self.feature_names,
        )

        table = res["disagreement_table"]
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["Rule"], "All points")
        self.assertEqual(table.iloc[0]["Drift Conf."], "100.0%")
        self.assertEqual(table.iloc[0]["Coverage"], "100.0%")


if __name__ == "__main__":
    unittest.main()
