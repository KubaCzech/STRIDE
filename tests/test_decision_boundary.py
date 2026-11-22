import sys
import types
import unittest
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from stride.xai.boundary.analysis import DecisionBoundaryDriftAnalyzer


class _RecordingSSNP:
    """Stand-in for SSNP that records its training data and projects linearly."""

    fit_calls = []

    def __init__(self, **kwargs):
        self.n_features = None

    def fit(self, X, y):
        _RecordingSSNP.fit_calls.append((np.array(X), np.array(y)))
        self.n_features = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, :2]

    def inverse_transform(self, X):
        X = np.asarray(X)
        return np.hstack([X, np.zeros((len(X), self.n_features - 2))])


class TestDecisionBoundaryProjection(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X_before = rng.rand(60, 3)
        self.y_before = (self.X_before[:, 0] > 0.5).astype(int)
        self.X_after = rng.rand(40, 3) + 0.5
        self.y_after = (self.X_after[:, 1] > 1.0).astype(int)
        _RecordingSSNP.fit_calls = []

    def _analyze(self):
        fake_module = types.ModuleType("stride.xai.boundary.ssnp")
        fake_module.SSNP = _RecordingSSNP
        with mock.patch.dict(sys.modules, {"stride.xai.boundary.ssnp": fake_module}):
            analyzer = DecisionBoundaryDriftAnalyzer(self.X_before, self.y_before, self.X_after, self.y_after)
            return analyzer.analyze(model_class=LogisticRegression, grid_size=10)

    def test_projection_is_trained_once_on_both_windows(self):
        self._analyze()

        self.assertEqual(len(_RecordingSSNP.fit_calls), 1)
        X_fit, y_fit = _RecordingSSNP.fit_calls[0]
        scaler = MinMaxScaler().fit(self.X_before)
        expected_X = np.vstack([scaler.transform(self.X_before), scaler.transform(self.X_after)])
        np.testing.assert_allclose(X_fit, expected_X)
        np.testing.assert_array_equal(y_fit, np.concatenate([self.y_before, self.y_after]))

    def test_both_windows_share_the_projection(self):
        res = self._analyze()

        scaler = MinMaxScaler().fit(self.X_before)
        np.testing.assert_allclose(res["pre"]["X_2d"], scaler.transform(self.X_before)[:, :2])
        np.testing.assert_allclose(res["post"]["X_2d"], scaler.transform(self.X_after)[:, :2])


if __name__ == "__main__":
    unittest.main()
