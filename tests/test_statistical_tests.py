"""Tests for the statistical-test drift detector."""

import unittest
import warnings

import numpy as np
import pandas as pd

from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType

TESTS = [
    StatisticalTestType.KolmogorovSmirnov,
    StatisticalTestType.AD,
    StatisticalTestType.WassersteinDistance,
    StatisticalTestType.JensenShannon,
    StatisticalTestType.KullbackLeibler,
]
VALUE_KEYS = {
    "kolmogorov_smirnov": "p_value",
    "anderson_darling": "p_value",
    "wasserstein": "wd",
    "jensen_shannon": "js_div",
    "kullback_leibler": "kl_div",
}


def _block(seed: int, shift: float = 0.0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(shift, 1, n), "b": rng.normal(0, 1, n)})


class TestStatisticalTestsDriftDetector(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore", message="p-value (capped|floored)")
        warnings.filterwarnings("ignore", message="Parameter `variant` has been introduced")
        self.y = np.repeat([0, 1], 300)

    def test_shifted_feature_is_flagged_and_unchanged_one_is_not(self):
        detector = StatisticalTestsDriftDetector(_block(0), self.y, _block(1, shift=1.0), self.y)
        detector.detect(TESTS)

        for cl in (0, 1):
            self.assertTrue(detector.drift_details["kolmogorov_smirnov"][cl]["a"]["drift"])
            self.assertTrue(detector.drift_details["wasserstein"][cl]["a"]["drift"])
            self.assertFalse(detector.drift_details["wasserstein"][cl]["b"]["drift"])

    def test_class_missing_in_one_window_is_skipped(self):
        y_after = np.zeros(600, dtype=int)

        detector = StatisticalTestsDriftDetector(_block(0), self.y, _block(1), y_after)
        detector.detect(TESTS)

        for test, key in VALUE_KEYS.items():
            with self.subTest(test=test):
                missing = detector.drift_details[test][1]["a"]
                self.assertIsNone(missing[key])
                self.assertFalse(missing["drift"])
                self.assertIsNotNone(detector.drift_details[test][0]["a"][key])

    def test_feature_constant_in_both_windows_has_no_drift(self):
        before, after = _block(0).assign(b=1.0), _block(1).assign(b=1.0)

        detector = StatisticalTestsDriftDetector(before, self.y, after, self.y)
        detector.detect(TESTS)

        for test in VALUE_KEYS:
            with self.subTest(test=test):
                self.assertFalse(detector.drift_details[test][0]["b"]["drift"])
        self.assertEqual(detector.drift_details["anderson_darling"][0]["b"]["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
