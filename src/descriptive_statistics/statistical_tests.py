import numpy as np
import pandas as pd
from enum import Enum
from typing import Union

from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance, spearmanr, anderson_ksamp

from src.common import DataScaler, ScalingType

# TODO:
# * wizualizacja
# * dokumentacja


class StatisticalTestType(Enum):
    KolmogorovSmirnov = "kolmogorov_smirnov"
    KullbackLeibler = "kullback_leibler"
    WassersteinDistance = "wasserstein"
    JensenShannon = "jensen_shannon"
    Spearman = "spearman"
    AD = "anderson_darling"
    All = "all"


class StatisticalTestsDriftDetector:
    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: Union[np.ndarray, pd.Series],
        X_after: pd.DataFrame,
        y_after: Union[np.ndarray, pd.Series],
        decision_thr: float = 0.4,
        alpha: float = 0.05,
        bins: int = 30,
        kl_thr: float = 0.1,
        wasserstein_thr: float = 0.1,
        js_thr: float = 0.05,
        spearman_thr: float = 0.9,
        drift_thr: float = 0.2,
    ):
        self.X_before = X_before
        self.y_before = y_before
        self.X_after = X_after
        self.y_after = y_after

        self.decision_thr = decision_thr
        self.bins = bins
        self.alpha = alpha

        self.kl_thr = kl_thr
        self.wasserstein_thr = wasserstein_thr
        self.js_thr = js_thr
        self.spearman_thr = spearman_thr

        self.drift_thr = drift_thr

        self.drift_flag = None
        self.drift_flags = None
        self.drift_details = None

        self.labels = sorted(list(set(self.y_before).union(set(self.y_after))))

        assert sorted(self.X_before.columns.values) == sorted(self.X_after.columns.values), (
            "Columns must be the same between data blocks"
        )

        self._scale_data()

    def _scale_data(self):
        DS = DataScaler(ScalingType.MinMax)
        self.X_before_scaled = DS.fit_transform(self.X_before, return_df=True)
        self.X_after_scaled = DS.transform(self.X_after, return_df=True)

    def detect(self, test_type):
        if test_type == StatisticalTestType.All:
            test_type = [t for t in StatisticalTestType if t != StatisticalTestType.All]
        elif isinstance(test_type, StatisticalTestType):
            test_type = [test_type]
        elif isinstance(test_type, list):
            for t in test_type:
                if not isinstance(t, StatisticalTestType):
                    raise ValueError("Unsupported test type")

        drift_flags = {}
        details = {}

        for test in test_type:
            curr_drift_flag, curr_details = self._detect_single_statistic(test)

            drift_flags[test.value] = curr_drift_flag
            details[test.value] = curr_details

        self.drift_flag = sum(drift_flags.values()) / len(drift_flags) > self.decision_thr
        self.drift_flags = drift_flags
        self.drift_details = details
        return self.drift_flag

    def _detect_single_statistic(self, test):
        return getattr(self, f"_{test.value}_test")()

    def _kolmogorov_smirnov_test(self):
        details = {
            l: {col: {"drift": False, "p_value": None, "stat": None} for col in self.X_before.columns} for l in self.labels
        }

        for l in self.labels:
            for column in self.X_before.columns:
                stat, p_value = ks_2samp(self.X_before[self.y_before == l][column], self.X_after[self.y_after == l][column])
                details[l][column]["p_value"] = p_value
                details[l][column]["stat"] = stat
                if p_value < self.alpha:
                    details[l][column]["drift"] = True

        drift_flag = np.mean([details[l][col]["drift"] for col in self.X_before.columns for l in self.labels]) > self.drift_thr
        return drift_flag, details

    def _kullback_leibler_test(self):
        eps = 1e-10
        details = {l: {col: {"drift": False, "kl_div": None} for col in self.X_before_scaled.columns} for l in self.labels}

        for l in self.labels:
            for column in self.X_before_scaled.columns:
                old_dist, bin_edges = np.histogram(
                    self.X_before_scaled[self.y_before == l][column], bins=self.bins, density=True
                )
                new_dist, _ = np.histogram(self.X_after_scaled[self.y_after == l][column], bins=bin_edges, density=True)

                old_dist += eps
                new_dist += eps

                old_dist /= old_dist.sum()
                new_dist /= new_dist.sum()

                kl_div = np.sum(rel_entr(old_dist, new_dist))
                details[l][column]["kl_div"] = kl_div
                if kl_div > self.kl_thr:
                    details[l][column]["drift"] = True

        drift_flag = (
            np.mean([details[l][col]["drift"] for col in self.X_before_scaled.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _wasserstein_test(self):
        details = {l: {col: {"drift": False, "wd": None} for col in self.X_before_scaled.columns} for l in self.labels}

        for l in self.labels:
            for column in self.X_before_scaled.columns:
                wd = wasserstein_distance(
                    self.X_before_scaled[self.y_before == l][column], self.X_after_scaled[self.y_after == l][column]
                )
                details[l][column]["wd"] = wd
                if wd > self.wasserstein_thr:
                    details[l][column]["drift"] = True
        drift_flag = (
            np.mean([details[l][col]["drift"] for col in self.X_before_scaled.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _jensen_shannon_test(self):
        eps = 1e-10
        details = {l: {col: {"drift": False, "js_div": None} for col in self.X_before_scaled.columns} for l in self.labels}

        for l in self.labels:
            for column in self.X_before_scaled.columns:
                old_dist, bin_edges = np.histogram(
                    self.X_before_scaled[self.y_before == l][column], bins=self.bins, density=True
                )
                old_dist += eps

                new_dist = np.histogram(self.X_after_scaled[self.y_after == l][column], bins=bin_edges, density=True)[0] + eps

                old_dist /= old_dist.sum()
                new_dist /= new_dist.sum()

                js_div = jensenshannon(old_dist, new_dist) ** 2
                details[l][column]["js_div"] = js_div

                if js_div > self.js_thr:
                    details[l][column]["drift"] = True
        drift_flag = (
            np.mean([details[l][col]["drift"] for col in self.X_before_scaled.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _spearman_test(self):
        details = {l: {col: {"drift": False, "spearman_coeff": None} for col in self.X_before.columns} for l in self.labels}

        for l in self.labels:
            for column in self.X_before.columns:
                min_len = min(
                    len(self.X_before[self.y_before == l][column]), len(self.X_after[self.y_after == l][column])
                )  # sizes of distribution must be the same
                old_sample = self.X_before[self.y_before == l][column].values[:min_len]
                new_sample = self.X_after[self.y_after == l][column].values[:min_len]

                corr, _ = spearmanr(old_sample, new_sample)
                details[l][column]["spearman_coeff"] = corr
                if abs(corr) < self.spearman_thr:
                    details[l][column]["drift"] = True
        drift_flag = np.mean([details[l][col]["drift"] for col in self.X_before.columns for l in self.labels]) > self.drift_thr
        return drift_flag, details

    def _anderson_darling_test(self):
        details = {
            l: {col: {"drift": False, "p_value": None, "stat": None, "critical": None} for col in self.X_before.columns}
            for l in self.labels
        }

        for l in self.labels:
            for column in self.X_before.columns:
                stat, critical, p_value = anderson_ksamp(
                    [self.X_before[self.y_before == l][column], self.X_after[self.y_after == l][column]]
                )
                if p_value < self.alpha:
                    details[l][column]["drift"] = True
                details[l][column]["p_value"] = p_value
                details[l][column]["stat"] = stat
                details[l][column]["critical"] = critical
        drift_flag = np.mean([details[l][col]["drift"] for col in self.X_before.columns for l in self.labels]) > self.drift_thr
        return drift_flag, details
