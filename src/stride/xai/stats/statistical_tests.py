"""Statistical hypothesis testing and distance metrics for data drift detection."""

from enum import Enum
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy.stats import anderson_ksamp, ks_2samp, spearmanr, wasserstein_distance

from stride.common import DataScaler, ScalingType


class StatisticalTestType(Enum):
    """Supported two-sample statistical tests and divergence metrics."""

    KolmogorovSmirnov = "kolmogorov_smirnov"
    KullbackLeibler = "kullback_leibler"
    WassersteinDistance = "wasserstein"
    JensenShannon = "jensen_shannon"
    Spearman = "spearman"
    AD = "anderson_darling"
    All = "all"


class StatisticalTestsDriftDetector:
    r"""Detect data drift using non-parametric statistical hypothesis tests and distances.

    Applies statistical two-sample tests comparing marginal feature distributions
    conditioned on class labels $P(X \mid Y=y)$ across reference and detection windows.
    Supported tests include Kolmogorov-Smirnov ($D_{KS}$), Anderson-Darling, 1D
    Wasserstein distance ($\mathcal{W}_1$), Kullback-Leibler divergence ($D_{KL}$),
    Jensen-Shannon divergence ($D_{JS}$), and Spearman rank correlation ($r_s$).

    Attributes:
        decision_thr: Proportion of evaluated tests that must signal drift to declare overall alarm.
        alpha: Statistical significance level $\alpha$ for p-value hypothesis tests.
        drift_flag: Overall drift alarm status.
        drift_details: Nested dictionary containing per-test, per-class, and per-feature diagnostics.
    """

    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: np.ndarray | pd.Series,
        X_after: pd.DataFrame,
        y_after: np.ndarray | pd.Series,
        decision_thr: float = 0.4,
        alpha: float = 0.05,
        bins: int = 30,
        kl_thr: float = 0.1,
        wasserstein_thr: float = 0.1,
        js_thr: float = 0.05,
        spearman_thr: float = 0.9,
        drift_thr: float = 0.2,
    ) -> None:
        r"""Initialize the statistical tests drift detector.

        Args:
            X_before: Feature matrix for baseline reference window.
            y_before: Target labels for baseline reference window.
            X_after: Feature matrix for detection window.
            y_after: Target labels for detection window.
            decision_thr: Proportion of tests that must trigger drift to raise global alarm.
            alpha: Significance level $\alpha$ for hypothesis tests (e.g., $0.05$).
            bins: Number of histogram bins for discrete divergence calculations.
            kl_thr: Threshold on Kullback-Leibler divergence.
            wasserstein_thr: Threshold on normalized Wasserstein distance $\mathcal{W}_1$.
            js_thr: Threshold on Jensen-Shannon divergence.
            spearman_thr: Minimum rank correlation below which drift is flagged.
            drift_thr: Proportion of feature-class pairs required to declare test-level drift.
        """
        self.X_before = X_before
        self.y_before = np.asarray(y_before)
        self.X_after = X_after
        self.y_after = np.asarray(y_after)

        self.decision_thr = decision_thr
        self.bins = bins
        self.alpha = alpha

        self.kl_thr = kl_thr
        self.wasserstein_thr = wasserstein_thr
        self.js_thr = js_thr
        self.spearman_thr = spearman_thr
        self.drift_thr = drift_thr

        self.drift_flag: bool | None = None
        self.drift_flags: dict[str, bool] | None = None
        self.drift_details: dict[str, dict] | None = None

        self.labels = sorted(list(set(self.y_before).union(set(self.y_after))))
        self._scale_data()

    def _scale_data(self) -> None:
        scaler = DataScaler(ScalingType.MinMax)
        self.X_before_scaled = scaler.fit_transform(self.X_before, return_df=True)
        self.X_after_scaled = scaler.transform(self.X_after, return_df=True)

    def detect(self, test_type: list[StatisticalTestType] | StatisticalTestType) -> bool:
        """Execute selected statistical test(s) to evaluate distribution stability.

        Args:
            test_type: Single test or list of `StatisticalTestType` to compute.

        Returns:
            Boolean indicator `drift_flag` stating whether drift was detected.

        Raises:
            ValueError: If an unsupported test type is provided.
        """
        if test_type == StatisticalTestType.All:
            evaluated_tests = [t for t in StatisticalTestType if t != StatisticalTestType.All]
        elif isinstance(test_type, StatisticalTestType):
            evaluated_tests = [test_type]
        elif isinstance(test_type, list):
            for t in test_type:
                if not isinstance(t, StatisticalTestType):
                    raise ValueError("Unsupported test type")
            evaluated_tests = test_type
        else:
            raise ValueError("Unsupported test type")

        drift_flags = {}
        details = {}

        for test in evaluated_tests:
            curr_drift_flag, curr_details = self._detect_single_statistic(test)
            drift_flags[test.value] = curr_drift_flag
            details[test.value] = curr_details

        self.drift_flag = bool(sum(drift_flags.values()) / len(drift_flags) > self.decision_thr)
        self.drift_flags = drift_flags
        self.drift_details = details
        return self.drift_flag

    def _detect_single_statistic(self, test: StatisticalTestType) -> tuple[bool, dict]:
        test_method = getattr(self, f"_{test.value}_test")
        return test_method()

    def _kolmogorov_smirnov_test(self) -> tuple[bool, dict]:
        details = {
            l: {col: {"drift": False, "p_value": None, "stat": None} for col in self.X_before.columns} for l in self.labels
        }

        for l in self.labels:
            for column in self.X_before.columns:
                stat, p_value = ks_2samp(
                    self.X_before[self.y_before == l][column],
                    self.X_after[self.y_after == l][column],
                )
                details[l][column]["p_value"] = p_value
                details[l][column]["stat"] = stat
                if p_value < self.alpha:
                    details[l][column]["drift"] = True

        drift_flag = bool(
            np.mean([details[l][col]["drift"] for col in self.X_before.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _kullback_leibler_test(self) -> tuple[bool, dict]:
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

                kl_div = float(np.sum(rel_entr(old_dist, new_dist)))
                details[l][column]["kl_div"] = kl_div
                if kl_div > self.kl_thr:
                    details[l][column]["drift"] = True

        drift_flag = bool(
            np.mean([details[l][col]["drift"] for col in self.X_before_scaled.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _wasserstein_test(self) -> tuple[bool, dict]:
        details = {l: {col: {"drift": False, "wd": None} for col in self.X_before_scaled.columns} for l in self.labels}

        for l in self.labels:
            for column in self.X_before_scaled.columns:
                wd = float(
                    wasserstein_distance(
                        self.X_before_scaled[self.y_before == l][column],
                        self.X_after_scaled[self.y_after == l][column],
                    )
                )
                details[l][column]["wd"] = wd
                if wd > self.wasserstein_thr:
                    details[l][column]["drift"] = True

        drift_flag = bool(
            np.mean([details[l][col]["drift"] for col in self.X_before_scaled.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _jensen_shannon_test(self) -> tuple[bool, dict]:
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

                js_div = float(jensenshannon(old_dist, new_dist) ** 2)
                details[l][column]["js_div"] = js_div
                if js_div > self.js_thr:
                    details[l][column]["drift"] = True

        drift_flag = bool(
            np.mean([details[l][col]["drift"] for col in self.X_before_scaled.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _spearman_test(self) -> tuple[bool, dict]:
        details = {l: {col: {"drift": False, "spearman_coeff": None} for col in self.X_before.columns} for l in self.labels}

        for l in self.labels:
            for column in self.X_before.columns:
                min_len = min(
                    len(self.X_before[self.y_before == l][column]),
                    len(self.X_after[self.y_after == l][column]),
                )
                old_sample = self.X_before[self.y_before == l][column].values[:min_len]
                new_sample = self.X_after[self.y_after == l][column].values[:min_len]

                corr, _ = spearmanr(old_sample, new_sample)
                details[l][column]["spearman_coeff"] = corr
                if abs(corr) < self.spearman_thr:
                    details[l][column]["drift"] = True

        drift_flag = bool(
            np.mean([details[l][col]["drift"] for col in self.X_before.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details

    def _anderson_darling_test(self) -> tuple[bool, dict]:
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

        drift_flag = bool(
            np.mean([details[l][col]["drift"] for col in self.X_before.columns for l in self.labels]) > self.drift_thr
        )
        return drift_flag, details
