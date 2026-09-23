import numpy as np
import pandas as pd

from enum import Enum
from typing import Optional, Union
from collections import defaultdict


class StatisticsType(Enum):
    Mean = "mean"
    StandardDeviation = "std"
    Min = "min"
    Max = "max"
    Median = "median"
    ImbalanceRatio = "imbalance_ratio"
    All = "all"


class DescriptiveStatisticsDriftDetector:
    """
    Detects data drift based on descriptive statistics between two data blocks.

    Attributes
    ----------
    data_before : pd.DataFrame
        Data before potential drift, with a 'label' column appended.
    data_after : pd.DataFrame
        Data after potential drift, with a 'label' column appended.
    decision_thr : float
        Minimal change of descriptive statistic required to admit drift.
    drift_flag : bool
        Flag indicating if drift has been detected.
    drift_details : dict
        Detailed information about which features/classes/statistics triggered drift.
    stat_shifts : dict
        Numeric values of the shifts for each feature/class/statistic.
    """

    data_before: pd.DataFrame
    data_after: pd.DataFrame

    decision_thr: float
    drift_flag: bool

    drift_details: Optional[defaultdict[str, defaultdict[str, dict]]] = None
    stat_shifts: Optional[defaultdict[str, defaultdict[str, dict]]] = None

    stats: pd.DataFrame

    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: np.ndarray,
        X_after: pd.DataFrame,
        y_after: np.ndarray,
        decision_thr: float = 0.4,
    ) -> None:
        """
        Initialize the drift detector.

        Parameters
        ----------
        X_before : pd.DataFrame
            Feature values of first data block.
        y_before : np.ndarray
            Class labels of first data block.
        X_after : pd.DataFrame
            Feature values of second data block.
        y_after : np.ndarray
            Class labels of second data block.
        decision_thr : float
            Proportion of statistics that must be set to True to admit drift.
            Applies only when more than one statistic is checked.
        """
        self.data_before = pd.concat([X_before.copy(), pd.Series(y_before, name="label")], axis=1)
        self.data_after = pd.concat([X_after.copy(), pd.Series(y_after, name="label")], axis=1)

        self.decision_thr = decision_thr

        self.drift_details = None
        self.stat_shifts = None

        self.drift_flag = False

    @staticmethod
    def _get_empty_dict() -> dict:
        """
        Generate empty nested dictionary to store necessary information.

        Returns
        -------
        empty_dict : dict
            Empty, twice nested dictionary
        """
        return defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def _calculate_shift(old_value: float, new_value: float) -> float:
        """
        Calculate relative shift between old value and new value

        Parameters
        ----------
        old_value : float
            Value of a single statistic for first data block.
        new_value : float
            Value of a single statistic for second data block.

        Returns
        -------
        shift : float
            Relative shift of statistic between first and second data block.
        """
        eps = 1e-10
        shift = (new_value - old_value) / abs(old_value + eps)
        return shift

    @staticmethod
    def _check_shift(value: float, thr: float) -> bool:
        """
        Check if a numeric shift exceeds the given threshold.

        Parameters
        ----------
        value : float
            The computed shift.
        thr : float
            Threshold to determine if drift occurs.

        Returns
        -------
        bool
            True if the shift exceeds the threshold.
        """
        return value > thr

    def _detect_numeric_stat(
        self, data_before: pd.DataFrame, data_after: pd.DataFrame, stat: StatisticsType, thr: float
    ) -> bool:
        """
        Detect drift for numeric features based on a single statistic.

        Parameters
        ----------
        data_before : pd.DataFrame
            Features values and labels for first data block.
        data_after : pd.DataFrame
            Features values and labels for second data block.
        stat : StatisticsType
            Statistic to check (mean, median, std, etc.).
        thr : float
            Minimal shift of statistic between data blocks to consider drift.

        Returns
        -------
        drift_flag : bool
            Boolean flag saying whether the drift occurred or not.
        """
        agg = stat.value

        old_values = data_before.groupby("label").agg(agg)
        new_values = data_after.groupby("label").agg(agg)

        all_labels = old_values.index.union(new_values.index)
        all_features = old_values.columns

        # Handling disappearing/appearing class
        old_values = old_values.reindex(index=all_labels, columns=all_features, fill_value=0.0)
        new_values = new_values.reindex(index=all_labels, columns=all_features, fill_value=0.0)

        drift_flag = False

        for label in old_values.index:
            for feature in old_values.columns:
                shift = self._calculate_shift(old_values.loc[label, feature], new_values.loc[label, feature])
                drift = self._check_shift(shift, thr)

                self.drift_details[label][feature][stat.value] = drift
                self.stat_shifts[label][feature][stat.value] = shift
                drift_flag = drift_flag or drift

        return drift_flag

    def _detect_imbalance_ratio(self, data_before: pd.DataFrame, data_after: pd.DataFrame, thr: float) -> bool:
        """
        Detect drift in class distribution (imbalance ratio).

        Parameters
        ----------
        data_before : pd.DataFrame
            Features values and labels for first data block.
        data_after : pd.DataFrame
            Features values and labels for second data block.
        thr : float
            Minimal shift of statistic between data blocks to consider drift.

        Returns
        -------
        drift_flag : bool
            Boolean flag saying whether the drift occurred or not.
        """
        old_ir = data_before["label"].value_counts(normalize=True)
        new_ir = data_after["label"].value_counts(normalize=True)

        all_labels = old_ir.index.union(new_ir.index)

        old_ir = old_ir.reindex(all_labels, fill_value=0.0)
        new_ir = new_ir.reindex(all_labels, fill_value=0.0)

        drift_flag = False

        for label in old_ir.index.union(new_ir.index):
            shift = self._calculate_shift(old_ir[label], new_ir[label])
            drift = self._check_shift(shift, thr)

            self.drift_details[label]["__class_ratio__"]["imbalance_ratio"] = drift
            self.stat_shifts[label]["__class_ratio__"]["imbalance_ratio"] = shift
            drift_flag = drift_flag or drift

        return drift_flag

    def _detect_single_statistic(
        self, data_before: pd.DataFrame, data_after: pd.DataFrame, stat_type: StatisticsType, thr: float
    ) -> bool:
        """
        Detect drift for a single statistic type across all features and labels.

        This method evaluates either a numeric statistic (mean, std, min, max, median)
        or the class imbalance ratio. It updates the `drift_details` and `stat_shifts`
        dictionaries with per-label and per-feature drift information.

        Parameters
        ----------
        data_before : pd.DataFrame
            Features values and labels for first data block.
        data_after : pd.DataFrame
            Features values and labels for second data block.
        stat_type : StatisticType or list of object of class StatisticType
            Statistic that we want to check - available: min, median, mean, max, std, class ratio.
            We can check either one or many statistics. Use StatisticsType.All for all.
        thr : float
            How much the statistics must change to consider the drift.
        Returns
        -------
        drift_flag : bool
            True if drift is detected for any feature/label combination for the given statistic.

        Notes
        -----
        - Handles ImbalanceRatio and numeric statistics separately.
        - For numeric statistics, `data_before` and `data_after` are grouped by 'label', aggregated and checked per feature.
        - For `ImbalanceRatio`, the class distribution is compared.
        - Handles disappearing or newly appearing classes by treating missing values as 0.
        - Updates internal dictionaries:
            - `self.drift_details[label][feature][stat_type]`
            - `self.stat_shifts[label][feature][stat_type]`
        """
        if stat_type == StatisticsType.ImbalanceRatio:
            return self._detect_imbalance_ratio(data_before, data_after, thr)
        else:
            return self._detect_numeric_stat(data_before, data_after, stat_type, thr)

    def detect(
        self,
        stat_type: Union[list[StatisticsType], StatisticsType],
        thr: float = 0.2,
        features: Optional[list[str]] = None,
    ) -> tuple[bool, dict]:
        """
        Detect data drift based on descriptive statistics.

        Parameters
        ----------
        stat_type : StatisticType or list of object of class StatisticType
            Statistic that we want to check - available: min, median, mean, max, std, class ratio.
            We can check either one or many statistics. Use StatisticsType.All for all.
        thr : float
            How much the statistics must change to consider the drift, applies per statistic.
            Different from self.decision_thr.
        features : list or None
            List of features to check. If set to None, we check all features.

        Returns
        -------
        drift_flag : bool
            Boolean flag saying whether the drift occurred or not.
        details : dict
            Nested dictionary with drift information for each class/feature/statistic.
        """
        if features is not None:
            data_before = pd.concat([self.data_before[features], self.data_before["label"]], axis=1)
            data_after = pd.concat([self.data_after[features], self.data_after["label"]], axis=1)
        else:
            data_before = self.data_before.copy()
            data_after = self.data_after.copy()

        self.drift_details = self._get_empty_dict()
        self.stat_shifts = self._get_empty_dict()

        if stat_type == StatisticsType.All:
            stat_type = [s for s in StatisticsType if s != StatisticsType.All]
        elif isinstance(stat_type, StatisticsType):
            stat_type = [stat_type]
        elif isinstance(stat_type, list):
            for s in stat_type:
                if not isinstance(s, StatisticsType):
                    raise ValueError("Unsupported statistics type")

        drifts = []
        for st in stat_type:
            curr_drift = self._detect_single_statistic(data_before, data_after, st, thr)
            drifts.append(curr_drift)
        self.drift_flag = sum(drifts) / len(drifts) > self.decision_thr
        return self.drift_flag, self.drift_details

    def calculate_stats_before_after(self) -> None:
        """
        Compute descriptive statistics for features in both data_before and data_after.

        Returns
        -------
        combined : pd.DataFrame
            MultiIndex DataFrame with old and new statistics per label and feature.
            Levels: ('old'/'new', feature_name, statistic_name)
        """

        def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
            features = [c for c in df.columns if c != "label"]
            grouped = df.groupby("label")

            records = []

            for label, group in grouped:
                stats = {("label", "id"): label}

                for f in features:
                    # stats[(f, "min")] = group[f].min()
                    stats[(f, "median")] = group[f].median()
                    stats[(f, "mean")] = group[f].mean()
                    stats[(f, "std")] = group[f].std()
                    # stats[(f, "max")] = group[f].max()

                records.append(stats)

            stats_df = pd.DataFrame(records)
            stats_df.set_index(("label", "id"), inplace=True)
            return stats_df

        stats_old = compute_stats(self.data_before)
        stats_new = compute_stats(self.data_after)

        stats_old.columns = pd.MultiIndex.from_tuples([("old", f, s) for (f, s) in stats_old.columns])
        stats_new.columns = pd.MultiIndex.from_tuples([("new", f, s) for (f, s) in stats_new.columns])

        # tables merging and ordering
        combined = pd.concat([stats_old, stats_new], axis=1).sort_index(axis=1, level=[0, 1, 2])
        return combined
