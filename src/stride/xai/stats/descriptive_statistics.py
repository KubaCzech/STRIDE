"""Descriptive statistics drift detection across features and class conditional distributions."""

from collections import defaultdict
from enum import Enum
from typing import Any
import numpy as np
import pandas as pd


class StatisticsType(Enum):
    """Supported descriptive summary statistics."""

    Mean = "mean"
    StandardDeviation = "std"
    Min = "min"
    Max = "max"
    Median = "median"
    ImbalanceRatio = "imbalance_ratio"
    All = "all"


class DescriptiveStatisticsDriftDetector:
    """Detect data drift based on shifts in descriptive statistics across stream blocks.

    Evaluates conditional summary statistics ($mean$, $std$, $min$, $max$, $median$,
    class imbalance ratio) between two stream data windows. Determines relative
    percentage shifts per feature and class label, signaling drift when aggregate
    shifts violate user-defined decision thresholds.

    Attributes:
        data_before: Baseline data block with label column appended.
        data_after: Current detection data block with label column appended.
        decision_thr: Proportion of evaluated statistics that must flag drift to raise global alarm.
        drift_flag: Boolean indicator stating whether drift was detected.
        drift_details: Detailed map of per-class, per-feature statistic drift flags.
        stat_shifts: Computed relative numeric shifts.
    """

    data_before: pd.DataFrame
    data_after: pd.DataFrame
    decision_thr: float
    drift_flag: bool
    drift_details: defaultdict[Any, defaultdict[Any, dict[str, Any]]] | None
    stat_shifts: defaultdict[Any, defaultdict[Any, dict[str, Any]]] | None

    def __init__(
        self,
        X_before: pd.DataFrame,
        y_before: np.ndarray,
        X_after: pd.DataFrame,
        y_after: np.ndarray,
        decision_thr: float = 0.4,
    ) -> None:
        """Initialize the descriptive statistics drift detector.

        Args:
            X_before: Feature matrix of baseline reference block.
            y_before: Target labels of baseline reference block.
            X_after: Feature matrix of current detection block.
            y_after: Target labels of current detection block.
            decision_thr: Minimum fraction of triggered statistics required to declare drift.
        """
        self.data_before = pd.concat([X_before.copy(), pd.Series(y_before, name="label")], axis=1)
        self.data_after = pd.concat([X_after.copy(), pd.Series(y_after, name="label")], axis=1)

        self.decision_thr = decision_thr
        self.drift_details = None
        self.stat_shifts = None
        self.drift_flag = False

    @staticmethod
    def _get_empty_dict() -> defaultdict:
        return defaultdict(lambda: defaultdict(dict))

    @staticmethod
    def _calculate_shift(old_value: float, new_value: float) -> float:
        eps = 1e-10
        return float((new_value - old_value) / abs(old_value + eps))

    @staticmethod
    def _check_shift(value: float, thr: float) -> bool:
        return value > thr

    def _detect_numeric_stat(
        self, data_before: pd.DataFrame, data_after: pd.DataFrame, stat: StatisticsType, thr: float
    ) -> bool:
        agg = stat.value
        old_values = data_before.groupby("label").agg(agg)
        new_values = data_after.groupby("label").agg(agg)

        all_labels = old_values.index.union(new_values.index)
        all_features = old_values.columns

        old_values = old_values.reindex(index=all_labels, columns=all_features, fill_value=0.0)
        new_values = new_values.reindex(index=all_labels, columns=all_features, fill_value=0.0)

        drift_flag = False
        for label in old_values.index:
            for feature in old_values.columns:
                shift = self._calculate_shift(old_values.loc[label, feature], new_values.loc[label, feature])
                drift = self._check_shift(shift, thr)

                if self.drift_details is not None:
                    self.drift_details[label][feature][stat.value] = drift
                if self.stat_shifts is not None:
                    self.stat_shifts[label][feature][stat.value] = shift
                drift_flag = drift_flag or drift

        return drift_flag

    def _detect_imbalance_ratio(self, data_before: pd.DataFrame, data_after: pd.DataFrame, thr: float) -> bool:
        old_ir = data_before["label"].value_counts(normalize=True)
        new_ir = data_after["label"].value_counts(normalize=True)

        all_labels = old_ir.index.union(new_ir.index)
        old_ir = old_ir.reindex(all_labels, fill_value=0.0)
        new_ir = new_ir.reindex(all_labels, fill_value=0.0)

        drift_flag = False
        for label in old_ir.index.union(new_ir.index):
            shift = self._calculate_shift(old_ir[label], new_ir[label])
            drift = self._check_shift(shift, thr)

            if self.drift_details is not None:
                self.drift_details[label]["__class_ratio__"]["imbalance_ratio"] = drift
            if self.stat_shifts is not None:
                self.stat_shifts[label]["__class_ratio__"]["imbalance_ratio"] = shift
            drift_flag = drift_flag or drift

        return drift_flag

    def _detect_single_statistic(
        self, data_before: pd.DataFrame, data_after: pd.DataFrame, stat_type: StatisticsType, thr: float
    ) -> bool:
        if stat_type == StatisticsType.ImbalanceRatio:
            return self._detect_imbalance_ratio(data_before, data_after, thr)
        return self._detect_numeric_stat(data_before, data_after, stat_type, thr)

    def detect(
        self,
        stat_type: list[StatisticsType] | StatisticsType,
        thr: float = 0.2,
        features: list[str] | None = None,
    ) -> tuple[bool, dict]:
        """Detect data drift based on descriptive statistics shifts.

        Args:
            stat_type: Single `StatisticsType` or collection of statistics to evaluate.
            thr: Relative shift magnitude threshold exceeding baseline to trigger individual drift.
            features: Optional subset of feature column names to evaluate.

        Returns:
            Tuple of `(drift_flag, details_dict)` where `drift_flag` indicates overall alarm.

        Raises:
            ValueError: If an unsupported statistics type is passed.
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
            evaluated_stats = [s for s in StatisticsType if s != StatisticsType.All]
        elif isinstance(stat_type, StatisticsType):
            evaluated_stats = [stat_type]
        elif isinstance(stat_type, list):
            for s in stat_type:
                if not isinstance(s, StatisticsType):
                    raise ValueError("Unsupported statistics type")
            evaluated_stats = stat_type
        else:
            raise ValueError("Unsupported statistics type")

        drifts = []
        for st in evaluated_stats:
            curr_drift = self._detect_single_statistic(data_before, data_after, st, thr)
            drifts.append(curr_drift)

        self.drift_flag = bool(sum(drifts) / len(drifts) > self.decision_thr)
        return self.drift_flag, self.drift_details

    def calculate_stats_before_after(self) -> pd.DataFrame:
        """Compute summary statistics for features across both baseline and detection blocks.

        Returns:
            MultiIndex DataFrame with baseline ('old') and detection ('new') statistics per label.
        """

        def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
            features = [c for c in df.columns if c != "label"]
            grouped = df.groupby("label")
            records = []

            for label, group in grouped:
                stats: dict[tuple[str, str], Any] = {("label", "id"): label}
                for f in features:
                    stats[(f, "median")] = group[f].median()
                    stats[(f, "mean")] = group[f].mean()
                    stats[(f, "std")] = group[f].std()
                records.append(stats)

            stats_df = pd.DataFrame(records)
            stats_df.set_index(("label", "id"), inplace=True)
            return stats_df

        stats_old = compute_stats(self.data_before)
        stats_new = compute_stats(self.data_after)

        stats_old.columns = pd.MultiIndex.from_tuples([("old", f, s) for (f, s) in stats_old.columns])
        stats_new.columns = pd.MultiIndex.from_tuples([("new", f, s) for (f, s) in stats_new.columns])

        combined = pd.concat([stats_old, stats_new], axis=1).sort_index(axis=1, level=[0, 1, 2])
        return combined
