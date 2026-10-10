"""Feature importance drift analysis, temporal discriminators, and selective localization."""

from typing import Any
import warnings
import numpy as np
import pandas as pd

from .base import AbstainStrategy
from .methods import calculate_feature_importance


class FeatureImportanceDriftAnalyzer:
    r"""Analyze data drift and concept drift attributions using feature importance methods.

    Trains temporal discriminators to distinguish between reference ($t=0$) and
    detection ($t=1$) windows. Evaluates feature attributions via Permutation Feature
    Importance (PFI), SHAP, or LIME. Distinguishes Covariate Shift ($P(X)$ changes)
    from Real Concept Drift ($P(Y \mid X)$ changes) and supports selective classification
    with conformal abstention to isolate the drift locus $S_{\text{drift}}$.

    Attributes:
        feature_names: Names of feature columns, or generated identifiers.
        X_before: Pre-drift feature matrix.
        y_before: Pre-drift target array.
        X_after: Post-drift feature matrix.
        y_after: Post-drift target array.

    Examples:
        >>> analyzer = FeatureImportanceDriftAnalyzer(X_ref, y_ref, X_det, y_det)
        >>> res = analyzer.compute_drift_importance(importance_method="permutation")
        >>> print(res["importance_mean"])
    """

    def __init__(
        self,
        X_before: np.ndarray | pd.DataFrame,
        y_before: np.ndarray | pd.Series,
        X_after: np.ndarray | pd.DataFrame,
        y_after: np.ndarray | pd.Series,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize the analyzer with pre-drift and post-drift data splits.

        Args:
            X_before: Feature matrix from the reference window.
            y_before: Target values from the reference window.
            X_after: Feature matrix from the detection window.
            y_after: Target values from the detection window.
            feature_names: Optional sequence of feature names. If None and inputs
                are DataFrames, column names are extracted.
        """
        if feature_names is None and hasattr(X_before, "columns"):
            feature_names = X_before.columns.tolist()

        self.feature_names = feature_names

        if hasattr(X_before, "values"):
            X_before = X_before.values
        if hasattr(y_before, "values"):
            y_before = y_before.values
        if hasattr(X_after, "values"):
            X_after = X_after.values
        if hasattr(y_after, "values"):
            y_after = y_after.values

        self.X_before = np.asarray(X_before)
        self.y_before = np.asarray(y_before)
        self.X_after = np.asarray(X_after)
        self.y_after = np.asarray(y_after)

    def _prepare_drift_data(self, include_target: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if include_target:
            X_combined = np.concatenate([self.X_before, self.X_after])
            y_combined = np.concatenate([self.y_before, self.y_after])
            X_features = np.column_stack([X_combined, y_combined])
        else:
            X_features = np.concatenate([self.X_before, self.X_after])

        n_samples_before = len(self.X_before)
        n_samples_after = len(self.X_after)
        time_labels = np.array([0] * n_samples_before + [1] * n_samples_after)

        if include_target:
            feature_names_for_calc = (self.feature_names + ["Y"]) if self.feature_names else None
            if feature_names_for_calc is None:
                feature_names_for_calc = [f"Feature {i}" for i in range(X_features.shape[1])]
        else:
            feature_names_for_calc = self.feature_names
            if feature_names_for_calc is None:
                feature_names_for_calc = [f"Feature {i}" for i in range(X_features.shape[1])]

        return X_features, time_labels, feature_names_for_calc

    def _filter_importance_results(
        self,
        fi_result: dict[str, Any],
        include_target: bool,
        feature_names_for_calc: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        if include_target:
            if "importances_mean" in fi_result:
                fi_result["importances_mean"] = fi_result["importances_mean"][:-1]

            if "importances_std" in fi_result:
                fi_result["importances_std"] = fi_result["importances_std"][:-1]

            if "importances" in fi_result:
                if fi_result["importances"].shape[0] == len(feature_names_for_calc):
                    fi_result["importances"] = fi_result["importances"][:-1]
                elif len(fi_result["importances"].shape) > 1 and fi_result["importances"].shape[1] == len(
                    feature_names_for_calc
                ):
                    fi_result["importances"] = fi_result["importances"][:, :-1]

            feature_names_returned = self.feature_names if self.feature_names else feature_names_for_calc[:-1]
        else:
            feature_names_returned = feature_names_for_calc

        return fi_result, feature_names_returned

    @staticmethod
    def _localize_by_confidence_threshold(model: Any, X_features: np.ndarray, tau: float = 0.15) -> np.ndarray:
        r"""Localize drift locus using confidence threshold margin on discriminator probabilities.

        Samples with $|P(T=1 \mid x) - 0.5| > \tau$ form the accepted drift locus $S_{\text{drift}}$.
        Samples with $|P(T=1 \mid x) - 0.5| \le \tau$ are abstained.

        Args:
            model: Trained temporal discriminator model.
            X_features: Combined feature matrix.
            tau: Indifference margin threshold in $[0.0, 0.5]$.

        Returns:
            Boolean mask array where True indicates membership in $S_{\text{drift}}$.

        Raises:
            ValueError: If `tau` is not within $[0.0, 0.5]$.
        """
        if not (0.0 <= tau <= 0.5):
            raise ValueError(f"tau must be in [0.0, 0.5], got {tau}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_features)
            if hasattr(model, "classes_"):
                class_1_indices = np.where(model.classes_ == 1)[0]
                p1 = proba[:, class_1_indices[0]] if len(class_1_indices) > 0 else proba[:, -1]
            else:
                p1 = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        elif hasattr(model, "decision_function"):
            df = model.decision_function(X_features)
            p1 = 1.0 / (1.0 + np.exp(-df))
        else:
            preds = model.predict(X_features)
            p1 = preds.astype(float)

        return np.abs(p1 - 0.5) > tau

    @staticmethod
    def _localize_by_conformal(
        model_class: Any,
        model_params: dict[str, Any],
        X_features: np.ndarray,
        time_labels: np.ndarray,
        alpha: float = 0.05,
        n_bootstraps: int = 25,
        random_state: int | None = 42,
    ) -> np.ndarray:
        r"""Localize drift locus using conformal predictions with out-of-bag calibration across bootstraps.

        Derives conformal p-values:
            $$p_{\text{drifting}}(x) = \min_{y \in \{0, 1\}} p_y(x)$$
        where samples with $p_{\text{drifting}}(x) < \alpha$ reject $H_0$ (invariant) and form $S_{\text{drift}}$.

        Args:
            model_class: Model estimator class for bootstrap training.
            model_params: Initialization parameters passed to `model_class`.
            X_features: Combined feature matrix.
            time_labels: Binary temporal indicators ($0$ for reference, $1$ for detection).
            alpha: Conformal significance level in $(0.0, 1.0)$.
            n_bootstraps: Number of bootstrap iterations.
            random_state: Seed for pseudo-random bootstrap sampling.

        Returns:
            Boolean mask array where True indicates rejection of invariant null hypothesis.

        Raises:
            ValueError: If `alpha` is not within $(0.0, 1.0)$.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}")

        n_samples = len(X_features)
        rng = np.random.RandomState(random_state)
        p_drift_matrix = []

        for _ in range(n_bootstraps):
            bootstrap_seed = rng.randint(0, 1000000)
            boot_rng = np.random.RandomState(bootstrap_seed)

            in_bag = boot_rng.choice(n_samples, size=n_samples, replace=True)
            in_bag_set = set(in_bag)
            oob = np.array([i for i in range(n_samples) if i not in in_bag_set])

            if len(oob) < 5 or len(np.unique(time_labels[in_bag])) < 2 or len(np.unique(time_labels[oob])) < 2:
                continue

            boot_params = dict(model_params)
            if "random_state" in boot_params:
                boot_params["random_state"] = bootstrap_seed

            boot_model = model_class(**boot_params)
            boot_model.fit(X_features[in_bag], time_labels[in_bag])

            if hasattr(boot_model, "predict_proba"):
                oob_proba = boot_model.predict_proba(X_features[oob])
                full_proba = boot_model.predict_proba(X_features)

                classes = getattr(boot_model, "classes_", np.array([0, 1]))
                c0_idx = np.where(classes == 0)[0][0] if 0 in classes else 0
                c1_idx = np.where(classes == 1)[0][0] if 1 in classes else (1 if oob_proba.shape[1] > 1 else 0)

                oob_p0 = oob_proba[:, c0_idx]
                oob_p1 = oob_proba[:, c1_idx]

                full_p0 = full_proba[:, c0_idx]
                full_p1 = full_proba[:, c1_idx]
            else:
                oob_preds = boot_model.predict(X_features[oob])
                full_preds = boot_model.predict(X_features)
                oob_p1 = (oob_preds == 1).astype(float)
                oob_p0 = 1.0 - oob_p1
                full_p1 = (full_preds == 1).astype(float)
                full_p0 = 1.0 - full_p1

            oob_y = time_labels[oob]
            s_cal = np.where(oob_y == 1, 1.0 - oob_p1, 1.0 - oob_p0)
            sorted_s_cal = np.sort(s_cal)
            n_cal = len(sorted_s_cal)

            s_test_0 = 1.0 - full_p0
            s_test_1 = 1.0 - full_p1

            count_ge_0 = n_cal - np.searchsorted(sorted_s_cal, s_test_0, side="left")
            p0 = (1.0 + count_ge_0) / (n_cal + 1.0)

            count_ge_1 = n_cal - np.searchsorted(sorted_s_cal, s_test_1, side="left")
            p1 = (1.0 + count_ge_1) / (n_cal + 1.0)

            p_drift_b = np.minimum(p0, p1)
            p_drift_matrix.append(p_drift_b)

        if not p_drift_matrix:
            return np.ones(n_samples, dtype=bool)

        p_drifting = np.median(np.array(p_drift_matrix), axis=0)
        return p_drifting < alpha

    def compute_drift_importance(
        self,
        importance_method: str = "permutation",
        include_target: bool = True,
        model_class: Any | None = None,
        model_params: dict[str, Any] | None = None,
        abstain_strategy: str | None = None,
        tau: float = 0.15,
        alpha: float = 0.05,
        min_samples: int = 20,
        n_bootstraps: int = 25,
        random_state: int = 42,
    ) -> dict[str, Any]:
        r"""Compute data drift or concept drift feature attributions with optional model abstention.

        When `include_target=True`, analyzes Concept Drift ($P(Y \mid X)$ shifts) by
        concatenating features and labels $[X, Y]$ to classify the temporal window.
        When `include_target=False`, analyzes Covariate Shift ($P(X)$ shifts) using
        only feature representations $X$.

        Args:
            importance_method: Method to calculate feature importance ("permutation", "shap", "lime").
            include_target: Whether to append the target label $Y$ into the discriminator input matrix.
            model_class: Classifier wrapper class used for temporal discrimination. Defaults to `MLPModel`.
            model_params: Optional initialization dictionary passed to `model_class`.
            abstain_strategy: Optional strategy to isolate drift locus (None, "confidence_threshold", "conformal").
            tau: Indifference margin threshold for confidence_threshold abstention.
            alpha: Significance level for conformal prediction hypothesis testing.
            min_samples: Minimum samples in drift locus required before falling back to full dataset.
            n_bootstraps: Number of bootstrap iterations for conformal calibration.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary containing:
                - 'model': Trained classifier instance.
                - 'accuracy': Overall discriminator classification accuracy.
                - 'importance_result': Detailed attribution metrics dictionary.
                - 'importance_mean': Mean importance array across evaluated features.
                - 'importance_std': Standard deviation array across permutations.
                - 'feature_names': List of feature names corresponding to the scores.
                - 'drift_coverage': Proportion of instances in the drift locus ($|S_{\text{drift}}| / N$).
                - 'selective_accuracy': Accuracy evaluated strictly on accepted drift locus instances.
                - 'rejection_rate': Proportion of abstained invariant instances ($1.0 - \text{coverage}$).
                - 'drifting_mask': Boolean array indicating which samples belong to the drift locus.
                - 'drift_locus_fallback': Boolean indicating whether fallback to full dataset occurred.

        Raises:
            ValueError: If `abstain_strategy` is unknown or parameter ranges are violated.
        """
        X_features, time_labels, feature_names_for_calc = self._prepare_drift_data(include_target)

        if model_class is None:
            from stride.models.mlp import MLPModel

            model_class = MLPModel

        if model_params is None:
            model_params = {}

        model = model_class(**model_params)
        model.fit(X_features, time_labels)
        accuracy = float(model.score(X_features, time_labels))

        drift_locus_fallback = False
        n_samples = len(X_features)

        if abstain_strategy is None:
            drifting_mask = np.ones(n_samples, dtype=bool)
            drift_coverage = 1.0
            rejection_rate = 0.0
            selective_accuracy = accuracy
            X_eval = X_features
            time_eval = time_labels
        elif abstain_strategy == AbstainStrategy.CONFIDENCE_THRESHOLD:
            drifting_mask = self._localize_by_confidence_threshold(model, X_features, tau=tau)
            drift_coverage = float(np.mean(drifting_mask))
            rejection_rate = float(1.0 - drift_coverage)
        elif abstain_strategy == AbstainStrategy.CONFORMAL:
            drifting_mask = self._localize_by_conformal(
                model_class,
                model_params,
                X_features,
                time_labels,
                alpha=alpha,
                n_bootstraps=n_bootstraps,
                random_state=random_state,
            )
            drift_coverage = float(np.mean(drifting_mask))
            rejection_rate = float(1.0 - drift_coverage)
        else:
            raise ValueError(
                f"Unknown abstain_strategy: '{abstain_strategy}'. "
                f"Supported strategies are: {AbstainStrategy.all_available()} or None."
            )

        if abstain_strategy is not None:
            n_drift = int(np.sum(drifting_mask))
            if n_drift > 0:
                preds = model.predict(X_features[drifting_mask])
                selective_accuracy = float(np.mean(preds == time_labels[drifting_mask]))
            else:
                selective_accuracy = accuracy

            if n_drift >= min_samples:
                X_eval = X_features[drifting_mask]
                time_eval = time_labels[drifting_mask]
            else:
                warnings.warn(
                    f"Identified drift locus contains only {n_drift} samples (< min_samples={min_samples}). "
                    f"Falling back to full-dataset evaluation for feature importance.",
                    UserWarning,
                    stacklevel=2,
                )
                X_eval = X_features
                time_eval = time_labels
                drift_locus_fallback = True

        fi_result = calculate_feature_importance(
            model, X_eval, time_eval, method=importance_method, feature_names=feature_names_for_calc
        )

        fi_result, feature_names_returned = self._filter_importance_results(fi_result, include_target, feature_names_for_calc)

        importance_mean = fi_result["importances_mean"]
        importance_std = fi_result["importances_std"]

        return {
            "model": model,
            "accuracy": accuracy,
            "importance_result": fi_result,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
            "feature_names": feature_names_returned,
            "drift_coverage": drift_coverage,
            "selective_accuracy": selective_accuracy,
            "rejection_rate": rejection_rate,
            "drifting_mask": drifting_mask,
            "drift_locus_fallback": drift_locus_fallback,
        }

    def compute_predictive_importance_shift(
        self,
        importance_method: str = "permutation",
        model_class: Any | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute how predictive feature importance shifts before and after drift.

        Trains separate classifiers on reference and detection data, evaluates feature
        attributions for predicting label $Y$, and returns comparative importance scores.

        Args:
            importance_method: Method to calculate feature importance ("permutation", "shap", "lime").
            model_class: Classifier class to fit on splits. Defaults to `MLPModel`.
            model_params: Optional initialization dictionary passed to `model_class`.

        Returns:
            Dictionary containing:
                - 'model_before': Classifier trained on pre-drift window.
                - 'model_after': Classifier trained on post-drift window.
                - 'accuracy_before': Accuracy on pre-drift window.
                - 'accuracy_after': Accuracy on post-drift window.
                - 'fi_before': Feature importance results for pre-drift model.
                - 'fi_after': Feature importance results for post-drift model.
        """
        X_features_before = self.X_before
        X_features_after = self.X_after

        if model_class is None:
            from stride.models.mlp import MLPModel

            model_class = MLPModel

        if model_params is None:
            model_params = {}

        model_before = model_class(**model_params)
        model_before.fit(X_features_before, self.y_before)
        acc_before = float(model_before.score(X_features_before, self.y_before))

        model_after = model_class(**model_params)
        model_after.fit(X_features_after, self.y_after)
        acc_after = float(model_after.score(X_features_after, self.y_after))

        fi_before = calculate_feature_importance(
            model_before, X_features_before, self.y_before, method=importance_method, feature_names=self.feature_names
        )

        fi_after = calculate_feature_importance(
            model_after, X_features_after, self.y_after, method=importance_method, feature_names=self.feature_names
        )

        return {
            "model_before": model_before,
            "model_after": model_after,
            "accuracy_before": acc_before,
            "accuracy_after": acc_after,
            "fi_before": fi_before,
            "fi_after": fi_after,
        }
