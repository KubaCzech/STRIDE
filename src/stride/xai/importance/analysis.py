import warnings
import numpy as np
from .base import AbstainStrategy
from .methods import calculate_feature_importance


class FeatureImportanceDriftAnalyzer:
    """
    Analyzer for detecting and explaining drift using feature importance methods.

    This class provides methods to analyze data drift (changes in P(X)),
    concept drift (changes in P(Y|X)), and predictive importance shifts.
    It encapsulates the data splits (before/after drift) and feature names.
    """

    def __init__(self, X_before, y_before, X_after, y_after, feature_names=None):
        """
        Initialize the analyzer with data splits.

        Parameters
        ----------
        X_before : array-like or pd.DataFrame
            Features from the window before the drift.
        y_before : array-like or pd.Series
            Target values from the window before the drift.
        X_after : array-like or pd.DataFrame
            Features from the window after the drift.
        y_after : array-like or pd.Series
            Target values from the window after the drift.
        feature_names : list, optional
            List of feature names. If None and input is DataFrame, columns are used.
        """
        # Prepare features and labels
        if feature_names is None and hasattr(X_before, "columns"):
            feature_names = X_before.columns.tolist()

        self.feature_names = feature_names

        # Convert to numpy if pandas
        if hasattr(X_before, "values"):
            X_before = X_before.values
        if hasattr(y_before, "values"):
            y_before = y_before.values
        if hasattr(X_after, "values"):
            X_after = X_after.values
        if hasattr(y_after, "values"):
            y_after = y_after.values

        self.X_before = X_before
        self.y_before = y_before
        self.X_after = X_after
        self.y_after = y_after

    def _prepare_drift_data(self, include_target):
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

    def _filter_importance_results(self, fi_result, include_target, feature_names_for_calc):
        if include_target:
            # 1. Update arrays in fi_result
            if "importances_mean" in fi_result:
                fi_result["importances_mean"] = fi_result["importances_mean"][:-1]

            if "importances_std" in fi_result:
                fi_result["importances_std"] = fi_result["importances_std"][:-1]

            if "importances" in fi_result:
                # Check shape to determine axis to slice
                if fi_result["importances"].shape[0] == len(feature_names_for_calc):
                    fi_result["importances"] = fi_result["importances"][:-1]
                elif len(fi_result["importances"].shape) > 1 and fi_result["importances"].shape[1] == len(
                    feature_names_for_calc
                ):
                    fi_result["importances"] = fi_result["importances"][:, :-1]

            # Features to return (exclude Y)
            feature_names_returned = self.feature_names if self.feature_names else feature_names_for_calc[:-1]
        else:
            feature_names_returned = feature_names_for_calc

        return fi_result, feature_names_returned

    @staticmethod
    def _localize_by_confidence_threshold(model, X_features: np.ndarray, tau: float = 0.15) -> np.ndarray:
        """
        Localize drift using confidence threshold margin on discriminator probabilities.

        Samples with |P(T=1|x) - 0.5| > tau form the drift locus mask S_drift.
        Samples with |P(T=1|x) - 0.5| <= tau are abstained (invariant region L^c).
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
        model_class,
        model_params: dict,
        X_features: np.ndarray,
        time_labels: np.ndarray,
        alpha: float = 0.05,
        n_bootstraps: int = 25,
        random_state: int | None = 42,
    ) -> np.ndarray:
        """
        Localize drift using conformal predictions with out-of-bag calibration across bootstraps.

        Following Hinder et al. (ESANN 2026), derives conformal p-values:
            p_drifting(x) = min_{y in {0, 1}} p_y(x)
        where points with p_drifting(x) < alpha reject H0 ('non-drifting') and form S_drift.
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

            # Ensure OOB has samples and both splits contain both classes
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

            # Non-conformity score on calibration (OOB) samples: s_i = 1 - P(T = T_i | x_i)
            oob_y = time_labels[oob]
            s_cal = np.where(oob_y == 1, 1.0 - oob_p1, 1.0 - oob_p0)
            sorted_s_cal = np.sort(s_cal)
            n_cal = len(sorted_s_cal)

            # Test non-conformity scores for candidate classes 0 and 1
            s_test_0 = 1.0 - full_p0
            s_test_1 = 1.0 - full_p1

            # Vectorized conformal p-values: (1 + count(s_cal >= s_test)) / (n_cal + 1)
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
        importance_method="permutation",
        include_target=True,
        model_class=None,
        model_params=None,
        abstain_strategy=None,
        tau=0.15,
        alpha=0.05,
        min_samples=20,
        n_bootstraps=25,
        random_state=42,
    ):
        """
        Compute drift analysis (data drift or concept drift) importance with optional model abstention.

        If include_target is True, this analyzes Concept Drift (P(Y|X) changes) by using both
        features and target (X, Y) to classify time periods.
        If include_target is False, this analyzes Data Drift (P(X) changes) by using only
        features (X) to classify time periods.

        Parameters
        ----------
        importance_method : str, default="permutation"
            Method to calculate feature importance ("permutation", "shap", "lime").
        include_target : bool, default=True
            Whether to include the target variable 'Y' in the analysis.
        model_class : class, optional
            Class of the model to use for classification. Defaults to MLPModel.
        model_params : dict, optional
            Parameters to initialize the model.
        abstain_strategy : str or None, default=None
            Abstention strategy to isolate the drift locus:
            - None: No abstention, full dataset evaluated.
            - "confidence_threshold": Indifference margin thresholding (|P(T=1|x) - 0.5| > tau).
            - "conformal": Conformal prediction hypothesis testing (p_drifting < alpha).
        tau : float, default=0.15
            Indifference margin threshold for confidence_threshold abstention.
        alpha : float, default=0.05
            Significance level for conformal prediction hypothesis testing.
        min_samples : int, default=20
            Minimum samples in drift locus required to evaluate conditional feature importance.
            If fewer, gracefully falls back to full-dataset evaluation with a diagnostic warning.
        n_bootstraps : int, default=25
            Number of bootstrap calibrations for conformal prediction.
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        dict
            A dictionary containing:
            - 'model': The trained classifier.
            - 'accuracy': The overall accuracy on all time-period instances.
            - 'importance_result': Full feature importance results.
            - 'importance_mean': Mean importance scores.
            - 'importance_std': Standard deviation of importance scores.
            - 'feature_names': List of feature names.
            - 'drift_coverage': Proportion of instances in the drift locus (|S_drift| / N).
            - 'selective_accuracy': Accuracy evaluated strictly on accepted drift locus instances.
            - 'rejection_rate': Proportion of abstained invariant instances (1.0 - drift_coverage).
            - 'drifting_mask': Boolean array indicating which samples belong to the drift locus.
            - 'drift_locus_fallback': Boolean indicating whether fallback to full dataset was triggered.
        """
        # Prepare Data
        X_features, time_labels, feature_names_for_calc = self._prepare_drift_data(include_target)

        # Train Model
        if model_class is None:
            from stride.models.mlp import MLPModel

            model_class = MLPModel

        if model_params is None:
            model_params = {}

        model = model_class(**model_params)
        model.fit(X_features, time_labels)
        accuracy = float(model.score(X_features, time_labels))

        # Localize Drift via Abstention
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

        # Calculate Feature Importance
        fi_result = calculate_feature_importance(
            model, X_eval, time_eval, method=importance_method, feature_names=feature_names_for_calc
        )

        # Filter results
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

    def compute_predictive_importance_shift(self, importance_method="permutation", model_class=None, model_params=None):
        """
        Compute how predictive feature importance shifts before and after drift.

        This method trains two separate models: one on 'before' data and one on
        'after' data. It then calculates feature importance for both models to
        predict the target variable 'Y'. Changes in feature importance rankings
        or magnitudes indicate that the underlying predictive mechanism has shifted.

        Parameters
        ----------
        importance_method : str, default="permutation"
            Method to calculate feature importance.
        model_class : class, optional
            Class of the model to use. Defaults to MLPModel.
        model_params : dict, optional
            Parameters for the model.

        Returns
        -------
        dict
            A dictionary containing:
            - 'model_before': Model trained on pre-drift data.
            - 'model_after': Model trained on post-drift data.
            - 'accuracy_before': Accuracy of the pre-drift model on pre-drift data.
            - 'accuracy_after': Accuracy of the post-drift model on post-drift data.
            - 'fi_before': Feature importance results for the pre-drift model.
            - 'fi_after': Feature importance results for the post-drift model.
        """
        X_features_before = self.X_before
        X_features_after = self.X_after

        # Train Models
        if model_class is None:
            from stride.models.mlp import MLPModel

            model_class = MLPModel

        if model_params is None:
            model_params = {}

        # Model trained BEFORE drift
        model_before = model_class(**model_params)
        model_before.fit(X_features_before, self.y_before)
        acc_before = model_before.score(X_features_before, self.y_before)

        # Model trained AFTER drift
        model_after = model_class(**model_params)
        model_after.fit(X_features_after, self.y_after)
        acc_after = model_after.score(X_features_after, self.y_after)

        # Feature Importance for BEFORE drift
        fi_before = calculate_feature_importance(
            model_before, X_features_before, self.y_before, method=importance_method, feature_names=self.feature_names
        )

        # Feature Importance for AFTER drift
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
