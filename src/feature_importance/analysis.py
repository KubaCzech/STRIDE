import numpy as np
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

    def compute_drift_importance(
        self, importance_method="permutation", include_target=True, model_class=None, model_params=None
    ):
        """
        Compute drift analysis (data drift or concept drift) importance.

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

        Returns
        -------
        dict
            A dictionary containing:
            - 'model': The trained classifier.
            - 'accuracy': The accuracy of determining the time period.
            - 'importance_result': Full feature importance results.
            - 'importance_mean': Mean importance scores.
            - 'importance_std': Standard deviation of importance scores.
            - 'feature_names': List of feature names.
        """
        # Prepare Data
        X_features, time_labels, feature_names_for_calc = self._prepare_drift_data(include_target)

        # Train Model
        if model_class is None:
            from src.models.mlp import MLPModel

            model_class = MLPModel

        if model_params is None:
            model_params = {}

        model = model_class(**model_params)
        model.fit(X_features, time_labels)
        accuracy = model.score(X_features, time_labels)

        # Calculate Feature Importance
        fi_result = calculate_feature_importance(
            model, X_features, time_labels, method=importance_method, feature_names=feature_names_for_calc
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
            from src.models.mlp import MLPModel

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
