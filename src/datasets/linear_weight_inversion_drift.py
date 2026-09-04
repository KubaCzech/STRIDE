import numpy as np
import pandas as pd
from .base import BaseDataset
from .utils import apply_sigmoid_drift


class LinearWeightInversionDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "linear_weight_inversion_drift"

    @property
    def display_name(self) -> str:
        return "Linear Weight Inversion Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({"n_features": 11, "n_drift_features": 5, "drift_width": 1})
        return params

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "n_samples_before",
                "type": "int",
                "label": "Number of Samples Before Drift",
                "default": 1000,
                "min_value": 100,
                "step": 100,
                "help": "Number of samples generated before the concept drift occurs.",
            },
            {
                "name": "n_samples_after",
                "type": "int",
                "label": "Number of Samples After Drift",
                "default": 1000,
                "min_value": 100,
                "step": 100,
                "help": "Number of samples generated after the concept drift occurs.",
            },
            {
                "name": "n_features",
                "type": "int",
                "label": "Number of Features (n_features)",
                "default": 11,
                "min_value": 2,
                "step": 1,
                "help": "Total number of features for the dataset. Must be >= 2.",
            },
            {
                "name": "n_drift_features",
                "type": "int",
                "label": "Number of Drifting Features (n_drift_features)",
                "default": 5,
                "min_value": 1,
                "step": 1,
                "help": "Number of features that will drift. Must be <= n_features.",
            },
            {
                "name": "drift_width",
                "type": "int",
                "label": "Drift Width (drift_width)",
                "default": 1,
                "min_value": 1,
                "step": 1,
                "help": "Width of the concept drift (number of samples).",
            },
        ]

    def generate(
        self,
        n_samples_before=1000,
        n_samples_after=1000,
        n_features=11,
        n_drift_features=5,
        random_seed=42,
        drift_width=1,
        **kwargs,
    ):
        """
        Generate synthetic data with Linear Weight Inversion Drift.
        """
        # Validation
        if n_drift_features > n_features:
            raise ValueError("n_drift_features cannot exceed n_features")
        if n_drift_features == 0:
            raise ValueError("n_drift_features must be greater than 0 to observe drift")

        np.random.seed(random_seed)
        total_samples = n_samples_before + n_samples_after
        drift_point = n_samples_before
        feature_names = [f"X{i + 1}" for i in range(n_features)]

        # --- Feature Generation (NO Data Drift) ---
        # Generate all features from a stable distribution (Uniform [0, 1])
        X = np.random.uniform(0, 1, (total_samples, n_features))

        # --- Define Weights for Equal Magnitude Concept Drift ---

        # Base weight for non-drifting features (e.g., X3, X4, ...)
        # Set to 1.0 so they have equal importance to drifting features (which are 1.0 or -1.0)
        stable_weight = 1.0
        stable_weights = np.full(n_features - n_drift_features, stable_weight)

        # Weights for the drifting features (e.g., X1, X2)
        # 1. Before Drift: All drifting features have a strong positive influence.
        drift_weight_before = 1.0
        weights_drift_before = np.full(n_drift_features, drift_weight_before)
        W_before = np.concatenate([weights_drift_before, stable_weights])

        # 2. After Drift: All drifting features have an equally strong
        # negative influence.
        drift_weight_after = -1.0
        weights_drift_after = np.full(n_drift_features, drift_weight_after)
        W_after = np.concatenate([weights_drift_after, stable_weights])

        # --- Calculate Scores and Labels ---

        # We calculate "concept 1" (Pre) labels for ALL samples
        scores_pre = X @ W_before + np.random.normal(0, 0.1, total_samples)
        threshold_pre = np.median(scores_pre)
        y_pre = (scores_pre > threshold_pre).astype(int)

        # We calculate "concept 2" (Post) labels for ALL samples
        scores_post = X @ W_after + np.random.normal(0, 0.1, total_samples)
        threshold_post = np.median(scores_post)
        y_post = (scores_post > threshold_post).astype(int)

        # --- Mix Labels Based on Drift Width (Sigmoid) ---
        y_array = apply_sigmoid_drift(y_pre, y_post, drift_point, drift_width)

        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y_array, name="Y")

        return X_df, y_series
