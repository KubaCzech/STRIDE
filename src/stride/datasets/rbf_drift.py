from .base import BaseDataset
from .utils import apply_sigmoid_drift
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


class RBFDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "rbf_drift"

    @property
    def display_name(self) -> str:
        return "RBF Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            {
                "n_windows_before": 1,
                "n_windows_after": 1,
                "n_features": 4,
                "gamma": 30.0,
                "noise": 0.0,
                "cluster_std": 0.05,
                "random_seed": 42,
                "drift_width": 1,
            }
        )
        return params

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "n_windows_before",
                "type": "int",
                "label": "Windows Before Drift",
                "default": 1,
                "min_value": 1,
                "step": 1,
                "help": "Number of windows before the specific drift point.",
            },
            {
                "name": "n_windows_after",
                "type": "int",
                "label": "Windows After Drift",
                "default": 1,
                "min_value": 1,
                "step": 1,
                "help": "Number of windows after the specific drift point.",
            },
            {
                "name": "gamma",
                "type": "float",
                "label": "Gamma",
                "default": 30.0,
                "min_value": 0.1,
                "step": 0.1,
                "help": "Gamma parameter for RBF kernel.",
            },
            {
                "name": "noise",
                "type": "float",
                "label": "Label Noise",
                "default": 0.0,
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.01,
                "help": "Ratio of labels to flip.",
            },
            {
                "name": "cluster_std",
                "type": "float",
                "label": "Cluster Std Dev",
                "default": 0.05,
                "min_value": 0.01,
                "step": 0.01,
                "help": "Standard deviation of the Gaussian clusters.",
            },
            {
                "name": "random_seed",
                "type": "int",
                "label": "Random Seed",
                "default": 42,
                "min_value": 0,
                "step": 1,
                "help": "Seed for reproducible random generation.",
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

    def _generate_cluster_data(self, centers, total_samples, std):
        """Generate Gaussian clusters with guaranteed samples per cluster."""
        n_centers = len(centers)
        if n_centers == 0:
            return np.zeros((total_samples, 0))  # Should not happen with default logic

        base = total_samples // n_centers
        remainder = total_samples % n_centers

        X_list = []
        for i, c in enumerate(centers):
            n_cluster = base + (1 if i < remainder else 0)
            X_list.append(np.random.normal(loc=c, scale=std, size=(n_cluster, len(c))))

        if not X_list:
            return np.zeros((total_samples, len(centers[0]) if len(centers) > 0 else 0))

        return np.vstack(X_list)

    def _generate_labels(self, X, centers_0, centers_1, gamma):
        """Assign class based on stronger RBF response."""
        K0 = np.max(rbf_kernel(X, centers_0, gamma=gamma), axis=1)
        K1 = np.max(rbf_kernel(X, centers_1, gamma=gamma), axis=1)
        return (K1 > K0).astype(int)

    def _add_label_noise(self, y, ratio):
        n_flip = int(ratio * len(y))
        if n_flip > 0:
            idx = np.random.choice(len(y), n_flip, replace=False)
            y[idx] = 1 - y[idx]
        return y

    def generate(
        self,
        n_samples_before=2000,
        n_samples_after=2000,
        gamma=30.0,
        noise=0.0,
        cluster_std=0.05,
        random_seed=42,
        drift_width=1,
        **kwargs,
    ) -> tuple[pd.DataFrame, pd.Series]:

        np.random.seed(random_seed)

        # Define 4 cluster centers in 4D → 2 per class
        # (Taken from reference implementation)
        centers_class0 = np.array([[0.2, 0.8, 0.2, 0.8], [0.8, 0.2, 0.8, 0.2]])

        centers_class1 = np.array([[0.2, 0.2, 0.8, 0.8], [0.8, 0.8, 0.2, 0.2]])

        total_samples = n_samples_before + n_samples_after
        samples_total = total_samples  # Alias for clarity

        # 1. Generate full length data for Concept 1 (Pre)
        X1 = self._generate_cluster_data(np.vstack([centers_class0, centers_class1]), samples_total, cluster_std)
        y1 = self._generate_labels(X1, centers_class0, centers_class1, gamma)

        # Shuffle Concept 1
        p1 = np.random.permutation(samples_total)
        X1 = X1[p1]
        y1 = y1[p1]

        # 2. Generate full length data for Concept 2 (Post)
        X2 = self._generate_cluster_data(np.vstack([centers_class0, centers_class1]), samples_total, cluster_std)
        # Swapped centers for labels
        y2 = self._generate_labels(X2, centers_class1, centers_class0, gamma)

        # Shuffle Concept 2
        p2 = np.random.permutation(samples_total)
        X2 = X2[p2]
        y2 = y2[p2]

        # Add noise independently
        y1 = self._add_label_noise(y1, noise)
        y2 = self._add_label_noise(y2, noise)

        # 3. Probabilistic Mixing using utility function
        X = apply_sigmoid_drift(X1, X2, n_samples_before, drift_width)
        y = apply_sigmoid_drift(y1, y2, n_samples_before, drift_width)

        # Convert to Pandas
        feature_names = [f"x{i + 1}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")

        return X_df, y_series
