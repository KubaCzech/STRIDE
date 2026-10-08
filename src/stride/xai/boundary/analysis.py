"""Decision boundary migration analysis between streaming data windows."""

from typing import Any
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from stride.exceptions import OptionalDependencyError
from stride.xai.boundary.disagreement import compute_disagreement_analysis


class DummyProjector:
    """Passthrough projector for 2D datasets."""

    def fit(self, X: np.ndarray, y: Any = None) -> "DummyProjector":
        """Fit passthrough projector (no-op).

        Args:
            X: Input feature array.
            y: Ignored target values.

        Returns:
            Fitted instance.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return input unmodified.

        Args:
            X: Input 2D feature array.

        Returns:
            Identical array.
        """
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Return 2D points unmodified.

        Args:
            X: 2D feature array.

        Returns:
            Identical array.
        """
        return X


class DecisionBoundaryDriftAnalyzer:
    r"""Analyze decision boundary shifts between data stream windows.

    Projects high-dimensional feature spaces $\mathbb{R}^D$ into an interpretable 2D
    manifold $\mathbb{R}^2$ using Self-Supervised Neighbor Projection (SSNP) or passthrough
    scaling. Evaluates classifiers trained before ($f_{\text{pre}}$) and after ($f_{\text{post}}$)
    drift, reconstructs dense decision grids, and trains Disagreement Decision Trees
    to extract geometric boundary shift explanations.

    Attributes:
        random_state: Pseudo-random generator seed.
        X_before: Pre-drift feature array of shape `(n_samples, n_features)`.
        y_before: Pre-drift target labels.
        X_after: Post-drift feature array.
        y_after: Post-drift target labels.
    """

    def __init__(
        self,
        X_before: np.ndarray | pd.DataFrame,
        y_before: np.ndarray | pd.Series,
        X_after: np.ndarray | pd.DataFrame,
        y_after: np.ndarray | pd.Series,
        random_state: int = 42,
    ) -> None:
        """Initialize the decision boundary analyzer with window splits.

        Args:
            X_before: Reference feature data before drift.
            y_before: Reference target labels before drift.
            X_after: Detection feature data after drift.
            y_after: Detection target labels after drift.
            random_state: Random state used to ensure reproducible projections.
        """
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        try:
            import tensorflow as tf

            tf.random.set_seed(self.random_state)
        except Exception:
            pass

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

    def analyze(
        self,
        model_class: Any | None = None,
        model_params: dict[str, Any] | None = None,
        grid_size: int = 300,
        ssnp_epochs: int = 10,
        ssnp_patience: int = 5,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute decision boundary shifts and disagreement explanation trees.

        Args:
            model_class: Classifier class used to map decision boundaries. Defaults to `MLPModel`.
            model_params: Initialization keyword arguments passed to `model_class`.
            grid_size: Number of sampling intervals along each axis of the 2D grid ($N \times N$).
            ssnp_epochs: Training epochs for SSNP autoencoder projection network.
            ssnp_patience: Early stopping patience epochs for SSNP training.
            feature_names: Optional sequence of human-readable feature column names.

        Returns:
            Dictionary containing:
                - 'pre': Pre-drift results dictionary with classifier, 2D coordinates, and grid surfaces.
                - 'post': Post-drift results dictionary.
                - 'ssnp_model': Fitted projection model instance.
                - 'grid_size': Evaluated grid dimension.
                - 'disagreement': Disagreement analysis report and decision tree rules.
                - 'is_2d': Whether input data was natively 2-dimensional.

        Raises:
            OptionalDependencyError: If input dimension $> 2$ and `tensorflow` is not installed.
        """
        scaler = MinMaxScaler()
        X_before_scaled = scaler.fit_transform(self.X_before)
        X_after_scaled = scaler.transform(self.X_after)

        is_2d = self.X_before.shape[1] == 2

        if is_2d:
            ssnp: Any = DummyProjector()
        else:
            try:
                from stride.xai.boundary.ssnp import SSNP

                ssnp = SSNP(epochs=ssnp_epochs, patience=ssnp_patience, verbose=0)
                ssnp.fit(X_before_scaled, self.y_before)
            except Exception as err:
                raise OptionalDependencyError(
                    package_name="tensorflow",
                    feature_name="High-dimensional decision boundary projection (SSNP)",
                    extra_name="deeplearning",
                ) from err

        X_before_2d = ssnp.transform(X_before_scaled)
        X_after_2d = ssnp.transform(X_after_scaled)

        if model_class is None:
            from stride.models.mlp import MLPModel

            model_class = MLPModel

        if model_params is None:
            model_params = {}

        model_params["random_state"] = self.random_state

        def process_window(
            X_train: np.ndarray, y_train: np.ndarray, X_2d_train: np.ndarray, grid_bounds: Any = None
        ) -> dict[str, Any]:
            clf = model_class(**model_params)
            clf.fit(X_train, y_train)

            if grid_bounds is None:
                xmin, xmax = float(np.min(X_2d_train[:, 0])), float(np.max(X_2d_train[:, 0]))
                ymin, ymax = float(np.min(X_2d_train[:, 1])), float(np.max(X_2d_train[:, 1]))
                x_margin = (xmax - xmin) * 0.1
                y_margin = (ymax - ymin) * 0.1
                bounds = (xmin - x_margin, xmax + x_margin, ymin - y_margin, ymax + y_margin)
            else:
                bounds = grid_bounds

            xmin, xmax, ymin, ymax = bounds

            x_intrvls = np.linspace(xmin, xmax, num=grid_size)
            y_intrvls = np.linspace(ymin, ymax, num=grid_size)

            xx, yy = np.meshgrid(x_intrvls, y_intrvls)
            pts = np.c_[xx.ravel(), yy.ravel()]

            batch_size = 50000
            n_pts = len(pts)

            probs_list = []
            labels_list = []

            for i in range(0, n_pts, batch_size):
                batch_pts = pts[i : i + batch_size]
                batch_high_dim = ssnp.inverse_transform(batch_pts)

                batch_probs = clf.predict_proba(batch_high_dim)
                batch_labels = clf.predict(batch_high_dim)

                if hasattr(batch_probs, "max"):
                    batch_alpha = batch_probs.max(axis=1)
                else:
                    batch_alpha = np.ones(len(batch_labels))

                probs_list.append(batch_alpha)
                labels_list.append(batch_labels)

            probs_flat = np.concatenate(probs_list)
            labels_flat = np.concatenate(labels_list)

            prob_grid = probs_flat.reshape(grid_size, grid_size)
            label_grid = labels_flat.reshape(grid_size, grid_size)

            return {
                "clf": clf,
                "X_train": X_train,
                "y_train": y_train,
                "X_2d": X_2d_train,
                "grid_probs": prob_grid,
                "grid_labels": label_grid,
                "grid_bounds": bounds,
            }

        result_pre = process_window(X_before_scaled, self.y_before, X_before_2d)
        result_post = process_window(X_after_scaled, self.y_after, X_after_2d)

        b = result_post["grid_bounds"]
        x_intrvls = np.linspace(b[0], b[1], num=grid_size)
        y_intrvls = np.linspace(b[2], b[3], num=grid_size)
        xx, yy = np.meshgrid(x_intrvls, y_intrvls)
        pts_2d = np.c_[xx.ravel(), yy.ravel()]

        X_grid_high_scaled = ssnp.inverse_transform(pts_2d)

        disagreement_results = compute_disagreement_analysis(
            clf_pre=result_pre["clf"],
            clf_post=result_post["clf"],
            X_eval_raw=self.X_after,
            X_eval_scaled=X_after_scaled,
            X_grid_high_scaled=X_grid_high_scaled,
            feature_names=feature_names,
            scaler=scaler,
        )

        return {
            "pre": result_pre,
            "post": result_post,
            "ssnp_model": ssnp,
            "grid_size": grid_size,
            "disagreement": disagreement_results,
            "is_2d": is_2d,
        }
