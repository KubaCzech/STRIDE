import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.decision_boundary.ssnp import SSNP
from src.decision_boundary.disagreement import compute_disagreement_analysis


class DummyProjector:
    """Passthrough projector for data that is already 2D."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X

class DecisionBoundaryDriftAnalyzer:
    def __init__(self, X_before, y_before, X_after, y_after, random_state=42):
        self.random_state = random_state
        # 0. Enforce Determinism
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        # 1. Prepare Data
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

    def analyze(self, model_class=None, model_params=None, grid_size=300, ssnp_epochs=10, ssnp_patience=5, feature_names=None):
        """
        Compute decision boundary analysis using SSNP for dimensionality reduction
        and a classifier for the decision boundary. Handles both pre and post drift windows.
        """

        # Normalize Data (Fit on Pre, Transform both)
        scaler = MinMaxScaler()
        X_before_scaled = scaler.fit_transform(self.X_before)
        X_after_scaled = scaler.transform(self.X_after)

        is_2d = self.X_before.shape[1] == 2
        
        if is_2d:
            ssnp = DummyProjector()
        else:
            # SSNP is used to find a 2D projection that preserves class structure.
            ssnp = SSNP(epochs=ssnp_epochs, patience=ssnp_patience, verbose=0)
            ssnp.fit(X_before_scaled, self.y_before)

        # Project points to 2D (if 2D already, this just returns the scaled data)
        X_before_2d = ssnp.transform(X_before_scaled)
        X_after_2d = ssnp.transform(X_after_scaled)

        # 3. Setup Classifier
        if model_class is None:
            from src.models.mlp import MLPModel
            model_class = MLPModel

        if model_params is None:
            model_params = {}

        # Force random_state for classifier determinism
        model_params['random_state'] = self.random_state

        # Helper to train and predict grid
        def process_window(X_train, y_train, X_2d_train, grid_bounds=None):
            # Train Classifier on High-Dim Data
            clf = model_class(**model_params)
            clf.fit(X_train, y_train)

            # Define Grid Bounds (if not provided, calculate from train data)
            if grid_bounds is None:
                xmin, xmax = np.min(X_2d_train[:, 0]), np.max(X_2d_train[:, 0])
                ymin, ymax = np.min(X_2d_train[:, 1]), np.max(X_2d_train[:, 1])
                # Add some margin
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

            # Inverse Transform 2D Grid -> High Dim
            # Process in batches to avoid OOM
            batch_size = 50000
            n_pts = len(pts)

            probs_list = []
            labels_list = []
            high_dim_list = []

            for i in range(0, n_pts, batch_size):
                batch_pts = pts[i:i+batch_size]
                batch_high_dim = ssnp.inverse_transform(batch_pts)
                
                # Keep high dim points if needed (e.g. for disagreement)
                
                # Predict
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

            # Reshape to grid
            prob_grid = probs_flat.reshape(grid_size, grid_size)
            label_grid = labels_flat.reshape(grid_size, grid_size)

            return {
                'clf': clf,
                'X_train': X_train,
                'y_train': y_train,
                'X_2d': X_2d_train,
                'grid_probs': prob_grid,
                'grid_labels': label_grid,
                'grid_bounds': bounds,
            }

        # 4. Process Pre and Post
        # We determine a unified grid bound based on BOTH pre and post 2D projections
        # to ensure the visualizations are comparable or cover the drift.
        
        result_pre = process_window(X_before_scaled, self.y_before, X_before_2d)
        
        # Use Post bounds for Post window
        result_post = process_window(X_after_scaled, self.y_after, X_after_2d)

        # 5. Compute Disagreement Analysis 
        # We generate a grid specifically on the Post window bounds
        # and compute disagreement on THIS grid to train the explainer tree.
        
        # Re-generate grid points for the disagreement analysis (High-D Manifold)
        b = result_post['grid_bounds']
        x_intrvls = np.linspace(b[0], b[1], num=grid_size)
        y_intrvls = np.linspace(b[2], b[3], num=grid_size)
        xx, yy = np.meshgrid(x_intrvls, y_intrvls)
        pts_2d = np.c_[xx.ravel(), yy.ravel()]
        
        # Inverse transform entire grid to High-D (Scaled)
        X_grid_high_scaled = ssnp.inverse_transform(pts_2d)
        
        disagreement_results = compute_disagreement_analysis(
            clf_pre=result_pre['clf'],
            clf_post=result_post['clf'],
            X_eval_raw=self.X_after,       # Used for unscaling map
            X_eval_scaled=X_after_scaled,  # Used for drift rate calc on real data
            X_grid_high_scaled=X_grid_high_scaled, # Used for training the Viz Tree 
            feature_names=feature_names,
            scaler=scaler
        )

        return {
            'pre': result_pre,
            'post': result_post,
            'ssnp_model': ssnp,
            'grid_size': grid_size,
            'disagreement': disagreement_results,
            'is_2d': is_2d
        }