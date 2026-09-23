import numpy as np
from sklearn.inspection import permutation_importance
import shap
from lime.lime_tabular import LimeTabularExplainer
from .base import FeatureImportanceMethod


def calculate_feature_importance(model, X, y, method="permutation", feature_names=None, n_repeats=30, random_state=42):
    """
    Calculate feature importance using specified method.

    Parameters
    ----------
    model : estimator
        Trained model
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    method : str, default="permutation"
        Method to use: "permutation", "shap", or "lime"
    feature_names : list, optional
        Names of features
    n_repeats : int, default=30
        Number of repeats for permutation importance
    random_state : int, default=42
        Random state for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - 'importances_mean': mean importance scores
        - 'importances_std': standard deviation of importance scores
        - 'importances': full array of importance values (for boxplots)
        - 'method': method used
    """
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    if method == FeatureImportanceMethod.PFI:
        return _calculate_pfi(model, X, y, n_repeats, random_state)

    elif method == FeatureImportanceMethod.SHAP:
        return _calculate_shap(model, X, feature_names)

    elif method == FeatureImportanceMethod.LIME:
        return _calculate_lime(model, X, y, feature_names, random_state)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'permutation', 'shap', or 'lime'")


def _calculate_pfi(model, X, y, n_repeats=30, random_state=42):
    """Calculate Permutation Feature Importance."""
    pfi_result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)

    return {
        "importances_mean": pfi_result.importances_mean,
        "importances_std": pfi_result.importances_std,
        "importances": pfi_result.importances,
        "method": "PFI",
    }


def _calculate_shap(model, X, feature_names):
    """Calculate SHAP values."""
    # Use a subset for efficiency if dataset is large
    background_size = min(100, len(X))
    background = shap.sample(X, background_size)

    # Create explainer
    explainer = shap.KernelExplainer(model.predict_proba, background)

    # Calculate SHAP values (use subset if too large)
    sample_size = min(500, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_indices]

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, SHAP returns a list
    # [class_0_values, class_1_values]
    # We use class 1 (positive class) for binary classification
    # For multiclass, this logic would need to adapt
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # Handle possible 3D array for some model types
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Ensure shap_values is 2D: (n_samples, n_features)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(-1, 1)

    # Calculate mean absolute SHAP values
    abs_shap_values = np.abs(shap_values)
    importances_mean = np.mean(abs_shap_values, axis=0)
    importances_std = np.std(abs_shap_values, axis=0)

    # Ensure arrays are 1D with correct length (n_features)
    importances_mean = np.atleast_1d(importances_mean).flatten()
    importances_std = np.atleast_1d(importances_std).flatten()

    # Ensure importances array is (n_features, n_samples)
    # for consistency with PFI
    if abs_shap_values.shape[1] == len(feature_names):
        importances_array = abs_shap_values.T
    else:
        importances_array = abs_shap_values

    return {
        "importances_mean": importances_mean,
        "importances_std": importances_std,
        "importances": importances_array,
        "method": "SHAP",
    }


def _calculate_lime(model, X, y, feature_names, random_state=42):
    """Calculate LIME feature importance."""
    np.random.seed(random_state)

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X, feature_names=feature_names, class_names=["Class 0", "Class 1"], mode="classification", random_state=random_state
    )

    # Calculate LIME explanations for a sample of instances
    sample_size = min(100, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)

    all_importances = []

    for idx in sample_indices:
        exp = explainer.explain_instance(X[idx], model.predict_proba, num_features=len(feature_names))

        # Extract feature importances (absolute values)
        importance_dict = dict(exp.as_list())
        importances = []
        for feat_name in feature_names:
            # Find matching feature (LIME may modify feature names)
            matching_imp = 0
            for key, val in importance_dict.items():
                if feat_name in key or key.startswith(feat_name):
                    matching_imp = abs(val)
                    break
            importances.append(matching_imp)

        all_importances.append(importances)

    all_importances = np.array(all_importances)

    return {
        "importances_mean": np.mean(all_importances, axis=0),
        "importances_std": np.std(all_importances, axis=0),
        "importances": all_importances.T,
        "method": "LIME",
    }
