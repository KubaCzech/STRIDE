"""Base estimator interfaces for STRIDE classifiers."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BaseModel(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for all STRIDE classifier models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the model."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the model."""
        pass

    @abstractmethod
    def get_model(self) -> Any:
        """Return the underlying sklearn-compatible model instance."""
        pass

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseModel":
        """Fit the classifier on training data."""
        self.model_ = self.get_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X."""
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        return self.model_.predict_proba(X)

    def score(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
        """Return the mean accuracy on test data and labels."""
        return self.model_.score(X, y)

    def get_settings_schema(self) -> list[dict]:
        """
        Deprecated: Return a schema describing the settings for this model.

        UI schemas should be managed by the application layer (e.g. dashboard.config.model_schemas).
        """
        import warnings

        warnings.warn(
            "get_settings_schema() on model classes is deprecated and will be removed in a future release. "
            "UI schemas should be managed by dashboard.config.model_schemas.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from dashboard.config.model_schemas import get_model_schema

            return get_model_schema(self.name)
        except ImportError:
            return []

    def get_available_settings(self) -> dict[str, dict]:
        """
        Deprecated: Return available named preset settings for this model.

        UI presets should be managed by the application layer (e.g. dashboard.config.model_schemas).
        """
        import warnings

        warnings.warn(
            "get_available_settings() on model classes is deprecated and will be removed in a future release. "
            "UI presets should be managed by dashboard.config.model_schemas.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from dashboard.config.model_schemas import get_model_available_settings

            return get_model_available_settings(self.name)
        except ImportError:
            return {}
