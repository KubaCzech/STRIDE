from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
import json
import os


class BaseModel(BaseEstimator, ClassifierMixin, ABC):
    """Abstract base class for all models."""

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
    def get_model(self):
        """Return the underlying sklearn-compatible model."""
        pass

    def fit(self, X, y):
        """Fit the model."""
        self.model_ = self.get_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        """Predict class labels."""
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model_.predict_proba(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return self.model_.score(X, y)

    def get_settings_schema(self) -> list[dict]:
        """
        Return a schema describing the settings for this model.
        """
        return []

    def get_available_settings(self) -> dict[str, dict]:
        """
        Return available named settings for this model.
        """
        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")

        try:
            with open(settings_path, "r") as f:
                all_settings = json.load(f)
            return all_settings.get(self.name, {})
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
