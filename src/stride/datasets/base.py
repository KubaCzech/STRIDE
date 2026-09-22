"""Abstract base class and contract for all STRIDE synthetic and imported datasets."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseDataset(ABC):
    """Abstract base class for all data streams and drift generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the dataset."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for the dataset."""
        pass

    @abstractmethod
    def generate(self, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generate the data stream before and after drift.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (X, y)
            X: pd.DataFrame with named features of shape (n_samples, n_features)
            y: pd.Series with target values of shape (n_samples,)
        """
        pass

    def get_params(self) -> dict:
        """Return default generation parameters for the dataset."""
        return {"n_samples_before": 1000, "n_samples_after": 1000, "random_seed": 42}

    def get_settings_schema(self) -> list[dict]:
        """
        Return a schema describing the settings for this dataset.
        Subclasses can provide parameter schemas for consumer dashboards.
        """
        return []

    def get_available_settings(self) -> dict[str, dict]:
        """
        Return available named preset settings for this dataset.
        Delegates to dashboard configuration if available.
        """
        try:
            from dashboard.config.dataset_schemas import get_dataset_available_settings

            return get_dataset_available_settings(self.name)
        except ImportError:
            return {}
