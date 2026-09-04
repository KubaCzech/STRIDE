from abc import ABC, abstractmethod
import pandas as pd


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

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
        Generate the dataset.

        Returns
        -------
        tuple
            (X, y)
            X: pd.DataFrame with named features
            y: pd.Series with target values
        """
        pass

    def get_params(self) -> dict:
        """
        Return default parameters for the dataset.
        Can be overridden by subclasses to provide specific parameters for the UI.
        """
        return {"n_samples_before": 1000, "n_samples_after": 1000, "random_seed": 42}

    def get_settings_schema(self) -> list[dict]:
        """
        Return a schema describing the settings for this dataset.

        Returns
        -------
        list[dict]
            A list of dictionaries, where each dictionary describes a setting.
            Format:
            {
                "name": "param_name",
                "type": "int" | "float" | "text" | "file" | "bool",
                "label": "Display Label",
                "default": value,
                "min_value": min, # optional
                "max_value": max, # optional
                "step": step,     # optional
                "allowed_types": [], # for file
                "help": "Tooltip text"
            }
        """
        return []

    def get_available_settings(self) -> dict[str, dict]:
        """
        Return available named settings for this dataset.

        Returns
        -------
        dict[str, dict]
            A dictionary mapping setting names to their parameter values.
        """
        import json
        import os

        settings_path = os.path.join(os.path.dirname(__file__), "settings.json")

        try:
            with open(settings_path, "r") as f:
                all_settings = json.load(f)
            return all_settings.get(self.name, {})
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
