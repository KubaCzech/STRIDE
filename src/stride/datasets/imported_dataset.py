import pandas as pd
from .base import BaseDataset
from .dataset_registry import DatasetRegistry


class ImportedCSVDataset(BaseDataset):
    def __init__(self, name: str, registry_info: dict, registry: DatasetRegistry):
        self._name = name
        self.registry_info = registry_info
        self.registry = registry

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return f"{self._name} (Imported)"

    def get_params(self) -> dict:
        # Params are largely fixed for loaded datasets, but we can return helpful info
        return {
            "target_column": self.registry_info.get("target_column"),
            "features": self.registry_info.get("selected_features"),
        }

    def get_settings_schema(self) -> list[dict]:
        # Imported datasets have fixed structure, maybe allow renaming target?
        # For now, keep it simple.
        return []

    def get_available_settings(self) -> dict:
        return {}

    def generate(self, **kwargs):
        """
        Load data from the stored CSV file.
        """
        file_path = self.registry.get_dataset_path(self._name)
        if not file_path:
            raise FileNotFoundError(f"File for dataset {self._name} not found.")

        df = pd.read_csv(file_path)

        target_col = self.registry_info.get("target_column", "target")
        selected_features = self.registry_info.get("selected_features")

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        y = df[target_col]

        if selected_features:
            # Filter columns, keeping only those selected (and checking existence)
            valid_features = [f for f in selected_features if f in df.columns and f != target_col]
            X = df[valid_features]
        else:
            X = df.drop(columns=[target_col])

        return X, y
