import pandas as pd
from .base import BaseDataset


class CSVDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "csv_dataset"

    @property
    def display_name(self) -> str:
        return "Load from CSV"

    def get_params(self) -> dict:
        return {"file_path": None, "target_column": "target", "drift_point": None}

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "file_path",
                "type": "file",
                "label": "Upload CSV File",
                "allowed_types": ["csv"],
                "help": "Upload a CSV file containing the dataset.",
            },
            {
                "name": "target_column",
                "type": "text",
                "label": "Target Column Name",
                "default": "target",
                "help": "Name of the column containing the target variable.",
            },
        ]

    def generate(self, file_path=None, target_column="target", drift_point=None, **kwargs):
        """
        Load data from a CSV file.
        """
        if file_path is None:
            raise ValueError("file_path must be provided for CSVDataset")

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in CSV")

        y = df[target_column]
        X = df.drop(columns=[target_column])

        return X, y
