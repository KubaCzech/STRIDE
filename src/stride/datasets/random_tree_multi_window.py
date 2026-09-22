from .protree_data.stream_generators import RandomTree
from .base import BaseDataset
import pandas as pd


class RandomTreeMultiWindowDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "random_tree_multi_window"

    @property
    def display_name(self) -> str:
        return "Random Tree Multi-Window"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            {
                "num_windows": 100,
                "drift_positions": [28000, 52000, 70000],
                "drift_duration": 1,
                "n_informative": 4,
                "n_redundant": 2,
            }
        )
        return params

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "num_windows",
                "type": "int",
                "label": "Number of Windows",
                "default": 100,
                "min_value": 2,
                "step": 1,
                "help": "Total number of windows to generate.",
            },
            {
                "name": "drift_positions",
                "type": "text",
                "label": "Drift Positions (comma-separated sample numbers)",
                "default": "28000, 52000, 70000",
                "help": "Enter sample positions where drifts occur, e.g., '28000, 52000, 70000'. Leave empty for no drifts.",
            },
            {
                "name": "drift_duration",
                "type": "int",
                "label": "Drift Duration (samples)",
                "default": 1,
                "min_value": 1,
                "step": 100,
                "help": "Duration of each drift transition in samples.",
            },
            {
                "name": "n_informative",
                "type": "int",
                "label": "Number of Informative Features",
                "default": 4,
                "min_value": 1,
                "step": 1,
                "help": "Number of informative features in the dataset.",
            },
            {
                "name": "n_redundant",
                "type": "int",
                "label": "Number of Redundant Features",
                "default": 2,
                "min_value": 0,
                "step": 1,
                "help": "Number of redundant features in the dataset.",
            },
        ]

    def generate(
        self,
        num_windows=100,
        window_length=1000,
        drift_positions=None,
        drift_duration=1,
        n_informative=4,
        n_redundant=2,
        random_seed=42,
        **kwargs,
    ):
        """
        Generate synthetic data stream using protree's RandomTree generator.

        Parameters
        ----------
        num_windows : int
            Number of windows to generate
        window_length : int
            Number of samples per window
        drift_positions : list or str
            List of sample positions where drifts occur, or comma-separated string
        drift_duration : int
            Duration of drift transition in samples
        n_informative : int
            Number of informative features
        n_redundant : int
            Number of redundant features
        random_seed : int
            Random seed for reproducibility
        """

        # Parse drift positions if string
        if isinstance(drift_positions, str):
            if drift_positions.strip():
                drift_positions = [int(x.strip()) for x in drift_positions.split(",")]
            else:
                drift_positions = []
        elif drift_positions is None:
            drift_positions = []

        # Create RandomTree generator
        ds = RandomTree(
            drift_position=drift_positions if drift_positions else 500,
            drift_duration=drift_duration,
            seed=random_seed,
            n_informative=n_informative,
            n_redundant=n_redundant,
        )

        # Generate all windows
        all_x = []
        all_y = []

        for i in range(num_windows):
            x_block, y_block = zip(*ds.take(window_length))
            all_x.extend(x_block)
            all_y.extend(y_block)

        # Convert to DataFrame and Series
        # The x_block items are dictionaries with feature names as keys
        X = pd.DataFrame(all_x)
        y = pd.Series(all_y, name="target")

        return X, y
