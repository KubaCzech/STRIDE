from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel


class RandomForestModel(BaseModel):
    """Wrapper for RandomForestClassifier."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model_ = None

    @property
    def name(self) -> str:
        return "random_forest"

    @property
    def display_name(self) -> str:
        return "Random Forest"

    def get_model(self):
        max_depth = self.max_depth
        if max_depth == 0:
            max_depth = None

        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
        )

    def get_settings_schema(self) -> list[dict]:
        return [
            {
                "name": "n_estimators",
                "type": "int",
                "label": "Number of Estimators",
                "default": 100,
                "min_value": 10,
                "step": 10,
                "help": "The number of trees in the forest.",
            },
            {
                "name": "max_depth",
                "type": "int",
                "label": "Max Depth",
                "default": 0,  # 0 will mean None in our logic if we handle it
                "min_value": 0,
                "step": 1,
                "help": "The maximum depth of the tree. 0 means unlimited.",
            },
            {
                "name": "min_samples_split",
                "type": "int",
                "label": "Min Samples Split",
                "default": 2,
                "min_value": 2,
                "step": 1,
                "help": "The minimum number of samples required to split an internal node.",
            },
        ]
