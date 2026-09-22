from sklearn.neural_network import MLPClassifier
import ast
from .base import BaseModel


class MLPModel(BaseModel):
    """Wrapper for MLPClassifier."""

    def __init__(self, hidden_layer_sizes=(10, 10), max_iter=500, alpha=1e-5, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state
        self.model_ = None

    @property
    def name(self) -> str:
        return "mlp"

    @property
    def display_name(self) -> str:
        return "MLP Classifier"

    def get_model(self):
        # Parse hidden_layer_sizes if it's a string
        hidden_layers = self.hidden_layer_sizes
        if isinstance(hidden_layers, str):
            try:
                hidden_layers = ast.literal_eval(hidden_layers)
            except (ValueError, SyntaxError):
                # Fallback or error handling if parsing fails
                # For now, let's assume valid input or default to (10, 10)
                hidden_layers = (10, 10)

        return MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=self.random_state,
            solver="adam",
        )
