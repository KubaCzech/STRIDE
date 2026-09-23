"""Classifier model wrappers compatible with Scikit-learn estimator interface."""

from .base import BaseModel
from .mlp import MLPModel
from .random_forest import RandomForestModel

MODELS = {"mlp": MLPModel, "random_forest": RandomForestModel}

__all__ = ["BaseModel", "MLPModel", "RandomForestModel", "MODELS"]
