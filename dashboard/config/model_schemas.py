"""UI Schemas and default presets for models in the Streamlit dashboard."""

MODEL_SETTINGS_SCHEMAS = {
    "mlp": [
        {
            "name": "hidden_layer_sizes",
            "type": "list_of_int",
            "label": "Hidden Layer Sizes",
            "default": [10, 10],
            "help": "Specify the number of neurons for each hidden layer.",
        },
        {
            "name": "max_iter",
            "type": "int",
            "label": "Max Iterations",
            "default": 500,
            "min_value": 10,
            "step": 10,
            "help": "Maximum number of iterations.",
        },
        {
            "name": "alpha",
            "type": "float",
            "label": "Alpha (L2 penalty)",
            "default": 0.00001,
            "min_value": 0.0,
            "step": 0.00001,
            "format": "%.5f",
            "help": "L2 penalty (regularization term) parameter.",
        },
    ],
    "random_forest": [
        {
            "name": "n_estimators",
            "type": "int",
            "label": "Number of Estimators",
            "default": 100,
            "min_value": 1,
            "step": 10,
            "help": "The number of trees in the forest.",
        },
        {
            "name": "max_depth",
            "type": "int",
            "label": "Max Depth (0 for None)",
            "default": 0,
            "min_value": 0,
            "step": 1,
            "help": "The maximum depth of the tree. 0 means nodes are expanded until all leaves are pure.",
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
    ],
}

MODEL_PRESETS = {
    "mlp": {
        "Default": {
            "default": True,
            "hidden_layer_sizes": [10, 10],
            "max_iter": 500,
            "alpha": 0.00001,
        },
        "Deep Network": {
            "hidden_layer_sizes": [50, 50, 50],
            "max_iter": 1000,
            "alpha": 0.0001,
        },
    },
    "random_forest": {
        "Default": {
            "default": True,
            "n_estimators": 100,
            "max_depth": 0,
            "min_samples_split": 2,
        },
        "Simple Forest": {
            "n_estimators": 10,
            "max_depth": 5,
            "min_samples_split": 5,
        },
    },
}


def get_model_settings_schema(model_name: str) -> list[dict]:
    """Return UI settings schema for a specific model."""
    return MODEL_SETTINGS_SCHEMAS.get(model_name, [])


get_model_schema = get_model_settings_schema


def get_model_available_settings(model_name: str) -> dict[str, dict]:
    """Return available preset configurations for a specific model."""
    return MODEL_PRESETS.get(model_name, {})
