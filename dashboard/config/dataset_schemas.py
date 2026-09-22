"""UI Schemas and default presets for datasets in the Streamlit dashboard."""

DATASET_PRESETS = {
    "hyperplane_drift": {
        "Default": {
            "default": True,
            "n_features": 2,
            "n_drift_features": 2,
            "noise_percentage": 0.05,
            "drift_noise_percentage": 0.1,
            "mag_change": 0.2,
            "drift_width": 1,
        }
    },
    "random_rbf_drift": {
        "Default": {
            "default": True,
            "n_features": 10,
            "n_classes": 2,
            "n_centroids": 50,
            "change_speed": 0.0,
            "n_drift_centroids": 50,
            "drift_width": 1,
        },
        "Fast Drift": {
            "n_features": 10,
            "n_classes": 2,
            "n_centroids": 50,
            "change_speed": 0.87,
            "n_drift_centroids": 50,
            "drift_width": 1,
        },
    },
    "rbf_drift": {
        "Default": {
            "default": True,
            "n_features": 4,
            "gamma": 30.0,
            "noise": 0.0,
            "cluster_std": 0.05,
            "drift_width": 1,
        }
    },
    "sea_drift": {
        "Default": {
            "default": True,
        }
    },
    "electricity_drift": {
        "Default": {
            "default": True,
            "size_of_block": 2000,
            "starting_point": -1,
        }
    },
    "airlines_drift": {
        "Default": {
            "default": True,
            "size_of_block": 2000,
            "starting_point": -1,
        }
    },
    "forest_drift": {
        "Default": {
            "default": True,
            "size_of_block": 2000,
            "starting_point": -1,
        }
    },
    "linear_weight_inversion_drift": {
        "Default": {
            "default": True,
            "n_features": 11,
            "n_drift_features": 5,
        }
    },
}

DATASET_SETTINGS_SCHEMAS: dict[str, list[dict]] = {
    "hyperplane_drift": [
        {
            "name": "n_windows_before",
            "type": "int",
            "label": "Number of Windows Before Drift",
            "default": 1,
            "min_value": 0,
            "step": 1,
            "help": "Number of windows generated before the concept drift occurs.",
        },
        {
            "name": "n_windows_after",
            "type": "int",
            "label": "Number of Windows After Drift",
            "default": 1,
            "min_value": 0,
            "step": 1,
            "help": "Number of windows generated after the concept drift occurs.",
        },
        {
            "name": "n_features",
            "type": "int",
            "label": "Number of Features (n_features)",
            "default": 2,
            "min_value": 2,
            "step": 1,
            "help": "Total number of features for the hyperplane. Must be >= 2.",
        },
        {
            "name": "n_drift_features",
            "type": "int",
            "label": "Number of Drifting Features (n_drift_features)",
            "default": 2,
            "min_value": 2,
            "step": 1,
            "help": "Number of features that will drift. Must be <= n_features.",
        },
        {
            "name": "noise_percentage",
            "type": "float",
            "label": "Noise Percentage (noise_percentage)",
            "default": 0.05,
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.01,
            "help": "Probability of label noise for the initial stream.",
        },
        {
            "name": "drift_noise_percentage",
            "type": "float",
            "label": "Drift Noise Percentage (drift_noise_percentage)",
            "default": 0.1,
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.01,
            "help": "Probability of label noise for the drift stream.",
        },
        {
            "name": "mag_change",
            "type": "float",
            "label": "Magnitude of Change (mag_change)",
            "default": 0.2,
            "min_value": 0.0,
            "step": 0.01,
            "help": "Magnitude of change for drifting features.",
        },
        {
            "name": "drift_width",
            "type": "int",
            "label": "Drift Width (drift_width)",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Width of the concept drift (number of samples).",
        },
    ],
    "random_rbf_drift": [
        {
            "name": "n_windows_before",
            "type": "int",
            "label": "Number of Windows Before Drift",
            "default": 1,
            "min_value": 0,
            "step": 1,
            "help": "Number of windows generated before the concept drift occurs.",
        },
        {
            "name": "n_windows_after",
            "type": "int",
            "label": "Number of Windows After Drift",
            "default": 1,
            "min_value": 0,
            "step": 1,
            "help": "Number of windows generated after the concept drift occurs.",
        },
        {
            "name": "n_features",
            "type": "int",
            "label": "Number of Features",
            "default": 10,
            "min_value": 2,
            "step": 1,
            "help": "Number of numerical features.",
        },
        {
            "name": "n_centroids",
            "type": "int",
            "label": "Number of Centroids",
            "default": 50,
            "min_value": 1,
            "step": 1,
            "help": "Total number of centroids.",
        },
        {
            "name": "change_speed",
            "type": "float",
            "label": "Change Speed",
            "default": 0.0,
            "min_value": 0.0,
            "step": 0.01,
            "help": "Speed of drift (0.0 = no drift).",
        },
        {
            "name": "n_drift_centroids",
            "type": "int",
            "label": "Drifting Centroids",
            "default": 50,
            "min_value": 0,
            "step": 1,
            "help": "Number of centroids that drift.",
        },
        {
            "name": "drift_width",
            "type": "int",
            "label": "Drift Width (drift_width)",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Width of the concept drift (number of samples).",
        },
    ],
    "rbf_drift": [
        {
            "name": "n_windows_before",
            "type": "int",
            "label": "Windows Before Drift",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Number of windows before the specific drift point.",
        },
        {
            "name": "n_windows_after",
            "type": "int",
            "label": "Windows After Drift",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Number of windows after the specific drift point.",
        },
        {
            "name": "gamma",
            "type": "float",
            "label": "Gamma",
            "default": 30.0,
            "min_value": 0.1,
            "step": 0.1,
            "help": "Gamma parameter for RBF kernel.",
        },
        {
            "name": "noise",
            "type": "float",
            "label": "Label Noise",
            "default": 0.0,
            "min_value": 0.0,
            "max_value": 1.0,
            "step": 0.01,
            "help": "Ratio of labels to flip.",
        },
        {
            "name": "cluster_std",
            "type": "float",
            "label": "Cluster Std Dev",
            "default": 0.05,
            "min_value": 0.01,
            "step": 0.01,
            "help": "Standard deviation of the Gaussian clusters.",
        },
        {
            "name": "random_seed",
            "type": "int",
            "label": "Random Seed",
            "default": 42,
            "min_value": 0,
            "step": 1,
            "help": "Seed for reproducible random generation.",
        },
        {
            "name": "drift_width",
            "type": "int",
            "label": "Drift Width (drift_width)",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Width of the concept drift (number of samples).",
        },
    ],
    "sea_drift": [
        {
            "name": "n_windows_before",
            "type": "int",
            "label": "Number of Windows Before Drift",
            "default": 1,
            "min_value": 0,
            "step": 1,
            "help": "Number of windows generated before the concept drift occurs.",
        },
        {
            "name": "n_windows_after",
            "type": "int",
            "label": "Number of Windows After Drift",
            "default": 1,
            "min_value": 0,
            "step": 1,
            "help": "Number of windows generated after the concept drift occurs.",
        },
        {
            "name": "n_features",
            "type": "int",
            "label": "Number of Features in the stream",
            "default": 3,
            "min_value": 2,
            "step": 1,
            "help": "Number of features to be generated in the stream.",
        },
        {
            "name": "drift_width",
            "type": "int",
            "label": "Drift Width (drift_width)",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Width of the concept drift (number of samples).",
        },
    ],
    "linear_weight_inversion_drift": [
        {
            "name": "n_samples_before",
            "type": "int",
            "label": "Number of Samples Before Drift",
            "default": 1000,
            "min_value": 100,
            "step": 100,
            "help": "Number of samples generated before the concept drift occurs.",
        },
        {
            "name": "n_samples_after",
            "type": "int",
            "label": "Number of Samples After Drift",
            "default": 1000,
            "min_value": 100,
            "step": 100,
            "help": "Number of samples generated after the concept drift occurs.",
        },
        {
            "name": "n_features",
            "type": "int",
            "label": "Number of Features (n_features)",
            "default": 11,
            "min_value": 2,
            "step": 1,
            "help": "Total number of features for the dataset. Must be >= 2.",
        },
        {
            "name": "n_drift_features",
            "type": "int",
            "label": "Number of Drifting Features (n_drift_features)",
            "default": 5,
            "min_value": 1,
            "step": 1,
            "help": "Number of features that will drift. Must be <= n_features.",
        },
        {
            "name": "drift_width",
            "type": "int",
            "label": "Drift Width (drift_width)",
            "default": 1,
            "min_value": 1,
            "step": 1,
            "help": "Width of the concept drift (number of samples).",
        },
    ],
    "river_dataset": [
        {
            "name": "size_of_block",
            "type": "int",
            "label": "Size of Block",
            "default": 2000,
            "min_value": 100,
            "step": 100,
            "help": "Number of samples to extract from the dataset.",
        },
        {
            "name": "starting_point",
            "type": "int",
            "label": "Starting Point",
            "default": -1,
            "min_value": -1,
            "step": 100,
            "help": "Starting index in the stream. If -1, a random point is chosen.",
        },
    ],
    "csv_dataset": [
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
    ],
    "multi_window_standard": [
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
    ],
}

# Alias river datasets
DATASET_SETTINGS_SCHEMAS["electricity_drift"] = DATASET_SETTINGS_SCHEMAS["river_dataset"]
DATASET_SETTINGS_SCHEMAS["airlines_drift"] = DATASET_SETTINGS_SCHEMAS["river_dataset"]
DATASET_SETTINGS_SCHEMAS["forest_drift"] = DATASET_SETTINGS_SCHEMAS["river_dataset"]

# Alias multi-window datasets
for key in [
    "plane_multi_window",
    "sea_multi_window",
    "rbf_multi_window",
    "sine_multi_window",
    "stagger_multi_window",
    "random_tree_multi_window",
    "mixed_multi_window",
]:
    DATASET_SETTINGS_SCHEMAS[key] = DATASET_SETTINGS_SCHEMAS["multi_window_standard"]


def get_dataset_available_settings(dataset_name: str) -> dict[str, dict]:
    """Return available preset configurations for a specific dataset."""
    return DATASET_PRESETS.get(dataset_name, {})


def get_dataset_schema(dataset_name: str, window_len: int | None = None) -> list[dict]:
    """Return UI settings schema for a specific dataset, adapting window lengths if requested."""
    schema = [item.copy() for item in DATASET_SETTINGS_SCHEMAS.get(dataset_name, [])]

    if window_len and dataset_name in ["custom_normal", "custom_3d_drift", "sea_drift"]:
        adapted_schema = []
        for item in schema:
            new_item = item.copy()
            if item["name"] == "n_samples_before":
                new_item["name"] = "n_windows_before"
                new_item["label"] = "Number of Windows Before Drift"
                new_item["default"] = int(item["default"] / window_len) if item.get("default") else 1
                new_item["min_value"] = 1
                new_item["step"] = 1
                new_item["help"] = "Number of windows generated before the concept drift occurs."
            elif item["name"] == "n_samples_after":
                new_item["name"] = "n_windows_after"
                new_item["label"] = "Number of Windows After Drift"
                new_item["default"] = int(item["default"] / window_len) if item.get("default") else 1
                new_item["min_value"] = 1
                new_item["step"] = 1
                new_item["help"] = "Number of windows generated after the concept drift occurs."
            adapted_schema.append(new_item)
        return adapted_schema

    return schema
