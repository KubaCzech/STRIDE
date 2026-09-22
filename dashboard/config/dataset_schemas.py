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


def get_dataset_available_settings(dataset_name: str) -> dict[str, dict]:
    """Return available preset configurations for a specific dataset."""
    return DATASET_PRESETS.get(dataset_name, {})
