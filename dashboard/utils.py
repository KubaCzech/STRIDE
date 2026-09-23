"""Dashboard utility functions."""

from dashboard.config.dataset_schemas import get_dataset_schema


def get_dataset_settings_schema(dataset, window_len):
    """
    Get the settings schema for a dataset, adapting it for the dashboard if necessary.
    """
    name = dataset.name if hasattr(dataset, "name") else str(dataset)
    return get_dataset_schema(name, window_len=window_len)
