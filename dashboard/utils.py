def get_dataset_settings_schema(dataset, window_len):
    """
    Get the settings schema for a dataset, adapting it for the dashboard if necessary.
    """
    if dataset.name != "custom_normal" and dataset.name != "custom_3d_drift" and dataset.name != "sea_drift":
        return dataset.get_settings_schema()

    original_schema = dataset.get_settings_schema()
    schema_to_render = []
    for item in original_schema:
        new_item = item.copy()
        if item["name"] == "n_samples_before":
            new_item["name"] = "n_windows_before"
            new_item["label"] = "Number of Windows Before Drift"
            new_item["default"] = int(item["default"] / window_len) if item["default"] else 1
            new_item["min_value"] = 1
            new_item["step"] = 1
            new_item["help"] = "Number of windows generated before the concept drift occurs."
        elif item["name"] == "n_samples_after":
            new_item["name"] = "n_windows_after"
            new_item["label"] = "Number of Windows After Drift"
            new_item["default"] = int(item["default"] / window_len) if item["default"] else 1
            new_item["min_value"] = 1
            new_item["step"] = 1
            new_item["help"] = "Number of windows generated after the concept drift occurs."
        schema_to_render.append(new_item)
    return schema_to_render
