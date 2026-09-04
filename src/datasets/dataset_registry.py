import os
import json
from typing import List, Dict, Optional


class DatasetRegistry:
    def __init__(self, data_dir: str = "data/imported_datasets", registry_file: str = "registry.json"):
        # Ensure paths are absolute or relative to the project root as needed.
        # Assuming run from project root.
        self.data_dir = data_dir
        self.registry_path = os.path.join(self.data_dir, registry_file)

        self._ensure_data_dir()
        self._load_registry()

    def _ensure_data_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    def save_dataset(self, name: str, file_obj, target_column: str, selected_features: List[str] = None):
        """
        Save a new dataset.
        file_obj: file-like object (e.g. from st.file_uploader) or bytes
        """
        safe_filename = f"{name.replace(' ', '_')}.csv"
        file_path = os.path.join(self.data_dir, safe_filename)

        # Save the file
        with open(file_path, "wb") as f:
            if hasattr(file_obj, "read"):
                # Reset pointer just in case
                file_obj.seek(0)
                f.write(file_obj.read())
            else:
                f.write(file_obj)

        # Update registry
        self.registry[name] = {
            "name": name,
            "filename": safe_filename,
            "target_column": target_column,
            "selected_features": selected_features,
        }
        self._save_registry()

    def list_datasets(self) -> Dict[str, dict]:
        return self.registry

    def delete_dataset(self, name: str):
        if name in self.registry:
            dataset_info = self.registry[name]
            file_path = os.path.join(self.data_dir, dataset_info["filename"])

            if os.path.exists(file_path):
                os.remove(file_path)

            del self.registry[name]
            self._save_registry()

    def get_dataset_info(self, name: str) -> Optional[dict]:
        return self.registry.get(name)

    def get_dataset_path(self, name: str) -> Optional[str]:
        info = self.get_dataset_info(name)
        if info:
            return os.path.join(self.data_dir, info["filename"])
        return None
