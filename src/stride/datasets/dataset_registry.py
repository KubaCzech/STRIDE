import json
import os
from typing import Any


class DatasetRegistry:
    def __init__(self, data_dir: str = "data/imported_datasets", registry_file: str = "registry.json") -> None:
        # Ensure paths are absolute or relative to the project root as needed.
        # Assuming run from project root.
        self.data_dir = data_dir
        self.registry_path = os.path.join(self.data_dir, registry_file)

        self._ensure_data_dir()
        self._load_registry()

    def _ensure_data_dir(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _load_registry(self) -> None:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save_registry(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    def save_dataset(
        self,
        name: str,
        file_obj: Any,
        target_column: str,
        selected_features: list[str] | None = None,
    ) -> None:
        """
        Save a new imported CSV dataset.
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
            "type": "imported_csv",
            "target_column": target_column,
            "selected_features": selected_features,
        }
        self._save_registry()

    def save_semi_synthetic_dataset(
        self,
        name: str,
        df: Any,
        target_column: str,
        recipe: dict,
        selected_features: list[str] | None = None,
    ) -> None:
        """
        Save a semi-synthetic dataset synthesized from two empirical concepts.
        """
        safe_filename = f"{name.replace(' ', '_')}.csv"
        file_path = os.path.join(self.data_dir, safe_filename)

        # Save DataFrame to CSV
        if hasattr(df, "to_csv"):
            df.to_csv(file_path, index=False)

        # Update registry
        feature_cols = (
            selected_features
            if selected_features is not None
            else [c for c in df.columns if c not in (target_column, "_concept")]
        )

        self.registry[name] = {
            "name": name,
            "filename": safe_filename,
            "type": "semi_synthetic",
            "display_name": f"{name} (Semi-Synthetic)",
            "target_column": target_column,
            "selected_features": feature_cols,
            "recipe": recipe,
        }
        self._save_registry()

    def list_datasets(self) -> dict[str, dict]:
        return self.registry

    def delete_dataset(self, name: str) -> None:
        if name in self.registry:
            dataset_info = self.registry[name]
            file_path = os.path.join(self.data_dir, dataset_info["filename"])

            if os.path.exists(file_path):
                os.remove(file_path)

            del self.registry[name]
            self._save_registry()

    def get_dataset_info(self, name: str) -> dict | None:
        return self.registry.get(name)

    def get_dataset_path(self, name: str) -> str | None:
        info = self.get_dataset_info(name)
        if info:
            return os.path.join(self.data_dir, info["filename"])
        return None
