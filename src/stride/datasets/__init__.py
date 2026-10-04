from .base import BaseDataset
from .benchmarks import get_sample_wine_quality_data
from .drift_blending import DriftBlender, DriftSchedule
from .feature_matching import FeatureAlignmentResult, FeatureMatcher
from .hyperplane_drift import HyperplaneDriftDataset
from .linear_weight_inversion_drift import LinearWeightInversionDriftDataset
from .mixed_multi_window import MixedMultiWindowDataset
from .plane_multi_window import PlaneMultiWindowDataset
from .random_tree_multi_window import RandomTreeMultiWindowDataset
from .rbf_drift import RBFDriftDataset
from .rbf_multi_window import RbfMultiWindowDataset
from .river_dataset import RiverDataset, RiverDatasetType
from .sea_drift import SeaDriftDataset
from .sea_multi_window import SeaMultiWindowDataset
from .semi_synthetic import SemiSyntheticDriftDataset
from .sine_multi_window import SineMultiWindowDataset
from .stagger_multi_window import StaggerMultiWindowDataset

from .dataset_registry import DatasetRegistry
from .imported_dataset import ImportedCSVDataset


def load_datasets() -> dict[str, BaseDataset]:
    base_datasets = [
        SeaDriftDataset(),
        HyperplaneDriftDataset(),
        LinearWeightInversionDriftDataset(),
        RBFDriftDataset(),
        RbfMultiWindowDataset(),
        SineMultiWindowDataset(),
        MixedMultiWindowDataset(),
        PlaneMultiWindowDataset(),
        RandomTreeMultiWindowDataset(),
        SeaMultiWindowDataset(),
        StaggerMultiWindowDataset(),
        RiverDataset(RiverDatasetType.ELECTRICITY.value),
    ]

    datasets_dict: dict[str, BaseDataset] = {d.name: d for d in base_datasets}

    # Load imported and semi-synthetic datasets from registry
    registry = DatasetRegistry()
    for name, info in registry.list_datasets().items():
        if info.get("type") == "semi_synthetic":
            datasets_dict[name] = SemiSyntheticDriftDataset.from_registry_info(name, info, registry)
        else:
            datasets_dict[name] = ImportedCSVDataset(name, info, registry)

    return datasets_dict


DATASETS = load_datasets()


def reload_datasets() -> None:
    new_datasets = load_datasets()
    DATASETS.clear()
    DATASETS.update(new_datasets)


def get_dataset(name: str) -> BaseDataset | None:
    return DATASETS.get(name)


def get_all_datasets() -> list[BaseDataset]:
    return list(DATASETS.values())


__all__ = [
    "BaseDataset",
    "DatasetRegistry",
    "ImportedCSVDataset",
    "SeaDriftDataset",
    "HyperplaneDriftDataset",
    "LinearWeightInversionDriftDataset",
    "RBFDriftDataset",
    "RbfMultiWindowDataset",
    "SineMultiWindowDataset",
    "MixedMultiWindowDataset",
    "PlaneMultiWindowDataset",
    "RandomTreeMultiWindowDataset",
    "SeaMultiWindowDataset",
    "StaggerMultiWindowDataset",
    "RiverDataset",
    "RiverDatasetType",
    "FeatureMatcher",
    "FeatureAlignmentResult",
    "DriftBlender",
    "DriftSchedule",
    "SemiSyntheticDriftDataset",
    "get_sample_wine_quality_data",
    "DATASETS",
    "load_datasets",
    "reload_datasets",
    "get_dataset",
    "get_all_datasets",
]
