from .base import BaseDataset
from .sea_drift import SeaDriftDataset
from .hyperplane_drift import HyperplaneDriftDataset
from .linear_weight_inversion_drift import LinearWeightInversionDriftDataset
from .rbf_drift import RBFDriftDataset
from .rbf_multi_window import RbfMultiWindowDataset
from .sine_multi_window import SineMultiWindowDataset
from .mixed_multi_window import MixedMultiWindowDataset
from .plane_multi_window import PlaneMultiWindowDataset
from .random_tree_multi_window import RandomTreeMultiWindowDataset
from .sea_multi_window import SeaMultiWindowDataset
from .stagger_multi_window import StaggerMultiWindowDataset

from .river_dataset import RiverDataset, RiverDatasetType
from .dataset_registry import DatasetRegistry
from .imported_dataset import ImportedCSVDataset


def load_datasets():
    base_datasets = [
        SeaDriftDataset(),
        HyperplaneDriftDataset(),
        LinearWeightInversionDriftDataset(),
        # RandomRBFDriftDataset(),
        RBFDriftDataset(),
        RbfMultiWindowDataset(),
        SineMultiWindowDataset(),
        MixedMultiWindowDataset(),
        PlaneMultiWindowDataset(),
        RandomTreeMultiWindowDataset(),
        SeaMultiWindowDataset(),
        StaggerMultiWindowDataset(),
        # CSVDataset(),
        RiverDataset(RiverDatasetType.ELECTRICITY.value),
        # RiverDataset(RiverDatasetType.AIRLIENES.value),
        # RiverDataset(RiverDatasetType.FOREST_COVER_TYPE.value)
    ]

    datasets_dict = {d.name: d for d in base_datasets}

    # Load imported datasets
    registry = DatasetRegistry()
    for name, info in registry.list_datasets().items():
        datasets_dict[name] = ImportedCSVDataset(name, info, registry)

    return datasets_dict


DATASETS = load_datasets()


def reload_datasets():
    new_datasets = load_datasets()
    DATASETS.clear()
    DATASETS.update(new_datasets)


def get_dataset(name: str) -> BaseDataset:
    return DATASETS.get(name)


def get_all_datasets() -> list[BaseDataset]:
    return list(DATASETS.values())
