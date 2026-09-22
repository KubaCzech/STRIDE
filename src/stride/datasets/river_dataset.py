import numpy as np
import itertools

from enum import Enum
from river import datasets
from .base import BaseDataset
from .utils import generate_river_data, generate_river_data_with_selection


class RiverDatasetType(Enum):
    ELECTRICITY = "Electricity"
    AIRLINES = "Airlines"
    BIKES = "Bikes"


class RiverDataset(BaseDataset):
    def __init__(self, name: str = RiverDatasetType.ELECTRICITY.value):
        self.dataset_name = name
        if self.dataset_name == RiverDatasetType.ELECTRICITY.value:
            self.dataset = datasets.Elec2()
        elif self.dataset_name == RiverDatasetType.AIRLINES.value:
            self.dataset = datasets.AirlinePassengers()
        elif self.dataset_name == RiverDatasetType.BIKES.value:
            self.dataset = datasets.Bikes()
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

    @property
    def name(self) -> str:
        return f"{self.dataset_name.lower()}_drift"

    @property
    def display_name(self) -> str:
        return f"{self.dataset_name} Drift"

    def get_settings_schema(self) -> list[dict]:
        return [
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
        ]

    def get_params(self) -> dict:
        return {"size_of_block": 2000, "starting_point": -1, "random_seed": 42}

    def generate(self, size_of_block=2000, starting_point=None, random_seed=42, **kwargs):
        size_of_dataset = self.dataset.n_samples
        np.random.seed(random_seed)

        if starting_point is None or starting_point == -1:
            starting_point = np.random.randint(0, size_of_dataset - size_of_block)
        else:
            starting_point = max(0, min(starting_point, size_of_dataset - size_of_block))

        stream = itertools.islice(iter(self.dataset), starting_point, None)

        if self.dataset_name == RiverDatasetType.ELECTRICITY.value:
            feature_names = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
            return generate_river_data_with_selection(stream, size_of_block, feature_names)

        return generate_river_data(stream, size_of_block)


RiverDataset(RiverDatasetType.ELECTRICITY.value)
RiverDataset(RiverDatasetType.AIRLINES.value)
