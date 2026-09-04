from typing import TypeAlias, Literal

from src.datasets.protree_data.river_generators import Sea, Rbf, Mixed, Stagger
from src.datasets.protree_data.stream_generators import Sine, Plane, RandomTree

TNamedStream: TypeAlias = Literal[
    "sine1",
    "sine500",
    "plane100",
    "plane1000",
    "plane5000",
    "random_tree20",
    "random_tree500",
    "rbf1",
    "rbf3000",
    "sea1",
    "stagger1",
    "mixed1",
]


class INamedStream:
    pass


class NamedStreamGeneratorFactory:
    @staticmethod
    def create(name: TNamedStream) -> INamedStream:
        name = "".join([n.capitalize() for n in name.split("_")])
        return globals()[f"{name}"]()


class Sine1(Sine, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[42900, 55500, 67200, 81100], drift_duration=1, seed=42, informative_attrs=(3, 2))


class Sine500(Sine, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[41300, 60000, 73000], drift_duration=500, seed=42, informative_attrs=(3, 2))


class Plane100(Plane, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[55500, 77200, 83500, 94000], drift_duration=100, seed=42)


class Plane1000(Plane, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[48500, 72200, 88700], drift_duration=1000, seed=42)


class Plane5000(Plane, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[43000, 75000], drift_duration=5000, seed=42)


class RandomTree20(RandomTree, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[42100, 49800, 79500], drift_duration=20, seed=42, n_informative=4, n_redundant=2)


class RandomTree500(RandomTree, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[50000, 70000, 90000], drift_duration=500, seed=42, n_informative=2, n_redundant=1)


class Rbf1(Rbf, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[47100, 63000, 88800], drift_duration=1, seed=42, n_informative=4, n_centroids=9)


class Rbf3000(Rbf, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[44000, 71000], drift_duration=3000, seed=42, n_informative=4, n_centroids=9)


class Sea1(Sea, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[55000, 78500], drift_duration=1, seed=42)


class Mixed1(Mixed, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[49900], classification_function=1, drift_duration=1, seed=42)


class Stagger1(Stagger, INamedStream):
    def __init__(self):
        super().__init__(drift_position=[43500, 69500], classification_function=0, drift_duration=1, seed=42)
