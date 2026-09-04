from __future__ import annotations

from typing import Literal

import numpy as np
from river.datasets import synth

from src.datasets.protree_data.stream_generators import IStreamGenerator


class Sea(IStreamGenerator):
    """SEA synthetic dataset, based on the River library implementation:
    https://riverml.xyz/latest/api/datasets/synth/SEA/

    The simulated drift is a transition between two classification functions available in the dataset.

    :param drift_position: The position of the drift.
    :param drift_duration: The width of the drift.
    :param seed: Random seed.
    """

    def __init__(self, drift_position: int | list[int] = 500, drift_duration: int = 0, seed: int = 42) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self.drift_position = drift_position if isinstance(drift_position, (list, tuple)) else [drift_position]
        self.drift_duration = drift_duration
        self.seed = seed
        self._iter_counter = 0
        self._iter_drift_remaining = 0
        self._data_streams = (synth.SEA(variant=3, seed=seed).__iter__(), None)
        np.random.seed(seed)

    def update_drift(self) -> None:
        if self._iter_counter in self.drift_position:
            self._data_streams = (self._data_streams[0], synth.SEA(variant=np.random.randint(0, 4), seed=self.seed).__iter__())
            self._iter_drift_remaining = self.drift_duration
        if self._iter_drift_remaining == 1:
            self._data_streams = self._data_streams[1], None
        self._iter_drift_remaining -= 1

    def _normalise(self, x: dict[str, float]) -> dict[str, float]:
        return {k: v / 10 for k, v in x.items()}

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < self.drift_factor():
                x, y = next(self._data_streams[1])
            else:
                x, y = next(self._data_streams[0])
            x = self._normalise(x)
            result.append((x, y))
        return result


class Rbf(IStreamGenerator):
    def __init__(
        self,
        drift_position: int | list[int] = 500,
        drift_duration: int = 1,
        seed: int = 42,
        n_informative: int = 5,
        n_centroids: int = 11,
    ) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self._data_stream = synth.RandomRBF(
            seed_model=seed, seed_sample=seed, n_features=n_informative, n_centroids=n_centroids, n_classes=3
        ).__iter__()
        self._flip = False

    def _permute_y(self, y: int) -> int:
        return (y + 1) % 3

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x, y = next(self._data_stream)
            if self._flip:
                y = self._permute_y(y)
            if self._iter_drift_remaining > 0 and np.random.uniform(0, 1) < self.drift_factor():
                y = self._permute_y(y)

            result.append((x, y))
            self._iter_counter += 1
        return result

    def update_drift(self):
        if self._iter_counter in self.drift_position:
            self._iter_drift_remaining = self.drift_duration
        if self._iter_drift_remaining == 1:
            self._flip = not self._flip
        self._iter_drift_remaining -= 1


class Stagger(IStreamGenerator):
    def __init__(
        self,
        drift_position: int | list[int] = 500,
        drift_duration: int = 1,
        classification_function: Literal[0, 1, 2] = 0,
        seed: int = 42,
    ) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self._data_stream = synth.STAGGER(classification_function=classification_function, balance_classes=True, seed=seed)
        self._data_stream_iter = self._data_stream.__iter__()

    def _normalise(self, x: dict[str, float]) -> dict[str, float]:
        return {
            **{f"size_{i}": int(i == x["size"]) for i in range(3)},
            **{f"color_{i}": int(i == x["color"]) for i in range(3)},
            **{f"shape_{i}": int(i == x["shape"]) for i in range(3)},
        }

    def take(self, n: int) -> list[tuple[dict[str, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x, y = next(self._data_stream_iter)
            x = self._normalise(x)
            result.append((x, y))
            self._iter_counter += 1
        return result

    def update_drift(self):
        if self._iter_counter in self.drift_position:
            self._data_stream.generate_drift()
            self._iter_drift_remaining = self.drift_duration
        self._iter_drift_remaining -= 1


class Mixed(IStreamGenerator):
    def __init__(
        self,
        drift_position: int | list[int] = 500,
        drift_duration: int = 1,
        classification_function: Literal[0, 1] = 1,
        seed: int = 42,
    ) -> None:
        super().__init__(drift_position, drift_duration, seed)
        self._data_stream = synth.Mixed(classification_function=classification_function, balance_classes=False, seed=seed)
        self._data_stream_iter = self._data_stream.__iter__()

    def _normalise(self, x: dict[int, float]) -> dict[int, float]:
        x[0] = int(x[0])
        x[1] = int(x[1])
        return x

    def take(self, n: int) -> list[tuple[dict[int, float], int]]:
        result = []
        for i in range(n):
            self.update_drift()
            x, y = next(self._data_stream_iter)
            x = self._normalise(x)
            result.append((x, y))
            self._iter_counter += 1
        return result

    def update_drift(self):
        if self._iter_counter in self.drift_position:
            self._data_stream.generate_drift()
            self._iter_drift_remaining = self.drift_duration
        self._iter_drift_remaining -= 1
