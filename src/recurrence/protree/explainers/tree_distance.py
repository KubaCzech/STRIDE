import warnings
from abc import ABC
from copy import deepcopy
from typing import Iterable

import numpy as np
import pandas as pd
from river.forest import ARFClassifier
from sklearn.ensemble import RandomForestClassifier as _SKLearnRandomForestClassifier

from src.recurrence.protree import TModel, TDataPoint, TDataBatch, TTarget, TPrototypes
from src.recurrence.protree.explainers.utils import parse_input, _type_to_np_dtype, predict_leaf_one
from src.recurrence.protree.metrics.classification import balanced_accuracy
from src.recurrence.protree.utils import iloc, get_x_belonging_to_cls, flatten_prototypes, parse_prototypes


class IModelAdapter(ABC):
    @property
    def n_trees(self) -> int:
        pass

    def get_leave_indices(self, x: TDataBatch) -> np.ndarray:
        pass

    def get_model_predictions(self, x: TDataBatch) -> np.ndarray:
        pass


class SKLearnAdapter(IModelAdapter):
    def __init__(self, model: TModel) -> None:
        self.model = model

    @property
    def n_trees(self) -> int:
        return self.model.n_estimators

    def get_leave_indices(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.apply(parse_input(x))

    def get_model_predictions(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(parse_input(x))


class RiverAdapter(IModelAdapter):
    def __init__(self, model: ARFClassifier) -> None:
        self.model = model

    @property
    def n_trees(self) -> int:
        return self.model.n_models

    def get_leave_indices(self, x: list[dict[str, int | float]]) -> np.ndarray:
        return np.array([predict_leaf_one(self.model, x_) for x_ in x])

    def get_model_predictions(self, x: list[dict[str, int | float]]) -> np.ndarray:
        return np.array([self.model.predict_one(x_) for x_ in x])


class ModelAdapterBuilder:
    def __init__(self, model: TModel) -> None:
        self.model = model

    def __call__(self, *args, **kwargs) -> IModelAdapter:
        if isinstance(self.model, _SKLearnRandomForestClassifier):
            return SKLearnAdapter(model=self.model)
        elif isinstance(self.model, ARFClassifier):
            return RiverAdapter(model=self.model)
        raise ValueError("Unsupported model class.")


class IExplainer(ABC):
    def __init__(self, model: TModel, *args, **kwargs) -> None:
        self.model = ModelAdapterBuilder(model)()

    def pair_similarity(self, x1: TDataPoint, x2: TDataPoint) -> float:
        nodes1 = self.model.get_leave_indices(parse_input(x1))
        nodes2 = self.model.get_leave_indices(parse_input(x2))

        return sum((nodes1 == nodes2)[0]) / self.model.n_trees

    def similarity_matrix(self, batch: TDataBatch) -> np.ndarray:
        nodes = self.model.get_leave_indices(batch)

        shape = len(batch)
        matrix = np.zeros((shape, shape), dtype=float)
        for i in range(shape):
            matrix[i] = (nodes[i] == nodes).sum(axis=1)
        matrix /= self.model.n_trees
        return matrix

    def distance_matrix(self, batch: TDataBatch) -> np.ndarray:
        return 1 - self.similarity_matrix(batch)

    def _create_distance_matrices(
        self, x: TDataBatch, y: TTarget, classes: Iterable[int | str]
    ) -> dict[int | str, np.ndarray]:
        distances = {cls: None for cls in classes}
        for cls in classes:
            class_x = get_x_belonging_to_cls(x, y, cls)
            distances[cls] = self.distance_matrix(class_x)
        return distances

    def predict_with_prototypes(self, x: TDataBatch, prototypes: TPrototypes) -> np.ndarray:
        predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
        similarity = np.zeros((len(x)))
        x_nodes = self.model.get_leave_indices(x)

        for cls in prototypes:
            for _, prototype in (
                prototypes[cls].iterrows() if hasattr(prototypes[cls], "iterrows") else enumerate(prototypes[cls])
            ):
                prototype_leaves = self.model.get_leave_indices([prototype])
                prototype_similarity = (prototype_leaves == x_nodes).sum(axis=1) / self.model.n_trees
                mask = prototype_similarity > similarity
                predictions[mask] = cls
                similarity[mask] = prototype_similarity[mask]
        return predictions

    def get_prototype_assignment(self, x: TDataBatch, prototypes: TPrototypes) -> np.ndarray:
        prototypes = deepcopy(prototypes)
        prototypes = parse_prototypes(prototypes)
        prototypes = flatten_prototypes(prototypes)
        x_nodes = self.model.get_leave_indices(x)

        predictions = np.ones((len(x), 1)) * (-1)
        similarity = np.zeros((len(x)))

        for idx, prototype in prototypes.iterrows():
            prototype_leaves = self.model.get_leave_indices([prototype])
            prototype_similarity = (prototype_leaves == x_nodes).sum(axis=1) / self.model.n_trees
            mask = prototype_similarity > similarity
            predictions[mask] = idx
            similarity[mask] = prototype_similarity[mask]
        return predictions

    @staticmethod
    def _score(y: pd.DataFrame, y_hat: np.ndarray) -> float:
        if isinstance(y, pd.DataFrame):
            y = y.values.squeeze(1)
        elif isinstance(y, list):
            y = np.array(y)
        return balanced_accuracy(y, y_hat)

    def score_with_prototypes(self, x: TDataBatch, y: TTarget, prototypes: TPrototypes) -> float:
        y_hat = self.predict_with_prototypes(x, prototypes)
        return IExplainer._score(y, y_hat)

    def score(self, x: pd.DataFrame, y: pd.DataFrame) -> float:
        y_hat = self.model.get_model_predictions(x)
        return IExplainer._score(y, y_hat)

    def get_classes(self, y: TTarget) -> set[int | str]:
        if isinstance(y, (list, tuple)):
            return set(y)
        if isinstance(y, pd.DataFrame):
            classes = set()
            for col in y:
                classes.update(set([y_.item() if hasattr(y_, "item") else y_ for y_ in y[col].unique()]))
            return classes
        if isinstance(y, np.ndarray):
            return set(y.tolist())


class G_KM(IExplainer):
    def __init__(self, model: TModel, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.n_prototypes = int(n_prototypes)

    def _find_classwise_prototype(self, matrix: np.ndarray, prototypes: list[int]) -> int:
        mask = np.in1d(range(matrix.shape[0]), prototypes)
        current_partial_distances = np.minimum.reduce(matrix[mask], axis=0) if prototypes else np.inf
        candidate_distances = np.minimum(matrix[~mask], current_partial_distances).sum(axis=1)
        return np.where(~mask)[0][np.argmin(candidate_distances)]

    def _find_single_class_prototypes(self, matrix: np.ndarray) -> list[int]:
        prototypes = []
        for _ in range(self.n_prototypes):
            prototypes.append(self._find_classwise_prototype(matrix, prototypes))
        return prototypes

    def select_prototypes(self, x: TDataBatch, y: TTarget | None = None) -> TPrototypes:
        y = y or self.model.get_model_predictions(x)

        classes = self.get_classes(y)
        prototypes = {cls: [] for cls in classes}
        distances = self._create_distance_matrices(x, y, classes)

        for cls in classes:
            class_x = get_x_belonging_to_cls(x, y, cls)
            indices = self._find_single_class_prototypes(distances[cls])
            prototypes[cls] = iloc(class_x, indices)
        return prototypes


class SM_A(IExplainer):
    def __init__(self, model: TModel, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.n_prototypes = int(n_prototypes)

    def _find_classwise_prototype(self, matrix: np.ndarray, prototypes: list[int]) -> tuple[int, float]:
        if not prototypes:
            idx = np.argmin(matrix.sum(axis=1))
            improvement = matrix.shape[0] - np.sum(matrix[idx])
            return idx, improvement

        mask = np.in1d(range(matrix.shape[0]), prototypes)
        current_partial_distances = matrix[mask].min(axis=0)
        original_distance = current_partial_distances.sum()
        candidates = matrix[~mask]
        candidate_distances = np.minimum(candidates, current_partial_distances).sum(axis=1)
        if not len(candidate_distances):
            return -1, 0.0
        idx = np.argmin(candidate_distances)
        improvement = original_distance - candidate_distances[idx]
        original_idx = np.where(~mask)[0][np.argmin(candidate_distances)]
        return original_idx, improvement

    def _find_prototype(self, distances: dict[int | str, np.ndarray], prototypes: TPrototypes) -> tuple[int | float, int]:
        prototype: tuple[float, int | str, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2]

    def select_prototypes(self, x: TDataBatch, y: TTarget | None = None) -> TPrototypes:
        y = y or self.model.get_model_predictions(x)

        classes = self.get_classes(y)
        prototypes = {cls: [] for cls in classes}

        distances = self._create_distance_matrices(x, y, classes)
        for _ in range(self.n_prototypes):
            cls, idx = self._find_prototype(distances, prototypes)
            prototypes[cls].append(idx)

        for cls in classes:
            prototypes[cls] = iloc(get_x_belonging_to_cls(x, y, cls), prototypes[cls])

        return prototypes


class SM_WA(SM_A):
    def __init__(self, model: TModel, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(model=model, n_prototypes=n_prototypes, *args, **kwargs)

    def _find_prototype(
        self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
    ) -> tuple[int | float, int]:
        prototype: tuple[float, int | str, int] = (-np.inf, -1, -1)
        for cls in distances:
            idx, improvement = self._find_classwise_prototype(distances[cls], prototypes[cls])
            improvement = (1 / distances[cls].shape[0]) * improvement
            if improvement > prototype[0]:
                prototype = (improvement, cls, idx)
        return prototype[1], prototype[2]


class SG(SM_A):
    def __init__(self, n_prototypes: int = 3, *args, **kwargs) -> None:
        super().__init__(n_prototypes=n_prototypes, *args, **kwargs)

    def _find_prototype(
        self, prototypes: TPrototypes, x: TDataBatch, y: TTarget, accuracy: float
    ) -> tuple[int | str, int, float]:
        prototype: tuple[int | str, int] = (-1, -1)
        for cls in prototypes:
            class_x = x[y.values == cls]
            for idx in range(len(class_x)):
                if idx not in prototypes[cls]:
                    temp_prototypes = prototypes.copy()
                    temp_prototypes[cls].append(idx)
                    for _cls in temp_prototypes:
                        temp_prototypes[_cls] = iloc(get_x_belonging_to_cls(x, y, cls), temp_prototypes[_cls])
                    temp_accuracy = self.score_with_prototypes(x, y, temp_prototypes)
                    if temp_accuracy > accuracy:
                        prototype = (cls, idx)
        return prototype[1], prototype[2], accuracy

    def select_prototypes(self, x: TDataBatch, y: TTarget | None = None) -> TPrototypes:
        y = y or self.model.get_model_predictions(x)

        classes = self.get_classes(y)
        prototypes = {cls: [] for cls in classes}
        accuracy = 0

        for _ in range(self.model.n_prototypes):
            cls, idx, accuracy = self._find_prototype(prototypes, x, y, accuracy)
            prototypes[cls].append(idx)

        for cls in classes:
            prototypes[cls] = iloc(get_x_belonging_to_cls(x, y, cls), prototypes[cls])

        return prototypes


class APete(SM_A):
    def __init__(self, model: TModel, alpha: float = 0.05) -> None:
        super().__init__(model)
        self.alpha = float(alpha)

    def _find_prototype(
        self, distances: dict[int | str, np.ndarray], prototypes: dict[int | str, list[int]]
    ) -> tuple[int | float, int, float]:
        delta_prim = -np.inf
        prototype_idx = -1
        prototype_cls = -1
        for cls in distances:
            idx, delta = self._find_classwise_prototype(distances[cls], prototypes[cls])
            if delta > delta_prim:
                delta_prim = delta
                prototype_idx = idx
                prototype_cls = cls
        return prototype_cls, prototype_idx, delta_prim

    def select_prototypes(self, x: TDataBatch, y: TTarget | None = None) -> TPrototypes:
        y = y or self.model.get_model_predictions(x)

        prev_improvement = 0.0
        classes = self.get_classes(y)
        prototypes = {cls: [] for cls in classes}

        distances = self._create_distance_matrices(x, y, classes)
        while True:
            cls, idx, improvement = self._find_prototype(distances, prototypes)
            prototypes[cls].append(idx)

            protos = {}
            for cls in classes:
                protos[cls] = iloc(get_x_belonging_to_cls(x, y, cls), prototypes[cls])

            if improvement == 0 or np.abs(prev_improvement - improvement) / improvement <= self.alpha:
                return protos
            prev_improvement = improvement
