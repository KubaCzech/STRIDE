from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SKKMeans

from src.recurrence.protree import TDataBatch, TTarget, TModel, TPrototypes
from src.recurrence.protree.explainers.tree_distance import IExplainer, ModelAdapterBuilder
from src.recurrence.protree.explainers.utils import _type_to_np_dtype
from src.recurrence.protree.meta import RANDOM_SEED
from src.recurrence.protree.utils import get_x_belonging_to_cls, iloc


class KMeans(IExplainer):
    def __init__(
        self,
        model: TModel,
        n_prototypes: int = 3,
        init: str = "k-means++",
        n_init: str = "auto",
        max_iter: int = 300,
        tol: float = 0.0001,
        verbose: int = 0,
        random_state: int | None = RANDOM_SEED,
        copy_x: bool = True,
        algorithm: str = "lloyd",
        *args,
        **kwargs,
    ) -> None:
        self.model = ModelAdapterBuilder(model)()
        self.prototypes = {}
        self._kmeans = {}
        self._kwargs = dict(
            n_clusters=n_prototypes,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )

    def _fit_select_prototypes(self, x: TDataBatch, y: TTarget) -> None:
        classes = self.get_classes(y)
        for c in classes:
            x_partial = get_x_belonging_to_cls(x, y, c)
            x_partial_np = (
                x_partial.values if isinstance(x_partial, pd.DataFrame) else pd.DataFrame.from_records(x_partial).values
            )

            c_kwargs = self._kwargs.copy()
            c_kwargs["n_clusters"] = min(int(c_kwargs["n_clusters"]), len(x_partial_np))

            self._kmeans[c] = SKKMeans(**c_kwargs).fit(x_partial_np)
            cluster_centers = self._kmeans[c].cluster_centers_
            distances = np.linalg.norm(x_partial_np[:, np.newaxis, :] - cluster_centers[np.newaxis, :], axis=2)
            closest_indices = []
            for id_d in range(distances.shape[1]):
                closest_idx = np.argmin(distances[:, id_d])
                distances[closest_idx, :] = np.inf
                closest_indices.append(closest_idx)
            self.prototypes[c] = iloc(x_partial, closest_indices)

    def fit(self, x: pd.DataFrame) -> KMeans:
        y = pd.DataFrame({"target": self.model.get_model_predictions(x)})

        if not self.prototypes:
            self._fit_select_prototypes(x, y)
        return self

    def select_prototypes(self, x: TDataBatch, y: TTarget | None = None) -> TPrototypes:

        x = deepcopy(x)
        if not y:
            if isinstance(x, pd.DataFrame):
                y = pd.DataFrame({"target": self.model.get_model_predictions(x)})
            else:
                y = self.model.get_model_predictions(x).tolist()

        if not self.prototypes:
            self._fit_select_prototypes(x, y)
        return self.prototypes

    def predict_with_prototypes(self, x: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> np.ndarray:
        predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
        distance = np.ones((len(x))) * np.inf
        for cls in prototypes:
            for _, cluster_centers in prototypes[cls].iterrows():
                distances = np.linalg.norm(
                    x.values[:, np.newaxis, :] - cluster_centers.values[np.newaxis, :], axis=2
                ).squeeze()
                mask = distances < distance
                predictions[mask] = cls
                distance = np.minimum(distances, distance)
        return predictions

    def score_with_prototypes(self, x: pd.DataFrame, y: pd.DataFrame, prototypes: dict[int | str, pd.DataFrame]) -> float:
        y_hat = self.predict_with_prototypes(x, prototypes)
        return IExplainer._score(y, y_hat)
