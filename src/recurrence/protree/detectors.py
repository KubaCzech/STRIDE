from __future__ import annotations

from copy import deepcopy
from typing import Type, Literal, Callable

import numpy as np
from river.base import DriftDetector
from river.forest import ARFClassifier
from sklearn.ensemble import RandomForestClassifier

from src.recurrence.protree import TPrototypes, TDataBatch, TTarget
from src.recurrence.protree.explainers import TExplainer, APete


class RaceP(DriftDetector):
    """RACE-P: Real-time Analysis of Concept Evolution with Prototypes"""

    def __init__(
        self,
        model: RandomForestClassifier | ARFClassifier,
        prototype_selector: Type[TExplainer] = APete,
        prototype_selector_kwargs: dict | None = None,
        measure: Literal[
            "mutual_information", "rand_index", "completeness", "fowlkes_mallows", "centroid_displacement", "minimal_distance"
        ] = "centroid_displacement",
        strategy: Literal["class", "total"] = "total",
        distance: Literal["l2", "tree"] = "l2",
        assign_to: Literal["class", "prototype"] = "prototype",
        grace_period: int = 20,
        const: float = 3.0,
    ) -> None:
        super().__init__()
        self.assign_to = assign_to
        self.grace_period = grace_period
        self.const = const
        self.distance = distance
        self.measure = measure
        self.model = model
        self.prototype_selector = prototype_selector
        self.prototype_selector_kwargs = prototype_selector_kwargs or {}
        self.strategy = strategy
        self.metric_value: float | dict[int | str, float] | None = None

        self._iter_counter: int = 0
        self._drift_detected: bool = False
        self._direction: str | None = None

        self._init_state()

    def _init_state(self) -> None:
        self.x_blocks: tuple[TDataBatch, TDataBatch] = (None, None)
        self.y_blocks: tuple[TTarget, TTarget] = (None, None)

        self._reference_window: list[float] | list[dict[int | str, float]] = []
        self._reference_mean: float | dict[int | str, float] | None = None
        self._reference_std: float | dict[int | str, float] | None = None

        self.explainers: tuple[TExplainer | None, TExplainer | None] = (None, None)
        self.prototypes: tuple[TPrototypes, TPrototypes] = ({}, {})

    def update(self, x: TDataBatch, y: TTarget) -> None:
        self._update_block(x, y)
        self._drift_detected = False
        self._update_explainers()
        self._find_prototypes()
        if self._iter_counter >= 1:
            self.metric_value = self._compute_metric()
            if (self.grace_period // 2) <= self._iter_counter < self.grace_period:
                self._update_metric_stats(self.metric_value)
            elif self.grace_period <= self._iter_counter:
                self._test(self.metric_value)
        self._iter_counter += 1

    def _update_block(self, x: TDataBatch, y: TTarget) -> None:
        self.x_blocks = (self.x_blocks[1], x)
        self.y_blocks = (self.y_blocks[1], y)

    def _update_explainers(self) -> None:
        self.explainers = (
            deepcopy(self.explainers[1]),
            self.prototype_selector(deepcopy(self.model), **self.prototype_selector_kwargs),
        )

    def _find_prototypes(self) -> None:
        self.prototypes = (
            deepcopy(self.prototypes[1]),
            self.explainers[1].select_prototypes(self.x_blocks[1], self.y_blocks[1]),
        )

    def _update_metric_stats(self, metric: float | dict[str | int, float]) -> None:
        self._reference_window.append(metric)

        if self._iter_counter == self.grace_period - 1:
            if isinstance(metric, float):
                self._reference_mean = np.mean(self._reference_window)
                self._reference_std = np.std(self._reference_window)
            elif isinstance(metric, dict):
                self._reference_mean = {key: np.mean([m[key] for m in self._reference_window]) for key in metric}
                self._reference_std = {key: np.std([m[key] for m in self._reference_window]) for key in metric}

    def _reset(self):
        super()._reset()
        self._init_state()

    def _test_value(self, value: float, key: int | str | None = None) -> bool:
        if key is not None:
            if self._direction == "increase":
                return value > (self._reference_mean[key] + self.const * self._reference_std[key])
            else:
                return value < (self._reference_mean[key] - self.const * self._reference_std[key])

        if self._direction == "increase":
            return value > (self._reference_mean + self.const * self._reference_std)
        else:
            return value < (self._reference_mean - self.const * self._reference_std)

    def _test(self, metric: float | dict[int | str, float]) -> None:
        # strategy: class
        if isinstance(metric, dict):
            for key in metric:
                if self._test_value(metric[key], key=key):
                    self._drift_detected = True
                    break

        # strategy: total
        elif isinstance(metric, float):
            if self._test_value(metric, key=None):
                self._drift_detected = True

    def _compute_metric(self) -> float | dict[int | str, float]:
        if self.measure in ["mutual_information", "rand_index", "completeness", "fowlkes_mallows"]:
            import src.recurrence.protree.metrics.compare

            self._direction = "decrease"

            metric = getattr(src.recurrence.protree.metrics.compare, self.measure)
            return self._compute_cluster_metric(metric)

        elif self.measure == "prototype_reassignment_impact":
            from src.recurrence.protree.metrics.compare import prototype_reassignment_impact

            self._direction = "increase"
            if self.distance == "tree":
                return prototype_reassignment_impact(
                    *self.prototypes, self.x_blocks[1], self.y_blocks[1], explainer=self.explainers[1]
                )
            return prototype_reassignment_impact(*self.prototypes, self.x_blocks[1], self.y_blocks[1])

        elif self.measure == "minimal_distance":
            self._direction = "increase"

            if self.strategy == "class":
                from src.recurrence.protree.metrics.compare import classwise_mean_minimal_distance

                return self._compute_spatial_metric(classwise_mean_minimal_distance)

            elif self.strategy == "total":
                from src.recurrence.protree.metrics.compare import mean_minimal_distance

                return self._compute_spatial_metric(mean_minimal_distance)

        elif self.measure == "centroid_displacement":
            self._direction = "increase"

            if self.strategy == "class":
                from src.recurrence.protree.metrics.compare import centroids_displacements

                return self._compute_spatial_metric(centroids_displacements)

            elif self.strategy == "total":
                from src.recurrence.protree.metrics.compare import mean_centroid_displacement

                return self._compute_spatial_metric(mean_centroid_displacement)

    def _build_cluster_kwargs(self) -> dict:
        kwargs = {"a": self.prototypes[0], "b": self.prototypes[1], "x": self.x_blocks[1], "assign_to": self.assign_to}

        if self.distance == "tree":
            kwargs["explainer_a"] = self.explainers[0]
            kwargs["explainer_b"] = self.explainers[1]

        return kwargs

    def _compute_cluster_metric(self, metric: Callable[..., float]) -> float | dict[int | str, float]:
        return metric(**self._build_cluster_kwargs())

    def _compute_spatial_metric(self, metric: Callable[..., float | dict[int | str, float]]) -> float | dict[int | str, float]:
        return metric(*self.prototypes)
