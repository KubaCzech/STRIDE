from __future__ import annotations

from copy import deepcopy
from typing import Literal
import numpy as np
import pandas as pd

from src.recurrence.protree import TPrototypes, TDataBatch, TTarget
from src.recurrence.protree.explainers import TExplainer


class FullWindowStorage:
    """Stores data for every window during the stream, enabling distance computations."""

    def __init__(self):
        """Initialize storage for all windows."""
        self.windows: dict[int, tuple[TDataBatch, TTarget]] = {}
        self.prototypes: dict[int, TPrototypes] = {}
        self.explainers: dict[int, TExplainer] = {}
        self.drift_detected: dict[int, bool] = {}

    def store_window(
        self, iteration: int, x: TDataBatch, y: TTarget, prototypes: TPrototypes, explainer: TExplainer, drift: bool
    ) -> None:
        """Store window data for a given iteration."""
        self.windows[iteration] = (deepcopy(x), deepcopy(y))
        self.prototypes[iteration] = deepcopy(prototypes)
        self.explainers[iteration] = deepcopy(explainer)
        self.drift_detected[iteration] = drift

    def get_all_iterations(self) -> list[int]:
        """Get all stored iteration numbers."""
        return sorted(self.windows.keys())

    def get_window_data(self, iteration: int) -> tuple[TDataBatch, TTarget, TPrototypes, TExplainer]:
        """Get all data for a specific window.

        Returns:
            Tuple of (x, y, prototypes, explainer)
        """
        if iteration not in self.windows:
            raise ValueError(f"Window at iteration {iteration} not found")

        x, y = self.windows[iteration]
        prototypes = self.prototypes[iteration]
        explainer = self.explainers[iteration]
        return x, y, prototypes, explainer

    def compare_two_windows(
        self,
        iter_a: int,
        iter_b: int,
        measure: Literal[
            "mutual_information",
            "rand_index",
            "completeness",
            "fowlkes_mallows",
            "centroid_displacement",
            "minimal_distance",
            "prototype_reassignment_impact",
        ] = "centroid_displacement",
        strategy: Literal["class", "total"] = "total",
        distance: Literal["l2", "tree"] = "l2",
    ) -> float:
        """Compare two specific windows.

        Args:
            iter_a: First iteration number
            iter_b: Second iteration number
            measure: Comparison metric to use
            strategy: "class" for class-wise, "total" for overall
            distance: "l2" for Euclidean, "tree" for tree-based distance

        Returns:
            Distance/similarity score between the two windows
        """
        x_a, y_a, proto_a, exp_a = self.get_window_data(iter_a)
        x_b, y_b, proto_b, exp_b = self.get_window_data(iter_b)

        return self._compute_metric(proto_a, proto_b, exp_a, exp_b, x_b, y_b, measure, strategy, distance)

    def compare_window_to_all(
        self,
        target_iter: int,
        measure: Literal[
            "mutual_information",
            "rand_index",
            "completeness",
            "fowlkes_mallows",
            "centroid_displacement",
            "minimal_distance",
            "prototype_reassignment_impact",
        ] = "centroid_displacement",
        strategy: Literal["class", "total"] = "total",
        distance: Literal["l2", "tree"] = "l2",
    ) -> pd.Series:
        """Compare one window against all other stored windows.

        Args:
            target_iter: Iteration number to compare
            measure: Comparison metric to use
            strategy: "class" for class-wise, "total" for overall
            distance: "l2" for Euclidean, "tree" for tree-based distance

        Returns:
            Series with iteration numbers as index and distances as values
        """
        x_target, y_target, proto_target, exp_target = self.get_window_data(target_iter)

        iterations = self.get_all_iterations()
        distances = {}

        for iter_num in iterations:
            x, y, proto, exp = self.get_window_data(iter_num)

            score = self._compute_metric(proto_target, proto, exp_target, exp, x, y, measure, strategy, distance)
            distances[iter_num] = score

        return pd.Series(distances, name=f"distance_from_{target_iter}")

    def compute_distance_matrix(
        self,
        measure: Literal[
            "mutual_information",
            "rand_index",
            "completeness",
            "fowlkes_mallows",
            "centroid_displacement",
            "minimal_distance",
            "prototype_reassignment_impact",
        ] = "centroid_displacement",
        strategy: Literal["class", "total"] = "total",
        distance: Literal["l2", "tree"] = "l2",
        verbose=False,
    ) -> pd.DataFrame:
        """Compute full distance matrix between all stored windows.

        Args:
            measure: Comparison metric to use
            strategy: "class" for class-wise, "total" for overall
            distance: "l2" for Euclidean, "tree" for tree-based distance

        Returns:
            DataFrame where rows and columns are iteration numbers, values are distances
        """
        iterations = self.get_all_iterations()
        n = len(iterations)

        if verbose:
            print(f"Computing {n}x{n} distance matrix using {measure}...")
        matrix = np.zeros((n, n))

        for i, iter_a in enumerate(iterations):
            if i % 10 == 0:
                if verbose:
                    print(f"  Progress: {i}/{n}")

            x_a, y_a, proto_a, exp_a = self.get_window_data(iter_a)

            for j, iter_b in enumerate(iterations):
                if i == j:
                    matrix[i, j] = 0.0
                    continue

                if matrix[j, i] != 0.0:
                    # Matrix is symmetric
                    matrix[i, j] = matrix[j, i]
                    continue

                x_b, y_b, proto_b, exp_b = self.get_window_data(iter_b)

                score = self._compute_metric(proto_a, proto_b, exp_a, exp_b, x_b, y_b, measure, strategy, distance)
                matrix[i, j] = score

        if verbose:
            print("Complete!")
        return pd.DataFrame(matrix, index=iterations, columns=iterations)

    def _compute_metric(
        self, prototypes_a, prototypes_b, explainer_a, explainer_b, x, y, measure, strategy, distance
    ) -> float:
        """Compute similarity metric between two prototype sets."""

        if measure in ["mutual_information", "rand_index", "completeness", "fowlkes_mallows"]:
            import src.recurrence.protree.metrics.compare as compare_metrics

            metric = getattr(compare_metrics, measure)

            kwargs = {"a": prototypes_a, "b": prototypes_b, "x": x, "assign_to": "prototype"}

            if distance == "tree":
                kwargs["explainer_a"] = explainer_a
                kwargs["explainer_b"] = explainer_b

            return metric(**kwargs)

        elif measure == "prototype_reassignment_impact":
            from src.recurrence.protree.metrics.compare import prototype_reassignment_impact

            if distance == "tree":
                return prototype_reassignment_impact(prototypes_a, prototypes_b, x, y, explainer=explainer_b)
            return prototype_reassignment_impact(prototypes_a, prototypes_b, x, y)

        elif measure == "minimal_distance":
            if strategy == "class":
                from src.recurrence.protree.metrics.compare import classwise_mean_minimal_distance

                result = classwise_mean_minimal_distance(prototypes_a, prototypes_b)
                return np.mean(list(result.values()))
            else:
                from src.recurrence.protree.metrics.compare import mean_minimal_distance

                return mean_minimal_distance(prototypes_a, prototypes_b)

        elif measure == "centroid_displacement":
            if strategy == "class":
                from src.recurrence.protree.metrics.compare import centroids_displacements

                result = centroids_displacements(prototypes_a, prototypes_b)
                return np.mean(list(result.values()))
            else:
                from src.recurrence.protree.metrics.compare import mean_centroid_displacement

                return mean_centroid_displacement(prototypes_a, prototypes_b)

        return 0.0
