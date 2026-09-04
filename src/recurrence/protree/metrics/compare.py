from typing import Literal

import numpy as np
import pandas as pd
from icecream import ic

from src.recurrence.protree import TDataBatch, TPrototypes, TTarget
from src.recurrence.protree.explainers.tree_distance import IExplainer
from src.recurrence.protree.explainers.utils import _type_to_np_dtype
from src.recurrence.protree.utils import parse_batch, parse_prototypes, flatten_prototypes


def get_euclidean_predictions(x: TDataBatch, prototypes: TPrototypes) -> np.ndarray:
    """Generate predictions based on Euclidean distance between prototypes and the input data.

    :param x: Input data.
    :type x: TDataBatch
    :param prototypes: Prototypes.
    :type prototypes: TPrototypes

    :returns: Assignment of the input data to classes based on Euclidean distance to the nearest prototype.
    :rtype: np.ndarray

    """

    prototypes = parse_prototypes(prototypes)
    x = parse_batch(x)

    predictions = (np.ones((len(x), 1)) * (-1)).astype(_type_to_np_dtype(prototypes))
    distance = np.ones((len(x))) * np.inf

    for cls in prototypes:
        if isinstance(prototypes[cls], list):
            prototypes[cls] = pd.DataFrame.from_records(prototypes[cls])
        for idx in prototypes[cls].index:
            dist = np.linalg.norm(x - prototypes[cls].loc[idx], axis=1)
            mask = dist < distance
            predictions[mask] = cls
            distance[mask] = dist[mask]
    predictions = predictions.flatten()
    return predictions


def _get_predictions(x: TDataBatch, prototypes: TPrototypes, explainer: IExplainer | None = None) -> np.ndarray:
    if explainer:
        return explainer.predict_with_prototypes(x, prototypes)
    return get_euclidean_predictions(x, prototypes)


def get_euclidean_prototype_assignment(x: TDataBatch, prototypes: TPrototypes) -> np.ndarray:
    """For each data point, finds the nearest prototype based on Euclidean distance.

    :param x: Input data.
    :type x: TDataBatch
    :param prototypes: Prototypes.
    :type prototypes: TPrototypes

    :returns: Assignment of the input data to nearest prototypes based on Euclidean distance.
    :rtype: np.ndarray

    """

    prototypes = parse_prototypes(prototypes)
    prototypes = flatten_prototypes(prototypes)
    x = parse_batch(x)

    distance_matrix = np.linalg.norm(x.values[:, None, :] - prototypes.values[None, ...], axis=-1)
    return distance_matrix.argmin(axis=1)


def _get_nearest_prototypes(x: TDataBatch, prototypes: TPrototypes, explainer: IExplainer) -> np.ndarray:
    if explainer:
        return explainer.get_prototype_assignment(x, prototypes)
    return get_euclidean_prototype_assignment(x, prototypes)


def _get_assignment(
    a: TPrototypes,
    b: TPrototypes,
    x: TDataBatch,
    explainer_a: IExplainer | None = None,
    explainer_b: IExplainer | None = None,
    assign_to: Literal["prototype", "class"] = "class",
) -> tuple[np.ndarray, np.ndarray]:
    if explainer_a and not explainer_b:
        explainer_b = explainer_a

    match assign_to:
        case "prototype":
            assignment_a = _get_nearest_prototypes(x, a, explainer_a)
            assignment_b = _get_nearest_prototypes(x, b, explainer_b)
        case "class":
            assignment_a = _get_predictions(x, a, explainer_a)
            assignment_b = _get_predictions(x, b, explainer_b)
        case _:
            raise ValueError(f"Invalid value for assign_to: {assign_to}")

    return assignment_a.reshape(len(assignment_a)), assignment_b.reshape(len(assignment_b))


def mutual_information(
    a: TPrototypes,
    b: TPrototypes,
    x: TDataBatch,
    explainer_a: IExplainer | None = None,
    explainer_b: IExplainer | None = None,
    assign_to: Literal["prototype", "class"] = "class",
) -> float:
    """Calculate the similarity metric based on mutual information between two sets of prototypes, with respect to the input
    data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based on
    Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer_a: Explainer to generate predictions for the set of prototypes a (Optional).
    :type explainer_a: IExplainer | None
    :param explainer_b: Explainer to generate predictions for the set of predictions b. If explainer_a is provided and
    explainer_b is not provided, explainer_a is used for both prototypes a and b (Optional).
    :type explainer_b: IExplainer | None
    :param assign_to: Whether to compute mutual information based on prototype or class assignment (default = "class").
    :type assign_to: Literal["prototype", "class"]

    :returns: The similarity metric between the two sets of prototypes based on mutual information.
    :rtype: float

    """

    from sklearn.metrics import adjusted_mutual_info_score

    assignment_a, assignment_b = _get_assignment(a, b, x, explainer_a, explainer_b, assign_to)

    return adjusted_mutual_info_score(assignment_a, assignment_b, average_method="arithmetic")


def rand_index(
    a: TPrototypes,
    b: TPrototypes,
    x: TDataBatch,
    explainer_a: IExplainer | None = None,
    explainer_b: IExplainer | None = None,
    assign_to: Literal["prototype", "class"] = "class",
) -> float:
    """Calculate the similarity metric based on the Rand index between two sets of prototypes, with respect to the input
    data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based on
    Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer_a: Explainer to generate predictions for the set of prototypes a (Optional).
    :type explainer_a: IExplainer | None
    :param explainer_b: Explainer to generate predictions for the set of predictions b. If explainer_a is provided and
    explainer_b is not provided, explainer_a is used for both prototypes a and b (Optional).
    :type explainer_b: IExplainer | None
    :param assign_to: Whether to compute mutual information based on prototype or class assignment (default = "class").
    :type assign_to: Literal["prototype", "class"]

    :returns: The similarity metric between the two sets of prototypes based on the Rand index.
    :rtype: float

    """

    from sklearn.metrics import adjusted_rand_score

    assignment_a, assignment_b = _get_assignment(a, b, x, explainer_a, explainer_b, assign_to)

    return adjusted_rand_score(assignment_a, assignment_b)


def fowlkes_mallows(
    a: TPrototypes,
    b: TPrototypes,
    x: TDataBatch,
    explainer_a: IExplainer | None = None,
    explainer_b: IExplainer | None = None,
    assign_to: Literal["prototype", "class"] = "class",
) -> float:
    """Calculate the similarity metric based on the Fowlkes-Mallows index between two sets of prototypes, with respect to the
    input data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based
    on Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer_a: Explainer to generate predictions for the set of prototypes a (Optional).
    :type explainer_a: IExplainer | None
    :param explainer_b: Explainer to generate predictions for the set of predictions b. If explainer_a is provided and
    explainer_b is not provided, explainer_a is used for both prototypes a and b (Optional).
    :type explainer_b: IExplainer | None
    :param assign_to: Whether to compute mutual information based on prototype or class assignment (default = "class").
    :type assign_to: Literal["prototype", "class"]

    :returns: The similarity metric between the two sets of prototypes based on the Fowlkes-Mallows index.
    :rtype: float

    """

    from sklearn.metrics import fowlkes_mallows_score

    if explainer_a and not explainer_b:
        explainer_b = explainer_a

    assignment_a, assignment_b = _get_assignment(a, b, x, explainer_a, explainer_b, assign_to)

    return fowlkes_mallows_score(assignment_a, assignment_b)


def completeness(
    a: TPrototypes,
    b: TPrototypes,
    x: TDataBatch,
    explainer_a: IExplainer | None = None,
    explainer_b: IExplainer | None = None,
    assign_to: Literal["prototype", "class"] = "class",
) -> float:
    """Calculate the similarity metric based on the completeness score between two sets of prototypes, with respect to the
    input data. If an explainer is provided, the predictions are based on the explainer, otherwise, the predictions are based
    on Euclidean distance.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param explainer_a: Explainer to generate predictions for the set of prototypes a (Optional).
    :type explainer_a: IExplainer | None
    :param explainer_b: Explainer to generate predictions for the set of predictions b. If explainer_a is provided and
    explainer_b is not provided, explainer_a is used for both prototypes a and b (Optional).
    :type explainer_b: IExplainer | None
    :param assign_to: Whether to compute mutual information based on prototype or class assignment (default = "class").
    :type assign_to: Literal["prototype", "class"]

    :returns: The similarity metric between the two sets of prototypes based on the completeness score.
    :rtype: float

    """

    from sklearn.metrics import completeness_score

    if explainer_a and not explainer_b:
        explainer_b = explainer_a

    assignment_a, assignment_b = _get_assignment(a, b, x, explainer_a, explainer_b, assign_to)

    return completeness_score(assignment_a, assignment_b)


def centroids_displacements(a: TPrototypes, b: TPrototypes, penalty: float = 10.0) -> dict[int | str, float]:
    """Calculate the displacement of the centroids between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 10.0).
    :type penalty: float
    """

    classes = set(a.keys()).union(b.keys())

    a = parse_prototypes(a)
    b = parse_prototypes(b)

    distances = {}

    for cls in classes:
        # both empty
        if (cls not in a or len(a[cls]) == 0) and (cls not in b or len(b[cls]) == 0):
            distances[cls] = 0.0

        # one empty
        elif cls not in a or len(a[cls]) == 0 or cls not in b or len(b[cls]) == 0:
            distances[cls] = penalty

        # both not empty
        else:
            distances[cls] = np.linalg.norm((a[cls].mean() - b[cls].mean()).values)
    return distances


def mean_centroid_displacement(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> float:
    """Calculate the mean displacement of the centroids between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 2.0).
    :type penalty: float

    :returns: The mean displacement of the centroids between the two sets of prototypes.
    :rtype: float

    """

    displacements = centroids_displacements(a, b, penalty)
    return np.mean(list(displacements.values()))


def _l2_distance_matrix(a: TPrototypes, b: TPrototypes, cls: int | float) -> np.ndarray:
    axis = int(len(a[cls]) > len(b[cls]))
    a[cls] = parse_batch(a[cls])
    b[cls] = parse_batch(b[cls])
    distance_matrix = np.linalg.norm(a[cls].values[:, None, ...] - b[cls].values[None, ...], axis=-1)
    return distance_matrix.min(axis=axis).tolist()


def mean_minimal_distance(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> float:
    """Calculate the mean minimal distance between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 2.0).
    :type penalty: float

    :returns: The mean minimal distance between the two sets of prototypes.
    :rtype: float

    """

    classes = set(a.keys()).union(b.keys())
    distances = []
    for cls in classes:
        # both empty
        if (cls not in a or len(a[cls]) == 0) and (cls not in b or len(b[cls]) == 0):
            continue

        # one empty
        elif cls not in a or len(a[cls]) == 0:
            distances.extend([penalty] * len(b[cls]))
        elif cls not in b or len(b[cls]) == 0:
            distances.extend([penalty] * len(a[cls]))

        # both not empty
        else:
            distances.extend(_l2_distance_matrix(a, b, cls))

    return np.mean(distances)


def classwise_mean_minimal_distance(a: TPrototypes, b: TPrototypes, penalty: float = 2.0) -> dict[int | str, float]:
    """Calculate the class-wise mean minimal distance between two sets of prototypes.

    :param a: First set of prototypes.
    :type a: TPrototypes
    :param b: Second set of prototypes.
    :type b: TPrototypes
    :param penalty: Penalty for empty classes (default = 2.0).
    :type penalty: float

    :returns: The class-wise mean minimal distance between the two sets of prototypes.
    :rtype: dict[int | str, float]

    """

    classes = set(a.keys()).union(b.keys())
    distances = {}

    for cls in classes:
        # both empty
        if (cls not in a or len(a[cls]) == 0) and (cls not in b or len(b[cls]) == 0):
            distances[cls] = 0.0

        # one empty
        elif cls not in a or len(a[cls]) == 0 or cls not in b or len(b[cls]) == 0:
            distances[cls] = penalty * len(b[cls] if cls in b and len(b[cls]) > 0 else a[cls])

        # both not empty
        else:
            distances[cls] = np.mean(_l2_distance_matrix(a, b, cls))
    return distances


def _get_accuracy(prototypes: TPrototypes, x: TDataBatch, y: TTarget, explainer: IExplainer | None = None) -> float:
    """Get accuracy of predictions based on the given prototypes using the Euclidean distance or a custom explainer.

    :param prototypes: Prototypes.
    :type prototypes: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param y: True class labels for the input data.
    :type y: TTarget
    :param explainer: Explainer to generate predictions (Optional).
    :type explainer: IExplainer

    :returns: The accuracy of the predictions based on the given prototypes.
    :rtype: float

    """

    from sklearn.metrics import accuracy_score

    predictions = _get_predictions(x, prototypes, explainer)
    return accuracy_score(y, predictions)


def _one_way_swap_delta(
    prototypes_a: TPrototypes, prototypes_b: TPrototypes, x: TDataBatch, y: TTarget, explainer: IExplainer | None = None
) -> float:
    baseline_accuracy = _get_accuracy(prototypes_b, x, y, explainer)

    prototypes_a = parse_prototypes(prototypes_a)
    prototypes_b = parse_prototypes(prototypes_b)

    accuracy_changes = []
    for cls, prototypes in prototypes_a.items():
        try:
            if len(prototypes) == 0:
                continue

            for idx, prototype in prototypes.iterrows():
                temp_prototypes = prototypes_b.copy()
                if cls not in temp_prototypes:
                    temp_prototypes[cls] = pd.DataFrame()
                temp_prototypes[cls] = pd.concat([temp_prototypes[cls], prototype.to_frame().T]).reset_index(drop=True)

                new_accuracy = _get_accuracy(temp_prototypes, x, y, explainer)
                accuracy_changes.append(np.abs(baseline_accuracy - new_accuracy))
        except Exception as e:
            ic(prototypes)
            raise e
    if accuracy_changes:
        return np.mean(accuracy_changes)
    else:
        return 0.0


def prototype_reassignment_impact(
    prototypes_a: TPrototypes, prototypes_b: TPrototypes, x: TDataBatch, y: TTarget, explainer: IExplainer | None = None
) -> float:
    """
    Calculate a distance metric between two sets of prototypes by measuring the change in accuracy when prototypes from
    one set are temporarily added to the other set. This metrics is symmetric as it calculates changes in accuracy in both
    directions (from prototypes a to b and from prototypes b to a).

    :param prototypes_a: First set of prototypes.
    :type prototypes_a: TPrototypes
    :param prototypes_b: Second set of prototypes.
    :type prototypes_b: TPrototypes
    :param x: Input data.
    :type x: TDataBatch
    :param y: True class labels for the input data.
    :type y: TTarget
    :param explainer: Explainer to generate predictions (Optional).
    :type explainer: IExplainer

    :returns: A distance metric score, with higher values indicating greater divergence between the two sets of prototypes.
    :rtype: float
    """
    one_way_deterioration_a = _one_way_swap_delta(prototypes_a, prototypes_b, x, y, explainer)
    one_way_deterioration_b = _one_way_swap_delta(prototypes_b, prototypes_a, x, y, explainer)
    return (one_way_deterioration_a + one_way_deterioration_b) / 2
