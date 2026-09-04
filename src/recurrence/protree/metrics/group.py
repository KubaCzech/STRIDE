from typing import Callable

import numpy as np
import pandas as pd

from src.recurrence.protree import TPrototypes, TDataBatch, TTarget
from src.recurrence.protree.explainers.tree_distance import IExplainer
from src.recurrence.protree.metrics.classification import balanced_accuracy


def fidelity_with_model(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> float:
    y_model = explainer.model.get_model_predictions(x)
    y_prototypes = explainer.predict_with_prototypes(x, prototypes)
    return balanced_accuracy(y_model, y_prototypes)


def contribution(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> float:
    from src.recurrence.protree.metrics.individual import individual_contribution

    diversity = 0
    n_prototypes = 0
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            try:
                diversity += individual_contribution(prototypes, cls, idx, explainer, x)
                n_prototypes += 1
            except ValueError:
                continue
    return diversity / n_prototypes


def _dist(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget, func: Callable) -> float:
    dist = 0
    n_prototypes = 0
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            dist += func(prototypes, cls, idx, explainer, x, y)
            n_prototypes += 1
    return dist / n_prototypes


def mean_in_distribution(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget) -> float:
    from src.recurrence.protree.metrics.individual import individual_in_distribution

    return _dist(prototypes, explainer, x, y, individual_in_distribution)


def mean_out_distribution(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget) -> float:
    from src.recurrence.protree.metrics.individual import individual_out_distribution

    return _dist(prototypes, explainer, x, y, individual_out_distribution)


def mean_entropy_hubness(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> float:
    from numpy import mean

    entropies = vector_entropy_hubness(prototypes, explainer, x)
    return mean(list(entropies.values())).item()


def vector_entropy_hubness(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> dict[str | bool | int, float]:
    from numpy import log2
    from scipy.stats import entropy

    from src.recurrence.protree.metrics.individual import hubness

    y = explainer.model.get_model_predictions(x)

    hub_score = {cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            hub_score[cls].append(hubness(prototypes, cls, idx, explainer, x, y))
    entropies = {
        cls: (entropy(hub_score[cls], base=2) / max(log2(len(hub_score[cls]) + 1e-6), 1 + 1e-6)).item()
        if sum(hub_score[cls]) != 0
        else 0.0
        for cls in hub_score
    }
    return entropies


def vector_consistent_votes(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> dict[str | bool | int, float]:
    from src.recurrence.protree.metrics.individual import consistent_votes

    consistency = {cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            consistency[cls].append(consistent_votes(prototypes, cls, idx, explainer, x))
    return consistency


def vector_voting_frequency(prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch) -> dict[str | bool | int, float]:
    from src.recurrence.protree.metrics.individual import voting_frequency

    votes = {cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            votes[cls].append(voting_frequency(prototypes, cls, idx, explainer, x))
    return votes


def vector_in_distribution(
    prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget
) -> dict[str | bool | int, float]:
    from src.recurrence.protree.metrics.individual import individual_in_distribution

    in_distribution = {cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            in_distribution[cls].append(individual_in_distribution(prototypes, cls, idx, explainer, x, y))
    return in_distribution


def vector_out_distribution(
    prototypes: TPrototypes, explainer: IExplainer, x: TDataBatch, y: TTarget
) -> dict[str | bool | int, float]:
    from src.recurrence.protree.metrics.individual import individual_out_distribution

    out_distribution = {cls.item() if isinstance(cls, np.ndarray) else cls: [] for cls in prototypes}
    for cls in prototypes:
        for idx in prototypes[cls].index if isinstance(prototypes[cls], pd.DataFrame) else range(len(prototypes[cls])):
            out_distribution[cls].append(individual_out_distribution(prototypes, cls, idx, explainer, x, y))
    return out_distribution
