from copy import deepcopy

import numpy as np
import pandas as pd

from src.recurrence.protree import TPrototypes, TTarget, TDataBatch
from src.recurrence.protree.explainers.tree_distance import IExplainer
from src.recurrence.protree.utils import get_x_belonging_to_cls, get_re_idx, flatten_prototypes, get_x_not_belonging_to_cls


def individual_contribution(prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer, x: TDataBatch) -> float:
    from src.recurrence.protree.metrics.group import fidelity_with_model

    prototypes_without = deepcopy(prototypes)
    if isinstance(prototypes[cls], pd.DataFrame):
        prototypes_without[cls].drop(idx, inplace=True)
    else:
        prototypes_without[cls].pop(idx)

    result = fidelity_with_model(prototypes, explainer, x) - fidelity_with_model(prototypes_without, explainer, x)
    return result


def voting_frequency(prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer, x: TDataBatch) -> float:
    prototypes_flat = flatten_prototypes(prototypes)
    re_idx = get_re_idx(prototypes, cls, idx)

    x_leaves = explainer.model.get_leave_indices(x)
    proto_leaves = explainer.model.get_leave_indices(prototypes_flat)
    neighbourhood = []
    for i in range(len(proto_leaves)):
        neighbourhood.append((x_leaves == proto_leaves[i]).sum(axis=1))
    return (np.vstack(neighbourhood).argmax(axis=0) == re_idx).sum().item() / len(x) if len(x) > 0 else 0.0


def consistent_votes(prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer, x: TDataBatch) -> float:
    re_idx = get_re_idx(prototypes, cls, idx)
    prototypes_flat = flatten_prototypes(prototypes)

    classification = explainer.model.get_model_predictions(x)
    mask = classification == cls

    x_leaves = explainer.model.get_leave_indices(x)
    proto_leaves = explainer.model.get_leave_indices(prototypes_flat)
    neighbourhood = []
    for i in range(len(proto_leaves)):
        neighbourhood.append((x_leaves == proto_leaves[i]).sum(axis=1))
    votes = np.vstack(neighbourhood).argmax(axis=0) == re_idx
    correct = np.logical_and(votes, mask).sum()
    return (correct / votes.sum()).item() if votes.sum() > 0 else 0.0


def hubness(prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer, x: TDataBatch, y: TTarget) -> float:
    sub_x = get_x_belonging_to_cls(x, y, cls)

    if not len(sub_x):
        return 0.0

    re_idx = get_re_idx(prototypes, cls, idx, in_class_only=True)

    x_cls_leaves = explainer.model.get_leave_indices(sub_x)
    proto_leaves = explainer.model.get_leave_indices(prototypes[cls])

    neighbourhood = []
    for i in range(len(proto_leaves)):
        neighbourhood.append((x_cls_leaves == proto_leaves[i]).sum(axis=1))
    return (np.vstack(neighbourhood).argmax(axis=0) == re_idx).sum().item() / len(sub_x) if len(sub_x) > 0 else 0.0


def _mean_similarity(sub_x: TDataBatch, prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer) -> float:
    x_cls_leaves = explainer.model.get_leave_indices(sub_x)
    if isinstance(prototypes[cls], pd.DataFrame):
        proto_leaves = explainer.model.get_leave_indices(pd.DataFrame(prototypes[cls].loc[idx]).transpose())
    else:
        proto_leaves = explainer.model.get_leave_indices([prototypes[cls][idx]])
    return ((x_cls_leaves == proto_leaves).sum() / np.prod(x_cls_leaves.shape)).item()


def individual_in_distribution(
    prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer, x: TDataBatch, y: TTarget
) -> float:
    sub_x = get_x_belonging_to_cls(x, y, cls)
    return _mean_similarity(sub_x, prototypes, cls, idx, explainer)


def individual_out_distribution(
    prototypes: TPrototypes, cls: str | int, idx: int, explainer: IExplainer, x: TDataBatch, y: TTarget
) -> float:
    sub_x = get_x_not_belonging_to_cls(x, y, cls)
    return _mean_similarity(sub_x, prototypes, cls, idx, explainer)
