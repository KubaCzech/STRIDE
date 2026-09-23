from __future__ import annotations

from enum import Enum
from typing import TypeAlias, Union

from stride.xai.recurrence.protree.explainers.naive import KMeans
from stride.xai.recurrence.protree.explainers.tree_distance import SM_A, SM_WA, SG, G_KM, APete

TExplainer: TypeAlias = Union[KMeans, G_KM, SM_A, SM_WA, SG, APete]


class Explainer(Enum):
    KMeans = KMeans
    G_KM = G_KM
    SM_A = SM_A
    SM_WA = SM_WA
    SG = SG
    APete = APete
