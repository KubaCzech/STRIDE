from typing import Dict, Any, Type

import numpy as np
import pandas as pd
from river.forest import ARFClassifier
from river.tree import HoeffdingTreeClassifier
from river.tree.nodes.branch import DTBranch


def _1d_np_to_2d_np(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, axis=0)


def parse_input(x: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(x, list) and len(x) == 1:
        return np.array(x)
    match type(x):
        case np.ndarray:
            match len(x.shape):
                case 1:
                    return _1d_np_to_2d_np(x)
                case 2:
                    return x
                case _:
                    raise ValueError("Invalid input size.")
        case pd.Series:
            return _1d_np_to_2d_np(x.to_numpy())
        case pd.DataFrame:
            return x.to_numpy()
        case _:
            raise TypeError(f"Invalid input type: {type(x)}.")


def _type_to_np_dtype(x: Dict[Any, Any]) -> Type | str:
    val = list(x.keys())[0]
    if isinstance(val, str):
        return "<U50"
    return type(val)


def predict_leaf_one(model: ARFClassifier | HoeffdingTreeClassifier, x: dict[str, float | int]) -> int:
    if isinstance(model, ARFClassifier):
        return [predict_leaf_one(model_, x) for model_ in model]

    if model._root is not None:
        if isinstance(model._root, DTBranch):
            leaf = model._root.traverse(x, until_leaf=True)
        else:
            leaf = model._root
        return id(leaf)
