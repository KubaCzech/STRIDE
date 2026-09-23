from typing import Any

import numpy as np
import pandas as pd

from stride.xai.recurrence.protree import TDataBatch, TTarget, TPrototypes


def pprint_dict(dict_: dict, indent_level: int = 0) -> None:
    indent = "\t" * indent_level
    for key, value in dict_.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            pprint_dict(value, indent_level + 1)
        else:
            print(f"{indent}{key}: {value}")


def parse_int_float_str(value) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def get_x_belonging_to_cls(x: TDataBatch, y: TTarget, cls: int | str) -> TDataBatch:
    if isinstance(x, pd.DataFrame) and isinstance(y, (np.ndarray, pd.DataFrame)):
        if isinstance(y, np.ndarray):
            return x[y == cls]
        return x[(y == cls).any(axis=1)]
    if isinstance(x, (list, tuple)) and isinstance(y, (np.ndarray, list, tuple)):
        return [x_ for x_, y_ in zip(x, y) if y_ == cls]
    raise ValueError("x and y have to both be of the same type, one of pd.DataFrame or list")


def get_x_not_belonging_to_cls(x: TDataBatch, y: TTarget, cls: int | str) -> TDataBatch:
    if isinstance(x, pd.DataFrame) and isinstance(y, (np.ndarray, pd.DataFrame)):
        if isinstance(y, np.ndarray):
            return x[(y != cls)]
        return x[(y != cls).any(axis=1)]
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return [x_ for x_, y_ in zip(x, y) if y_ != cls]
    raise ValueError("x and y have to both be of the same type, one of pd.DataFrame or list")


def iloc(x: pd.DataFrame | list[Any], indices: list[int]) -> pd.DataFrame | list[Any]:
    if isinstance(x, pd.DataFrame):
        return x.iloc[indices, :]
    if isinstance(x, (list, tuple)):
        return [x_ for i, x_ in enumerate(x) if i in indices]


def parse_batch(batch: TDataBatch) -> pd.DataFrame:
    if isinstance(batch, pd.DataFrame):
        return batch
    return pd.DataFrame.from_records(batch)


def parse_prototypes(prototypes: TPrototypes) -> dict[int | str, pd.DataFrame]:
    output = {}
    for cls in prototypes:
        output[cls] = parse_batch(prototypes[cls])
    return output


def get_re_idx(prototypes: TPrototypes, cls: str | int, idx: int, in_class_only: bool = False) -> int:
    if isinstance(prototypes[cls], pd.DataFrame):
        if in_class_only:
            re_idx = prototypes[cls].index.to_list().index(idx)
        else:
            re_idx = flatten_prototypes(prototypes).index.to_list().index(idx)
    if isinstance(prototypes[cls], list):
        if in_class_only:
            re_idx = idx
        else:
            re_idx = 0
            for cls_ in prototypes:
                for idx_ in prototypes[cls_]:
                    if cls == cls_ and idx == idx_:
                        break
                    re_idx += 1
    return re_idx


def flatten_prototypes(prototypes: TPrototypes) -> pd.DataFrame | list[dict[str, int | float]]:
    if isinstance(prototypes[list(prototypes.keys())[0]], pd.DataFrame):
        return pd.concat([prototypes[cls] for cls in prototypes], ignore_index=False)
