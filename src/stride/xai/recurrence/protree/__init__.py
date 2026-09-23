from typing import TypeAlias

import numpy as np
import pandas as pd
from river.forest import ARFClassifier
from sklearn.ensemble import RandomForestClassifier as _SKLearnRandomForestClassifier

TModel: TypeAlias = ARFClassifier | _SKLearnRandomForestClassifier
TDataPoint: TypeAlias = pd.Series | np.ndarray | dict[str, int | float]
TDataBatch: TypeAlias = pd.DataFrame | np.ndarray | list[dict[str, int | float]]
TTarget: TypeAlias = pd.DataFrame | list[int | str]
TPrototypes: TypeAlias = dict[str | int, pd.DataFrame | dict[str, int | float]]
