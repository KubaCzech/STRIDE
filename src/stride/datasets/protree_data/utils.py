from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy


class BinaryScaler:
    def __init__(self) -> None:
        self.mapper: dict[str, float] = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> BinaryScaler:
        binary_columns = x.columns[x.nunique() == 2]
        for c in binary_columns:
            conditional_entropies = []
            for cls in y["target"].unique():
                mask = y["target"] == cls
                p_x_given_y = x[mask].loc[:, c].values.mean()
                entropy_x_given_y = entropy([p_x_given_y, 1 - p_x_given_y])
                weight = mask.sum() / len(y)
                conditional_entropies.append(entropy_x_given_y * weight)
            self.mapper[c] = (1 - sum(conditional_entropies)) / np.log(y["target"].nunique())
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for c in self.mapper:
            x[c] *= self.mapper[c]
        return x

    def fit_transform(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        self.fit(x, y)
        return self.transform(x)
