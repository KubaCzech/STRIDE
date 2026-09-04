from __future__ import annotations

import numpy as np
import pandas as pd


class MultilabelHotEncoder:
    def __init__(self) -> None:
        """MultilabelHotEncoder is a class designed to hot encode a DataFrame with multiple columns containing categorical
        labels. It supports fitting to the unique labels in the data and transforming the data into a binary-encoded format.

        Example:
        >>> encoder = MultilabelHotEncoder()
        >>> data = pd.DataFrame({'target1': ['foo', 'foo', 'bar'],
                                 'target2': [None, 'bar', None]})
        >>> encoded_data = encoder.fit_transform(data)
        >>> print(encoded_data)
           foo  bar
        0    1    0
        1    1    1
        2    0    1
        """
        self.columns = set()

    def fit(self, y: pd.DataFrame) -> MultilabelHotEncoder:
        """Fit the encoder to the unique labels in the input DataFrame.

        :param y: The input DataFrame with categorical labels.

        :return: The fitted encoder instance.
        """
        for col in y:
            self.columns.update(set(y[col].unique()))
        self.columns.remove(np.nan)
        self.columns = list(self.columns)
        return self

    def transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame into a binary-encoded format.

        :param y: The input DataFrame with categorical labels.

        :return: A new DataFrame with binary-encoded columns.
        """
        y_transformed = pd.DataFrame(data=np.zeros((y.shape[0], len(self.columns)), dtype=int), columns=self.columns)
        for col in self.columns:
            y_transformed[col] = (y == col).any(axis=1).astype(int)
        return y_transformed

    def fit_transform(self, y: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the encoder and transform the input DataFrame in a single step.

        :param y: The input DataFrame with categorical labels.

        :return: A new DataFrame with binary-encoded columns.
        """
        self.fit(y)
        return self.transform(y)
