"""Common preprocessing, scaling, and dimensionality reduction utilities."""

from enum import Enum
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from umap import UMAP


class ScalingType(Enum):
    MinMax = "min_max"
    Standard = "standard"


class ReducerType(Enum):
    # Linear methods
    PCA = "PCA"
    ICA = "ICA"
    FA = "FA"
    LDA = "LDA"

    # Non-linear methods
    TSNE = "t_SNE"
    UMAP = "UMAP"
    LLE = "LLE"
    MDS = "MDS"


NON_TRANSFORMABLES = {ReducerType.TSNE, ReducerType.LLE, ReducerType.MDS}


class DataScaler:
    """Wrapper around sklearn scalers with a unified interface."""

    scaler: TransformerMixin
    _is_fitted: bool

    def __init__(self, scaling_type: ScalingType):
        if scaling_type == ScalingType.MinMax:
            self.scaler = MinMaxScaler()
        elif scaling_type == ScalingType.Standard:
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaling type: {scaling_type}")

        self._is_fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray) -> "DataScaler":
        """Fit the scaler on the given data."""
        self.scaler.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray, return_df: bool = True) -> np.ndarray | pd.DataFrame:
        """Transform data using the fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'transform'.")

        transformed = self.scaler.transform(X)
        if return_df and isinstance(X, pd.DataFrame):
            return pd.DataFrame(transformed, columns=X.columns, index=X.index)
        return transformed

    def fit_transform(self, X: pd.DataFrame | np.ndarray, return_df: bool = True) -> np.ndarray | pd.DataFrame:
        """Fit the scaler and transform the data."""
        self.fit(X)
        return self.transform(X, return_df=return_df)


class DataDimensionsReducer:
    """
    Unified interface for dimensionality reduction.

    Provides a single interface to perform both linear and nonlinear
    dimensionality reduction using PCA, ICA, FA, LDA, t-SNE, UMAP, LLE, and MDS.
    """

    reducer_type: ReducerType
    n_components: int
    reducer: Any
    _is_fitted: bool

    def __init__(self, reducer_type: ReducerType, n_components: int = 2):
        self.reducer_type = reducer_type
        self.n_components = n_components
        self.reducer = self._create_reducer()
        self._is_fitted = False

    def _create_reducer(self) -> Any:
        if self.reducer_type == ReducerType.PCA:
            return PCA(n_components=self.n_components)
        elif self.reducer_type == ReducerType.ICA:
            return FastICA(n_components=self.n_components)
        elif self.reducer_type == ReducerType.FA:
            return FactorAnalysis(n_components=self.n_components)
        elif self.reducer_type == ReducerType.LDA:
            return LDA(n_components=self.n_components)
        elif self.reducer_type == ReducerType.TSNE:
            return TSNE(n_components=self.n_components, init="pca", learning_rate="auto", random_state=42)
        elif self.reducer_type == ReducerType.UMAP:
            return UMAP(n_components=self.n_components, random_state=42, transform_seed=42)
        elif self.reducer_type == ReducerType.LLE:
            return LocallyLinearEmbedding(n_components=self.n_components, n_neighbors=max(5, self.n_components + 1))
        elif self.reducer_type == ReducerType.MDS:
            return MDS(n_components=self.n_components, n_init=4, init="classical_mds", random_state=42)
        else:
            raise ValueError(f"Unsupported reducer type: {self.reducer_type}")

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None, return_df: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Fit the dimensionality reduction model and return transformed data."""
        if self.reducer_type == ReducerType.LDA:
            if y is None:
                raise ValueError("LDA requires class labels 'y' for fitting.")
            if self.n_components > len(np.unique(y)) - 1:
                raise ValueError("n_components must be less than number of classes - 1 for LDA.")
            reduced_data = self.reducer.fit_transform(X, y)
        else:
            reduced_data = self.reducer.fit_transform(X)

        self._is_fitted = True
        if return_df and isinstance(X, pd.DataFrame):
            return pd.DataFrame(reduced_data, columns=[f"component_{i + 1}" for i in range(self.n_components)], index=X.index)
        return reduced_data

    def transform(self, X: pd.DataFrame | np.ndarray, return_df: bool = False) -> np.ndarray | pd.DataFrame:
        """Transform new data X using the already fitted reducer."""
        if not self._is_fitted:
            raise RuntimeError("The reducer has not been fitted yet. Call 'fit_transform' first.")

        if self.reducer_type in NON_TRANSFORMABLES:
            raise RuntimeError(
                f"The reducer type {self.reducer_type} does not support 'transform' after fitting. Use fit_transform instead."
            )

        reduced_data = self.reducer.transform(X)
        if return_df and isinstance(X, pd.DataFrame):
            return pd.DataFrame(reduced_data, columns=[f"component_{i + 1}" for i in range(self.n_components)], index=X.index)
        return reduced_data


__all__ = [
    "ScalingType",
    "ReducerType",
    "NON_TRANSFORMABLES",
    "DataScaler",
    "DataDimensionsReducer",
]
