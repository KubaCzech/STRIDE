import numpy as np
import pandas as pd

from typing import Union, Any

from enum import Enum
from umap import UMAP

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE, LocallyLinearEmbedding, MDS
from sklearn.decomposition import FastICA, FactorAnalysis, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class ScalingType(Enum):
    MinMax = "min_max"
    Standard = "standard"


class ReducerType(Enum):
    # For linear data
    PCA = "PCA"
    ICA = "ICA"
    FA = "FA"
    LDA = "LDA"

    # Non-linear
    TSNE = "t_SNE"
    UMAP = "UMAP"
    LLE = "LLE"
    MDS = "MDS"
    # Autoencoders/Isomap/kernel PCA could be added here


NON_TRANSFORMABLES = {ReducerType.TSNE, ReducerType.LLE, ReducerType.MDS}


class DataScaler:
    """
    Wrapper around sklearn scalers with a unified interface.

    Parameters
    ----------
    scaling_type : ScalingType
        Type of scaling to apply.

    Attributes
    ----------
    scaler : TransformerMixin
        The scaler instance.
    _is_fitted : bool
        Flag indicating if the scaler has been fitted.
    """

    scaler: TransformerMixin
    _is_fitted: bool

    def __init__(self, scaling_type: ScalingType):
        if scaling_type == ScalingType.MinMax:
            self.scaler = MinMaxScaler()
        elif scaling_type == ScalingType.Standard:
            self.scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaling type")

        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the scaler on the given data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        """
        self.scaler.fit(X)
        self._is_fitted = True

    def transform(self, X: pd.DataFrame, return_df: bool = True) -> np.ndarray:
        """
        Transform data using the fitted scaler.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Scaled data.

        Notes
        -----
            One scaler can be fitted once but used many times

        Raises
        ------
        RuntimeError
            If scaler has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit' before 'transform'.")

        if return_df:
            return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame, return_df: bool = True) -> np.ndarray:
        """
        Fit the scaler and transform the data.

        Parameters
        ----------
        X : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Scaled data.
        """
        self.fit(X)

        if return_df:
            return pd.DataFrame(self.transform(X), columns=X.columns, index=X.index)
        return self.transform(X)


class DataDimensionsReducer:
    """
    Unified interface for dimensionality reduction.

    This class provides a single interface to perform both linear and nonlinear
    dimensionality reduction using popular algorithms like PCA, ICA, FA, LDA,
    t-SNE, UMAP, LLE, and MDS.

    Attributes
    ----------
    reducer_type : ReducerType
        The type of dimensionality reduction algorithm to use.
    n_components : int
        The number of components/dimensions to reduce the data to.
    reducer : Any
        The instantiated reducer object (e.g., PCA, TSNE, UMAP).
    _is_fitted : bool
        Flag indicating whether the reducer has been fitted.

    Notes
    -----
    - TSNE, MDS, and LLE are embedding-only methods and do NOT support transform().
    - UMAP supports transform() after fitting.
    - LDA requires class labels `y` for fitting.
    - This class is NOT fully sklearn Pipeline-compatible due to embedding-only methods.
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
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray] = None, return_df: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Fit the dimensionality reduction model to X (and y for LDA) and return the transformed data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to reduce.
        y : pd.Series or np.ndarray, optional
            Class labels for LDA. Required if reducer_type is LDA.
        return_df : bool, default False
            If True, returns a pandas DataFrame; otherwise returns a NumPy array.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The reduced data.

        Raises
        ------
        ValueError
            If LDA is used without labels or n_components is too high.
        """
        if self.reducer_type == ReducerType.LDA:
            if y is None:
                raise ValueError("LDA requires class labels 'y' for fitting.")

            if self.n_components > len(np.unique(y)) - 1:
                raise ValueError("n_components must be less than number of classes - 1 for LDA.")
            reduced_data = self.reducer.fit_transform(X, y)
        else:
            reduced_data = self.reducer.fit_transform(X)

        self._is_fitted = True
        if return_df:
            return pd.DataFrame(reduced_data, columns=[f"component_{i + 1}" for i in range(self.n_components)])
        return reduced_data

    def transform(self, X: pd.DataFrame, return_df: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform new data X using the already fitted reducer.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.
        return_df : bool, default False
            If True, returns a pandas DataFrame; otherwise returns a NumPy array.

        Returns
        -------
        np.ndarray or pd.DataFrame
            The transformed data.

        Raises
        ------
        RuntimeError
            If called on a reducer that has not been fitted.
        RuntimeError
            If called on embedding-only reducers (TSNE, MDS, LLE) that do not support transform().
        """
        if not self._is_fitted:
            raise RuntimeError("The reducer has not been fitted yet. Call 'fit_transform' first.")

        if self.reducer_type in NON_TRANSFORMABLES:
            raise RuntimeError(
                f"The reducer type {self.reducer_type} does not support 'transform' after fitting. Use fit_transform instead."
            )

        reduced_data = self.reducer.transform(X)
        if return_df:
            return pd.DataFrame(reduced_data, columns=[f"component_{i + 1}" for i in range(self.n_components)])
        return reduced_data
