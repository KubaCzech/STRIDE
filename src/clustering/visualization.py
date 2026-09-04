import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from functools import wraps
from matplotlib.patches import Patch
from typing import Sequence, Union, Callable
from matplotlib.colors import ListedColormap, BoundaryNorm
from src.common import DataDimensionsReducer, ReducerType

colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
    "#c49c94",  # light brown
    "#f7b6d2",  # light pink
    "#c7c7c7",  # light gray
    "#dbdb8d",  # light yellow-green
    "#9edae5",  # light cyan
]

color_map = {i: colors[i] for i in range(len(colors))}


def reduce_dimensions(
    n_components: int = 2,
    reducer: ReducerType = ReducerType.PCA,
) -> Callable:
    """
    Decorator to automatically reduce the dimensionality of pandas DataFrame arguments
    passed to the decorated function.

    The decorator:
    - Detects positional and keyword arguments of type `pd.DataFrame`
    - Reduces the number of columns to `n_components` if the DataFrame has more columns
    - Uses a single shared reducer (lazy initialization)
    - Applies `fit_transform` on the first occurrence of a DataFrame, then `transform` for subsequent ones

    Non-DataFrame arguments are passed through unchanged.

    Parameters
    ----------
    n_components : int, default=2
        Target number of dimensions (columns) after reduction.
        If the DataFrame has columns less than or equal to `n_components`, no reduction is performed.

    reducer : ReducerType, default=ReducerType.PCA
        Type of dimensionality reduction algorithm used by `DataDimensionsReducer` (e.g., PCA, UMAP, t-SNE).

    Returns
    -------
    Callable
        A decorator that automatically transforms DataFrame arguments before calling the function.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ddr = None  # lazy init

            def reduce_df(X):
                nonlocal ddr
                if not isinstance(X, pd.DataFrame):
                    return X

                if X.shape[1] <= n_components:
                    return X

                if ddr is None:
                    ddr = DataDimensionsReducer(reducer_type=reducer, n_components=n_components)
                    return ddr.fit_transform(X, return_df=True)

                return ddr.transform(X, return_df=True)

            # positional args
            new_args = [reduce_df(arg) for arg in args]

            # keyword args
            for k, v in kwargs.items():
                kwargs[k] = reduce_df(v)

            return func(*new_args, **kwargs)

        return wrapper

    return decorator


def _plot_clusters(X: pd.DataFrame, labels: Sequence[Union[float, int]], title: str) -> None:
    """
    Plot clusters with different colors for 2D data.

    Parameters
    ----------
    X : pd.DataFrame
        Data points to plot.
    labels : Sequence[int or float]
        Cluster labels for each data point.
    title : str
        Title of the plot. Usually indicating before or after drift.

    Notes
    -----
    Global color_map is used to ensure consistent coloring across plots.
    Function used internally by other plotting functions.
    """
    if hasattr(X, "values"):
        X = X.values
    if hasattr(labels, "values"):
        labels = labels.values

    # Ensure labels are integers
    if labels.dtype.kind == "f":
        labels = labels.astype(int)

    unique_labels = sorted(np.unique(labels))

    for ul in unique_labels:
        color = color_map.get(ul, "#333333")  # Default to dark gray if missing
        plt.scatter(X[labels == ul, 0], X[labels == ul, 1], color=color, label=f"Cluster {ul}", s=30)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()


@reduce_dimensions()
def plot_drift_clustered(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    _y_before: Sequence[Union[int, float]],
    _y_after: Sequence[Union[int, float]],
    labels_before: Sequence[Union[int, float]],
    labels_after: Sequence[Union[int, float]],
    show: bool = False,
    save=None,
) -> None:
    """
    Plot clusters from first and second data block to visualize drift for 2D data.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature values from first data block.
    X_after : pd.DataFrame
        Feature values from second data block.
    _y_before : Sequence[Union[int, float]]
        Class labels for first data block.
    _y_after : Sequence[Union[int, float]]
        Class labels for first second block.
    labels_before : Sequence[int or float]
        Cluster labels for first data block.
    labels_after : Sequence[int or float]
        Cluster labels for second data block.
    show : bool, default=False
        If True, display the plot immediately. Otherwise, the plot can be
        further modified or returned for later display.
    """
    plt.figure(figsize=(12, 5))

    # Normalize inputs
    if hasattr(labels_before, "values"):
        labels_before = labels_before.values
    if hasattr(labels_after, "values"):
        labels_after = labels_after.values

    if labels_before.dtype.kind == "f":
        labels_before = labels_before.astype(int)
    if labels_after.dtype.kind == "f":
        labels_after = labels_after.astype(int)

    # First data block
    plt.subplot(1, 2, 1)
    _plot_clusters(X_before, labels_before, "Before Drift")

    # Second data block
    plt.subplot(1, 2, 2)
    _plot_clusters(X_after, labels_after, "After Drift")

    plt.tight_layout()
    if save is not None and isinstance(save, str):
        plt.savefig(save)
    if show:
        plt.show()


@reduce_dimensions()
def plot_clusters_by_class(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: Sequence[Union[int, float]],
    y_after: Sequence[Union[int, float]],
    cluster_labels_before: Sequence[Union[int, float]],
    cluster_labels_after: Sequence[Union[int, float]],
    show: bool = False,
    save=None,
) -> None:
    """
    Plot clusters separated by class labels to visualize drift per class (not overall) for 2D data.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature values from first data block.
    X_after : pd.DataFrame
        Feature values from second data block.
    y_before : Sequence[int or float]
        Class labels for first data block.
    y_after : Sequence[int or float]
        Class labels for second data block.
    cluster_labels_before : Sequence[int or float]
        Cluster labels for first data block.
    cluster_labels_after : Sequence[int or float]
        Cluster labels for second data block.
    show : bool, default=False
        If True, display the plot immediately. Otherwise, the plot can be
        further modified or returned for later display.

    Notes
    -----
    Creates a subplot for each class, showing clusters before and after drift.
    """
    classes = sorted(set(y_before).union(set(y_after)))
    n_classes = len(classes)

    fig, axes = plt.subplots(n_classes, 2, figsize=(12, 5 * n_classes))

    # If there is only 1 class, axes is not 2D; this line fixes that
    if n_classes == 1:
        axes = np.array([axes])

    for row, cl in enumerate(classes):
        # first data block
        ax_before = axes[row, 0]
        plt.sca(ax_before)

        mask_before = np.array(y_before) == cl
        _plot_clusters(X_before[mask_before], cluster_labels_before[mask_before], title=f"Before Drift – Class {int(cl)}")

        # second data block
        ax_after = axes[row, 1]
        plt.sca(ax_after)

        mask_after = np.array(y_after) == cl
        _plot_clusters(X_after[mask_after], cluster_labels_after[mask_after], title=f"After Drift – Class {int(cl)}")

    plt.tight_layout()
    if save is not None and isinstance(save, str):
        plt.savefig(save)
    if show:
        plt.show()


@reduce_dimensions()
def plot_centers_shift(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    _y_before: Sequence[Union[int, float]],
    _y_after: Sequence[Union[int, float]],
    cluster_labels_old: Sequence[Union[int, float]],
    cluster_labels_new: Sequence[Union[int, float]],
    show: bool = False,
    save=None,
) -> None:
    """
    Plot shifts of cluster centroids between two data blocks for 2D data.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature values from first data block.
    X_after : pd.DataFrame
        Feature values from second data block.
    y_before : Sequence[int or float]
        Class labels for first data block.
    y_after : Sequence[int or float]
        Class labels for second data block.
    cluster_labels_before : Sequence[int or float]
        Cluster labels for first data block.
    cluster_labels_after : Sequence[int or float]
        Cluster labels for second data block.
    show : bool, default=False
        If True, display the plot immediately. Otherwise, the plot can be
        further modified or returned for later display.
    """
    unique_labels_old = set(cluster_labels_old)
    unique_labels_new = set(cluster_labels_new)

    all_labels = sorted(list(unique_labels_old.union(unique_labels_new)))

    # TODO: zmienic rozmiar
    plt.figure(figsize=(8, 8))

    # For legend purposes
    plt.scatter([], [], marker="x", color="black", label="Center (before)")
    plt.scatter([], [], marker="o", color="black", label="Center (after)")

    for label in all_labels:
        plt.scatter([], [], marker="o", color=color_map[label], label=f"Cluster {label}")

    for label in all_labels:
        center_old = None
        center_new = None

        if label in unique_labels_old:
            center_old = X_before[cluster_labels_old == label].mean().values

        if label in unique_labels_new:
            center_new = X_after[cluster_labels_new == label].mean().values

        # cluster shifted
        if center_old is not None and center_new is not None:
            plt.plot(
                [center_old[0], center_new[0]],
                [center_old[1], center_new[1]],
                linewidth=2,
                linestyle="--",
                color=color_map[label],
            )
            plt.scatter(*center_old, marker="x", color=color_map[label], s=60)
            plt.scatter(*center_new, marker="o", color=color_map[label], s=60)

        # cluster disappeared (only old)
        elif center_old is not None:
            plt.scatter(*center_old, marker="x", color=color_map[label], s=60)

        # cluster appeared (only new)
        elif center_new is not None:
            plt.scatter(*center_new, marker="o", color=color_map[label], s=60)

    plt.title("Cluster centroid shifts")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.legend()
    if save is not None and isinstance(save, str):
        plt.savefig(save)
    if show:
        plt.show()


def plot_clustering_heatmap(stats_shifts, threshold, show=False, save=None):
    def make_dataframe_from_dict(dictt):
        df = (
            pd.DataFrame.from_dict(dictt, orient="index")
            .stack()
            .apply(pd.Series)
            .reset_index()
            .rename(columns={"level_0": "cluster", "level_1": "feature"})
        )
        return df

    stats_shifts_flattened = make_dataframe_from_dict(stats_shifts)
    stats_shifts_flattened["X"] = (
        "Cluster" + stats_shifts_flattened["cluster"].astype(int).astype(str) + ":" + stats_shifts_flattened["feature"]
    )
    # stats = ["min", "mean", "median", "max", "std"]
    stats = ["mean", "median", "std"]

    heatmap_data = stats_shifts_flattened.set_index("X")[stats].T

    hm_na = heatmap_data.isna()
    hm_value = abs(heatmap_data) > threshold

    heatmap_encoded = np.zeros(heatmap_data.shape)
    heatmap_encoded[hm_value] = 1
    heatmap_encoded[hm_na] = 2

    heatmap_to_plot = pd.DataFrame(heatmap_encoded, columns=heatmap_data.columns)
    heatmap_to_plot.index = heatmap_data.index

    cmap = ListedColormap(["#ff6b6b", "#6bff81", "#7E7B7B"])

    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(14, 5))
    sns.heatmap(heatmap_to_plot, cmap=cmap, norm=norm, cbar=False, linewidths=0.5, linecolor="white")

    plt.xlabel("(Cluster, Feature)")
    plt.ylabel("Descriptive statistics")
    plt.title("Threshold and Missing-Value Diagnostics Across Clusters and Features")

    legend_elements = [
        Patch(facecolor="#ff6b6b", edgecolor="black", label="Within threshold"),
        Patch(facecolor="#6bff81", edgecolor="black", label=f"|value| > {threshold}"),
        Patch(facecolor="#7E7B7B", edgecolor="black", label="Missing value (NaN)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", title="Legend", frameon=True)
    if save is not None and isinstance(save, str):
        plt.savefig(save)
    if show:
        plt.show()
