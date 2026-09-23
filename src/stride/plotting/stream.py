"""High-level stream visualisation functions.

Public API:
    - :func:`plot_feature_target_relationship`
    - :func:`plot_class_distribution`
    - :func:`plot_feature_space`
    - :func:`visualize_data_stream`
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from ._renderers import _plot_distribution_comparison


def plot_feature_target_relationship(
    X,
    n_features,
    feature_names,
    X_before,
    X_after,
    y_before,
    y_after,
    class_colors,
    title="Feature vs Target Relationship",
    viz_type="violin",
) -> Figure:
    """Create a figure showing feature distributions grouped by class and window.

    Each subplot corresponds to one feature.  Within each subplot, distributions
    are stacked horizontally — one pair (Before / After) per class.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Full feature matrix (used only for shape information).
    n_features : int
        Number of features.
    feature_names : list[str]
        Feature names.
    X_before : np.ndarray
        Feature matrix for the *before* window.
    X_after : np.ndarray
        Feature matrix for the *after* window.
    y_before : np.ndarray
        Class labels for the *before* window.
    y_after : np.ndarray
        Class labels for the *after* window.
    class_colors : dict
        Mapping ``{class_id: color_string}``.
    title : str, default="Feature vs Target Relationship"
        Figure title.
    viz_type : str, default="violin"
        Distribution plot type: ``'violin'``, ``'box'``, or ``'scatter'``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    unique_classes = sorted(np.unique(np.concatenate([y_before, y_after])))
    n_classes = len(unique_classes)

    fig, axes = plt.subplots(n_features, 1, figsize=(10, (1.5 * n_classes + 0.5) * n_features), squeeze=False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)

    for i in range(n_features):
        ax = axes[i, 0]
        feat_name = feature_names[i]

        plot_data = []
        for idx, cls in enumerate(unique_classes):
            color = class_colors[cls]
            base_pos = idx * 2.5

            plot_data.append(
                {
                    "label": f"Class {cls}\n(Before)",
                    "values": X_before[y_before == cls, i],
                    "color": color,
                    "alpha": 0.3,
                    "position": base_pos,
                }
            )
            plot_data.append(
                {
                    "label": f"Class {cls}\n(After)",
                    "values": X_after[y_after == cls, i],
                    "color": color,
                    "alpha": 0.8,
                    "position": base_pos + 1,
                }
            )

        _plot_distribution_comparison(ax, viz_type, plot_data, feat_name)
        ax.set_title(f"{feat_name} Distributions", fontsize=12)

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    return fig


def plot_class_distribution(class_dist_before, class_dist_after, class_colors, title="Class Distribution") -> Figure:
    """Create a figure showing class distribution before and after a window.

    Parameters
    ----------
    class_dist_before : dict
        ``{class_id: proportion}`` for the *before* window.
    class_dist_after : dict
        ``{class_id: proportion}`` for the *after* window.
    class_colors : dict
        Mapping ``{class_id: color_string}``.
    title : str, default="Class Distribution"
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(12, 6))
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)

    classes = sorted(class_colors.keys())
    labels = [f"Class {c}" for c in classes]
    colors = [class_colors[c] for c in classes]

    ax_before.bar(labels, [class_dist_before.get(c, 0) for c in classes], color=colors, alpha=0.7, edgecolor="black")
    ax_before.set_ylabel("Proportion")
    ax_before.set_title("Before")
    ax_before.set_ylim([0, 1])
    ax_before.grid(True, alpha=0.3, axis="y")

    ax_after.bar(labels, [class_dist_after.get(c, 0) for c in classes], color=colors, alpha=0.7, edgecolor="black")
    ax_after.set_ylabel("Proportion")
    ax_after.set_title("After")
    ax_after.set_ylim([0, 1])
    ax_after.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


def plot_feature_space(
    n_features, feature_names, X_before, X_after, y_before, y_after, class_colors, title="Feature Space"
) -> Figure:
    """Create a figure showing the feature space for two windows (1D, 2D, or PCA).

    Parameters
    ----------
    n_features : int
        Number of features.
    feature_names : list[str]
        Feature names.
    X_before : np.ndarray
        Feature matrix for the *before* window.
    X_after : np.ndarray
        Feature matrix for the *after* window.
    y_before : np.ndarray
        Class labels for the *before* window.
    y_after : np.ndarray
        Class labels for the *after* window.
    class_colors : dict
        Mapping ``{class_id: color_string}``.
    title : str, default="Feature Space"
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 7))
    fs_title_suffix = ""

    unique_classes = sorted(class_colors.keys())

    if n_features == 1:
        for cls in unique_classes:
            ax_before.hist(X_before[y_before == cls], bins=30, alpha=0.5, label=f"Class {cls}", color=class_colors[cls])
            ax_after.hist(X_after[y_after == cls], bins=30, alpha=0.5, label=f"Class {cls}", color=class_colors[cls])
        ax_before.set_xlabel(feature_names[0])
        ax_after.set_xlabel(feature_names[0])
        ax_before.set_ylabel("Frequency")
        ax_after.set_ylabel("Frequency")

    elif n_features == 2:
        for cls in unique_classes:
            ax_before.scatter(
                X_before[y_before == cls, 0],
                X_before[y_before == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
            ax_after.scatter(
                X_after[y_after == cls, 0],
                X_after[y_after == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
        ax_before.set_xlabel(feature_names[0])
        ax_before.set_ylabel(feature_names[1])
        ax_after.set_xlabel(feature_names[0])
        ax_after.set_ylabel(feature_names[1])

    else:
        pca = PCA(n_components=2, random_state=42)
        X_combined = np.concatenate([X_before, X_after])
        X_2d = pca.fit_transform(X_combined)

        X_2d_before = X_2d[: len(X_before)]
        X_2d_after = X_2d[len(X_before) :]

        for cls in unique_classes:
            ax_before.scatter(
                X_2d_before[y_before == cls, 0],
                X_2d_before[y_before == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
            ax_after.scatter(
                X_2d_after[y_after == cls, 0],
                X_2d_after[y_after == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
        ax_before.set_xlabel("Principal Component 1")
        ax_before.set_ylabel("Principal Component 2")
        ax_after.set_xlabel("Principal Component 1")
        ax_after.set_ylabel("Principal Component 2")
        fs_title_suffix = " (PCA)"

    ax_before.set_title("Before")
    ax_before.legend()
    ax_before.grid(True, alpha=0.3)

    ax_after.set_title("After")
    ax_after.legend()
    ax_after.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title + fs_title_suffix, fontsize=16, fontweight="bold", y=1.0)

    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    return fig


def visualize_data_stream(
    X,
    y,
    window_before_start,
    window_after_start,
    window_length,
    feature_names,
    title_feat_target="Feature vs Target Relationship",
    title_class_dist="Class Distribution",
    title_feat_space="Feature Space",
    viz_type="violin",
) -> list[Figure]:
    """Visualize the data stream for two specific windows.

    Creates three separate figures:

    1. Feature-target relationships (violin / box / scatter per feature per class)
    2. Class distributions (bar chart before vs after)
    3. Feature space (1D histogram, 2D scatter, or PCA projection)

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Full feature matrix.
    y : array-like, shape (n_samples,)
        Class labels.
    window_before_start : int
        Start index (inclusive) of the *before* window.
    window_after_start : int
        Start index (inclusive) of the *after* window.
    window_length : int
        Number of samples in each window.
    feature_names : list[str]
        Feature names.
    title_feat_target : str
        Title for the feature-target figure.
    title_class_dist : str
        Title for the class-distribution figure.
    title_feat_space : str
        Title for the feature-space figure.
    viz_type : str, default="violin"
        Distribution plot type: ``'violin'``, ``'box'``, or ``'scatter'``.

    Returns
    -------
    list[matplotlib.figure.Figure]
        List of three figures in the order described above.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    X_before = X[window_before_start : window_before_start + window_length]
    y_before = y[window_before_start : window_before_start + window_length]
    X_after = X[window_after_start : window_after_start + window_length]
    y_after = y[window_after_start : window_after_start + window_length]

    n_features = X.shape[1]
    unique_classes = np.unique(np.concatenate([y_before, y_after]))

    class_dist_before = {cls: np.mean(y_before == cls) for cls in unique_classes}
    class_dist_after = {cls: np.mean(y_after == cls) for cls in unique_classes}

    if len(unique_classes) <= 2:
        _base = {0: "#FF6B6B", 1: "#4ECDC4"}
        class_colors = {cls: _base.get(i, plt.cm.tab10(i)) for i, cls in enumerate(sorted(unique_classes))}
    else:
        cmap = plt.cm.get_cmap("tab10")
        class_colors = {cls: cmap(i % 10) for i, cls in enumerate(sorted(unique_classes))}

    return [
        plot_feature_target_relationship(
            X,
            n_features,
            feature_names,
            X_before,
            X_after,
            y_before,
            y_after,
            class_colors,
            title=title_feat_target,
            viz_type=viz_type,
        ),
        plot_class_distribution(class_dist_before, class_dist_after, class_colors, title=title_class_dist),
        plot_feature_space(
            n_features, feature_names, X_before, X_after, y_before, y_after, class_colors, title=title_feat_space
        ),
    ]
