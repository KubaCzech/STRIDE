import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def _plot_distribution_comparison(ax, viz_type, data_dict, feature_name):
    """
    Helper to plot distributions for comparison on a single axis.

    Args:
        ax: Matplotlib axis
        viz_type: 'violin', 'box', or 'scatter'
        data_dict: list of dicts with keys ['label', 'values', 'color', 'alpha', 'position']
        feature_name: Name of the feature (for x-label)
    """
    positions = [d["position"] for d in data_dict]
    values = [d["values"] for d in data_dict]

    # Filter out empty data to prevent plotting errors
    valid_indices = [i for i, v in enumerate(values) if len(v) > 0]
    if not valid_indices:
        return

    valid_values = [values[i] for i in valid_indices]
    valid_positions = [positions[i] for i in valid_indices]
    valid_colors = [data_dict[i]["color"] for i in valid_indices]
    valid_alphas = [data_dict[i]["alpha"] for i in valid_indices]

    if viz_type == "violin":
        _plot_violin(ax, valid_values, valid_positions, valid_colors, valid_alphas)
    elif viz_type == "box":
        _plot_box(ax, valid_values, valid_positions, valid_colors, valid_alphas)
    else:  # scatter
        # Scatter needs original data_dict for labels logic in loop, or simplified
        _plot_scatter(ax, data_dict, valid_indices)

    # Formatting
    ax.set_yticks(positions)
    ax.set_yticklabels([d["label"] for d in data_dict])
    ax.set_xlabel(feature_name)
    ax.grid(True, alpha=0.3, axis="x")

    # Remove top/right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_violin(ax, values, positions, colors, alphas):
    # Horizontal violin plot
    # Improvement: Add quantiles (25%, 75%) to see the IQR, and Median.
    quantiles_list = [[0.25, 0.75] for _ in values]

    parts = ax.violinplot(
        values, positions=positions, vert=False, showmeans=False, showextrema=False, showmedians=True, quantiles=quantiles_list
    )

    # Style the violin bodies
    if "bodies" in parts:
        for pc, color, alpha in zip(parts["bodies"], colors, alphas):
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

    # Style the medians (Solid White for contrast)
    if "cmedians" in parts:
        parts["cmedians"].set_edgecolor("white")
        parts["cmedians"].set_linewidth(2)
        parts["cmedians"].set_alpha(1.0)

    # Style the quantiles (Dashed Black for IQR boundaries)
    if "cquantiles" in parts:
        parts["cquantiles"].set_edgecolor("black")
        parts["cquantiles"].set_linestyle("--")
        parts["cquantiles"].set_linewidth(1)
        parts["cquantiles"].set_alpha(0.6)


def _plot_box(ax, values, positions, colors, alphas):
    # Horizontal box plot
    bplot = ax.boxplot(values, positions=positions, vert=False, patch_artist=True, widths=0.6)

    for patch, color, alpha in zip(bplot["boxes"], colors, alphas):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor("black")

    for median in bplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)


def _plot_scatter(ax, data_dict, valid_indices):
    # Scatter with jitter on Y-axis (which represents the groups)
    for i in valid_indices:
        d = data_dict[i]
        # Add random jitter to the Y-position
        y_jitter = np.random.normal(d["position"], 0.08, size=len(d["values"]))

        ax.scatter(
            d["values"],
            y_jitter,
            alpha=d["alpha"],
            s=20,
            label=d["label"] if i == 0 else "",  # Label once if needed
            color=d["color"],
            edgecolors="none",
        )


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
):
    """
    Creates a figure showing feature distributions, grouped by Class and Window.

    Layout:
        One subplot per feature.
        Inside each subplot, distributions are stacked horizontally:
        - For each class: (Before vs After)
    """
    unique_classes = sorted(np.unique(np.concatenate([y_before, y_after])))
    n_classes = len(unique_classes)

    # Create taller subplots to accommodate stacked distributions per feature
    fig, axes = plt.subplots(n_features, 1, figsize=(10, (1.5 * n_classes + 0.5) * n_features), squeeze=False)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)

    for i in range(n_features):
        ax = axes[i, 0]
        feat_name = feature_names[i]

        plot_data = []
        for idx, cls in enumerate(unique_classes):
            color = class_colors[cls]
            base_pos = idx * 2.5  # Space groups apart

            # --- Before ---
            plot_data.append(
                {
                    "label": f"Class {cls}\n(Before)",
                    "values": X_before[y_before == cls, i],
                    "color": color,
                    "alpha": 0.3,  # Faded
                    "position": base_pos,
                }
            )
            # --- After ---
            plot_data.append(
                {
                    "label": f"Class {cls}\n(After)",
                    "values": X_after[y_after == cls, i],
                    "color": color,
                    "alpha": 0.8,  # Solid
                    "position": base_pos + 1,
                }
            )

        _plot_distribution_comparison(ax, viz_type, plot_data, feat_name)
        ax.set_title(f"{feat_name} Distributions", fontsize=12)

    plt.tight_layout()
    # Adjust top margin for the main title
    fig.subplots_adjust(top=0.94)
    return fig


def plot_class_distribution(class_dist_before, class_dist_after, class_colors, title="Class Distribution"):
    """
    Creates a figure showing class distribution.
    """
    fig, (ax_class_before, ax_class_after) = plt.subplots(1, 2, figsize=(12, 6))
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.0)

    classes = sorted(class_colors.keys())
    labels = [f"Class {c}" for c in classes]
    colors = [class_colors[c] for c in classes]

    # Class distributions - Before
    ax_class_before.bar(labels, [class_dist_before.get(c, 0) for c in classes], color=colors, alpha=0.7, edgecolor="black")
    ax_class_before.set_ylabel("Proportion")
    ax_class_before.set_title("Before")
    ax_class_before.set_ylim([0, 1])
    ax_class_before.grid(True, alpha=0.3, axis="y")

    # Class distributions - After
    ax_class_after.bar(labels, [class_dist_after.get(c, 0) for c in classes], color=colors, alpha=0.7, edgecolor="black")
    ax_class_after.set_ylabel("Proportion")
    ax_class_after.set_title("After")
    ax_class_after.set_ylim([0, 1])
    ax_class_after.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    # Increased top margin
    fig.subplots_adjust(top=0.88)
    return fig

    plt.tight_layout()
    # Increased top margin
    fig.subplots_adjust(top=0.88)
    return fig


def plot_feature_space(n_features, feature_names, X_before, X_after, y_before, y_after, class_colors, title="Feature Space"):
    """
    Creates a figure showing feature space (1D, 2D, or PCA).
    """
    fig, (ax_fs_before, ax_fs_after) = plt.subplots(1, 2, figsize=(14, 7))
    fs_title_suffix = ""

    unique_classes = sorted(class_colors.keys())

    if n_features == 1:
        # 1D plot (Histogram)
        for cls in unique_classes:
            ax_fs_before.hist(X_before[y_before == cls], bins=30, alpha=0.5, label=f"Class {cls}", color=class_colors[cls])
            ax_fs_after.hist(X_after[y_after == cls], bins=30, alpha=0.5, label=f"Class {cls}", color=class_colors[cls])
        ax_fs_before.set_xlabel(feature_names[0])
        ax_fs_after.set_xlabel(feature_names[0])
        ax_fs_before.set_ylabel("Frequency")
        ax_fs_after.set_ylabel("Frequency")

    elif n_features == 2:
        # 2D plot
        for cls in unique_classes:
            ax_fs_before.scatter(
                X_before[y_before == cls, 0],
                X_before[y_before == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
            ax_fs_after.scatter(
                X_after[y_after == cls, 0],
                X_after[y_after == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
        ax_fs_before.set_xlabel(feature_names[0])
        ax_fs_before.set_ylabel(feature_names[1])
        ax_fs_after.set_xlabel(feature_names[0])
        ax_fs_after.set_ylabel(feature_names[1])

    else:
        # > 2D plot (Use PCA)
        pca = PCA(n_components=2, random_state=42)
        # Fit PCA on combined data to ensure same projection
        X_combined = np.concatenate([X_before, X_after])
        X_2d = pca.fit_transform(X_combined)

        X_2d_before = X_2d[: len(X_before)]
        X_2d_after = X_2d[len(X_before) :]

        for cls in unique_classes:
            ax_fs_before.scatter(
                X_2d_before[y_before == cls, 0],
                X_2d_before[y_before == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
            ax_fs_after.scatter(
                X_2d_after[y_after == cls, 0],
                X_2d_after[y_after == cls, 1],
                alpha=0.5,
                s=20,
                label=f"Class {cls}",
                color=class_colors[cls],
            )
        ax_fs_before.set_xlabel("Principal Component 1")
        ax_fs_before.set_ylabel("Principal Component 2")
        ax_fs_after.set_xlabel("Principal Component 1")
        ax_fs_after.set_ylabel("Principal Component 2")
        fs_title_suffix = " (PCA)"

    ax_fs_before.set_title("Before")
    ax_fs_before.legend()
    ax_fs_before.grid(True, alpha=0.3)

    ax_fs_after.set_title("After")
    ax_fs_after.legend()
    ax_fs_after.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title + fs_title_suffix, fontsize=16, fontweight="bold", y=1.0)

    plt.tight_layout()
    # Increased top margin
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
):
    """
    Visualize the data stream for two specific windows.

    Creates four separate figures:
    1. Feature distributions over time
    2. Feature-target relationships
    3. Class distributions
    4. 2D (PCA) Feature Space

    Parameters
    ----------
    X : array-like (n_samples, n_features)
        Feature matrix
    y : array-like (n_samples,)
        Binary class labels
    window_before_start : int
        Start index for the first window
    window_after_start : int
        Start index for the second window
    window_length : int
        Length of the windows
    feature_names : list
        Names of features
    """
    # Convert to numpy if pandas
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    # Define windows
    start_before = window_before_start
    end_before = start_before + window_length

    start_after = window_after_start
    end_after = start_after + window_length

    # Slice data for analysis
    X_before = X[start_before:end_before]
    y_before = y[start_before:end_before]

    X_after = X[start_after:end_after]
    y_after = y[start_after:end_after]

    n_features = X.shape[1]

    # Create time steps for plotting
    # time_steps_before = np.arange(start_before, start_before + len(X_before))
    # time_steps_after = np.arange(start_after, start_after + len(X_after))

    # Identify unique classes
    unique_classes = np.unique(np.concatenate([y_before, y_after]))

    # Calculate class distributions
    class_dist_before = {cls: np.mean(y_before == cls) for cls in unique_classes}
    class_dist_after = {cls: np.mean(y_after == cls) for cls in unique_classes}

    # Colors for the classes
    # If more than 2 classes, use a colormap
    if len(unique_classes) <= 2:
        class_colors = {0: "#FF6B6B", 1: "#4ECDC4"}
        # Handle cases where classes are not 0 and 1
        class_colors = {cls: class_colors.get(i, plt.cm.tab10(i)) for i, cls in enumerate(sorted(unique_classes))}
    else:
        cmap = plt.cm.get_cmap("tab10")
        class_colors = {cls: cmap(i % 10) for i, cls in enumerate(sorted(unique_classes))}

    figs = []

    # 1. Feature vs Target Relationship
    figs.append(
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
        )
    )

    # 3. Class Distribution
    figs.append(plot_class_distribution(class_dist_before, class_dist_after, class_colors, title=title_class_dist))

    # 4. Feature Space
    figs.append(
        plot_feature_space(
            n_features, feature_names, X_before, X_after, y_before, y_after, class_colors, title=title_feat_space
        )
    )

    return figs
