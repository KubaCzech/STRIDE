"""Low-level matplotlib rendering primitives for stream visualisation.

These are private helpers used internally by :mod:`.stream`.  They are not
part of the public ``stride.plotting`` API.
"""

import numpy as np
import matplotlib.pyplot as plt


def _plot_violin(ax: plt.Axes, values: list, positions: list, colors: list, alphas: list) -> None:
    """Draw horizontal violin plots on *ax*."""
    quantiles_list = [[0.25, 0.75] for _ in values]

    parts = ax.violinplot(
        values, positions=positions, vert=False, showmeans=False, showextrema=False, showmedians=True, quantiles=quantiles_list
    )

    if "bodies" in parts:
        for pc, color, alpha in zip(parts["bodies"], colors, alphas):
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

    if "cmedians" in parts:
        parts["cmedians"].set_edgecolor("white")
        parts["cmedians"].set_linewidth(2)
        parts["cmedians"].set_alpha(1.0)

    if "cquantiles" in parts:
        parts["cquantiles"].set_edgecolor("black")
        parts["cquantiles"].set_linestyle("--")
        parts["cquantiles"].set_linewidth(1)
        parts["cquantiles"].set_alpha(0.6)


def _plot_box(ax: plt.Axes, values: list, positions: list, colors: list, alphas: list) -> None:
    """Draw horizontal box plots on *ax*."""
    bplot = ax.boxplot(values, positions=positions, vert=False, patch_artist=True, widths=0.6)

    for patch, color, alpha in zip(bplot["boxes"], colors, alphas):
        patch.set_facecolor(color)
        patch.set_alpha(alpha)
        patch.set_edgecolor("black")

    for median in bplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)


def _plot_scatter(ax: plt.Axes, data_dict: list, valid_indices: list) -> None:
    """Draw jittered scatter plots on *ax*."""
    for i in valid_indices:
        d = data_dict[i]
        y_jitter = np.random.normal(d["position"], 0.08, size=len(d["values"]))

        ax.scatter(
            d["values"],
            y_jitter,
            alpha=d["alpha"],
            s=20,
            label=d["label"] if i == 0 else "",
            color=d["color"],
            edgecolors="none",
        )


def _plot_distribution_comparison(
    ax: plt.Axes,
    viz_type: str,
    data_dict: list,
    feature_name: str,
) -> None:
    """Plot distributions for comparison on a single axis.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axis.
    viz_type : str
        ``'violin'``, ``'box'``, or ``'scatter'``.
    data_dict : list[dict]
        Each entry has keys ``'label'``, ``'values'``, ``'color'``, ``'alpha'``, ``'position'``.
    feature_name : str
        Feature name used as the x-axis label.
    """
    positions = [d["position"] for d in data_dict]
    values = [d["values"] for d in data_dict]

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
        _plot_scatter(ax, data_dict, valid_indices)

    ax.set_yticks(positions)
    ax.set_yticklabels([d["label"] for d in data_dict])
    ax.set_xlabel(feature_name)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
