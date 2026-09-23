import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, Optional, Sequence, Callable
from enum import Enum
from functools import wraps
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde, probplot

# TODO 2: legendy w histogramie, kde, ecdf
# TODO 3: wspólna oś y, jesli nie to zrobic sharey
# TODO 4: nazwy osi


class PlotOptions(Enum):
    Mean = "mean"
    Median = "median"
    Both = "both"


def plot(title: str, sharey: bool = True, palette: Optional[Sequence[str]] = None) -> Callable:
    """Decorator that wraps a plotting function into a grid of subplots
    over features and class labels.

    The decorated function is called once per subplot and must accept
    the following keyword arguments:
    - ax : matplotlib.axes.Axes
    - old_vals : pd.Series
    - new_vals : pd.Series
    - feature : str
    - cl : list
    - colors : sequence of str

    Parameters
    ----------
    title : str
        Title of the entire figure.
    sharey : bool, default=True
        Whether all subplots share the Y axis.
    palette : sequence of str, optional
        Color palette used for plots. If None, Matplotlib default colors are used.

    Returns
    -------
    Callable
        A decorator that converts a single-axes plotting function
        into a grid-based plot returning a :class:`matplotlib.figure.Figure`.
    """

    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(
            X_before: pd.DataFrame,
            y_before: Union[pd.Series, np.ndarray],
            X_after: pd.DataFrame,
            y_after: Union[pd.Series, np.ndarray],
            *args,
            save=None,
            **kwargs,
        ) -> Figure:
            if hasattr(y_before, "values"):
                y_before = y_before.values
            if hasattr(y_after, "values"):
                y_after = y_after.values

            if palette is None:
                default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            else:
                default_colors = palette
            kwargs["default_colors"] = default_colors

            features = X_before.columns
            classes = sorted([int(i) for i in set(y_before).union(set(y_after))])

            fig, axes = plt.subplots(len(features), len(classes), figsize=(6 * len(classes), 5 * len(features)), sharey=sharey)

            # normalize axes shape
            if len(features) == 1:
                axes = axes[None, :]
            if len(classes) == 1:
                axes = axes[:, None]

            for i, feature in enumerate(features):
                for j, cl in enumerate(classes):
                    ax = axes[i, j]

                    old_vals = X_before.loc[y_before == cl, feature].dropna()
                    new_vals = X_after.loc[y_after == cl, feature].dropna()

                    if old_vals.empty or new_vals.empty:
                        ax.set_visible(False)
                        continue

                    plot_func(
                        ax=ax,
                        old_vals=old_vals,
                        new_vals=new_vals,
                        feature=feature,
                        cl=cl,
                        colors=default_colors,
                        *args,
                        **kwargs,
                    )

                    if i == 0:
                        ax.set_title(f"Class {int(cl)}")
                    if j == 0:
                        ax.set_ylabel(feature)

            fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            if save is not None and isinstance(save, str):
                plt.savefig(save)
            return fig

        return wrapper

    return decorator


# 1. Boxplot
@plot("Boxplot – Before vs After")
def plot_boxplot(
    ax: plt.Axes, old_vals: pd.Series, new_vals: pd.Series, feature: str, show_: PlotOptions, save=None, **_
) -> None:
    """Draw a boxplot comparing distributions of two data blocks.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature data from first data block.
    y_before : pd.Series or np.ndarray
        Class labels from first data block.
    X_after : pd.DataFrame
        Feature data from second data block.
    y_after : pd.Series or np.ndarray
        Class labels from second data block.
    show_ : PlotOptions, default=PlotOptions.Median
        Option to show Mean, Median or Both in the boxplot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    median_color, mean_color = None, None
    legend_elements = []

    if show_ == PlotOptions.Median:
        bp = ax.boxplot([old_vals, new_vals], tick_labels=["Before", "After"])
        median_color = bp["medians"][0].get_color()
    elif show_ == PlotOptions.Mean:
        bp = ax.boxplot(
            [old_vals, new_vals],
            tick_labels=["Before", "After"],
            showmeans=True,
            meanline=True,
            medianprops=dict(visible=False),
        )
        mean_color = bp["means"][0].get_color()
    elif show_ == PlotOptions.Both:
        bp = ax.boxplot([old_vals, new_vals], tick_labels=["Before", "After"], showmeans=True, meanline=True)
        median_color = bp["medians"][0].get_color()
        mean_color = bp["means"][0].get_color()
    else:
        raise ValueError("Unsupported Boxplot option")
    ax.set_ylabel("Value of feature")

    if median_color is not None:
        legend_elements.append(Line2D([0], [0], color=median_color, lw=1, label="Median"))
    if mean_color is not None:
        legend_elements.append(Line2D([0], [0], color=mean_color, lw=1, linestyle="--", label="Mean"))
    ax.legend(handles=legend_elements)


# 2. Histogram
@plot("Histogram - Before vs After")
def plot_histogram(
    ax: plt.Axes, old_vals: pd.Series, new_vals: pd.Series, colors: Sequence[str], bins: int = 30, save=None, **_
) -> None:
    """Draw overlapping histograms for distributions of two data blocks.

    Parameters
    ----------
    X_before : pd.DataFrame
    y_before : np.ndarray or pd.Series
    X_after : pd.DataFrame
    y_after : np.ndarray or pd.Series

    Returns
    -------
    matplotlib.figure.Figure
    """
    all_vals = np.concatenate([old_vals, new_vals])
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    ax.hist(old_vals, bins=bin_edges, alpha=0.6, density=False, label="Before", color=colors[0])
    ax.hist(new_vals, bins=bin_edges, alpha=0.6, density=False, label="After", color=colors[1])
    both_color = np.mean([to_rgb(colors[0]), to_rgb(colors[1])], axis=0)

    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label="Before"),
        Patch(facecolor=colors[1], alpha=0.6, label="After"),
        Patch(facecolor=both_color, alpha=0.6, label="Both"),
    ]
    ax.set_xlabel("Value of feature")
    ax.set_ylabel("Number of appearences")
    ax.legend(handles=legend_elements)


# 3. Violin Plot
@plot("Violin Plot – Before vs After")
def plot_violin(ax: plt.Axes, old_vals: pd.Series, new_vals: pd.Series, show_: PlotOptions, **_) -> None:
    """Draw a violin plot comparing distributions of two data blocks.

    Parameters
    ----------
    X_before : pd.DataFrame
    y_before : np.ndarray or pd.Series
    X_after : pd.DataFrame
    y_after : np.ndarray or pd.Series
    show_ : PlotOptions, default=PlotOptions.Median

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = [old_vals, new_vals]
    if show_ == PlotOptions.Median:
        ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=True)
    elif show_ == PlotOptions.Mean:
        ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=False)
    else:
        raise ValueError("Unsupported Violin plot option")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Before", "After"])


# 4. QQ Plot
@plot("QQ Plot – Before vs After")
def plot_qq(ax: plt.Axes, old_vals: pd.Series, new_vals: pd.Series, **_) -> None:
    """Draw a QQ-plot comparing distributions of two data blocks.

    Parameters
    ----------
    X_before : pd.DataFrame
    y_before : np.ndarray or pd.Series
    X_after : pd.DataFrame
    y_after : np.ndarray or pd.Series

    Returns
    -------
    matplotlib.figure.Figure
    """
    ax.set_ylim(-0.05, 1.05)
    probplot(old_vals, dist="norm", plot=ax)
    probplot(new_vals, dist="norm", plot=ax)


# 5. KDE Plot
@plot("KDE Plot – Before vs After")
def plot_kde(ax: plt.Axes, old_vals: pd.Series, new_vals: pd.Series, colors: Sequence[str], **_) -> None:
    """Draw a Kernel Density Estimation plot comparing distributions of two data blocks.

    Parameters
    ----------
    X_before : pd.DataFrame
    y_before : np.ndarray or pd.Series
    X_after : pd.DataFrame
    y_after : np.ndarray or pd.Series

    Returns
    -------
    matplotlib.figure.Figure
    """
    kde_old = gaussian_kde(old_vals)
    kde_new = gaussian_kde(new_vals)

    x_min = min(min(old_vals), min(new_vals))
    x_max = max(max(old_vals), max(new_vals))
    x_range = np.linspace(x_min, x_max, 1000)

    ax.plot(x_range, kde_old(x_range), label="Before", color=colors[0])
    ax.plot(x_range, kde_new(x_range), label="After", color=colors[1])
    ax.legend()


# 6. ECDF Plot
@plot("ECDF – Before vs After")
def plot_ecdf(ax: plt.Axes, old_vals: pd.Series, new_vals: pd.Series, colors: Sequence[str], **_) -> None:
    """Draw an Empirical CDF plot comparing distributions of two data blocks.

    Parameters
    ----------
    X_before : pd.DataFrame
    y_before : np.ndarray or pd.Series
    X_after : pd.DataFrame
    y_after : np.ndarray or pd.Series

    Returns
    -------
    matplotlib.figure.Figure
    """

    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    ax.plot(*ecdf(old_vals), label="Before", color=colors[0])
    ax.plot(*ecdf(new_vals), label="After", color=colors[1])
    ax.legend()
