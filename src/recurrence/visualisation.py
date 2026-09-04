import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.recurrence.full_window_storage import FullWindowStorage


def _compute_global_prototype_range(storage, windows_to_compare, max_prototypes):
    """Compute global min/max across all prototypes for consistent y-axis scaling."""
    global_min = float("inf")
    global_max = -float("inf")

    for window_nr in windows_to_compare:
        x, y, prototypes, explainer = storage.get_window_data(window_nr)
        for class_name in set(y):
            proto_list = prototypes[class_name]
            # Limit prototypes if max_prototypes is set
            if max_prototypes is not None:
                proto_list = proto_list[:max_prototypes]

            for prototype in proto_list:
                values = [v for _, v in sorted(prototype.items(), key=lambda x: x[0])]
                local_min = min(values)
                local_max = max(values)
                global_min = min(global_min, local_min)
                global_max = max(global_max, local_max)

    margin = 0.05 * (global_max - global_min)
    return global_min - margin, global_max + margin


def _plot_class_prototypes(ax, prototypes, class_name, max_prototypes, used_min, used_max):
    """Plot prototypes for a specific class in a subplot."""
    proto_list = prototypes[class_name]
    num_prototypes = len(proto_list)

    # Limit prototypes if max_prototypes is set
    if max_prototypes is not None:
        proto_list = proto_list[:max_prototypes]

    for prototype in proto_list:
        items = sorted(prototype.items(), key=lambda x: x[0])
        feature_names = [str(k) for k, _ in items]
        feature_values = [v for _, v in items]

        ax.plot(feature_names, feature_values)
        ax.tick_params(axis="x", labelrotation=45)

    # Show count info
    count_text = f"prototype_count={num_prototypes}"
    if max_prototypes is not None and num_prototypes > max_prototypes:
        count_text += f" (showing {max_prototypes})"

    ax.text(0.02, 0.95, count_text, transform=ax.transAxes, fontsize=9, verticalalignment="top")

    ax.set_ylim(used_min, used_max)


def plot_prototype_comparison(storage: FullWindowStorage, windows_to_compare: list[int], max_prototypes: int = None):
    """Compare prototypes across multiple windows.

    Args:
        storage: FullWindowStorage instance
        windows_to_compare: List of window numbers to compare
        max_prototypes: Maximum number of prototypes to display per class (None = show all)
    """
    # Compute global min/max for consistent y-axis scaling
    used_min, used_max = _compute_global_prototype_range(storage, windows_to_compare, max_prototypes)

    # Get classes from first window for subplot setup
    x, y, prototypes, _ = storage.get_window_data(windows_to_compare[0])
    classes_sorted = sorted(list(set(prototypes.keys())))

    fig, ax = plt.subplots(len(classes_sorted), len(windows_to_compare), figsize=(12, 6), sharey=True)

    # Plot prototypes for each window and class
    for col, window_nr in enumerate(windows_to_compare):
        x, y, prototypes, explainer = storage.get_window_data(window_nr)

        for row, class_name in enumerate(classes_sorted):
            _plot_class_prototypes(ax[row, col], prototypes, class_name, max_prototypes, used_min, used_max)

    # Column titles (windows)
    for col, window_nr in enumerate(windows_to_compare):
        ax[0, col].set_title(f"Window {window_nr}")

    # Row titles (classes)
    for row, class_name in enumerate(classes_sorted):
        ax[row, 0].set_ylabel(f"Class {class_name}", rotation=90, labelpad=10)

    plt.tight_layout()
    plt.show()


def plot_cluster_timeline(labels, drift_locations=None, title="Cluster Timeline"):
    """Plot cluster timeline showing concept changes over time using Plotly.

    Args:
        labels: Array of cluster labels for each window
        drift_locations: List of window indices where drifts were detected
        title: Plot title
    """
    # Create color mapping for clusters
    unique_labels = sorted([x for x in set(labels) if x != -1])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]

    color_map = {-1: "#000000"}  # Black for outliers
    for i, label in enumerate(unique_labels):
        color_map[label] = colors[i % len(colors)]

    # Create figure
    fig = go.Figure()

    # Group consecutive windows with same label for continuous bars
    segments = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append({"start": start_idx, "end": i - 1, "label": current_label, "width": i - start_idx})
            current_label = labels[i]
            start_idx = i

    # Add the last segment
    segments.append({"start": start_idx, "end": len(labels) - 1, "label": current_label, "width": len(labels) - start_idx})

    # Add bars for each segment
    shown_labels = set()
    for seg in segments:
        label = seg["label"]
        label_name = "Outlier" if label == -1 else f"Concept {label}"
        show_legend = label not in shown_labels
        shown_labels.add(label)

        # Calculate center position for the bar
        center_x = (seg["start"] + seg["end"]) / 2

        fig.add_trace(
            go.Bar(
                x=[center_x],
                y=[1],
                width=[seg["width"]],
                marker_color=color_map[label],
                name=label_name,
                showlegend=show_legend,
                hovertemplate=f"Windows {seg['start']}-{seg['end']}<br>{label_name}<extra></extra>",
                legendgroup=str(label),
            )
        )

    # Add drift markers as shapes (vertical lines)
    shapes = []
    annotations = []
    if drift_locations:
        for drift_idx in drift_locations:
            # Add vertical line
            shapes.append(
                dict(
                    type="line", x0=drift_idx - 0.5, x1=drift_idx - 0.5, y0=0, y1=1, line=dict(color="red", width=5), yref="y"
                )
            )
            # Add annotation
            annotations.append(
                dict(
                    x=drift_idx - 0.5,
                    y=1.0,
                    text=f"DRIFT at window {drift_idx}",
                    showarrow=False,
                    font=dict(size=20, color="black"),
                    yref="paper",
                    yanchor="bottom",
                )
            )

    fig.update_layout(
        title=title,
        title_font=dict(size=20, color="black"),
        xaxis_title="Window",
        yaxis_title="",
        showlegend=True,
        height=300,
        barmode="overlay",
        bargap=0,
        yaxis=dict(showticklabels=False, range=[0, 1]),
        xaxis=dict(range=[-0.5, len(labels) - 0.5]),
        shapes=shapes,
        annotations=annotations,
        plot_bgcolor="white",
        margin=dict(t=80, b=40, l=40, r=40),
    )

    return fig


def plot_distance_to_all_windows(
    storage: FullWindowStorage, window_nr: int, drift_locations=None, measure="centroid_displacement", k_median=1
):
    """Plot distance from one window to all other windows.

    Args:
        storage: FullWindowStorage instance
        window_nr: Window number to compare against all others
        drift_locations: List of window indices where drifts were detected
        measure: Distance measure to use
        k_median: Width of median filter for smoothing (1 = no smoothing)
    """
    from src.recurrence.methods import median_mask

    data_to_plot = storage.compare_window_to_all(window_nr, measure=measure)
    data_to_plot = [float(x) for x in data_to_plot]

    if k_median > 1:
        data_to_plot = median_mask(data_to_plot, k_median)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(data_to_plot, linewidth=2)
    ax.axvline(x=window_nr, color="red", linestyle="--", linewidth=2, label="Inspected window")

    # Mark drift locations
    if drift_locations:
        for i, drift in enumerate(drift_locations):
            # Only add label to first drift line for legend
            label = "Detected drifts" if i == 0 else None
            ax.axvline(x=drift - 0.5, color="orange", linestyle=":", alpha=0.7, linewidth=3, label=label)

    ax.set_xticks(range(0, len(data_to_plot), 5))
    ax.set_xlabel("Window")
    ax.set_ylabel("Distance")
    ax.set_title(f"Distance from Window {window_nr} to All Other Windows")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _find_samples_closest_to_class(x_dicts, y_list, prototypes, class_name):
    """Find samples whose closest prototype is from the specified class."""
    samples_closest_to_this_class = []
    for i, sample in enumerate(x_dicts):
        min_dist = float("inf")
        closest_class = None

        # Check distance to all prototypes across all classes
        for cls, cls_protos in prototypes.items():
            for proto in cls_protos:
                common_features = set(proto.keys()) & set(sample.keys())
                if common_features:
                    dist = np.sqrt(sum((proto[f] - sample[f]) ** 2 for f in common_features))
                    if dist < min_dist:
                        min_dist = dist
                        closest_class = cls

        # If closest prototype is from the current class, include this sample
        if closest_class == class_name:
            samples_closest_to_this_class.append(i)

    return samples_closest_to_this_class


def _find_samples_closest_to_prototype(prototype, proto_list, x_dicts, samples_closest_to_class):
    """Find samples whose closest prototype within the class is the given prototype."""
    closest_samples = []

    for sample_idx in samples_closest_to_class:
        sample = x_dicts[sample_idx]

        # Calculate distance to current prototype
        common_features = set(prototype.keys()) & set(sample.keys())
        if common_features:
            dist_to_this = np.sqrt(sum((prototype[f] - sample[f]) ** 2 for f in common_features))
        else:
            dist_to_this = float("inf")

        # Check if this prototype is closest among all prototypes in the same class
        is_closest = True
        for other_proto in proto_list:
            if other_proto is prototype:
                continue
            common_features = set(other_proto.keys()) & set(sample.keys())
            if common_features:
                other_dist = np.sqrt(sum((other_proto[f] - sample[f]) ** 2 for f in common_features))
                if other_dist < dist_to_this:
                    is_closest = False
                    break

        if is_closest:
            closest_samples.append(sample_idx)

    return closest_samples


def _compute_prototype_feature_statistics(proto_list):
    """Compute feature statistics across all prototypes in a class."""
    feature_data = {}
    for prototype in proto_list:
        for feat, val in prototype.items():
            if feat not in feature_data:
                feature_data[feat] = []
            feature_data[feat].append(val)

    stats_rows = []
    for feat in sorted(feature_data.keys()):
        values = feature_data[feat]
        stats_rows.append(
            {"Feature": feat, "Mean": np.mean(values), "Std": np.std(values), "Min": np.min(values), "Max": np.max(values)}
        )

    return pd.DataFrame(stats_rows)


def _analyze_class_prototypes(class_name, prototypes, x_dicts, y_list):
    """Analyze prototypes for a specific class."""
    print(f"\n{'-' * 60}")
    print(f"Class {class_name}")
    print(f"{'-' * 60}")

    proto_list = prototypes[class_name]
    print(f"Number of prototypes: {len(proto_list)}")

    # Count samples closest to this class
    samples_closest_to_this_class = _find_samples_closest_to_class(x_dicts, y_list, prototypes, class_name)
    print(f"{len(samples_closest_to_this_class)} samples have their closest prototype from class {class_name}\n")

    # Analyze each prototype
    for proto_idx, prototype in enumerate(proto_list):
        closest_samples = _find_samples_closest_to_prototype(prototype, proto_list, x_dicts, samples_closest_to_this_class)

        # Count classes of closest samples
        class_counts = {}
        for sample_idx in closest_samples:
            sample_class = y_list[sample_idx]
            class_counts[sample_class] = class_counts.get(sample_class, 0) + 1

        # Display info
        class_breakdown = ", ".join([f"Class {cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
        print(f"  • Prototype {proto_idx + 1}: Closest to {len(closest_samples)} samples ({class_breakdown})")

    # Feature statistics across prototypes
    if len(proto_list) > 0:
        df_stats = _compute_prototype_feature_statistics(proto_list)
        print("\n  Feature Statistics Across Prototypes:")
        print(df_stats.to_string(index=False))


def plot_window_detail(storage: FullWindowStorage, window_nr: int):
    """Display detailed information about a specific window's prototypes.

    Args:
        storage: FullWindowStorage instance
        window_nr: Window number to analyze
    """
    x, y, prototypes, explainer = storage.get_window_data(window_nr)

    # Convert to list of dicts if needed
    if isinstance(x[0], dict):
        x_dicts = list(x)
    else:
        x_dicts = x

    y_list = list(y)

    print(f"\n{'=' * 60}")
    print(f"Window {window_nr} - Detailed Analysis")
    print(f"{'=' * 60}\n")

    print(f"Total samples: {len(x_dicts)}")
    print(f"Total prototypes: {sum(len(proto_list) for proto_list in prototypes.values())}")
    print(f"Classes: {sorted(set(y_list))}\n")

    # Analyze each class
    for class_name in sorted(set(y_list)):
        _analyze_class_prototypes(class_name, prototypes, x_dicts, y_list)

    print(f"\n{'=' * 60}\n")
