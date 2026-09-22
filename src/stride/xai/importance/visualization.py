import numpy as np
import matplotlib.pyplot as plt


def visualize_drift_importance(analysis_result, feature_names, plot_type="bar", include_target=True):
    """
    Visualize the results of drift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_drift_importance
    feature_names : list
        Names of features
    plot_type : str, default='bar'
        Type of plot to show: 'bar' or 'box'.
    include_target : bool, default=True
        Whether the target was included in analysis (for title).
    """
    fi_result = analysis_result["importance_result"]
    importance_mean = analysis_result["importance_mean"]
    importance_std = analysis_result["importance_std"]
    n_features = len(feature_names)

    drift_type = "Concept Drift (X+Y)" if include_target else "Data Drift (X-only)"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Feature Importance for {drift_type} Analysis - {fi_result['method']}", fontsize=14, fontweight="bold")

    if plot_type == "bar":
        # Bar plot
        x_pos = np.arange(n_features)
        ax.bar(x_pos, importance_mean, yerr=importance_std, color="#e74c3c", alpha=0.8, edgecolor="black", capsize=5)
        ax.set_ylabel(f"Importance Score ({fi_result['method']})")
        ax.set_title(f"Feature Importance for Detecting Time-Period ({drift_type})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

    elif plot_type == "box":
        # Box plot
        importances = fi_result["importances"]

        # Handle cases where importances might not be (n_features, n_samples)
        if importances.ndim == 2 and importances.shape[0] != n_features and importances.shape[1] == n_features:
            importances = importances.T

        # Ensure importances is (n_features, n_samples) before boxplot
        if importances.shape[0] == n_features:
            bp = ax.boxplot(
                [importances[i] for i in range(n_features)],
                tick_labels=feature_names,
                patch_artist=True,
                notch=True,
                showmeans=True,
            )

            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            for patch in bp["boxes"]:
                patch.set_facecolor("#e74c3c")
                patch.set_alpha(0.7)
            ax.set_ylabel(f"{fi_result['method']} Score")
            ax.set_xlabel("Features")
            ax.set_title(f"Distribution of {fi_result['method']} Scores")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                "Boxplot not available (SHAP/LIME 1D result)",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

    fig.tight_layout()
    return fig


def visualize_predictive_importance_shift(analysis_result, feature_names, plot_type="bar"):
    """
    Visualize the results of predictive importance shift analysis.

    Parameters
    ----------
    analysis_result : dict
        Result dictionary from compute_predictive_importance_shift
    feature_names : list
        Names of features
    plot_type : str, default='bar'
        Type of plot to show: 'bar' or 'box'.
    """
    fi_before = analysis_result["fi_before"]
    fi_after = analysis_result["fi_after"]
    n_features = len(feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Predictive Feature Importance (NN Before vs After Drift) - {fi_before['method']}", fontsize=14, fontweight="bold"
    )

    if plot_type == "bar":
        # Bar comparison
        x_pos = np.arange(n_features)
        width = 0.35
        ax.bar(
            x_pos - width / 2,
            fi_before["importances_mean"],
            width,
            yerr=fi_before["importances_std"],
            label="NN (Trained BEFORE Drift)",
            color="#1abc9c",
            alpha=0.8,
            edgecolor="black",
            capsize=5,
        )
        ax.bar(
            x_pos + width / 2,
            fi_after["importances_mean"],
            width,
            yerr=fi_after["importances_std"],
            label="NN (Trained AFTER Drift)",
            color="#f39c12",
            alpha=0.8,
            edgecolor="black",
            capsize=5,
        )
        ax.set_ylabel(f"Importance Score ({fi_before['method']})")
        ax.set_title("Feature Importance for Predicting Target Class")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    elif plot_type == "box":
        # Side-by-side box plots
        # Handle cases where importances might not be (n_features, n_samples)
        importances_before = fi_before["importances"]
        if (
            importances_before.ndim == 2
            and importances_before.shape[0] != n_features
            and importances_before.shape[1] == n_features
        ):
            importances_before = importances_before.T

        importances_after = fi_after["importances"]
        if (
            importances_after.ndim == 2
            and importances_after.shape[0] != n_features
            and importances_after.shape[1] == n_features
        ):
            importances_after = importances_after.T

        # Calculate positions for side-by-side boxplots
        positions_before = np.arange(n_features) * 2 - 0.2
        positions_after = np.arange(n_features) * 2 + 0.2

        # Check if boxplot can be drawn (importances is 2D and aligned)
        if importances_before.shape[0] == n_features and importances_after.shape[0] == n_features:
            bp1 = ax.boxplot(
                [importances_before[i] for i in range(n_features)],
                positions=positions_before,
                widths=0.3,
                patch_artist=True,
                showmeans=True,
                notch=True,
            )
            bp2 = ax.boxplot(
                [importances_after[i] for i in range(n_features)],
                positions=positions_after,
                widths=0.3,
                patch_artist=True,
                showmeans=True,
                notch=True,
            )

            for patch in bp1["boxes"]:
                patch.set_facecolor("#1abc9c")
                patch.set_alpha(0.7)
            for patch in bp2["boxes"]:
                patch.set_facecolor("#f39c12")
                patch.set_alpha(0.7)

            ax.set_ylabel(f"{fi_before['method']} Score")
            ax.set_xlabel("Features")
            ax.set_title(f"Distribution of {fi_before['method']} Scores")
            ax.set_xticks(np.arange(n_features) * 2)
            ax.set_xticklabels(feature_names)
            ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Before Drift", "After Drift"], loc="upper right")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                "Boxplot not available (SHAP/LIME 1D result)",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

    fig.tight_layout()
    return fig
