import traceback
import numpy as np
import pandas as pd
import streamlit as st
from typing import Sequence
import matplotlib.pyplot as plt

from stride.xai.clustering.clustering import ClusterBasedDriftDetector
from stride.xai.clustering.visualization import (
    plot_clusters_by_class,
    plot_drift_clustered,
    plot_centers_shift,
)

# Criteria in the order used by ClusterBasedDriftDetector for `weights` and `effective_weights`.
CRITERIA = ["Clusters appeared/disappeared", "Descriptive statistics", "Centroid shift", "Spread"]

DRIFT_STYLE = "background-color: rgba(231, 76, 60, 0.25)"


def _display_intro():
    """
    Renders the introductory text for the Clustering Analysis tab.
    """

    st.header("Clustering Analysis")
    st.markdown(
        """
    This tab uses cluster-based methods to detect and analyze concept drift.
    It compares the data distribution and cluster structures between two windows:
    **Before Drift** and **After Drift**.

    Within each class, X-Means finds the clusters of the *Before* window, and the
    *After* window is assigned to the nearest of those clusters, so both windows are
    compared on the same partition. Drift is assessed through four criteria: clusters
    that appeared or disappeared, centroid shifts, changes of per-cluster statistics,
    and changes of cluster spread. A permutation test (random re-splits of the pooled
    data) decides whether the observed drift strength is larger than chance.
    """
    )


def _render_detector_config() -> dict:
    """
    Renders the detector settings and returns them as keyword arguments of the detector.
    """
    with st.expander("Clustering Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            k_init = st.number_input(
                "Initial clusters per class (k_init)",
                min_value=1,
                max_value=10,
                value=2,
                help="Number of clusters X-Means starts from in every class.",
            )
            k_max = st.number_input(
                "Maximum clusters per class (k_max)",
                min_value=int(k_init),
                max_value=30,
                value=max(10, int(k_init)),
                help="Upper limit of the number of clusters X-Means may find in every class.",
            )
        with col2:
            n_permutations = st.number_input(
                "Permutations",
                min_value=0,
                max_value=500,
                value=50,
                step=10,
                help=(
                    "Random re-splits of the pooled windows used to calibrate the decision. "
                    "More permutations give a finer p-value (the smallest attainable is 1 / (permutations + 1)) "
                    "but take proportionally longer. 0 disables the test and compares the strength with 0.5."
                ),
            )
            permutation_alpha = st.select_slider(
                "Significance level",
                options=[0.01, 0.05, 0.1],
                value=0.05,
                help="Drift is reported when the permutation p-value is below this level.",
                disabled=n_permutations == 0,
            )
        with col3:
            matching = st.radio(
                "Cluster matching",
                options=["reference", "independent"],
                format_func=lambda m: {
                    "reference": "Assign to reference clusters",
                    "independent": "Independent clusterings (legacy)",
                }[m],
                help=(
                    "Reference: the After window is assigned to the clusters of the Before window. "
                    "Independent: both windows are clustered separately and centroids are paired "
                    "with the Hungarian algorithm."
                ),
            )
            use_alpha = st.checkbox(
                "Require significant per-cluster changes",
                value=False,
                help=(
                    "Additionally require a two-sample test (α = 0.01) that accounts for the cluster size before a "
                    "centroid, statistic or spread change counts. Removes spurious flags in small clusters at the "
                    "cost of some sensitivity."
                ),
            )
    return {
        "k_init": int(k_init),
        "k_max": int(k_max),
        "matching": matching,
        "n_permutations": int(n_permutations),
        "permutation_alpha": float(permutation_alpha),
        "alpha": 0.01 if use_alpha else None,
    }


def _display_drift_summary(detector: ClusterBasedDriftDetector):
    """
    Displays the drift decision, drift strength and permutation p-value.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        A detector on which ``detect()`` has been called.
    """
    if detector.drift_flag:
        st.success("**Drift detected** between the two windows.")
    else:
        st.info("**No drift detected** between the two windows.")

    col_flag, col_strength, col_p = st.columns(3)
    col_flag.metric("Drift Detected", "Yes" if detector.drift_flag else "No")
    col_strength.metric(
        "Drift Strength",
        f"{detector.strength_of_drift:.3f}",
        help="Weighted share of triggered checks across the four criteria (0 = none, 1 = all).",
    )
    if detector.p_value is not None:
        col_p.metric(
            "Permutation p-value",
            f"{detector.p_value:.3f}",
            help=(
                f"Share of {detector.n_permutations} random re-splits of the pooled data (within each class) whose "
                f"drift strength is at least as high as the observed one. Drift is reported when "
                f"p < {detector.permutation_alpha}."
            ),
        )
        null_q95 = float(np.quantile(detector.null_strengths, 0.95))
        st.caption(
            f"Without drift, 95% of re-splits reach a strength of at most {null_q95:.3f}; "
            f"the observed strength is {detector.strength_of_drift:.3f}."
        )
    else:
        col_p.metric("Decision Threshold", f"{detector.decision_thr:.2f}")


def _class_name(label) -> str:
    """Readable class label: integral numbers without decimals, anything else as text."""
    if isinstance(label, (bool, np.bool_)):
        return str(label)
    if isinstance(label, (int, float, np.integer, np.floating)) and float(label).is_integer():
        return str(int(label))
    return str(label)


def _criteria_counts(class_details: dict) -> tuple[np.ndarray, np.ndarray]:
    """Triggered and evaluated checks of one class, in the order of ``CRITERIA``."""
    stats = [
        flag
        for cluster in class_details["desc_stats_changes"].values()
        for feature in cluster.values()
        for flag in feature.values()
    ]
    groups = [
        [class_details["nr_of_clusters"]],
        stats,
        list(class_details["centroid_shift"].values()),
        list(class_details["avg_distance_to_center"].values()),
    ]
    triggered = np.array([sum(bool(f) for f in g) for g in groups])
    evaluated = np.array([len(g) for g in groups])
    return triggered, evaluated


def _config_details_dataframe(details: dict, weights: Sequence[float]) -> pd.DataFrame:
    """
    Summarises the checks of every criterion per class and their contribution to the strength.

    Parameters
    ----------
    details : dict
        Detailed drift information per class (``detector.drift_details``).
    weights : Sequence[float]
        Effective criterion weights (``detector.effective_weights``).

    Returns
    -------
    pd.DataFrame
        One row per criterion: weight, triggered/evaluated checks per class and the
        contribution to the drift strength. The contributions sum to the drift strength.
    """
    counts = {cl: _criteria_counts(d) for cl, d in details.items()}
    triggered = np.sum([t for t, _ in counts.values()], axis=0)
    evaluated = np.sum([e for _, e in counts.values()], axis=0)

    df = pd.DataFrame({"Criterion": CRITERIA, "Weight": np.asarray(weights, dtype=float)})
    for cl, (t, e) in counts.items():
        df[f"Class {_class_name(cl)}"] = [f"{ti}/{ei}" if ei else "–" for ti, ei in zip(t, e)]
    df["Contribution"] = triggered / np.maximum(evaluated, 1) * np.asarray(weights, dtype=float)
    return df


def _display_drift_details(details: dict, weights: Sequence[float]):
    """
    Displays the triggered checks per class and criterion.

    Parameters
    ----------
    details : dict
        Detailed drift information per class.
    weights : Sequence[float]
        Effective weights of the criteria.
    """
    st.subheader("Drift Details by Class")

    details_df = _config_details_dataframe(details, weights)

    st.dataframe(
        details_df,
        column_config={
            "Criterion": st.column_config.TextColumn("Criterion", help="Drift criterion"),
            "Weight": st.column_config.NumberColumn(
                "Weight",
                help="Weight of the criterion, renormalised over criteria with at least one evaluated check.",
                format="%.3f",
            ),
            **{
                col: st.column_config.TextColumn(col, help="Triggered / evaluated checks")
                for col in details_df.columns
                if col.startswith("Class ")
            },
            "Contribution": st.column_config.NumberColumn(
                "Contribution",
                help="Weight × share of triggered checks; the contributions sum to the drift strength.",
                format="%.4f",
            ),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Appeared and disappeared clusters count only towards the first criterion; centroid, statistics and "
        "spread are compared for clusters present in both windows. “–” means the criterion had nothing to evaluate."
    )


def _display_visualization(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: np.ndarray,
    y_after: np.ndarray,
    labels_old: Sequence[int],
    labels_new: Sequence[int],
    viz_type: str = "clusters by class",
):
    """
    Draws one of the cluster plots into the current matplotlib figure.

    X_before : pd.DataFrame
        Feature matrix for 'before' data block
    X_after : pd.DataFrame
        Feature matrix for 'after' data block
    y_before : np.ndarray
        Target variable for 'before' data block
    y_after : np.ndarray
        Target variable for 'after' data block
    labels_old : Sequence[int]
        Cluster labels for 'before' data block
    labels_new : Sequence[int]
        Cluster labels for 'after' data block
    viz_type : str
        Type of visualization to display
    """
    plot_func = CLUSTER_PLOT_FUNCTIONS.get(viz_type, plot_clusters_by_class)
    plot_func(X_before, X_after, y_before, y_after, labels_old, labels_new)


CLUSTER_PLOT_FUNCTIONS = {
    "clusters by class": plot_clusters_by_class,
    "clusters overall": plot_drift_clustered,
    "centroid shifts (plot)": plot_centers_shift,
}


# TABULAR DISPLAY FUNCTIONS
def _cluster_overview_dataframe(detector: ClusterBasedDriftDetector) -> pd.DataFrame:
    """
    One row per cluster: class, size in both windows, status, centroid shift and spread.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        A detector on which ``detect()`` has been called.
    """
    y_old, y_new = np.asarray(detector.y_old), np.asarray(detector.y_new)
    centroid_flags = {k: v for d in detector.drift_details.values() for k, v in d["centroid_shift"].items()}
    spread_flags = {k: v for d in detector.drift_details.values() for k, v in d["avg_distance_to_center"].items()}

    rows = []
    for k, shift in sorted(detector.cluster_shifts.items()):
        in_old = detector.cluster_labels_old == k
        in_new = detector.cluster_labels_new == k
        tracked = isinstance(shift, dict)
        rows.append(
            {
                "Cluster": int(k),
                "Class": _class_name((y_old[in_old] if in_old.any() else y_new[in_new])[0]),
                "Status": "tracked" if tracked else shift,
                "Samples Before": int(in_old.sum()),
                "Samples After": int(in_new.sum()),
                "Centroid Shift": shift["euclidean_distance"] if tracked else None,
                "Centroid Drift": bool(centroid_flags.get(k, False)),
                "Spread Before": detector.avg_distance_old.get(k),
                "Spread After": detector.avg_distance_new.get(k),
                "Spread Change": detector.avg_distance_shift.get(k) * 100 if tracked else None,
                "Spread Drift": bool(spread_flags.get(k, False)),
            }
        )
    return pd.DataFrame(rows)


def _display_cluster_overview_table(detector: ClusterBasedDriftDetector):
    """
    Displays the status, size, centroid shift and spread of every cluster.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        A detector on which ``detect()`` has been called.
    """
    df = _cluster_overview_dataframe(detector)
    styled = df.style.map(
        lambda s: DRIFT_STYLE if s in ("appeared", "disappeared") else "",
        subset=["Status"],
    )
    st.dataframe(
        styled,
        column_config={
            "Cluster": st.column_config.NumberColumn("Cluster", help="Global cluster ID, shared by both windows"),
            "Class": st.column_config.TextColumn("Class"),
            "Status": st.column_config.TextColumn(
                "Status",
                help=(
                    "tracked: present in both windows; appeared: a region of the After window not explained by the "
                    "Before clusters; disappeared: receives significantly fewer After samples than expected"
                ),
            ),
            "Samples Before": st.column_config.NumberColumn("Samples Before"),
            "Samples After": st.column_config.NumberColumn("Samples After"),
            "Centroid Shift": st.column_config.NumberColumn(
                "Centroid Shift",
                help=f"Euclidean distance between the centroids (standardised features); threshold {detector.thr_centroid_shift:.3f}",
                format="%.3f",
            ),
            "Centroid Drift": st.column_config.CheckboxColumn("Centroid Drift"),
            "Spread Before": st.column_config.NumberColumn(
                "Spread Before", help="Mean distance of the samples to their centroid", format="%.3f"
            ),
            "Spread After": st.column_config.NumberColumn("Spread After", format="%.3f"),
            "Spread Change": st.column_config.NumberColumn(
                "Spread Change",
                help=f"Relative change of the spread; threshold ±{detector.thr_avg_distance_to_center_change:.0%}",
                format="%+.1f%%",
            ),
            "Spread Drift": st.column_config.CheckboxColumn("Spread Drift"),
        },
        width="stretch",
        hide_index=True,
    )


def _stats_shift_dataframe(detector: ClusterBasedDriftDetector) -> pd.DataFrame:
    """Change of the per-cluster statistics of clusters present in both windows, in σ."""
    rows = []
    for k, shift in sorted(detector.cluster_shifts.items()):
        if not isinstance(shift, dict):
            continue
        for feature in detector.columns:
            stats = detector.stats_shifts[k][feature]
            rows.append({"Cluster": int(k), "Feature": str(feature), **{s: stats[s] for s in ("mean", "median", "std")}})
    return pd.DataFrame(rows, columns=["Cluster", "Feature", "mean", "median", "std"])


def _display_stats_shift_table(detector: ClusterBasedDriftDetector):
    """
    Displays the change of the per-cluster mean, median and standard deviation.

    Shifts are absolute changes of standardised features, i.e. expressed in
    standard deviations (σ) of each feature.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing statistical shift information.
    """
    df = _stats_shift_dataframe(detector)
    if df.empty:
        st.info("No cluster is present in both windows, so per-cluster statistics are not compared.")
        return

    thr = detector.thr_desc_stats
    st.caption(
        f"Change of each statistic between the windows for clusters present in both, in standard deviations (σ) "
        f"of the feature. Highlighted: |change| > {thr} σ."
    )
    styled = df.style.map(lambda v: DRIFT_STYLE if pd.notna(v) and abs(v) > thr else "", subset=["mean", "median", "std"])
    st.dataframe(
        styled,
        column_config={
            "Cluster": st.column_config.NumberColumn("Cluster"),
            "Feature": st.column_config.TextColumn("Feature"),
            **{s: st.column_config.NumberColumn(f"Δ {s}", format="%+.3f σ") for s in ("mean", "median", "std")},
        },
        width="stretch",
        hide_index=True,
    )


def _display_centroids_table(detector: ClusterBasedDriftDetector):
    """
    Displays the centroids of every cluster in both windows (standardised features).

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        The drift detector containing cluster centroids.
    """
    columns = [str(c) for c in detector.columns]
    rows = []
    for cluster_id in sorted(set(detector.centers_old) | set(detector.centers_new)):
        for window, centroids in [("Before", detector.centers_old), ("After", detector.centers_new)]:
            centroid = centroids.get(cluster_id)
            values = [None] * len(columns) if centroid is None else [float(v) for v in centroid]
            rows.append({"Cluster": int(cluster_id), "Window": window, **dict(zip(columns, values))})

    st.caption("Centroids in standardised feature space (mean 0, standard deviation 1 in the Before window).")
    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Cluster": st.column_config.NumberColumn("Cluster"),
            "Window": st.column_config.TextColumn("Window"),
            **{c: st.column_config.NumberColumn(c, format="%.3f") for c in columns},
        },
        width="stretch",
        hide_index=True,
    )


def _plot_permutation_test(detector: ClusterBasedDriftDetector):
    """Histogram of the drift strength under random re-splits with the observed strength."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(detector.null_strengths, bins=15, color="#95a5a6", edgecolor="white", label="Random re-splits (no drift)")
    ax.axvline(
        detector.strength_of_drift,
        color="#c0392b",
        linewidth=2,
        label=f"Observed strength (p = {detector.p_value:.3f})",
    )
    ax.set_xlabel("Drift strength")
    ax.set_ylabel("Number of re-splits")
    ax.set_title("Permutation test")
    ax.legend()
    fig.tight_layout()
    return fig


CLUSTER_TABLE_FUNCTIONS = {
    "cluster overview (table)": _display_cluster_overview_table,
    "statistics shifts (table)": _display_stats_shift_table,
    "centroids (table)": _display_centroids_table,
}


def _display(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: np.ndarray,
    y_after: np.ndarray,
    detector: ClusterBasedDriftDetector,
):
    """
    Displays clustering analysis results including visualizations and tables.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix for 'before' window
    X_after : pd.DataFrame
        Feature matrix for 'after' window
    y_before : np.ndarray
        Target variable for 'before' window
    y_after : np.ndarray
        Target variable for 'after' window
    detector : ClusterBasedDriftDetector
        The drift detector containing clustering results.
    """
    st.subheader("Clustering Visualization")

    labels_old = detector.cluster_labels_old
    labels_new = detector.cluster_labels_new

    if labels_old is None or labels_new is None:
        st.warning("Cluster labels not available for plotting.")
        return

    options = list(CLUSTER_PLOT_FUNCTIONS) + list(CLUSTER_TABLE_FUNCTIONS)
    if detector.null_strengths is not None:
        options.append("permutation test (plot)")

    col_viz_options, _ = st.columns([1, 2])
    with col_viz_options:
        _type = st.selectbox(
            "Select clustering analysis view",
            options=options,
            index=0,
            key="cluster_view_selector",
            help="Choose how clustering results and drift-related information should be visualized or summarized.",
        )

    if _type in CLUSTER_PLOT_FUNCTIONS:
        if X_before.shape[1] > 2:
            st.caption("More than two features: the plot shows the first two principal components (PCA).")
        _display_visualization(X_before, X_after, y_before, y_after, labels_old, labels_new, viz_type=_type)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    elif _type == "permutation test (plot)":
        fig = _plot_permutation_test(detector)
        st.pyplot(fig)
        plt.close(fig)
    else:
        CLUSTER_TABLE_FUNCTIONS[_type](detector)


@st.cache_data(show_spinner=False)
def _run_detector(X_before, y_before, X_after, y_after, settings: dict) -> ClusterBasedDriftDetector:
    """Run the detector once per pair of windows and settings; switching views reuses the result."""
    # Fixed random_state: identical windows always give identical clusters and p-value.
    # Permutations run sequentially: worker processes would each re-import stride (and
    # TensorFlow), which slows down and eventually stalls a long-running dashboard session.
    detector = ClusterBasedDriftDetector(X_before, y_before, X_after, y_after, random_state=42, **settings)
    detector.detect()
    return detector


def render_clustering_analysis_tab(X_before: pd.DataFrame, y_before: np.ndarray, X_after: pd.DataFrame, y_after: np.ndarray):
    """
    Renders the Clustering Analysis tab.

    Parameters
    ----------
    X_before : pd.DataFrame
        Feature matrix for 'before' window
    y_before : np.ndarray
        Target variable for 'before' window
    X_after : pd.DataFrame
        Feature matrix for 'after' window
    y_after : np.ndarray
        Target variable for 'after' window
    """
    _display_intro()
    settings = _render_detector_config()

    with st.spinner("Running cluster-based drift detection (with permutation test)..."):
        try:
            detector = _run_detector(X_before, y_before, X_after, y_after, settings)

            _display_drift_summary(detector)
            _display_drift_details(detector.drift_details, detector.effective_weights)
            _display(X_before, X_after, y_before, y_after, detector)

        except Exception as e:
            st.error(f"An error occurred during drift detection: {e}")
            st.text(traceback.format_exc())
