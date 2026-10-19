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
    plot_feature_shift_heatmap,
)

# Criteria in the order used by ClusterBasedDriftDetector for `weights` and `effective_weights`.
CRITERIA = ["Clusters appeared/disappeared", "Descriptive statistics", "Centroid shift", "Spread"]

DRIFT_STYLE = "background-color: rgba(231, 76, 60, 0.25)"
CHANGE_STYLES = {
    "disappeared": DRIFT_STYLE,
    "appeared": "background-color: rgba(52, 152, 219, 0.25)",
    "changed": "background-color: rgba(243, 156, 18, 0.25)",
}

# A tracked cluster "grew" / "shrank" when its share of the class at least doubles / halves
# and changes by at least this many percentage points.
SHARE_CHANGE_MIN = 0.05
# Number of features listed per cluster in the findings table.
TOP_FEATURES = 3


def _display_intro():
    """
    Renders the introductory text for the Clustering Analysis tab.
    """

    st.header("Clustering Analysis")
    st.markdown(
        """
    This tab explains **what** changed between the **Before Drift** and **After Drift**
    windows; whether and when drift occurs is the job of the drift detectors
    (see the *Drift Detection* tab).

    Within each class, X-Means finds the clusters of the *Before* window, and the
    *After* window is assigned to the nearest of those clusters, so both windows are
    described by the same clusters. The table below lists, per cluster, whether it
    disappeared, appeared, moved, changed its spread or its share of the class, and
    which features changed the most.
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
                    "Random re-splits of the pooled windows, used to check whether the changes found are larger "
                    "than sampling noise. More permutations give a finer p-value (the smallest attainable is "
                    "1 / (permutations + 1)) but take proportionally longer. 0 disables the check."
                ),
            )
            permutation_alpha = st.select_slider(
                "Significance level",
                options=[0.01, 0.05, 0.1],
                value=0.05,
                help="The changes count as larger than sampling noise when the permutation p-value is below this level.",
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


def _display_noise_check(detector: ClusterBasedDriftDetector):
    """
    States whether the changes found exceed what random re-splits of the windows produce.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        A detector on which ``detect()`` has been called.
    """
    if detector.p_value is None:
        st.caption("The permutation check is disabled, so the changes are not compared with sampling noise.")
        return

    n_total = len(detector.null_strengths)
    n_as_large = int(np.sum(detector.null_strengths >= detector.strength_of_drift - 1e-12))
    as_large = "none" if n_as_large == 0 else f"only {n_as_large}"
    if detector.p_value < detector.permutation_alpha:
        st.caption(
            f"✅ These changes are larger than sampling noise: {as_large} of {n_total} random re-splits of the pooled "
            f"windows produced changes as extensive (permutation p = {detector.p_value:.3f})."
        )
    else:
        st.warning(
            f"Changes as extensive also appear in {n_as_large} of {n_total} random re-splits of the pooled windows "
            f"(permutation p = {detector.p_value:.3f}), so the findings above may be sampling noise rather than drift."
        )


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
    st.markdown(
        "Each criterion counts the checks that exceeded their threshold; the weighted share of triggered checks "
        "is the drift strength that the permutation check compares with random re-splits of the windows."
    )
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


# TABULAR DISPLAY FUNCTIONS
def _format_shift(feature, value: float, statistic: str | None = None) -> str:
    """``"X1 +0.92σ"``, optionally with the statistic: ``"X1 std −0.31σ"``."""
    name = f"{feature} {statistic}" if statistic and statistic != "mean" else f"{feature}"
    return f"{name} {value:+.2f}σ".replace("-", "−")


def _changed_features(detector: ClusterBasedDriftDetector, k: int, flags: dict) -> str:
    """
    Features of a tracked cluster whose statistics changed beyond the threshold, largest change first.

    The change of the mean is reported when it exceeds the threshold; otherwise the largest
    flagged change of another statistic (the median is the noisiest of them).
    """
    changes = []
    for feature, stats in flags.items():
        flagged = [s for s, flag in stats.items() if flag]
        if flagged:
            shifts = detector.stats_shifts[k][feature]
            stat = "mean" if "mean" in flagged else max(flagged, key=lambda s: abs(shifts[s]))
            changes.append((feature, stat, shifts[stat]))
    changes.sort(key=lambda c: abs(c[2]), reverse=True)
    return ", ".join(_format_shift(f, v, s) for f, s, v in changes[:TOP_FEATURES])


def _cluster_location(detector: ClusterBasedDriftDetector, k: int, cl) -> str:
    """Features in which a cluster lies furthest from the mean of its class in the Before window (beyond the threshold)."""
    center = detector.centers_new[k] if detector.centers_old[k] is None else detector.centers_old[k]
    class_mean = np.asarray(detector.X_old)[np.asarray(detector.y_old) == cl].mean(axis=0)
    offset = np.asarray(center, dtype=float) - class_mean
    order = [i for i in np.argsort(-np.abs(offset))[:TOP_FEATURES] if abs(offset[i]) > detector.thr_desc_stats]
    return ", ".join(_format_shift(detector.columns[i], offset[i]) for i in order)


def _findings_dataframe(detector: ClusterBasedDriftDetector) -> pd.DataFrame:
    """
    One row per cluster describing what changed between the windows.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        A detector on which ``detect()`` has been called.

    Returns
    -------
    pd.DataFrame
        Class, cluster, the changes found (``Changes``), the cluster size and its share of the
        class in both windows, the centroid shift, the spread change, and the key features: for
        tracked clusters the features that changed most, for appeared and disappeared clusters the
        features in which the cluster lies furthest from its class. Rows are grouped by class;
        within a class, appeared and disappeared clusters come first, then the clusters with the
        most changes.
    """
    y_old, y_new = np.asarray(detector.y_old), np.asarray(detector.y_new)
    rows = []
    for class_order, (cl, details) in enumerate(detector.drift_details.items()):
        old_in_class, new_in_class = y_old == cl, y_new == cl
        clusters = set(detector.cluster_labels_old[old_in_class]) | set(detector.cluster_labels_new[new_in_class])
        for k in sorted(clusters):
            shift = detector.cluster_shifts[k]
            n_old = int(np.sum(detector.cluster_labels_old[old_in_class] == k))
            n_new = int(np.sum(detector.cluster_labels_new[new_in_class] == k))
            share_old = n_old / max(int(old_in_class.sum()), 1)
            share_new = n_new / max(int(new_in_class.sum()), 1)
            row = {
                "Class": _class_name(cl),
                "Cluster": int(k),
                "Size": f"{n_old} → {n_new}",
                "Share of Class": f"{share_old:.0%} → {share_new:.0%}",
                "Centroid Shift": None,
                "Spread Change": None,
            }
            if not isinstance(shift, dict):
                changes = [shift]
                row["Key Features"] = _cluster_location(detector, k, cl)
                severity = 1e6 + n_old + n_new
            else:
                changes = []
                if details["centroid_shift"].get(k):
                    changes.append("moved")
                stat_flags = details["desc_stats_changes"].get(k, {})
                if not changes and any(any(f.values()) for f in stat_flags.values()):
                    changes.append("statistics changed")
                if details["avg_distance_to_center"].get(k):
                    changes.append("wider" if detector.avg_distance_shift[k] > 0 else "narrower")
                if abs(share_new - share_old) >= SHARE_CHANGE_MIN:
                    if share_new >= 2 * share_old:
                        changes.append("grew")
                    elif share_new <= share_old / 2:
                        changes.append("shrank")
                row["Centroid Shift"] = shift["euclidean_distance"]
                row["Spread Change"] = detector.avg_distance_shift[k] * 100
                row["Key Features"] = _changed_features(detector, k, stat_flags)
                severity = len(changes) * 1e3 + shift["euclidean_distance"]
            row["Changes"] = ", ".join(changes) or "unchanged"
            row["_class_order"] = class_order
            row["_severity"] = severity
            rows.append(row)

    columns = ["Class", "Cluster", "Changes", "Size", "Share of Class", "Centroid Shift", "Spread Change", "Key Features"]
    df = pd.DataFrame(rows, columns=columns + ["_class_order", "_severity"])
    df = df.sort_values(["_class_order", "_severity", "Cluster"], ascending=[True, False, True])
    return df[columns].reset_index(drop=True)


def _changes_style(changes: str) -> str:
    """Background of the ``Changes`` cell: structural changes, other changes, or none."""
    if changes in CHANGE_STYLES:
        return CHANGE_STYLES[changes]
    return "" if changes == "unchanged" else CHANGE_STYLES["changed"]


def _display_findings(detector: ClusterBasedDriftDetector):
    """
    Displays the summary counts and the table of changes per cluster.

    Parameters
    ----------
    detector : ClusterBasedDriftDetector
        A detector on which ``detect()`` has been called.
    """
    st.subheader("What Changed")
    df = _findings_dataframe(detector)

    statuses = df["Changes"]
    col_ref, col_disappeared, col_appeared, col_changed = st.columns(4)
    col_ref.metric("Clusters Before", int(detector.number_of_clusters_old))
    col_disappeared.metric("Disappeared", int((statuses == "disappeared").sum()))
    col_appeared.metric("Appeared", int((statuses == "appeared").sum()))
    col_changed.metric(
        "Changed",
        int((~statuses.isin(["appeared", "disappeared", "unchanged"])).sum()),
        help="Clusters present in both windows that moved, changed their statistics, spread or share of the class.",
    )

    st.dataframe(
        df.style.map(_changes_style, subset=["Changes"]),
        column_config={
            "Class": st.column_config.TextColumn("Class"),
            "Cluster": st.column_config.NumberColumn(
                "Cluster",
                help="Clusters of the Before window are numbered first (0 … n-1); clusters that appear only in the After window follow",
            ),
            "Changes": st.column_config.TextColumn(
                "Changes",
                help=(
                    "disappeared: receives significantly fewer After samples than expected; "
                    "appeared: a region of the After window not explained by the Before clusters; "
                    f"moved: centroid shift above {detector.thr_centroid_shift:.2f}; "
                    f"statistics changed: a mean, median or standard deviation changed by more than "
                    f"{detector.thr_desc_stats} σ; "
                    f"wider / narrower: spread changed by more than {detector.thr_avg_distance_to_center_change:.0%}; "
                    f"grew / shrank: share of the class at least doubled / halved (by ≥ {SHARE_CHANGE_MIN:.0%} points)"
                ),
                width="medium",
            ),
            "Size": st.column_config.TextColumn("Size", help="Samples in the Before → After window"),
            "Share of Class": st.column_config.TextColumn(
                "Share of Class", help="Share of the samples of the class in the Before → After window"
            ),
            "Centroid Shift": st.column_config.NumberColumn(
                "Centroid Shift",
                help=(f"Distance between the centroids in standardised features; threshold {detector.thr_centroid_shift:.2f}"),
                format="%.2f",
            ),
            "Spread Change": st.column_config.NumberColumn(
                "Spread Change",
                help="Relative change of the mean distance of the samples to their centroid",
                format="%+.0f%%",
            ),
            "Key Features": st.column_config.TextColumn(
                "Key Features",
                help=(
                    "Tracked clusters: the features whose statistics changed most (mean unless stated, in standard "
                    "deviations σ). Appeared / disappeared clusters: where the cluster lies relative to the mean of "
                    "its class in the Before window."
                ),
                width="large",
            ),
        },
        width="stretch",
        hide_index=True,
    )
    _display_noise_check(detector)


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
            "Cluster": st.column_config.NumberColumn(
                "Cluster",
                help="Clusters of the Before window are numbered first (0 … n-1); clusters that appear only in the After window follow",
            ),
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
    """Histogram of the drift strength of random re-splits of the windows with the observed strength."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(detector.null_strengths, bins=15, color="#95a5a6", edgecolor="white", label="Random re-splits (sampling noise)")
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


def _show(fig):
    """Render a matplotlib figure and release it."""
    st.pyplot(fig)
    plt.close(fig)


def _display_visualizations(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: np.ndarray,
    y_after: np.ndarray,
    detector: ClusterBasedDriftDetector,
):
    """
    Displays the clusters of every class in both windows, the centroid shifts and the feature shift heatmap.

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
        A detector on which ``detect()`` has been called.
    """
    labels_old, labels_new = detector.cluster_labels_old, detector.cluster_labels_new

    st.subheader("Clusters Before and After")
    if X_before.shape[1] > 2:
        st.caption("More than two features: the plots show the first two principal components (PCA).")
    _show(plot_clusters_by_class(X_before, X_after, y_before, y_after, labels_old, labels_new))

    col_centroids, col_heatmap = st.columns(2)
    with col_centroids:
        st.subheader("Centroid Shifts")
        st.caption("× centroid in the Before window, ● in the After window; a dashed line joins the two.")
        _show(plot_centers_shift(X_before, X_after, y_before, y_after, labels_old, labels_new))
    with col_heatmap:
        st.subheader("Feature Changes per Cluster")
        # Features in the column order of the data (the statistics are keyed alphabetically).
        tracked = {
            k: {f: v[f] for f in detector.columns}
            for k, v in detector.stats_shifts.items()
            if isinstance(detector.cluster_shifts[k], dict)
        }
        if not tracked:
            st.info("No cluster is present in both windows, so there are no per-cluster feature changes to show.")
            return
        statistic = st.radio("Statistic", ["mean", "median", "std"], horizontal=True, key="cluster_heatmap_statistic")
        class_of = {row["Cluster"]: row["Class"] for _, row in _findings_dataframe(detector).iterrows()}
        fig = plot_feature_shift_heatmap(
            tracked,
            detector.thr_desc_stats,
            statistic=statistic,
            cluster_names={k: f"{k}\n(class {class_of[k]})" for k in tracked},
        )
        _show(fig)


def _display_details(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y_before: np.ndarray,
    y_after: np.ndarray,
    detector: ClusterBasedDriftDetector,
):
    """
    Displays the detailed tables and plots behind the findings in expanders.

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
        A detector on which ``detect()`` has been called.
    """
    st.subheader("Details")
    with st.expander("Cluster sizes, centroid shifts and spread"):
        _display_cluster_overview_table(detector)
    with st.expander("Statistics of every feature per cluster"):
        _display_stats_shift_table(detector)
    with st.expander("Centroids"):
        _display_centroids_table(detector)
    with st.expander("All clusters of both windows together"):
        _show(
            plot_drift_clustered(
                X_before, X_after, y_before, y_after, detector.cluster_labels_old, detector.cluster_labels_new
            )
        )
    with st.expander("Criteria and permutation check"):
        _display_drift_details(detector.drift_details, detector.effective_weights)
        if detector.null_strengths is not None:
            _show(_plot_permutation_test(detector))


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

    with st.spinner("Clustering both windows and comparing the clusters..."):
        try:
            detector = _run_detector(X_before, y_before, X_after, y_after, settings)

            _display_findings(detector)
            _display_visualizations(X_before, X_after, y_before, y_after, detector)
            _display_details(X_before, X_after, y_before, y_after, detector)

        except Exception as e:
            st.error(f"An error occurred during drift detection: {e}")
            st.text(traceback.format_exc())
