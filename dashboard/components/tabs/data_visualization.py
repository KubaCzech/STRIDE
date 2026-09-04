from src.descriptive_statistics.descriptive_statistics import DescriptiveStatisticsDriftDetector
from src.plotting import visualize_data_stream
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Generating data stream visualizations...")
def generate_plots(X, y, window_before_start, window_after_start, window_length, feature_names, viz_type="violin"):
    """Generates all visualization plots."""
    return visualize_data_stream(
        X,
        y,
        window_before_start,
        window_after_start,
        window_length,
        feature_names,
        title_feat_target=None,
        title_class_dist=None,
        title_feat_space=None,
        viz_type=viz_type,
    )


def _render_metrics(n_before, n_after, class_bal_before, class_bal_after):
    """Renders the top key metrics."""
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Samples (Before)", f"{n_before}")
    with m2:
        st.metric("Samples (After)", f"{n_after}")
    with m3:
        if not class_bal_before.empty:
            c1_perc_before = class_bal_before.to_dict().get(1, 0.0)
            st.metric("Label 1 Frequency (Before)", f"{c1_perc_before:.1%}")
        else:
            st.metric("Label 1 Frequency (Before)", "N/A")

    with m4:
        if not class_bal_after.empty:
            c1_perc_after = class_bal_after.to_dict().get(1, 0.0)
            delta = 0
            if not class_bal_before.empty:
                delta = c1_perc_after - class_bal_before.to_dict().get(1, 0.0)
            st.metric("Label 1 Frequency (After)", f"{c1_perc_after:.1%}", delta=f"{delta:.1%}", delta_color="off")
        else:
            st.metric("Label 1 Frequency (After)", "N/A")


def _render_visualization_plots(all_figs):
    """Renders the grid of visualization plots."""
    if not all_figs:
        st.error("No visualization plots were generated.")
        return

    # Unpack figures (Order based on src/plotting.py)
    # 0: Feature vs Target Relationship
    # 1: Class Distribution
    # 2: Feature Space
    if len(all_figs) >= 3:
        fig_feat_target = all_figs[0]
        fig_class_dist = all_figs[1]
        fig_feat_space = all_figs[2]

        # Row 1: Feature Space & Class Distribution
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown('<h4 style="text-align: center;">🗺️ Feature Space Distribution</h4>', unsafe_allow_html=True)
            st.pyplot(fig_feat_space)
        with c2:
            st.markdown('<h4 style="text-align: center;">📊 Class Distribution</h4>', unsafe_allow_html=True)
            # Stretch the plot to match the height of Feature Space
            fig_class_dist.set_size_inches(9.33, 7)
            st.pyplot(fig_class_dist)

        # Row 2: Detailed Feature Analysis
        # We only have Feature vs Target now, make it full width or centered
        st.markdown('<h4 style="text-align: center;">🎯 Feature vs Target Relationship</h4>', unsafe_allow_html=True)
        _, col_centered, _ = st.columns([1, 10, 1])
        with col_centered:
            st.pyplot(fig_feat_target)
    else:
        st.warning("Not all visualizations are available.")
        for fig in all_figs:
            st.pyplot(fig)


def _calculate_statistics_styled(X_before, y_before, X_after, y_after, feature_names):
    """Calculates descriptive statistics and returns styled(or plain) df."""
    df_before = pd.DataFrame(X_before, columns=feature_names)
    df_after = pd.DataFrame(X_after, columns=feature_names)

    detector = DescriptiveStatisticsDriftDetector(df_before, y_before, df_after, y_after)
    stats_combined = detector.calculate_stats_before_after()

    # Reformat for display
    # Transpose so rows are (Period, Feature, Stat)
    stats_T = stats_combined.T
    if stats_T.index.nlevels == 3:
        stats_T.index.names = ["Period", "Feature", "Stat"]
        stats_tidy = stats_T.reset_index()
        # Pivot: Index=[Feature, Stat], Columns=Period, Values=0
        stats_pivot = stats_tidy.pivot(index=["Feature", "Stat"], columns="Period", values=0)

        # Handle column renaming safer
        rename_map = {}
        if "old" in stats_pivot.columns:
            rename_map["old"] = "Before"
        if "new" in stats_pivot.columns:
            rename_map["new"] = "After"

        stats_pivot = stats_pivot.rename(columns=rename_map)

        # Ensure columns exist before selecting
        cols_to_show = []
        if "Before" in stats_pivot.columns:
            cols_to_show.append("Before")
        if "After" in stats_pivot.columns:
            cols_to_show.append("After")

        stats_pivot = stats_pivot[cols_to_show]

        # Calculate percent change if both exist
        if "Before" in stats_pivot.columns and "After" in stats_pivot.columns:
            stats_pivot["Change (%)"] = ((stats_pivot["After"] - stats_pivot["Before"]) / stats_pivot["Before"]) * 100

        return stats_pivot.style.format("{:.4f}")
    else:
        return stats_combined


def render_data_visualization_tab(
    X, y, X_before, y_before, X_after, y_after, feature_names, window_before_start, window_after_start, window_length
):
    """
    Renders the Data Stream Visualization tab.

    Parameters
    ----------
    X : array-like
        Feature matrix (full)
    y : array-like
        Target variable (full)
    X_before : array-like
        Feature matrix for 'before' window
    y_before : array-like
        Target variable for 'before' window
    X_after : array-like
        Feature matrix for 'after' window
    y_after : array-like
        Target variable for 'after' window
    feature_names : list
        List of feature names
    window_before_start : int
    window_after_start : int
    window_length : int
    """
    st.header("Dataset Visualization")

    # --- Metrics Section ---
    n_before = len(y_before)
    n_after = len(y_after)
    class_bal_before = pd.Series(y_before).value_counts(normalize=True)
    class_bal_after = pd.Series(y_after).value_counts(normalize=True)

    _render_metrics(n_before, n_after, class_bal_before, class_bal_after)
    st.markdown("---")

    # --- Visualizations ---
    # Add viz type selector
    with st.expander("Visualization Settings", expanded=False):
        col_viz_options, _ = st.columns([1, 2])
        with col_viz_options:
            viz_type = st.selectbox(
                "Feature vs Target Visualization Type",
                options=["violin", "box", "scatter"],
                index=0,  # Default to violin
                key="viz_type_selector",
                help="Choose how the relationship between features and target is visualized for each class in the windows.",
            )

    all_figs = generate_plots(X, y, window_before_start, window_after_start, window_length, feature_names, viz_type=viz_type)
    _render_visualization_plots(all_figs)

    # --- Descriptive Statistics ---
    st.markdown("---")
    with st.expander("🔢 Descriptive Statistics (Detailed table)", expanded=False):
        stats_display = _calculate_statistics_styled(X_before, y_before, X_after, y_after, feature_names)
        if hasattr(stats_display, "format") or isinstance(stats_display, (pd.DataFrame, pd.io.formats.style.Styler)):
            # If explicit width is necessary, st.dataframe handles Styler objects too
            st.dataframe(stats_display, width="stretch")
        else:
            st.dataframe(stats_display)
