import streamlit as st


def render_statistical_tests_tab(X_before, y_before, X_after, y_after, feature_names):
    """Renders the Statistical Tests tab (placeholder with roadmap / TODO)."""
    st.header("Statistical Tests")
    st.markdown(
        "Two-sample hypothesis testing and non-parametric statistical distance metrics "
        "comparing distributions before and after detected drift."
    )

    n_before = len(y_before) if y_before is not None else 0
    n_after = len(y_after) if y_after is not None else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Reference Window Samples", f"{n_before}")
    with c2:
        st.metric("Detection Window Samples", f"{n_after}")
    with c3:
        st.metric("Evaluated Features", f"{len(feature_names)}")

    st.markdown("---")

    with st.container(border=True):
        st.subheader("🚧 Under Construction (TODO)")
        st.info(
            "This module is currently in development. It will integrate core algorithms from "
            "`stride.xai.stats.StatisticalTestsDriftDetector` and `stride.xai.stats.visualization`."
        )

        st.markdown(
            """
            #### 📋 Planned Features & Statistical Tests:
            - **Hypothesis Testing & Distances**:
              - **Kolmogorov-Smirnov Test ($D_{KS}$)**: Non-parametric comparison of marginal CDFs.
              - **Anderson-Darling Test**: Sensitive tail-drift detection.
              - **Wasserstein Distance (Earth Mover's Distance $\\mathcal{W}_1$)**: Normalized optimal transport distance.
              - **Kullback-Leibler ($D_{KL}$) & Jensen-Shannon ($D_{JS}$) Divergence**: Information-theoretic distribution shift.
              - **Spearman Rank Correlation ($r_s$)**: Monotonic relationship shifts.
            - **Visual Distribution Diagnostics**:
              - Subplot grids comparing feature distributions across classes:
                - Boxplots (`plot_boxplot`)
                - Overlapping Histograms (`plot_histogram`)
                - Kernel Density Estimation (`plot_kde`)
                - Empirical Cumulative Distribution Functions (`plot_ecdf`)
            """
        )
