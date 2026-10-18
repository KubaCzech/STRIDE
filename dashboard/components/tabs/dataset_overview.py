import streamlit as st
import pandas as pd


def render_dataset_overview_tab(X, y, feature_names, dataset_instance=None, window_length=None):
    """Renders the Dataset Overview tab showing full data stream characteristics."""
    st.header("Dataset Overview")
    st.markdown("Global exploratory overview of the data stream prior to drift detection and window segmentation.")

    if X is None or len(X) == 0:
        st.warning("No dataset loaded. Please select or upload a dataset in the sidebar.")
        return

    n_samples = len(X)
    n_features = len(feature_names)
    y_series = pd.Series(y)
    unique_classes = y_series.nunique()
    class_counts = y_series.value_counts(normalize=True).to_dict()

    # --- Top KPIs ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Stream Samples", f"{n_samples:,}")
    with kpi2:
        st.metric("Number of Features", f"{n_features}")
    with kpi3:
        if window_length:
            n_windows = n_samples // window_length
            st.metric("Estimated Windows", f"{n_windows} (len: {window_length})")
        else:
            st.metric("Classes", f"{unique_classes}")
    with kpi4:
        label_1_freq = class_counts.get(1, class_counts.get(1.0, None))
        if label_1_freq is not None:
            st.metric("Label 1 Prevalence", f"{label_1_freq:.1%}")
        else:
            dominant_class = y_series.mode().iloc[0] if not y_series.empty else "N/A"
            st.metric("Dominant Class", f"{dominant_class}")

    st.markdown("---")

    # --- Ground Truth Drift Information (if synthetic) ---
    gt_drifts = getattr(dataset_instance, "ground_truth_drift_points", []) or st.session_state.get(
        "ground_truth_drift_points", []
    )
    drift_intervals = getattr(dataset_instance, "drift_intervals", []) or st.session_state.get("drift_intervals", [])
    drifting_features = getattr(dataset_instance, "drifting_features", []) or st.session_state.get("drifting_features", [])

    if gt_drifts or drift_intervals or drifting_features:
        with st.container(border=True):
            st.subheader("🎯 Ground Truth Drift Metadata")
            c_gt1, c_gt2, c_gt3 = st.columns(3)
            with c_gt1:
                st.markdown(f"**Known Drift Points:** `{gt_drifts if gt_drifts else 'None'}`")
            with c_gt2:
                st.markdown(f"**Drift Intervals:** `{drift_intervals if drift_intervals else 'None'}`")
            with c_gt3:
                st.markdown(f"**Drifting Features:** `{drifting_features if drifting_features else 'None'}`")

    # --- Dataset Preview & Summary ---
    c_preview, c_stats = st.tabs(["📋 Data Stream Preview", "📊 Feature Summary Statistics"])

    with c_preview:
        st.markdown("##### First 100 Samples of Data Stream")
        preview_df = pd.DataFrame(X.iloc[:100] if hasattr(X, "iloc") else X[:100], columns=feature_names).copy()
        preview_df["target"] = y[:100] if hasattr(y, "__getitem__") else list(y)[:100]
        st.dataframe(preview_df, width="stretch")

    with c_stats:
        st.markdown("##### Descriptive Statistics (Full Stream)")
        df_full = pd.DataFrame(X, columns=feature_names)
        st.dataframe(df_full.describe().T, width="stretch")
