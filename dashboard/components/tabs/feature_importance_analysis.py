import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.feature_importance import (
    FeatureImportanceMethod,
    FeatureImportanceDriftAnalyzer,
    visualize_drift_importance,
    visualize_predictive_importance_shift,
)


def render_feature_importance_analysis_tab(
    X_before, y_before, X_after, y_after, feature_names, model_class=None, model_params=None
):
    """
    Renders the Feature Importance Analysis tab.

    Parameters
    ----------
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
    model_class : class (optional)
        The model class to use for drift detection
    model_params : dict (optional)
        Parameters for the model
    """
    st.header("Feature Importance Analysis")

    # --- Controls Section ---

    # Configuration in an expander
    with st.expander("Analysis Settings", expanded=False):
        col_controls_1, col_controls_2, col_controls_3 = st.columns(3)

        with col_controls_1:
            # Select Feature Importance Method
            importance_method = st.selectbox(
                "Feature Importance Method",
                options=FeatureImportanceMethod.all_available(),
                format_func=lambda x: x.upper(),
                help="Select the method to calculate importance (e.g., Permutation, SHAP).",
            )

        with col_controls_2:
            # Plot Type Selector
            plot_type_display = st.selectbox(
                "Plot Type", options=["Bar Chart", "Box Plot"], index=0, help="Choose visualization type for the charts."
            )
            plot_type_map = {"Bar Chart": "bar", "Box Plot": "box"}
            selected_plot_type = plot_type_map[plot_type_display]

        with col_controls_3:
            # Checkbox for including target (Drift Analysis setting)
            st.write("")  # Add spacing to align with selectbox
            st.write("")
            include_target = st.checkbox(
                "Include Target (Y) in Drift Analysis",
                value=True,
                help="Checked: Concept Drift (P(Y|X)). Unchecked: Data Drift (P(X)).",
            )

    # Initialize DriftAnalyzer
    analyzer = FeatureImportanceDriftAnalyzer(X_before, y_before, X_after, y_after, feature_names=feature_names)

    # --- Analysis Section (Side-by-Side) ---
    col_drift, col_pred = st.columns(2)

    # --- Left Column: Drift Analysis ---
    with col_drift:
        drift_title = "Concept Drift (P(Y|X))" if include_target else "Data Drift (P(X))"

        if include_target:
            drift_help = """
            Goal: Detect changes in relationship between Features and Target.
            Method: Classification (X, Y) → Time Period.
            Interp: High importance = Feature contributing to drift.
            """
        else:
            drift_help = """
            Goal: Detect changes in Feature Distribution.
            Method: Classification (X) → Time Period.
            Interp: High importance = Feature contributing to drift.
            """

        st.subheader(f"{drift_title}", help=drift_help)

        with st.spinner(f"Running {drift_title}..."):
            # Compute Drift Analysis
            drift_result = analyzer.compute_drift_importance(
                importance_method=importance_method,
                include_target=include_target,
                model_class=model_class,
                model_params=model_params,
            )

            # Visualization
            fig_drift = visualize_drift_importance(
                drift_result, drift_result["feature_names"], plot_type=selected_plot_type, include_target=include_target
            )
            st.pyplot(fig_drift)
            plt.close(fig_drift)

            # Table
            st.markdown("**Importance Summary**")
            feature_names_result = drift_result["feature_names"]
            drift_df = pd.DataFrame(
                {
                    "Feature": feature_names_result,
                    "Mean Importance": drift_result["importance_mean"],
                    "Std Deviation": drift_result["importance_std"],
                }
            )
            drift_df = drift_df.sort_values("Mean Importance", ascending=False)
            st.dataframe(drift_df.style.format({"Mean Importance": "{:.4f}", "Std Deviation": "{:.4f}"}), width="stretch")

    # --- Right Column: Predictive Power Shift ---
    with col_pred:
        pred_help = """
        Goal: Compare model reliance on features before vs after.
        Method: Train Model(Before) vs Train Model(After).
        Interp: Change in importance ranking = Mechanism shift.
        """
        st.subheader("Predictive Power Shift", help=pred_help)

        with st.spinner("Running Predictive Shift Analysis..."):
            # Compute Predictive Shift
            shift_result = analyzer.compute_predictive_importance_shift(
                importance_method=importance_method, model_class=model_class, model_params=model_params
            )

            # Visualization
            fig_shift = visualize_predictive_importance_shift(shift_result, feature_names, plot_type=selected_plot_type)
            st.pyplot(fig_shift)
            plt.close(fig_shift)

            # Tables (Side-by-side inner columns for tables to save space, or stacked)
            # Stacked might be better for readability in the column

            st.markdown("**Importance: Before Drift**")
            pred_before_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Mean Importance": shift_result["fi_before"]["importances_mean"],
                    "Std Deviation": shift_result["fi_before"]["importances_std"],
                }
            )
            pred_before_df = pred_before_df.sort_values("Mean Importance", ascending=False)
            st.dataframe(
                pred_before_df.style.format({"Mean Importance": "{:.4f}", "Std Deviation": "{:.4f}"}), width="stretch"
            )

            st.markdown("**Importance: After Drift**")
            pred_after_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Mean Importance": shift_result["fi_after"]["importances_mean"],
                    "Std Deviation": shift_result["fi_after"]["importances_std"],
                }
            )
            pred_after_df = pred_after_df.sort_values("Mean Importance", ascending=False)
            st.dataframe(pred_after_df.style.format({"Mean Importance": "{:.4f}", "Std Deviation": "{:.4f}"}), width="stretch")
