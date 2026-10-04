import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from stride.xai.importance import (
    AbstainStrategy,
    FeatureImportanceMethod,
    FeatureImportanceDriftAnalyzer,
    visualize_drift_importance,
    visualize_predictive_importance_shift,
)


def render_feature_importance_analysis_tab(
    X_before, y_before, X_after, y_after, feature_names, model_class=None, model_params=None, dataset=None
):
    """
    Renders the Feature Importance Analysis tab.
    """
    st.header("Feature Importance Analysis")

    drifting_features = getattr(dataset, "drifting_features", None) or st.session_state.get("drifting_features", [])
    if drifting_features:
        st.info(
            f"🎯 **Ground Truth Drifting Features**: `{', '.join(drifting_features)}` "
            "— Compare whether top SHAP / Permutation importance shift aligns with these true physical drift loci."
        )

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

        st.markdown("---")
        col_abstain_1, col_abstain_2 = st.columns(2)

        with col_abstain_1:
            enable_abstain = st.checkbox(
                "Enable Drift Localization (Abstain Option)",
                value=True,
                help="Excludes points where P(T|X) ≈ 0.5 where pre- and post-drift distributions coincide, preventing noise-induced feature importance.",
            )
            if enable_abstain:
                strategy_options = [
                    (AbstainStrategy.CONFIDENCE_THRESHOLD, "Confidence Threshold (Margin τ)"),
                    (AbstainStrategy.CONFORMAL, "Conformal Predictions (p-value α)"),
                ]
                selected_strategy = st.selectbox(
                    "Localization Strategy",
                    options=[opt[0] for opt in strategy_options],
                    format_func=lambda s: dict(strategy_options)[s],
                    help="Method used to localize the drift subpopulation and reject invariant instances.",
                )
            else:
                selected_strategy = None

        with col_abstain_2:
            tau = 0.15
            alpha = 0.05
            if enable_abstain:
                if selected_strategy == AbstainStrategy.CONFIDENCE_THRESHOLD:
                    tau = st.slider(
                        "Abstention Indifference Margin (τ)",
                        min_value=0.05,
                        max_value=0.40,
                        value=0.15,
                        step=0.01,
                        help="Excludes points where |P(T=1|X) - 0.5| ≤ τ where pre- and post-drift distributions coincide.",
                    )
                elif selected_strategy == AbstainStrategy.CONFORMAL:
                    alpha = st.slider(
                        "Conformal Significance Level (α)",
                        min_value=0.01,
                        max_value=0.20,
                        value=0.05,
                        step=0.01,
                        help="Significance level for rejecting H0 ('non-drifting'). Points with min_y p_y(x) < α belong to the drift locus.",
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
                abstain_strategy=selected_strategy,
                tau=tau,
                alpha=alpha,
            )

            # Diagnostic KPI Metrics Display
            if enable_abstain and "drifting_mask" in drift_result:
                st.markdown("##### Drift Localization Diagnostics")
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric(
                        "Drift Coverage",
                        f"{drift_result['drift_coverage'] * 100:.1f}%",
                        help="Percentage of instances residing in the drift locus (spatial extent of drift).",
                    )
                with col_m2:
                    acc_delta = drift_result["selective_accuracy"] - drift_result["accuracy"]
                    st.metric(
                        "Selective Accuracy",
                        f"{drift_result['selective_accuracy'] * 100:.1f}%",
                        delta=f"{acc_delta * 100:+.1f}% vs overall",
                        help="Discriminator accuracy evaluated strictly on accepted drift locus instances.",
                    )
                with col_m3:
                    n_drift = int(np.sum(drift_result["drifting_mask"]))
                    n_total = len(drift_result["drifting_mask"])
                    st.metric(
                        "Drift Locus Samples",
                        f"{n_drift} / {n_total}",
                        help="Number of accepted instances in the drift locus.",
                    )

                if drift_result.get("drift_locus_fallback"):
                    st.warning(
                        "The drift locus contains too few instances (< min_samples). "
                        "Feature importance was evaluated on the full dataset as fallback."
                    )
                elif drift_result["drift_coverage"] < 0.25:
                    st.info(
                        "The detected drift is spatially localized to a subpopulation/subspace. "
                        "Feature importance reflects drivers strictly within this localized locus."
                    )
                elif drift_result["drift_coverage"] >= 0.85:
                    st.info("The detected drift is widespread/global across the data distribution.")

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
