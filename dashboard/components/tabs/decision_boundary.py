import streamlit as st
import matplotlib.pyplot as plt
import traceback
from src.decision_boundary.analysis import DecisionBoundaryDriftAnalyzer
from src.decision_boundary.visualization import visualize_decision_boundary, plot_categorical_drift_map


def _render_ssnp_config(X_before):
    is_2d = X_before.shape[1] == 2 if hasattr(X_before, 'shape') and len(X_before.shape) > 1 else False
    with st.expander("SSNP Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            ssnp_epochs = st.number_input("SSNP Epochs", min_value=1, max_value=200, value=10, step=5,
                                          help="Number of epochs for training the SSNP projector.",
                                          disabled=is_2d)
            if is_2d:
                st.caption("Data is 2D. SSNP bypassed.")
        with col2:
            grid_size = st.number_input("Grid Resolution", min_value=50, max_value=500, value=200, step=50,
                                        help="Resolution of the grid for visualizing probabilities.")
    return ssnp_epochs, grid_size


def _run_analysis_if_needed(
    X_before, y_before, X_after, y_after, model_class, model_params, ssnp_epochs, grid_size, feature_names=None
):
    # Initialize session state for results if not exists
    if 'decision_boundary_results' not in st.session_state:
        st.session_state.decision_boundary_results = None

    # Define current run parameters to detect changes
    current_run_params = {
        'ssnp_epochs': ssnp_epochs,
        'grid_size': grid_size,
        'model_class': getattr(model_class, '__name__', str(model_class)),
        'model_params': str(model_params),
        'X_shape': X_before.shape if hasattr(X_before, 'shape') else None
    }

    # Determine if analysis needs to be run
    should_run = False
    if st.session_state.decision_boundary_results is None:
        should_run = True
    elif 'decision_boundary_last_params' not in st.session_state:
        should_run = True
    elif st.session_state.decision_boundary_last_params != current_run_params:
        should_run = True

    if should_run:
        with st.spinner("Running Analysis (Auto-refresh)..."):
            try:
                analyzer = DecisionBoundaryDriftAnalyzer(X_before, y_before, X_after, y_after)
                results = analyzer.analyze(
                    model_class=model_class,
                    model_params=model_params,
                    ssnp_epochs=ssnp_epochs,
                    grid_size=grid_size,
                    feature_names=feature_names
                )
                # Store results and params in session state
                st.session_state.decision_boundary_results = results
                st.session_state.decision_boundary_last_params = current_run_params

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                # Optional: print stack trace for debugging
                st.text(traceback.format_exc())
                st.session_state.decision_boundary_results = None


def _display_results():
    if st.session_state.decision_boundary_results is not None:
        try:
            st.markdown("### Decision Boundaries (Pre vs Post)")

            res = st.session_state.decision_boundary_results
            is_2d = res.get('is_2d', False)

            # Create and display plot
            fig = visualize_decision_boundary(res)
            st.pyplot(fig)
            plt.close(fig)

            # Display Disagreement Analysis
            if 'disagreement' in res and res['disagreement']['drift_rate'] > 0:
                st.markdown("### Categorical Drift Map (Disagreement)")
                st.info(f"Drift Rate (Disagreement): {res['disagreement']['drift_rate']:.1%}")

                col_map, col_table = st.columns([1, 1])

                with col_map:
                    grid_bounds = res['post']['grid_bounds']
                    fig_map = plot_categorical_drift_map(
                        ssnp_model=res['ssnp_model'],
                        viz_tree=res['disagreement']['viz_tree'],
                        drift_leaf_ids=res['disagreement']['drift_leaf_ids'],
                        grid_bounds=grid_bounds,
                        grid_size=res['grid_size'],
                        is_2d=is_2d # Pass flag down
                    )
                    st.pyplot(fig_map)
                    plt.close(fig_map)

                with col_table:
                    st.markdown("#### Disagreement Rules")
                    rules_df = res['disagreement']['disagreement_table'].copy()
                    rules_df.index = rules_df.index + 1
                    st.dataframe(rules_df, width='stretch')
            elif 'disagreement' in res:
                st.success("No significant disagreement (drift) detected between pre-drift and post-drift models.")

        except Exception as e:
            st.error(f"Error displaying visualization: {e}")
            if st.button("Clear Results"):
                st.session_state.decision_boundary_results = None
                st.rerun()

def _render_decision_boundary_tab_content(X_before, y_before, X_after, y_after, model_class, model_params, feature_names=None):
    ssnp_epochs, grid_size = _render_ssnp_config(X_before)
    _run_analysis_if_needed(X_before, y_before, X_after, y_after, model_class,
                            model_params, ssnp_epochs, grid_size, feature_names)
    _display_results()

def render_decision_boundary_tab(X_before, y_before, X_after, y_after,
                                 model_class=None, model_params=None, feature_names=None):
    """
    Renders the Decision Boundary Analysis tab.

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
    model_class : class
        Classifier class
    model_params : dict
        Parameters for the classifier
    """
    st.header("Decision Boundary Analysis")
    st.markdown("""
    This tab visualizes the decision boundary of a classifier trained on the pre-drift and post-drift data.
    It uses **SSNP (Semi-Supervised Neural Projection)** to project the high-dimensional data into 2D while
    preserving the separation between classes.
    """)

    ssnp_epochs, grid_size = _render_ssnp_config(X_before)
    _run_analysis_if_needed(X_before, y_before, X_after, y_after, model_class,
                            model_params, ssnp_epochs, grid_size, feature_names)
    _display_results()