import inspect
import streamlit as st
import pandas as pd
from src.datasets import DATASETS, reload_datasets, DatasetRegistry
from src.models import MODELS
from dashboard.components.modals.dataset_settings import open_dataset_settings_modal
from dashboard.components.modals.model_settings import open_model_settings_modal


def _render_import_dataset_modal(dataset_key):
    if dataset_key == "➕ Import dataset...":

        @st.dialog("Import Dataset")
        def open_import_dataset_modal():
            st.write("Upload a CSV file to import a new dataset.")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                try:
                    # Read header only
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=0)
                    columns = df_preview.columns.tolist()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    return

                dataset_name = st.text_input("Dataset Name", value=uploaded_file.name.replace(".csv", ""))

                # Default target is the last column
                default_target_idx = len(columns) - 1 if columns else 0

                target_col = st.selectbox(
                    "Target Variable",
                    options=columns,
                    index=default_target_idx,
                    help="Select the column containing the target variable.",
                )

                # Default features are all except target
                available_features = [c for c in columns if c != target_col]
                features = st.multiselect(
                    "Features to Include",
                    options=available_features,
                    default=available_features,
                    help="Select the features to be used for the dataset.",
                )

                if st.button("Import Dataset"):
                    if not dataset_name:
                        st.error("Please provide a dataset name.")
                        return
                    if not features:
                        st.error("Please select at least one feature.")
                        return

                    # Save
                    registry = DatasetRegistry()
                    registry.save_dataset(dataset_name, uploaded_file, target_col, features)

                    # Reload datasets
                    reload_datasets()

                    # Update session state to select the new dataset
                    st.session_state.selected_dataset_key = dataset_name
                    st.success(f"Dataset '{dataset_name}' imported successfully!")
                    st.rerun()

        open_import_dataset_modal()
        # Stop execution so the rest of the app doesn't try to render with "Import dataset..." key
        st.stop()


def _render_dataset_selection():
    st.subheader("Dataset Selection")

    # 2. Select Dataset
    # Define Import Option
    IMPORT_OPTION = "➕ Import dataset..."

    dataset_options = list(DATASETS.keys()) + [IMPORT_OPTION]

    # Check if we should select a specific dataset (e.g. after import)
    index = 0
    if "selected_dataset_key" in st.session_state and st.session_state.selected_dataset_key in dataset_options:
        index = dataset_options.index(st.session_state.selected_dataset_key)

    col_ds1, col_ds2 = st.columns([0.75, 0.25], vertical_alignment="bottom")
    with col_ds1:
        dataset_key = st.selectbox(
            "Choose a Dataset",
            options=dataset_options,
            index=index,
            format_func=lambda x: x if x == IMPORT_OPTION else DATASETS[x].display_name,
            help="Select the synthetic dataset to analyze or import a new one.",
        )

    # Handle import modal
    _render_import_dataset_modal(dataset_key)

    # Process selection (if not stopped by modal)
    st.session_state.selected_dataset_key = dataset_key
    selected_dataset = DATASETS[dataset_key]

    # Add delete option for imported datasets
    registry = DatasetRegistry()
    if registry.get_dataset_info(dataset_key):
        if st.sidebar.button("🗑️ Delete Dataset", help="Permanently remove this imported dataset."):
            registry.delete_dataset(dataset_key)
            reload_datasets()
            st.session_state.selected_dataset_key = list(DATASETS.keys())[0]
            st.rerun()

    # Initialize session state for parameters if not exists
    if "dataset_params_by_dataset" not in st.session_state:
        st.session_state.dataset_params_by_dataset = {}

    # Switch or sync dataset parameters for the selected dataset
    if "current_dataset_key" not in st.session_state or st.session_state.current_dataset_key != dataset_key:
        st.session_state.current_dataset_key = dataset_key
        st.session_state.dataset_params = st.session_state.dataset_params_by_dataset.get(dataset_key, {}).copy()
    else:
        if "dataset_params" in st.session_state:
            st.session_state.dataset_params_by_dataset[dataset_key] = st.session_state.dataset_params

    with col_ds2:
        if st.button("⚙️", key="dataset_settings_btn", help="Configure dataset settings", width="stretch"):
            # Clear temporary settings widgets
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith("temp_dataset_param_")]
            for k in keys_to_clear:
                del st.session_state[k]
            open_dataset_settings_modal(selected_dataset, st.session_state.get("window_length", 1000), dataset_key)

    return dataset_key, selected_dataset, st.session_state.dataset_params


def _render_model_selection():
    # 5. Model Selection
    st.subheader("Model Configuration")

    col_m1, col_m2 = st.columns([0.75, 0.25], vertical_alignment="bottom")
    with col_m1:
        model_key = st.selectbox(
            "Choose a Model",
            options=list(MODELS.keys()),
            format_func=lambda x: MODELS[x]().display_name,
            help="Select the machine learning model to use for drift detection.",
        )

    selected_model_class = MODELS[model_key]

    # Maintain model parameters per model
    if "model_params_by_model" not in st.session_state:
        st.session_state.model_params_by_model = {}

    # Check if model changed or model_params not loaded for this model
    if "current_model_key" not in st.session_state or st.session_state.current_model_key != model_key:
        st.session_state.current_model_key = model_key
        st.session_state.model_params = st.session_state.model_params_by_model.get(model_key, {}).copy()
    else:
        # Keep model_params_by_model in sync with model_params
        if "model_params" in st.session_state:
            st.session_state.model_params_by_model[model_key] = st.session_state.model_params

    # Defensive filtering: pass only parameters accepted by selected_model_class.__init__
    valid_keys = set(inspect.signature(selected_model_class.__init__).parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in st.session_state.model_params.items() if k in valid_keys}
    st.session_state.model_params = filtered_params
    st.session_state.model_params_by_model[model_key] = filtered_params

    with col_m2:
        if st.button("⚙️", key="model_settings_btn", help="Configure model settings", width="stretch"):
            # Clear temporary settings widgets
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith("temp_model_")]
            for k in keys_to_clear:
                del st.session_state[k]
            open_model_settings_modal(selected_model_class)

    return selected_model_class, st.session_state.model_params


def render_sidebar_datasource_config():
    """
    Renders the configuration sidebar (Dataset, Model, Global Settings).
    Does NOT render window selection (requires data length).

    Returns a dictionary containing the configuration.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # 1. Global Window Settings
        st.subheader("Global Settings")
        window_length = st.number_input(
            "Window Length (Samples)", min_value=1, value=1000, help="Length of the analysis window in samples."
        )

        st.session_state.window_length = window_length  # Store for modal usage

        # Dataset Selection
        dataset_key, selected_dataset, dataset_params = _render_dataset_selection()

        # Model Selection
        selected_model_class, model_params = _render_model_selection()

    return {
        "window_length": window_length,
        "dataset_key": dataset_key,
        "dataset_params": dataset_params,
        "selected_model_class": selected_model_class,
        "model_params": model_params,
        "selected_features": st.session_state.get(f"selected_features_{dataset_key}", []),
    }


def render_sidebar_window_selection(max_samples, window_length):
    """
    Renders the Analysis Window Selection controls in the sidebar.
    Uses actual max_samples to enforce constraints.
    """
    with st.sidebar:
        st.subheader("Analysis Window Selection")

        max_windows = max(1, int(max_samples // window_length))

        col_w1, col_w2 = st.columns(2)

        # Get current values from session state to set dynamic limits and auto-correct
        curr_before_key = "window_before_input"
        curr_after_key = "window_after_input"

        # Initialize if not present (default 0 and 1)
        if curr_before_key not in st.session_state:
            st.session_state[curr_before_key] = 0
        if curr_after_key not in st.session_state:
            st.session_state[curr_after_key] = 1

        curr_before = st.session_state[curr_before_key]
        curr_after = st.session_state[curr_after_key]

        # Constraint 1: Absolute Max
        # Last index is max_windows - 1
        # Before can be at most max_windows - 2 (needs 1 spot for after)
        abs_max_before = max(0, max_windows - 2)
        abs_max_after = max(1, max_windows - 1)

        # Auto-correct absolute bounds
        if curr_before > abs_max_before:
            curr_before = abs_max_before
            st.session_state[curr_before_key] = curr_before

        if curr_after > abs_max_after:
            curr_after = abs_max_after
            st.session_state[curr_after_key] = curr_after

        # Constraint 2: Relative Order (Before < After)
        # We ensure min_after is at least Before + 1
        # If current After is too small, bump it up
        min_after = curr_before + 1
        if curr_after < min_after:
            curr_after = min_after
            # Verify we didn't exceed absolute max for After
            if curr_after > abs_max_after:
                # If bumping After exceeds max, we must push Before down
                curr_after = abs_max_after
                curr_before = curr_after - 1
                st.session_state[curr_before_key] = curr_before
            st.session_state[curr_after_key] = curr_after

        # Now Render Widgets with safe values
        with col_w1:
            window_before_start_windows = st.number_input(
                "Before",
                min_value=0,
                max_value=abs_max_before,
                # value=curr_before,  <-- Removed to avoid warning
                key=curr_before_key,
                help="Starting index for the first analysis window (in number of windows).",
            )

        with col_w2:
            window_after_start_windows = st.number_input(
                "After",
                min_value=window_before_start_windows + 1,  # Dynamic min based on widget
                max_value=abs_max_after,
                # value=curr_after,  <-- Removed to avoid warning
                key=curr_after_key,
                help="Starting index for the second analysis window (in number of windows).",
            )

        # Force constraint verification (visual warning just in case, though logic handles it)
        if window_before_start_windows >= window_after_start_windows:
            st.warning("Window 'Before' must be strictly smaller than 'After'.")

        # Calculate absolute indices
        window_before_start = window_before_start_windows * window_length
        window_after_start = window_after_start_windows * window_length

        return window_before_start, window_after_start
