import streamlit as st
from src.datasets import DatasetRegistry
from dashboard.components.settings import render_settings_from_schema
from dashboard.utils import get_dataset_settings_schema


def _get_preview_features(registry, selected_dataset, temp_dataset_params, window_length):
    """
    Helper to get available features for preview.
    """
    try:
        # Check if imported
        registry_info = registry.get_dataset_info(selected_dataset.name)
        if registry_info:
            return registry_info.get("selected_features", [])

        # Synthetic: generate small sample to get columns
        if st.session_state.dataset_params:
            gen_params = st.session_state.dataset_params.copy()
        else:
            gen_params = selected_dataset.get_params()

        if temp_dataset_params:
            gen_params.update(temp_dataset_params)

        # Handle window->sample conversion locally for preview
        if "n_windows_before" in gen_params:
            gen_params["n_samples_before"] = gen_params.pop("n_windows_before") * window_length
        if "n_windows_after" in gen_params:
            gen_params["n_samples_after"] = gen_params.pop("n_windows_after") * window_length

        # Override for minimal generation
        gen_params["n_samples_before"] = 5
        gen_params["n_samples_after"] = 5

        X_preview, _ = selected_dataset.generate(**gen_params)
        if X_preview is not None:
            return X_preview.columns.tolist()
    except Exception as e:
        st.warning(f"Could not preview features: {e}")
    return []


def _render_feature_selection(selected_dataset, dataset_key, registry, temp_dataset_params, window_length):
    """
    Renders feature selection UI.
    """
    st.subheader("Feature Selection")

    preview_features = _get_preview_features(registry, selected_dataset, temp_dataset_params, window_length)

    # Initialize selected features for this dataset if not in state
    feature_key = f"selected_features_{dataset_key}"
    if feature_key not in st.session_state:
        st.session_state[feature_key] = preview_features

    if preview_features:
        multiselect_key = f"multiselect_{dataset_key}"

        # Sanitize currently stored selections so they only contain existing preview features
        current_selection = st.session_state.get(feature_key, preview_features)
        valid_selection = [f for f in current_selection if f in preview_features]
        if not valid_selection:
            valid_selection = preview_features
        st.session_state[feature_key] = valid_selection

        # Also sanitize widget state if present to prevent StreamlitAPIException
        if multiselect_key in st.session_state:
            widget_val = [f for f in st.session_state[multiselect_key] if f in preview_features]
            if not widget_val:
                widget_val = valid_selection
            st.session_state[multiselect_key] = widget_val
            selected_feat = st.multiselect("Include Features", options=preview_features, key=multiselect_key)
        else:
            selected_feat = st.multiselect(
                "Include Features", options=preview_features, default=valid_selection, key=multiselect_key
            )
        st.session_state[feature_key] = selected_feat
    else:
        st.info("No features available to select.")
        st.session_state[feature_key] = []


def _on_setting_change_handler(available_settings):
    """Callback when settings dropdown changes"""
    selected = st.session_state.setting_selectbox
    if selected == "Not selected":
        st.session_state.selected_setting = None
        st.session_state.dataset_params = {}
    else:
        st.session_state.selected_setting = selected
        # Update the parameter values in session state
        st.session_state.dataset_params = available_settings[selected].copy()
        # Set flag to force update widgets
        st.session_state.force_update_widgets = True


def _get_cleaned_available_settings(selected_dataset):
    """Retrieves and cleans available settings from the dataset."""
    available_settings = selected_dataset.get_available_settings()
    cleaned_available_settings = {}
    default_preset_name = None

    if available_settings:
        for name, params in available_settings.items():
            if params.get("default"):
                default_preset_name = name
            cleaned_params = params.copy()
            cleaned_params.pop("default", None)
            cleaned_available_settings[name] = cleaned_params

    return cleaned_available_settings, default_preset_name


def _sync_selected_setting_with_params(cleaned_available_settings):
    """
    Reverse Lookup: If this is a fresh open (no temp keys), sync selected_setting with current params.
    """
    has_temp_keys = any(k.startswith("temp_dataset_param_") for k in st.session_state.keys())

    if not has_temp_keys and st.session_state.dataset_params:
        # Assume "Not selected" unless we find a match
        st.session_state.selected_setting = None

        current_params = st.session_state.dataset_params
        for name, preset_params in cleaned_available_settings.items():
            # Check if all preset params are present and equal in current_params
            is_match = True
            for k, v in preset_params.items():
                if current_params.get(k) != v:
                    is_match = False
                    break

            if is_match:
                st.session_state.selected_setting = name
                break


def _check_preset_modification(cleaned_available_settings):
    """
    Early detection of modifications to preset.
    """
    if (
        st.session_state.selected_setting
        and st.session_state.selected_setting in cleaned_available_settings
        and not st.session_state.force_update_widgets
    ):
        current_preset_params = cleaned_available_settings[st.session_state.selected_setting]
        is_modified = False

        for k, v in current_preset_params.items():
            widget_key = f"temp_dataset_param_{k}"
            # Only check if the widget key exists in session state
            if widget_key in st.session_state:
                if st.session_state[widget_key] != v:
                    is_modified = True
                    break

        if is_modified:
            st.session_state.selected_setting = None
            st.session_state.setting_selectbox = "Not selected"


def _render_preset_selectbox(cleaned_available_settings):
    """Renders the settings preset dropdown."""
    if not cleaned_available_settings:
        return

    setting_options = list(cleaned_available_settings.keys())

    # Only add "Not selected" if currently in that state
    if st.session_state.selected_setting is None:
        setting_options = ["Not selected"] + setting_options

    # Initialize the selectbox session state if not exists
    if "setting_selectbox" not in st.session_state:
        st.session_state.setting_selectbox = "Not selected"

    # Update selectbox value based on selected_setting
    if st.session_state.selected_setting in cleaned_available_settings:
        st.session_state.setting_selectbox = st.session_state.selected_setting
    elif st.session_state.selected_setting is None:
        st.session_state.setting_selectbox = "Not selected"

    st.selectbox(
        "Select Preset Settings",
        options=setting_options,
        key="setting_selectbox",
        on_change=_on_setting_change_handler,
        args=(cleaned_available_settings,),
        help="Choose a preset configuration or customize parameters manually.",
    )


@st.dialog("Dataset Settings")
def open_dataset_settings_modal(selected_dataset, window_length, dataset_key):
    st.write(f"Configure advanced settings for **{selected_dataset.display_name}**.")

    # 3. Settings Preset Selection
    cleaned_available_settings, default_preset_name = _get_cleaned_available_settings(selected_dataset)

    # Initialize session state for selected setting if not exists
    if "selected_setting" not in st.session_state:
        st.session_state.selected_setting = None

    _sync_selected_setting_with_params(cleaned_available_settings)

    # Auto-select default if nothing selected yet and default exists
    if st.session_state.selected_setting is None and default_preset_name and not st.session_state.dataset_params:
        st.session_state.selected_setting = default_preset_name
        st.session_state.dataset_params = cleaned_available_settings[default_preset_name].copy()
        st.session_state.force_update_widgets = True

    # Track if we need to force update the widgets (when preset changes)
    if "force_update_widgets" not in st.session_state:
        st.session_state.force_update_widgets = False

    _check_preset_modification(cleaned_available_settings)
    _render_preset_selectbox(cleaned_available_settings)

    # --- Dataset Settings ---
    schema_to_render = get_dataset_settings_schema(selected_dataset, window_length)

    # Render dataset settings with temporary key prefix
    initial_dataset_params = st.session_state.dataset_params.copy() if st.session_state.dataset_params else {}

    temp_dataset_params = render_settings_from_schema(
        schema_to_render,
        initial_values=initial_dataset_params,
        key_prefix="temp_dataset_",
        force_update=st.session_state.force_update_widgets,
    )

    # Reset force update flag
    if st.session_state.force_update_widgets:
        st.session_state.force_update_widgets = False

    # --- Feature Selection ---
    registry = DatasetRegistry()
    _render_feature_selection(selected_dataset, dataset_key, registry, temp_dataset_params, window_length)

    st.markdown("---")

    if st.button("Apply Dataset Changes"):
        # Update session state directly
        st.session_state.dataset_params = temp_dataset_params.copy()
        if "dataset_params_by_dataset" not in st.session_state:
            st.session_state.dataset_params_by_dataset = {}
        st.session_state.dataset_params_by_dataset[dataset_key] = temp_dataset_params.copy()
        st.rerun()
