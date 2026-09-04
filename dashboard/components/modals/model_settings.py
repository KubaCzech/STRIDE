import streamlit as st
from dashboard.components.settings import render_settings_from_schema


def _on_model_setting_change_handler(model_available_settings, model_name):
    """Callback when model settings dropdown changes"""
    selectbox_key = f"model_setting_selectbox_{model_name}"
    setting_key = f"selected_model_setting_{model_name}"
    force_key = f"force_update_model_widgets_{model_name}"

    selected = st.session_state.get(selectbox_key, "Not selected")
    if selected == "Not selected":
        st.session_state[setting_key] = None
        st.session_state.model_params = {}
    else:
        st.session_state[setting_key] = selected
        st.session_state.model_params = model_available_settings[selected].copy()
        st.session_state[force_key] = True

    if "model_params_by_model" not in st.session_state:
        st.session_state.model_params_by_model = {}
    st.session_state.model_params_by_model[model_name] = st.session_state.model_params.copy()


def _get_cleaned_model_settings(temp_model):
    """Retrieves and cleans available settings from the model."""
    model_available_settings = temp_model.get_available_settings()
    cleaned_available_settings = {}
    default_preset_name = None

    if model_available_settings:
        for name, params in model_available_settings.items():
            if params.get("default"):
                default_preset_name = name
            cleaned_params = params.copy()
            cleaned_params.pop("default", None)
            cleaned_available_settings[name] = cleaned_params

    return cleaned_available_settings, default_preset_name


def _sync_selected_model_setting(cleaned_available_settings, model_name):
    """Syncs selected setting with current params if fresh open."""
    has_temp_keys = any(k.startswith("temp_model_param_") for k in st.session_state.keys())
    setting_key = f"selected_model_setting_{model_name}"

    model_params_by_model = st.session_state.get("model_params_by_model", {})
    current_params = model_params_by_model.get(model_name, st.session_state.get("model_params", {}))

    if not has_temp_keys and current_params:
        st.session_state[setting_key] = None
        for name, preset_params in cleaned_available_settings.items():
            is_match = True
            for k, v in preset_params.items():
                if current_params.get(k) != v:
                    is_match = False
                    break
            if is_match:
                st.session_state[setting_key] = name
                break


def _check_model_preset_modification(cleaned_available_settings, model_name):
    """Checks if the current preset has been modified."""
    setting_key = f"selected_model_setting_{model_name}"
    selectbox_key = f"model_setting_selectbox_{model_name}"
    force_key = f"force_update_model_widgets_{model_name}"

    selected_setting = st.session_state.get(setting_key)
    if selected_setting and selected_setting in cleaned_available_settings and not st.session_state.get(force_key, False):
        current_preset_params = cleaned_available_settings[selected_setting]
        is_modified = False

        for k, v in current_preset_params.items():
            widget_key = f"temp_model_param_{k}"
            if widget_key in st.session_state:
                if st.session_state[widget_key] != v:
                    is_modified = True
                    break

        if is_modified:
            st.session_state[setting_key] = None
            st.session_state[selectbox_key] = "Not selected"


def _render_model_preset_selectbox(cleaned_available_settings, model_name):
    """Renders the model preset dropdown."""
    if not cleaned_available_settings:
        return

    setting_key = f"selected_model_setting_{model_name}"
    selectbox_key = f"model_setting_selectbox_{model_name}"
    selected_setting = st.session_state.get(setting_key)

    model_setting_options = list(cleaned_available_settings.keys())

    if selected_setting is None:
        model_setting_options = ["Not selected"] + model_setting_options

    if selectbox_key not in st.session_state:
        st.session_state[selectbox_key] = "Not selected"

    if selected_setting in cleaned_available_settings:
        st.session_state[selectbox_key] = selected_setting
    elif selected_setting is None:
        st.session_state[selectbox_key] = "Not selected"

    st.selectbox(
        "Select Model Preset",
        options=model_setting_options,
        key=selectbox_key,
        on_change=_on_model_setting_change_handler,
        args=(cleaned_available_settings, model_name),
        help="Choose a preset configuration for the model.",
    )


def _render_model_preset_selection(temp_model):
    """Renders the dropdown for selecting model presets."""
    model_name = temp_model.name
    setting_key = f"selected_model_setting_{model_name}"
    force_key = f"force_update_model_widgets_{model_name}"

    cleaned_available_settings, default_preset_name = _get_cleaned_model_settings(temp_model)

    # Initialize session state for selected model setting if not exists
    if setting_key not in st.session_state:
        st.session_state[setting_key] = None

    _sync_selected_model_setting(cleaned_available_settings, model_name)

    current_model_params = st.session_state.get("model_params_by_model", {}).get(model_name, {})

    # Auto-select default if nothing selected yet and default exists
    if st.session_state[setting_key] is None and default_preset_name and not current_model_params:
        st.session_state[setting_key] = default_preset_name
        preset_params = cleaned_available_settings[default_preset_name].copy()
        st.session_state.model_params = preset_params
        if "model_params_by_model" not in st.session_state:
            st.session_state.model_params_by_model = {}
        st.session_state.model_params_by_model[model_name] = preset_params.copy()
        st.session_state[force_key] = True

    # Track if we need to force update the model widgets
    if force_key not in st.session_state:
        st.session_state[force_key] = False

    _check_model_preset_modification(cleaned_available_settings, model_name)
    _render_model_preset_selectbox(cleaned_available_settings, model_name)


@st.dialog("Model Settings")
def open_model_settings_modal(selected_model_class):
    # Instantiate temporarily to get schema/settings/display_name
    temp_model = selected_model_class()
    model_name = temp_model.name
    force_key = f"force_update_model_widgets_{model_name}"

    st.write(f"Configure advanced settings for **{temp_model.display_name}**.")

    # 6. Model Preset Selection
    _render_model_preset_selection(temp_model)

    # --- Model Settings ---
    model_schema = temp_model.get_settings_schema()

    current_params = st.session_state.get("model_params_by_model", {}).get(model_name, st.session_state.get("model_params"))

    temp_model_params = render_settings_from_schema(
        model_schema,
        initial_values=current_params if current_params else None,
        key_prefix="temp_model_",
        force_update=st.session_state.get(force_key, False),
    )

    if st.session_state.get(force_key, False):
        st.session_state[force_key] = False

    if st.button("Apply Model Changes"):
        # Update session state
        st.session_state.model_params = temp_model_params
        if "model_params_by_model" not in st.session_state:
            st.session_state.model_params_by_model = {}
        st.session_state.model_params_by_model[model_name] = temp_model_params.copy()
        st.rerun()
