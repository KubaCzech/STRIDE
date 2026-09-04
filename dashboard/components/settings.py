import streamlit as st


def _render_list_of_int_input(setting, key, value, default, on_change):
    label = setting["label"]
    # Ensure value is a list
    current_list = value if isinstance(value, list) else list(value) if isinstance(value, tuple) else []
    if not current_list and default:
        current_list = default if isinstance(default, list) else list(default)

    # 1. Input for list length
    len_key = f"{key}_len"

    st.markdown(f"**{label} Configuration**")
    num_items = st.number_input(
        "Number of Layers",
        min_value=1,
        max_value=20,
        value=len(current_list) if len(current_list) > 0 else 1,
        step=1,
        key=len_key,
        on_change=on_change,
        help=f"Number of layers for {label}",
    )

    # Adjust list size locally to render correct number of inputs
    if len(current_list) < num_items:
        current_list.extend([10] * (num_items - len(current_list)))
    elif len(current_list) > num_items:
        current_list = current_list[:num_items]

    new_list = []
    for i in range(num_items):
        item_key = f"{key}_item_{i}"
        item_val = int(current_list[i])
        new_val = st.number_input(
            f"Layer {i + 1} Size", min_value=1, value=item_val, step=1, key=item_key, on_change=on_change
        )
        new_list.append(new_val)

    return new_list


def _render_simple_input(setting, key, on_change):
    label = setting["label"]
    help_text = setting.get("help", "")
    st_type = setting["type"]

    if st_type == "int":
        return st.number_input(
            label,
            min_value=setting.get("min_value"),
            max_value=setting.get("max_value"),
            step=setting.get("step", 1),
            help=help_text,
            key=key,
            on_change=on_change,
        )
    elif st_type == "float":
        return st.number_input(
            label,
            min_value=setting.get("min_value"),
            max_value=setting.get("max_value"),
            step=setting.get("step", 0.1),
            help=help_text,
            key=key,
            on_change=on_change,
        )
    elif st_type == "text":
        return st.text_input(label, help=help_text, key=key, on_change=on_change)
    elif st_type == "bool":
        return st.checkbox(label, help=help_text, key=key, on_change=on_change)
    elif st_type == "file":
        return st.file_uploader(label, type=setting.get("allowed_types"), help=help_text, key=key, on_change=on_change)
    return None


def render_settings_from_schema(
    schema: list[dict], on_change=None, initial_values=None, force_update=False, key_prefix=""
) -> dict:
    """
    Render Streamlit widgets based on a settings schema.

    Parameters
    ----------
    schema : list[dict]
        A list of dictionaries describing the settings.
    on_change : callable, optional
        Callback function to call when any parameter changes.
    initial_values : dict, optional
        Dictionary of initial values to use instead of defaults.
    force_update : bool, optional
        If True, force update widget values even if they already exist in session state.
    key_prefix : str, optional
        Prefix to add to widget keys to avoid collisions.

    Returns
    -------
    dict
        A dictionary of selected parameter values.
    """
    params = {}

    if not schema:
        return params

    for setting in schema:
        name = setting["name"]
        default = setting.get("default")
        key = f"{key_prefix}param_{name}"

        # Use initial value if provided, otherwise use default
        value = initial_values.get(name, default) if initial_values else default

        # Initialize session state if not exists, or force update if requested
        if key not in st.session_state:
            st.session_state[key] = value
        elif force_update and initial_values and name in initial_values:
            # Only update if force_update is True (when preset changes)
            st.session_state[key] = value

        if setting["type"] == "list_of_int":
            params[name] = _render_list_of_int_input(setting, key, value, default, on_change)
        else:
            params[name] = _render_simple_input(setting, key, on_change)

    return params
