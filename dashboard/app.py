import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'recurrence')))

import warnings  # noqa: E402
import logging  # noqa: E402
import streamlit as st  # noqa: E402
from dashboard.components.sidebar import render_sidebar_datasource_config, render_sidebar_window_selection  # noqa: E402

# Suppress TensorFlow oneDNN custom operations logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Suppress specific TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Also try to suppress via warnings just in case it's reachable that way
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*reset_default_graph.*')

from src.datasets import DATASETS  # noqa: E402
from dashboard.components.tabs import (  # noqa: E402
    render_data_visualization_tab,
    render_feature_importance_analysis_tab,
    render_drift_detection_tab,
    render_decision_boundary_tab,
    render_prototype_analysis_tab,
    render_clustering_analysis_tab
)
from dashboard.components.modals.info import show_info_modal  # noqa: E402


# --- App Configuration ---
st.set_page_config(
    page_title="Concept Drift Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---


# --- Main App ---

col_title, col_info = st.columns([0.9, 0.1])

with col_title:
    st.markdown("### 📊 Concept Drift Analysis Dashboard")

with col_info:
    if st.button("ℹ️ Info", help="Show dashboard information"):
        show_info_modal()

# --- Sidebar for User Input ---
# 1. Render Datasource Configuration (load dataset parameters)

datasource_config = render_sidebar_datasource_config()

window_length = datasource_config["window_length"]
dataset_key = datasource_config["dataset_key"]
dataset_params = datasource_config["dataset_params"]
selected_model_class = datasource_config["selected_model_class"]
model_params = datasource_config["model_params"]


# --- Data Generation ---
@st.cache_data
def generate_data(dataset_name, window_length_val, **kwargs):
    """Cached function to generate data."""
    dataset = DATASETS.get(dataset_name)
    if not dataset:
        st.error(f"Unknown dataset: {dataset_name}")
        return None, None

    gen_params = dataset.get_params()
    gen_params.update(kwargs)

    # Always add window_length
    gen_params["window_length"] = window_length_val

    # Convert window-based parameters to samples if present
    if "n_windows_before" in gen_params:
        gen_params["n_samples_before"] = gen_params.pop("n_windows_before") * window_length_val
    if "n_windows_after" in gen_params:
        gen_params["n_samples_after"] = gen_params.pop("n_windows_after") * window_length_val

    try:
        return dataset.generate(**gen_params)
    except Exception as e:
        st.error(f"Error generating data: {e}")
        return None, None


X, y = generate_data(
    dataset_key,
    window_length,
    **dataset_params
)

if X is not None:
    feature_names = X.columns.tolist()
else:
    feature_names = []
    # If data generation fails or no data, we can't really set window selection constraints properly.
    # But we should still render the inputs to avoid UI disappearing, just with default/fallback max.

# 2. Render Window Selection (now that we know data length)
if X is not None:
    max_samples = len(X)
else:
    max_samples = 1000  # Fallback

window_before_start, window_after_start = render_sidebar_window_selection(max_samples, window_length)

if X is None:
    st.warning("Please upload a CSV file or select a valid dataset to proceed.")
    st.stop()
# --- Feature Selection ---
# Feature selection is now handled in the sidebar configuration modal
selected_features = datasource_config.get("selected_features", [])

# If no features selected (or first run), default to all if empty, but sidebar should handle it.
if not selected_features and X is not None:
    # Fallback if state wasn't initialized
    selected_features = X.columns.tolist()

# Filter X and update feature_names
if selected_features:
    # Ensure selected features exist in X (in case params changed and X changed schema)
    valid_features = [f for f in selected_features if f in X.columns]
    if not valid_features:
        st.warning("Selected features no longer exist in the dataset. Resetting to all features.")
        valid_features = X.columns.tolist()
    X = X[valid_features]
    feature_names = valid_features
    st.session_state[f"selected_features_{dataset_key}"] = valid_features
else:
    # Just in case
    feature_names = X.columns.tolist()
    st.session_state[f"selected_features_{dataset_key}"] = feature_names

if not feature_names:
    st.error("No features selected or available.")
    st.stop()


# --- Tabs (Navigation) ---
tabs = [
    "Dataset",
    "Drift Detection",
    "Decision Boundary",
    "Feature Importance Analysis",
    "Clustering Analysis",
    "Prototype Analysis"
]

# Initialize session state for active tab if it doesn't exist
if "active_tab" not in st.session_state:
    st.session_state.active_tab = tabs[0]


# Custom CSS to style radio buttons as tabs
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


css_file = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
load_css(css_file)

# Use radio buttons for navigation to allow state persistence
active_tab = st.radio(
    "Select Analysis Module",
    tabs,
    horizontal=True,
    key="active_tab",
    label_visibility="collapsed"
)

# Prepare data slices for analysis tabs
start_before = window_before_start
end_before = start_before + window_length
start_after = window_after_start
end_after = start_after + window_length

# Ensure indices are within bounds
if end_before > len(X):
    st.error(f"Window 'Before' goes out of bounds: starts at {start_before}, ends at {end_before}, data length {len(X)}")
    st.stop()
if end_after > len(X):
    st.error(f"Window 'After' goes out of bounds: starts at {start_after}, ends at {end_after}, data length {len(X)}")
    st.stop()

X_before = X.iloc[start_before:end_before] if hasattr(X, "iloc") else X[start_before:end_before]
y_before = y.iloc[start_before:end_before] if hasattr(y, "iloc") else y[start_before:end_before]
X_after = X.iloc[start_after:end_after] if hasattr(X, "iloc") else X[start_after:end_after]
y_after = y.iloc[start_after:end_after] if hasattr(y, "iloc") else y[start_after:end_after]


if active_tab == tabs[0]:
    render_data_visualization_tab(X, y, X_before, y_before, X_after, y_after, feature_names,
                                  window_before_start, window_after_start, window_length)

elif active_tab == tabs[1]:
    render_drift_detection_tab(X, y, window_length,
                               model_class=selected_model_class,
                               model_params=model_params)

elif active_tab == tabs[2]:
    render_decision_boundary_tab(X_before, y_before, X_after, y_after,
                                 model_class=selected_model_class,
                                 model_params=model_params,
                                 feature_names=feature_names)

elif active_tab == tabs[3]:
    render_feature_importance_analysis_tab(X_before, y_before, X_after, y_after,
                                           feature_names,
                                           model_class=selected_model_class,
                                           model_params=model_params)

elif active_tab == tabs[4]:
    render_clustering_analysis_tab(X_before, y_before, X_after, y_after)

elif active_tab == tabs[5]:
    render_prototype_analysis_tab(X, y, window_length)
