import unittest
import sys
import os
import inspect

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import MLPModel, RandomForestModel  # noqa: E402
from src.datasets.linear_weight_inversion_drift import LinearWeightInversionDriftDataset  # noqA: E402
from streamlit.testing.v1 import AppTest  # noqA: E402


class TestDashboardState(unittest.TestCase):
    def test_model_parameter_filtering(self):
        """
        Verify that model parameter filtering strips unexpected keyword arguments
        and allows seamless model instantiation without TypeError.
        """
        mlp_params = {"hidden_layer_sizes": [10, 10], "max_iter": 500, "alpha": 0.00001, "random_state": 42}

        # Filtering MLP params for RandomForestModel
        rf_cls = RandomForestModel
        rf_valid_keys = set(inspect.signature(rf_cls.__init__).parameters.keys()) - {"self"}
        rf_filtered = {k: v for k, v in mlp_params.items() if k in rf_valid_keys}

        self.assertNotIn("hidden_layer_sizes", rf_filtered)
        self.assertNotIn("alpha", rf_filtered)
        self.assertNotIn("max_iter", rf_filtered)
        self.assertEqual(rf_filtered.get("random_state"), 42)

        # Ensure instantiating RandomForestModel with filtered params works
        rf_model = rf_cls(**rf_filtered)
        self.assertIsInstance(rf_model, RandomForestModel)

        # Filtering RF params for MLPModel
        rf_params = {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2, "random_state": 42}
        mlp_cls = MLPModel
        mlp_valid_keys = set(inspect.signature(mlp_cls.__init__).parameters.keys()) - {"self"}
        mlp_filtered = {k: v for k, v in rf_params.items() if k in mlp_valid_keys}

        self.assertNotIn("n_estimators", mlp_filtered)
        self.assertNotIn("max_depth", mlp_filtered)
        self.assertNotIn("min_samples_split", mlp_filtered)
        self.assertEqual(mlp_filtered.get("random_state"), 42)

        # Ensure instantiating MLPModel with filtered params works
        mlp_model = mlp_cls(**mlp_filtered)
        self.assertIsInstance(mlp_model, MLPModel)

    def test_feature_selection_sanitization_logic(self):
        """
        Verify that when dataset features decrease, the selected features are sanitized
        to only contain valid features present in the new preview.
        """
        dataset = LinearWeightInversionDriftDataset()
        X_11, _ = dataset.generate(n_features=11)
        features_11 = X_11.columns.tolist()
        self.assertEqual(len(features_11), 11)

        # Initially all 11 features selected
        selected_features = features_11.copy()

        # Generate with 10 features
        X_10, _ = dataset.generate(n_features=10)
        preview_10 = X_10.columns.tolist()
        self.assertEqual(len(preview_10), 10)

        # Sanitize logic as implemented in dataset_settings modal
        valid_selection = [f for f in selected_features if f in preview_10]
        if not valid_selection:
            valid_selection = preview_10

        self.assertEqual(len(valid_selection), 10)
        self.assertNotIn("X11", valid_selection)
        self.assertEqual(valid_selection, preview_10)

        # Test when user selected a specific subset including deleted feature
        user_selection = ["X1", "X3", "X11"]
        valid_subset = [f for f in user_selection if f in preview_10]
        self.assertEqual(valid_subset, ["X1", "X3"])

    def test_streamlit_multiselect_feature_reduction_regression(self):
        """
        Regression test simulating Streamlit rerun when reducing features in multiselect.
        Ensures StreamlitAPIException is not raised.
        """
        script = (
            "import streamlit as st\n\n"
            "n = st.number_input('n_features', min_value=1, value=11, key='n_feat')\n"
            "preview_features = [f'X{i+1}' for i in range(n)]\n\n"
            "dataset_key = 'linear_weight_inversion_drift'\n"
            "feature_key = f'selected_features_{dataset_key}'\n"
            "multiselect_key = f'multiselect_{dataset_key}'\n\n"
            "if feature_key not in st.session_state:\n"
            "    st.session_state[feature_key] = preview_features\n\n"
            "# Sanitization\n"
            "current_selection = st.session_state.get(feature_key, preview_features)\n"
            "valid_selection = [f for f in current_selection if f in preview_features]\n"
            "if not valid_selection:\n"
            "    valid_selection = preview_features\n"
            "st.session_state[feature_key] = valid_selection\n\n"
            "if multiselect_key in st.session_state:\n"
            "    widget_val = [f for f in st.session_state[multiselect_key] if f in preview_features]\n"
            "    if not widget_val:\n"
            "        widget_val = valid_selection\n"
            "    st.session_state[multiselect_key] = widget_val\n"
            "    sel = st.multiselect('Include Features', options=preview_features, key=multiselect_key)\n"
            "else:\n"
            "    sel = st.multiselect(\n"
            "        'Include Features', options=preview_features, default=valid_selection, key=multiselect_key\n"
            "    )\n\n"
            "st.session_state[feature_key] = sel\n"
        )

        at = AppTest.from_string(script)
        at.run()
        self.assertFalse(at.exception, f"Initial run raised exception: {at.exception}")
        self.assertEqual(len(at.multiselect(key="multiselect_linear_weight_inversion_drift").value), 11)

        # Decrease n_features to 10
        at.number_input(key="n_feat").set_value(10)
        at.run()
        self.assertFalse(at.exception, f"Reducing features raised exception: {at.exception}")
        self.assertEqual(len(at.multiselect(key="multiselect_linear_weight_inversion_drift").value), 10)

    def test_streamlit_model_switching_regression(self):
        """
        Regression test simulating model switching in the dashboard.
        Ensures model parameters are isolated per model and no TypeError is raised.
        """
        script = (
            "import inspect\n"
            "import streamlit as st\n"
            "from src.models import MODELS\n\n"
            "model_key = st.selectbox('Choose a Model', options=list(MODELS.keys()), key='model_choice')\n"
            "selected_model_class = MODELS[model_key]\n\n"
            "if 'model_params_by_model' not in st.session_state:\n"
            "    st.session_state['model_params_by_model'] = {}\n\n"
            "if 'current_model_key' not in st.session_state or st.session_state['current_model_key'] != model_key:\n"
            "    st.session_state['current_model_key'] = model_key\n"
            "    st.session_state['model_params'] = st.session_state['model_params_by_model'].get(model_key, {}).copy()\n"
            "else:\n"
            "    if 'model_params' in st.session_state:\n"
            "        st.session_state['model_params_by_model'][model_key] = st.session_state['model_params']\n\n"
            "valid_keys = set(inspect.signature(selected_model_class.__init__).parameters.keys()) - {'self'}\n"
            "filtered_params = {k: v for k, v in st.session_state['model_params'].items() if k in valid_keys}\n"
            "st.session_state['model_params'] = filtered_params\n"
            "st.session_state['model_params_by_model'][model_key] = filtered_params\n\n"
            "# Simulate tab creating model instance\n"
            "model_instance = selected_model_class(**filtered_params)\n"
            "st.session_state['model_instance'] = model_instance\n"
        )

        at = AppTest.from_string(script)
        at.run()
        self.assertFalse(at.exception, f"Initial run raised exception: {at.exception}")

        # Simulate applying custom MLP params
        at.session_state["model_params_by_model"]["mlp"] = {"hidden_layer_sizes": [50, 50], "max_iter": 1000, "alpha": 0.001}
        at.session_state["model_params"] = at.session_state["model_params_by_model"]["mlp"].copy()

        # Switch to random_forest without entering modal
        at.selectbox(key="model_choice").select("random_forest")
        at.run()
        self.assertFalse(at.exception, f"Switching to random_forest raised exception: {at.exception}")
        self.assertIsInstance(at.session_state["model_instance"], RandomForestModel)

        # Simulate applying custom RF params
        at.session_state["model_params_by_model"]["random_forest"] = {"n_estimators": 250, "max_depth": 10}
        at.session_state["model_params"] = at.session_state["model_params_by_model"]["random_forest"].copy()

        # Switch back to mlp
        at.selectbox(key="model_choice").select("mlp")
        at.run()
        self.assertFalse(at.exception, f"Switching back to mlp raised exception: {at.exception}")
        self.assertIsInstance(at.session_state["model_instance"], MLPModel)
        self.assertEqual(at.session_state["model_params"].get("hidden_layer_sizes"), [50, 50])


if __name__ == "__main__":
    unittest.main()
