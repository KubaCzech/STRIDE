"""Modal dialog for stitching two datasets into a semi-synthetic drift stream (Shaker Protocol)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stride.datasets import (
    DATASETS,
    DatasetRegistry,
    DriftBlender,
    FeatureMatcher,
    SemiSyntheticDriftDataset,
    get_sample_wine_quality_data,
    reload_datasets,
)


def _load_concept_source(label: str, key_prefix: str) -> tuple[pd.DataFrame | None, pd.Series | None, str]:
    """Helper to load a concept from existing dataset, uploaded CSV, or benchmark fixture."""
    st.markdown(f"#### {label}")
    source_type = st.radio(
        f"Source Type for {label}",
        options=["Existing Dataset", "Upload CSV", "Benchmark (Wine Quality)"],
        horizontal=True,
        key=f"{key_prefix}_source_type",
    )

    if source_type == "Existing Dataset":
        available_ds = [k for k in DATASETS.keys() if not k.startswith("➕")]
        selected_ds_name = st.selectbox(
            f"Select Dataset for {label}",
            options=available_ds,
            key=f"{key_prefix}_existing_name",
        )
        ds = DATASETS.get(selected_ds_name)
        if ds is not None:
            try:
                gen_params = ds.get_params()
                gen_params["n_samples_before"] = 300
                gen_params["n_samples_after"] = 300
                X, y = ds.generate(**gen_params)
                return X, y, selected_ds_name
            except Exception as e:
                st.error(f"Error loading {selected_ds_name}: {e}")
                return None, None, selected_ds_name

    elif source_type == "Upload CSV":
        uploaded_file = st.file_uploader(f"Choose CSV for {label}", type=["csv"], key=f"{key_prefix}_file")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                cols = list(df.columns)
                target_col = st.selectbox(
                    f"Target column for {label}",
                    options=cols,
                    index=len(cols) - 1,
                    key=f"{key_prefix}_target_col",
                )
                y = df[target_col]
                X = df.drop(columns=[target_col])
                return X, y, uploaded_file.name.replace(".csv", "")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return None, None, "Uploaded"

    else:  # Benchmark (Wine Quality)
        variant = st.selectbox(
            f"Wine Variant for {label}",
            options=["Red Wine", "White Wine"],
            index=0 if "A" in key_prefix.upper() or "1" in label else 1,
            key=f"{key_prefix}_wine_variant",
        )
        (X_red, y_red), (X_white, y_white) = get_sample_wine_quality_data(n_samples=300, random_state=42)
        if variant == "Red Wine":
            return X_red, y_red, "UCI Red Wine Quality"
        else:
            return X_white, y_white, "UCI White Wine Quality"

    return None, None, ""


@st.dialog("🔀 Stitch Datasets (Semi-Synthetic Drift)", width="large")
def open_dataset_stitcher_modal() -> None:
    """Renders the dataset stitching and feature alignment modal dialog."""
    st.markdown(
        """
        Synthesize non-stationary data streams from two empirical datasets or sub-populations
        $(\\mathcal{D}_A \\to \\mathcal{D}_B)$ with mathematically controlled drift transitions,
        ground-truth change points, and feature alignment following the **Shaker Protocol**
        *(Shaker & Hüllermeier, 2015)*.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Concept Selection",
            "2. Feature Alignment",
            "3. Drift Schedule",
            "4. Preview & Register",
        ]
    )

    # Tab 1: Concept Selection
    with tab1:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            X_a, y_a, name_a = _load_concept_source("Concept A (Pre-drift)", "stitch_concept_a")
            if X_a is not None:
                st.caption(f"Shape: {X_a.shape[0]} samples × {X_a.shape[1]} features")
        with col_c2:
            X_b, y_b, name_b = _load_concept_source("Concept B (Post-drift)", "stitch_concept_b")
            if X_b is not None:
                st.caption(f"Shape: {X_b.shape[0]} samples × {X_b.shape[1]} features")

    if X_a is None or X_b is None or y_a is None or y_b is None:
        st.info("Please select and load both Concept A and Concept B in Tab 1 to proceed.")
        return

    # Tab 2: Feature Alignment
    with tab2:
        st.markdown("#### Feature Alignment Strategy")
        strat_display = {
            "exact": "Exact Schema Matching (Matches common column names)",
            "statistical": "Statistical Bipartite Matching (Wasserstein / KS bipartite assignment)",
            "manual": "Manual Mapping Dictionary",
            "subspace": "Shared Latent Subspace (PCA Projection)",
        }
        match_strategy = st.selectbox(
            "Alignment Strategy",
            options=list(strat_display.keys()),
            format_func=lambda s: strat_display[s],
            key="stitch_match_strategy",
        )

        manual_mapping: dict[str, str] = {}
        if match_strategy == "manual":
            st.markdown("##### Manual Column Pairing")
            cols_a = list(X_a.columns)
            cols_b = list(X_b.columns)
            for c_a in cols_a:
                default_idx = cols_b.index(c_a) if c_a in cols_b else 0
                chosen_b = st.selectbox(
                    f"Map '{c_a}' (Concept A) to:",
                    options=cols_b,
                    index=default_idx,
                    key=f"stitch_map_{c_a}",
                )
                manual_mapping[c_a] = chosen_b

        st.markdown("---")
        st.markdown("#### Distribution Normalization")
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            align_distributions = st.checkbox(
                "Apply Distribution Normalization",
                value=False,
                help="Scales features to eliminate arbitrary scale differences.",
                key="stitch_align_dist",
            )
        with col_n2:
            scaler_type = st.selectbox(
                "Scaler Type",
                options=["standard", "minmax"],
                format_func=lambda s: "StandardScaler (Zero-Mean, Unit-Var)" if s == "standard" else "MinMaxScaler ([0, 1])",
                disabled=not align_distributions,
                key="stitch_scaler_type",
            )
        with col_n3:
            scaler_mode = st.selectbox(
                "Scaler Mode",
                options=["joint", "per_concept"],
                format_func=lambda m: (
                    "Joint (Preserves relative shifts)" if m == "joint" else "Per-Concept (Isolates P(y|x) shift)"
                ),
                disabled=not align_distributions,
                key="stitch_scaler_mode",
            )

    # Tab 3: Drift Schedule
    with tab3:
        st.markdown("#### Drift Transition Schedule")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sched_display = {
                "abrupt": "Abrupt (Hard switch at t_0)",
                "gradual": "Gradual (Sigmoidal Shaker Schedule)",
                "incremental": "Incremental (Nearest-Neighbor Linear Interpolation)",
                "recurring": "Recurring (Harmonic Periodic Modulation)",
            }
            schedule = st.selectbox(
                "Transition Schedule",
                options=list(sched_display.keys()),
                format_func=lambda s: sched_display[s],
                index=1,
                key="stitch_schedule",
            )
            n_samples = st.slider(
                "Total Stream Samples",
                min_value=200,
                max_value=4000,
                value=1000,
                step=100,
                key="stitch_n_samples",
            )
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=999999,
                value=42,
                key="stitch_seed",
            )

        with col_s2:
            t_0 = st.slider(
                "Inflection Point (t_0)",
                min_value=50,
                max_value=n_samples - 50,
                value=n_samples // 2,
                step=50,
                key="stitch_t_0",
            )
            drift_width = st.slider(
                "Transition Width (w)",
                min_value=10,
                max_value=min(n_samples, 600),
                value=200,
                step=10,
                key="stitch_width",
            )
            period = 500
            if schedule == "recurring":
                period = st.slider(
                    "Oscillation Period (T)",
                    min_value=100,
                    max_value=1000,
                    value=400,
                    step=50,
                    key="stitch_period",
                )

    # Tab 4: Preview & Register
    with tab4:
        st.markdown("#### Transition Probability Preview")
        t_axis = np.arange(n_samples)
        if schedule == "abrupt":
            prob_curve = np.zeros(n_samples)
            prob_curve[t_0:] = 1.0
        elif schedule == "gradual":
            prob_curve = 1.0 / (1.0 + np.exp(np.clip(-4.0 * (t_axis - t_0) / drift_width, -100, 100)))
        elif schedule == "incremental":
            t_s = max(0, int(t_0 - drift_width / 2))
            t_e = min(n_samples - 1, int(t_0 + drift_width / 2))
            prob_curve = np.zeros(n_samples)
            prob_curve[t_e:] = 1.0
            prob_curve[t_s:t_e] = (t_axis[t_s:t_e] - t_s) / max(1, (t_e - t_s))
        else:  # recurring
            prob_curve = 0.5 * (1.0 + np.sin(2.0 * np.pi * t_axis / period - np.pi / 2.0))

        fig_prob = go.Figure()
        fig_prob.add_trace(
            go.Scatter(
                x=t_axis,
                y=prob_curve,
                mode="lines",
                name="P(Concept B | t)",
                line=dict(color="#FF6B35", width=2.5),
            )
        )
        fig_prob.add_vline(x=t_0, line_dash="dash", line_color="red", annotation_text="t_0")
        if schedule in ("gradual", "incremental"):
            fig_prob.add_vrect(
                x0=max(0, t_0 - drift_width // 2),
                x1=min(n_samples - 1, t_0 + drift_width // 2),
                fillcolor="rgba(255, 107, 53, 0.15)",
                layer="below",
                line_width=0,
            )
        fig_prob.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=25, b=25),
            xaxis_title="Time Step t",
            yaxis_title="P(Concept B)",
            yaxis=dict(range=[-0.05, 1.05]),
        )
        st.plotly_chart(fig_prob, width="stretch")

        # Feature Alignment & Drifting Features Diagnostics
        try:
            matcher = FeatureMatcher(
                match_strategy=match_strategy,
                manual_mapping=manual_mapping if match_strategy == "manual" else None,
                align_distributions=align_distributions,
                scaler_type=scaler_type if align_distributions else None,
                scaler_mode=scaler_mode,
                random_state=random_seed,
            )
            align_res = matcher.align(X_a, X_b)

            st.markdown("##### Drifting Features Diagnostics")
            st.dataframe(align_res.drift_diagnostics, width="stretch")
            if align_res.drifting_features:
                st.success(
                    f"Detected {len(align_res.drifting_features)} drifting feature(s): {', '.join(align_res.drifting_features)}"
                )
            else:
                st.info("No statistically significant drifting features detected under KS test.")
        except Exception as err:
            st.error(f"Feature alignment error: {err}")
            return

        st.markdown("---")
        default_name = f"SemiSynth_{name_a[:10]}_{name_b[:10]}".replace(" ", "_").replace("-", "_")
        target_name = st.text_input("New Dataset Name", value=default_name, key="stitch_final_name")

        if st.button("🚀 Stitch & Register Dataset", type="primary", width="stretch"):
            if not target_name.strip():
                st.error("Please enter a valid dataset name.")
                return

            with st.spinner("Synthesizing and registering dataset..."):
                try:
                    blender = DriftBlender(
                        schedule=schedule,
                        n_samples=n_samples,
                        t_0=t_0,
                        w=drift_width,
                        period=period,
                        random_state=random_seed,
                    )
                    blend_res = blender.blend(
                        align_res.X_a_aligned,
                        y_a,
                        align_res.X_b_aligned,
                        y_b,
                    )

                    combined_df = blend_res.X.copy()
                    combined_df["target"] = blend_res.y
                    combined_df["_concept"] = blend_res.concept_stream

                    recipe = {
                        "source_a": name_a,
                        "source_b": name_b,
                        "match_strategy": match_strategy,
                        "manual_mapping": manual_mapping if match_strategy == "manual" else None,
                        "align_distributions": align_distributions,
                        "scaler_type": scaler_type,
                        "scaler_mode": scaler_mode,
                        "schedule": schedule,
                        "n_samples": n_samples,
                        "t_0": t_0,
                        "drift_width": drift_width,
                        "period": period,
                        "random_seed": random_seed,
                        "ground_truth_drift_points": blend_res.ground_truth_drift_points,
                        "drift_intervals": blend_res.drift_intervals,
                        "drifting_features": align_res.drifting_features,
                        "feature_mapping": align_res.feature_mapping,
                    }

                    registry = DatasetRegistry()
                    registry.save_semi_synthetic_dataset(
                        name=target_name,
                        df=combined_df,
                        target_column="target",
                        recipe=recipe,
                    )

                    reload_datasets()
                    st.session_state.selected_dataset_key = target_name
                    st.success(f"Dataset '{target_name}' registered successfully!")
                    st.rerun()

                except Exception as ex:
                    st.error(f"Failed to generate and register dataset: {ex}")
