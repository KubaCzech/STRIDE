import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from src.recurrence.methods import cluster_windows, get_drift_from_clusters, median_mask
from src.recurrence.visualisation import plot_cluster_timeline
from river import forest
from src.recurrence.protree.explainers import APete
from src.recurrence.full_window_storage import FullWindowStorage


def render_prototype_analysis_tab(X, y, window_length):  # noqa: C901
    """Main entry point for Prototype Analysis tab in the dashboard."""

    # Initialize session state
    if "prototype_storage" not in st.session_state:
        st.session_state.prototype_storage = None
    if "prototype_matrix" not in st.session_state:
        st.session_state.prototype_matrix = None
    if "prototype_labels" not in st.session_state:
        st.session_state.prototype_labels = None
    if "prototype_drift_locations" not in st.session_state:
        st.session_state.prototype_drift_locations = None
    if "prototype_dataset" not in st.session_state:
        st.session_state.prototype_dataset = None
    if "prototype_processing_complete" not in st.session_state:
        st.session_state.prototype_processing_complete = False

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data Processing", "Global View", "Window Comparison", "Local Analysis", "Distance Matrix"]
    )

    # ============================================================================
    # TAB 1: DATA PROCESSING
    # ============================================================================
    with tab1:
        st.header("Data Processing Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Stream Configuration")

            # Show info about dataset from dashboard
            total_samples = len(X)
            st.info(f"Using dataset from dashboard: {total_samples} samples")

            # Stream parameters (derived from dashboard data)
            num_windows = total_samples // window_length
            window_size = window_length

            st.metric("Number of Windows", num_windows)
            st.metric("Samples per Window", window_size)

            if num_windows < 10:
                st.warning(
                    "Warning: Number of windows is low. Consider increasing the number of windows. "
                    "Low number of windows might result in bad results during clustering."
                )

        with col2:
            st.subheader("Distance Calculation")

            measure = st.selectbox(
                "Distance Measure",
                options=["centroid_displacement", "prototype_reassignment_impact"],
                index=0,  # centroid_displacement is default
                help="Metric used to compare windows:\n"
                "- centroid_displacement: Recommended\n"
                "- prototype_reassignment_impact: Better, but slower",
                key="prototype_measure",
            )

            st.markdown("---")

            st.subheader("Clustering Configuration")
            median_filter_width = st.number_input(
                "Median Filter Width",
                min_value=1,
                max_value=15,
                value=3,
                step=2,
                help="Width of median filter for smoothing distance matrix before clustering (use odd numbers)",
                key="prototype_median_filter",
            )

            fix_outliers = st.checkbox(
                "Fix Outlier Labels",
                value=True,
                help="Smooth outlier detections by filling gaps between same labels",
                key="prototype_fix_outliers",
            )

        # Process button
        st.markdown("---")
        if st.button("Process Data", type="primary", key="prototype_process"):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Generate dataset from dashboard X, y
                status_text.text("Generating dataset...")
                X_np = X.to_numpy() if hasattr(X, "to_numpy") else X
                y_np = y.to_numpy() if hasattr(y, "to_numpy") else y

                dataset = []
                for i in range(num_windows):
                    start_idx = i * window_size
                    end_idx = start_idx + window_size
                    if end_idx <= total_samples:
                        x_block = X_np[start_idx:end_idx]
                        y_block = y_np[start_idx:end_idx]
                        dataset.append([x_block, y_block])
                    progress_bar.progress((i + 1) / (num_windows * 3))

                st.session_state.prototype_dataset = dataset

                # Setup model and storage
                status_text.text("Creating prototypes...")
                model = forest.ARFClassifier(seed=42)
                storage = FullWindowStorage()

                # Get feature names from dashboard
                feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{k}" for k in range(X.shape[1])]

                # Process each window
                for i in range(len(dataset)):
                    status_text.text(f"Processing window {i + 1}/{num_windows}...")
                    x_block, y_block = dataset[i]

                    # Convert to dicts for River
                    x_dicts = []
                    for row in x_block:
                        if hasattr(row, "tolist"):
                            row = row.tolist()
                        row_dict = {feature_names[k]: v for k, v in enumerate(row)}
                        x_dicts.append(row_dict)

                    y_list = y_block.tolist() if hasattr(y_block, "tolist") else list(y_block)

                    # Learn from data
                    for x, y_val in zip(x_dicts, y_list):
                        model.learn_one(x, y_val)

                    # Create prototypes
                    current_explainer = APete(model=model, alpha=0.01)
                    current_prototypes = current_explainer.select_prototypes(x_dicts, y_list)

                    # In case a class has 0 or 1 prototypes, run prototype selection just for this class to fix this
                    for class_name in set(y_block):
                        class_prototypes = current_prototypes[class_name]
                        if len(class_prototypes) in [0, 1, 2]:
                            # Change alpha threshold to make less prototypes
                            current_explainer = APete(model=model, alpha=0.05)

                            # Create prototypes for the class with missing prototypes
                            x_block_missing_class = []
                            for x, y in zip(x_dicts, y_list):
                                if y == class_name:
                                    x_block_missing_class.append(x)

                            y_block_missing_class = tuple([class_name] * len(x_block_missing_class))
                            x_block_missing_class = tuple(x_block_missing_class)

                            missing_prototypes = current_explainer.select_prototypes(
                                x_block_missing_class, y_block_missing_class
                            )
                            current_prototypes[class_name] = missing_prototypes[class_name]

                    # Store window data
                    storage.store_window(
                        iteration=i, x=x_dicts, y=y_list, prototypes=current_prototypes, explainer=None, drift=False
                    )

                    progress_bar.progress((num_windows + i + 1) / (num_windows * 3))

                # Compute distance matrix
                status_text.text("Computing distance matrix...")
                matrix = storage.compute_distance_matrix(measure=measure)
                progress_bar.progress(2 * num_windows / (num_windows * 3))

                # Cluster windows
                status_text.text("Clustering windows...")
                labels = cluster_windows(matrix, fix_outliers=fix_outliers, median_mask_width=median_filter_width)

                # Detect drifts
                drift_locations = get_drift_from_clusters(labels)

                # Store results in session state
                st.session_state.prototype_storage = storage
                st.session_state.prototype_matrix = matrix
                st.session_state.prototype_labels = labels
                st.session_state.prototype_drift_locations = drift_locations
                st.session_state.prototype_processing_complete = True

                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")

                st.success(f"""
                Processing Complete!
                - Stored {len(storage.get_all_iterations())} windows
                - Detected {len(drift_locations)} drifts at windows: {drift_locations}
                - Found {len(set(labels[labels != -1]))} distinct concepts
                """)

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

        # Show current status
        if st.session_state.prototype_processing_complete:
            st.info("✓ Data processed and ready for analysis. Navigate to other tabs to explore!")

            # Recompute Matrix button
            st.markdown("---")
            st.subheader("Recompute Distance Matrix and Clustering")
            st.write(
                "Use the parameters above to recompute the distance matrix and re-cluster",
                "windows without retraining the model on the same data.",
            )

            if st.button("Recompute", type="secondary", key="prototype_recompute"):
                try:
                    storage = st.session_state.prototype_storage

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Recompute distance matrix with current parameters
                    status_text.text("Recomputing distance matrix...")
                    matrix = storage.compute_distance_matrix(measure=measure)
                    progress_bar.progress(0.5)

                    # Re-cluster windows
                    status_text.text("Re-clustering windows...")
                    labels = cluster_windows(matrix, fix_outliers=fix_outliers, median_mask_width=median_filter_width)

                    # Detect drifts
                    drift_locations = get_drift_from_clusters(labels)

                    # Update session state
                    st.session_state.prototype_matrix = matrix
                    st.session_state.prototype_labels = labels
                    st.session_state.prototype_drift_locations = drift_locations

                    progress_bar.progress(1.0)
                    status_text.text("✅ Recomputation complete!")

                    st.success(f"""
                    Recomputation Complete!
                    - Used measure: {measure}
                    - Detected {len(drift_locations)} drifts at windows: {drift_locations}
                    - Found {len(set(labels[labels != -1]))} distinct concepts
                    """)

                except Exception as e:
                    st.error(f"Error during recomputation: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

    # ============================================================================
    # TAB 2: GLOBAL VIEW
    # ============================================================================
    with tab2:
        st.header("Global Stream Analysis")

        if not st.session_state.prototype_processing_complete:
            st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
        else:
            storage = st.session_state.prototype_storage
            labels = st.session_state.prototype_labels
            drift_locations = st.session_state.prototype_drift_locations

            # Global statistics
            st.subheader("Global Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Windows", len(storage.get_all_iterations()))
            with col2:
                st.metric("Detected Drifts", len(drift_locations))
            with col3:
                num_concepts = len(set(labels[labels != -1]))
                st.metric("Distinct Concepts", num_concepts)
            with col4:
                num_outliers = sum(labels == -1)
                st.metric("Outlier Windows", num_outliers)

            # Drift locations
            if drift_locations:
                st.info(f"🔄 **Detected drift locations (windows):** {', '.join(map(str, drift_locations))}")
            else:
                st.success("✓ No drifts detected - stream appears stable")

            # Additional statistics
            st.subheader("Stream Statistics")

            # Collect data across all windows
            total_samples = 0
            total_prototypes = 0
            all_classes = set()
            samples_per_window = []
            prototypes_per_window = []

            for i in storage.get_all_iterations():
                _, y, prototypes, _ = storage.get_window_data(i)
                total_samples += len(y)
                window_proto_count = sum(len(p) for p in prototypes.values())
                total_prototypes += window_proto_count
                all_classes.update(y)
                samples_per_window.append(len(y))
                prototypes_per_window.append(window_proto_count)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Samples", f"{total_samples:,}")
            with col2:
                st.metric("Total Prototypes", f"{total_prototypes:,}")
            with col3:
                st.metric("Avg Prototypes/Window", f"{np.mean(prototypes_per_window):.1f}")
            with col4:
                st.metric("Number of Classes", len(all_classes))

            # Prototype distribution statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                compression_ratio = (total_prototypes / total_samples) * 100
                st.metric("Compression Ratio", f"{compression_ratio:.2f}%")
            with col2:
                st.metric("Min Prototypes/Window", int(np.min(prototypes_per_window)))
            with col3:
                st.metric("Max Prototypes/Window", int(np.max(prototypes_per_window)))
            with col4:
                st.metric("Std Prototypes/Window", f"{np.std(prototypes_per_window):.2f}")

            st.markdown("---")

            # Class distribution over time
            st.subheader("Class Distribution Over Time")

            class_distributions = []
            all_classes = set()

            for i in storage.get_all_iterations():
                _, y, _, _ = storage.get_window_data(i)
                class_counts = pd.Series(y).value_counts()
                all_classes.update(class_counts.index)
                class_distributions.append(class_counts)

            # Create dataframe with percentages
            all_classes = sorted(list(all_classes))
            class_pct_data = []

            for i, dist in enumerate(class_distributions):
                total = sum(dist.values)
                row = {"window": i}
                for cls in all_classes:
                    row[f"class_{cls}"] = (dist.get(cls, 0) / total) * 100
                class_pct_data.append(row)

            df_class = pd.DataFrame(class_pct_data)

            fig, ax = plt.subplots(figsize=(12, 4))
            for cls in all_classes:
                ax.plot(df_class["window"], df_class[f"class_{cls}"], label=f"Class {cls}", linewidth=2)

            # Mark drift locations
            for drift in drift_locations:
                ax.axvline(x=drift, color="red", linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xlabel("Window")
            ax.set_ylabel("Percentage (%)")
            ax.set_title("Class Distribution Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # Prototype counts over time
            st.subheader("Prototype Counts Over Time")

            prototype_counts = []
            for i in storage.get_all_iterations():
                _, _, prototypes, _ = storage.get_window_data(i)
                row = {"window": i}
                for cls in all_classes:
                    row[f"class_{cls}"] = len(prototypes.get(cls, []))
                prototype_counts.append(row)

            df_proto = pd.DataFrame(prototype_counts)

            fig, ax = plt.subplots(figsize=(12, 4))
            for cls in all_classes:
                ax.plot(df_proto["window"], df_proto[f"class_{cls}"], label=f"Class {cls}", linewidth=2)

            # Mark drift locations
            for drift in drift_locations:
                ax.axvline(x=drift, color="red", linestyle="--", alpha=0.5, linewidth=1)

            ax.set_xlabel("Window")
            ax.set_ylabel("Number of Prototypes")
            ax.set_title("Prototype Counts Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # Concept timeline
            fig = plot_cluster_timeline(labels, drift_locations, title="Concept Timeline")
            st.plotly_chart(fig, width="stretch")

    # ============================================================================
    # TAB 3: COMPARISON
    # ============================================================================
    with tab3:
        st.header("Window Comparison")

        if not st.session_state.prototype_processing_complete:
            st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
        else:
            storage = st.session_state.prototype_storage
            all_iterations = storage.get_all_iterations()
            drift_locations = st.session_state.prototype_drift_locations

            st.write("Select multiple windows to compare their prototypes:")

            # Create default selection: window 0 + drift locations
            default_windows = [0] + drift_locations if drift_locations else all_iterations[: min(4, len(all_iterations))]
            # Ensure defaults are within valid range
            default_windows = [w for w in default_windows if w in all_iterations]

            # Multi-select for windows
            selected_windows = st.multiselect(
                "Select Windows to Compare",
                options=all_iterations,
                default=default_windows,
                help="Choose 2-6 windows for comparison",
            )

            if len(selected_windows) < 2:
                st.info("Please select at least 2 windows to compare.")
            elif len(selected_windows) > 6:
                st.warning("Comparing more than 6 windows may result in cluttered visualization.")
            else:
                # Display comparison plot
                st.subheader("Prototype Comparison")

                max_displayed_prototypes = st.slider(
                    "Maximum number of displayed prototypes",
                    min_value=1,
                    max_value=30,
                    value=10,
                    key="max_displayed_prototypes",
                )

                fig = plt.figure(figsize=(14, 8))

                # Get global min/max
                global_min = float("inf")
                global_max = -float("inf")

                for window_nr in selected_windows:
                    x, y, prototypes, explainer = storage.get_window_data(window_nr)
                    for class_name in set(y):
                        if class_name in prototypes:
                            for prototype in prototypes[class_name]:
                                values = [v for _, v in sorted(prototype.items(), key=lambda x: x[0])]
                                global_min = min(global_min, min(values))
                                global_max = max(global_max, max(values))

                margin = 0.05 * (global_max - global_min)
                used_min = global_min - margin
                used_max = global_max + margin

                # Get all classes
                all_classes = set()
                for window_nr in selected_windows:
                    _, y, prototypes, _ = storage.get_window_data(window_nr)
                    all_classes.update(prototypes.keys())
                classes_sorted = sorted(list(all_classes))

                # Create subplots
                n_rows = len(classes_sorted)
                n_cols = len(selected_windows)

                for row, class_name in enumerate(classes_sorted):
                    for col, window_nr in enumerate(selected_windows):
                        ax = plt.subplot(n_rows, n_cols, row * n_cols + col + 1)

                        x, y, prototypes, explainer = storage.get_window_data(window_nr)

                        if class_name in prototypes:
                            for prototype in prototypes[class_name][:max_displayed_prototypes]:
                                items = sorted(prototype.items(), key=lambda x: x[0])
                                feature_names = [str(k) for k, _ in items]
                                feature_values = [v for _, v in items]
                                ax.plot(feature_names, feature_values, alpha=0.7)

                            num_prototypes = len(prototypes[class_name])
                            ax.text(
                                0.02,
                                0.95,
                                f"prototype count={num_prototypes}",
                                transform=ax.transAxes,
                                fontsize=9,
                                verticalalignment="top",
                            )

                        ax.set_ylim(used_min, used_max)
                        ax.tick_params(axis="x", labelrotation=45)

                        if row == 0:
                            ax.set_title(f"Window {window_nr}", fontsize=10)
                        if col == 0:
                            ax.set_ylabel(f"Class {class_name}", fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Statistics table
                st.subheader("Window Statistics")

                stats_data = []
                for window_nr in selected_windows:
                    x, y, prototypes, _ = storage.get_window_data(window_nr)

                    stats = {
                        "Window": window_nr,
                        "Samples": len(y),
                        "Total Prototypes": sum(len(p) for p in prototypes.values()),
                    }

                    # Add per-class prototype counts
                    for class_name in classes_sorted:
                        stats[f"Class {class_name} Prototypes"] = len(prototypes.get(class_name, []))

                    # Class distribution
                    class_dist = pd.Series(y).value_counts()
                    for class_name in classes_sorted:
                        pct = (class_dist.get(class_name, 0) / len(y)) * 100
                        stats[f"Class {class_name} %"] = f"{pct:.1f}%"

                    stats_data.append(stats)

                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, width="stretch")

    # ============================================================================
    # TAB 4: LOCAL ANALYSIS
    # ============================================================================
    with tab4:
        st.header("Local Window Analysis")

        if not st.session_state.prototype_processing_complete:
            st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
        else:
            storage = st.session_state.prototype_storage
            all_iterations = storage.get_all_iterations()

            # Window selection
            selected_window = st.selectbox(
                "Select Window for Detailed Analysis", options=all_iterations, index=len(all_iterations) // 2
            )

            x, y, prototypes, _ = storage.get_window_data(selected_window)

            # Cluster label info
            window_label = labels[selected_window]
            if window_label == -1:
                cluster_info = "⚠️ This window is classified as an **outlier**"
            else:
                cluster_info = f"✓ This window belongs to **Cluster {window_label}**"

            # Check if this is a drift location
            is_drift = selected_window in drift_locations
            drift_info = " 🔄 **DRIFT DETECTED AT THIS WINDOW**" if is_drift else ""

            st.info(f"{cluster_info}{drift_info}")

            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Window ID", selected_window)
            with col2:
                st.metric("Total Samples", len(y))
            with col3:
                st.metric("Total Prototypes", sum(len(p) for p in prototypes.values()))
            with col4:
                st.metric("Number of Classes", len(prototypes))

            # Additional window statistics
            col1, col2, col3, col4 = st.columns(4)

            # Feature statistics from all samples
            x_dicts = []
            for sample in x:
                if isinstance(sample, dict):
                    x_dicts.append(sample)
                else:
                    x_dicts.append({i: float(v) for i, v in enumerate(sample)})

            # Get all feature values
            all_features = {}
            for sample in x_dicts:
                for feat, val in sample.items():
                    if feat not in all_features:
                        all_features[feat] = []
                    all_features[feat].append(val)

            with col1:
                # Compression ratio for this window
                compression = (sum(len(p) for p in prototypes.values()) / len(y)) * 100
                st.metric("Window Compression", f"{compression:.2f}%")
            with col2:
                # Entropy of class distribution
                class_counts = pd.Series(y).value_counts()
                probs = class_counts / len(y)
                entropy = -np.sum(probs * np.log2(probs))
                st.metric("Class Entropy", f"{entropy:.3f}")
            with col3:
                # Imbalance ratio (max class / min class)
                if len(class_counts) > 1:
                    imbalance = class_counts.max() / class_counts.min()
                    st.metric("Class Imbalance Ratio", f"{imbalance:.2f}")
                else:
                    st.metric("Class Imbalance Ratio", "N/A")
            with col4:
                pass

            st.markdown("---")

            # Class distribution
            st.subheader("Class Distribution in Window")
            class_counts = pd.Series(y).value_counts().sort_index()

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                class_counts.plot(kind="bar", ax=ax, color="steelblue")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.set_title(f"Class Distribution - Window {selected_window}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            with col2:
                st.write("**Class Statistics:**")
                for cls, count in class_counts.items():
                    pct = (count / len(y)) * 100
                    st.write(f"• Class {cls}: {count} samples ({pct:.1f}%)")

            # Prototype analysis
            st.subheader("Prototype Analysis")

            for class_name in sorted(prototypes.keys()):
                with st.expander(f"📌 Class {class_name} - {len(prototypes[class_name])} prototypes"):
                    proto_list = prototypes[class_name]

                    # Plot all prototypes for this class
                    fig, ax = plt.subplots(figsize=(10, 4))

                    for i, prototype in enumerate(proto_list):
                        items = sorted(prototype.items(), key=lambda x: x[0])
                        feature_names = [str(k) for k, _ in items]
                        feature_values = [v for _, v in items]
                        ax.plot(feature_names, feature_values, alpha=0.7)

                    ax.set_xlabel("Features")
                    ax.set_ylabel("Values")
                    ax.set_title(f"All Prototypes for Class {class_name}")
                    num_prototypes = len(proto_list)
                    ax.text(
                        0.02,
                        0.95,
                        f"prototype count={num_prototypes}",
                        transform=ax.transAxes,
                        fontsize=9,
                        verticalalignment="top",
                    )
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis="x", labelrotation=45)
                    st.pyplot(fig)
                    plt.close()

                    # Calculate closest samples for each prototype
                    st.write("**Prototype Coverage:**")

                    # Convert all samples to dictionaries for distance calculation
                    x_dicts = []
                    for sample in x:
                        if isinstance(sample, dict):
                            x_dicts.append(sample)
                        else:
                            # If sample is array-like, convert to dict
                            x_dicts.append({i: float(v) for i, v in enumerate(sample)})

                    # First, find all samples whose closest prototype (across ALL classes) is from the current class
                    samples_closest_to_this_class = []
                    for i, sample in enumerate(x_dicts):
                        min_dist = float("inf")
                        closest_class = None

                        # Check distance to all prototypes across all classes
                        for cls, cls_protos in prototypes.items():
                            for proto in cls_protos:
                                common_features = set(proto.keys()) & set(sample.keys())
                                if common_features:
                                    dist = np.sqrt(sum((proto[f] - sample[f]) ** 2 for f in common_features))
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_class = cls

                        # If closest prototype is from the current class, include this sample
                        if closest_class == class_name:
                            samples_closest_to_this_class.append(i)

                    st.write(
                        f"*{len(samples_closest_to_this_class)} samples have their closest prototype from class {class_name}*"
                    )

                    # Now for each prototype in this class, find which of the filtered samples are closest to it
                    for proto_idx, prototype in enumerate(proto_list):
                        # Calculate distances from this prototype to filtered samples only
                        closest_samples = []

                        for sample_idx in samples_closest_to_this_class:
                            sample = x_dicts[sample_idx]

                            # Calculate distance to current prototype
                            common_features = set(prototype.keys()) & set(sample.keys())
                            if common_features:
                                dist_to_this = np.sqrt(sum((prototype[f] - sample[f]) ** 2 for f in common_features))
                            else:
                                dist_to_this = float("inf")

                            # Check if this prototype is closest among all prototypes in the same class
                            is_closest = True
                            for other_proto in proto_list:
                                if other_proto is prototype:
                                    continue
                                common_features = set(other_proto.keys()) & set(sample.keys())
                                if common_features:
                                    other_dist = np.sqrt(sum((other_proto[f] - sample[f]) ** 2 for f in common_features))
                                    if other_dist < dist_to_this:
                                        is_closest = False
                                        break

                            if is_closest:
                                closest_samples.append(sample_idx)

                        # Count classes of closest samples
                        class_counts = {}
                        for sample_idx in closest_samples:
                            sample_class = y[sample_idx]
                            class_counts[sample_class] = class_counts.get(sample_class, 0) + 1

                        # Display info
                        class_breakdown = ", ".join([f"Class {cls}: {cnt}" for cls, cnt in sorted(class_counts.items())])
                        st.write(
                            f"• **Prototype {proto_idx + 1}**: Closest to {len(closest_samples)} samples ({class_breakdown})"
                        )

                    # Feature statistics across prototypes
                    if len(proto_list) > 0:
                        # Collect all feature values
                        feature_data = {}
                        for prototype in proto_list:
                            for feat, val in prototype.items():
                                if feat not in feature_data:
                                    feature_data[feat] = []
                                feature_data[feat].append(val)

                        stats_rows = []
                        for feat in sorted(feature_data.keys()):
                            values = feature_data[feat]
                            stats_rows.append(
                                {
                                    "Feature": feat,
                                    "Mean": np.mean(values),
                                    "Std": np.std(values),
                                    "Min": np.min(values),
                                    "Max": np.max(values),
                                }
                            )

                        st.write("**Feature Statistics Across Prototypes:**")
                        st.dataframe(pd.DataFrame(stats_rows), width="stretch")

            # Distance to all other windows
            st.subheader("Distance to All Windows")

            measure_local = st.selectbox(
                "Distance Measure",
                options=["centroid_displacement", "prototype_reassignment_impact"],
                index=0,  # centroid_displacement is default
                help="Metric used to compare windows:\n"
                "- centroid_displacement: Recommended\n"
                "- prototype_reassignment_impact: Better, but slower",
                key="prototype_measure_local",
            )

            k_median = st.slider("Median Filter Width", 1, 11, 1, 2, key="local_median_filter")

            fig, ax = plt.subplots(figsize=(12, 4))
            data_to_plot = storage.compare_window_to_all(selected_window, measure=measure_local)
            data_to_plot = [float(x) for x in data_to_plot]
            if k_median > 1:
                data_to_plot = median_mask(data_to_plot, k_median)

            ax.plot(data_to_plot, linewidth=2)
            ax.axvline(x=selected_window, color="red", linestyle="--", linewidth=2, label="Current Window")

            # Mark drift locations
            for drift in st.session_state.prototype_drift_locations:
                ax.axvline(x=drift, color="orange", linestyle=":", alpha=0.5, linewidth=1)

            ax.set_xlabel("Window")
            ax.set_ylabel("Distance")
            ax.set_title(f"Distance from Window {selected_window} to All Other Windows")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    # ============================================================================
    # TAB 5: DISTANCE MATRIX
    # ============================================================================
    with tab5:
        st.header("Distance Matrix Visualization")

        if not st.session_state.prototype_processing_complete:
            st.warning("⚠️ Please process data in the 'Data Processing' tab first.")
        else:
            matrix = st.session_state.prototype_matrix
            drift_locations = st.session_state.prototype_drift_locations

            st.write(f"Matrix shape: {matrix.shape}")

            # Matrix visualization options
            show_drift_markers = st.checkbox("Show Drift Markers", value=True)

            fig, ax = plt.subplots(figsize=(14, 12))

            # Create heatmap
            sns.heatmap(matrix, cmap="RdYlGn_r", square=True, linewidths=0, cbar_kws={"label": "Distance"}, ax=ax)

            # Mark drift positions if enabled
            if show_drift_markers and drift_locations:
                for drift_iter in drift_locations:
                    if drift_iter in matrix.index:
                        idx = list(matrix.index).index(drift_iter)
                        ax.axhline(y=idx, color="blue", linewidth=3, alpha=0.7)
                        ax.axvline(x=idx, color="blue", linewidth=3, alpha=0.7)

            ax.set_title("Window Distance Matrix")
            ax.set_xlabel("Window")
            ax.set_ylabel("Window")

            st.pyplot(fig)
            plt.close()

            # Matrix statistics
            st.subheader("Matrix Statistics")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Distance", f"{matrix.values.mean():.4f}")
            with col2:
                st.metric("Std Distance", f"{matrix.values.std():.4f}")
            with col3:
                st.metric("Max Distance", f"{matrix.values.max():.4f}")
