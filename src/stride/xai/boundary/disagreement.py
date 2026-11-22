import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _get_feature_bounds(feature_names, X_data=None):
    feature_mins = {}
    feature_maxs = {}
    if X_data is not None:
        if isinstance(X_data, pd.DataFrame):
            for col in feature_names:
                if col in X_data.columns:
                    feature_mins[col] = X_data[col].min()
                    feature_maxs[col] = X_data[col].max()
        else:
            for i, col in enumerate(feature_names):
                feature_mins[col] = X_data[:, i].min()
                feature_maxs[col] = X_data[:, i].max()
    return feature_mins, feature_maxs


def _get_unscaler_map_from_scaler(feature_names, scaler):
    """
    Extracts linear unscaling parameters (m, c) directly from a fitted MinMaxScaler.
    Formula: X_raw = X_scaled * (1/scale_) - (min_/scale_)
    So: m = 1/scale_, c = -min_/scale_
    """
    unscalers = {}

    # Safety check if scaler is fitted
    if not hasattr(scaler, "scale_") or not hasattr(scaler, "min_"):
        return {}

    # Vectorized calculation of slope (m) and intercept (c)
    m_vec = 1.0 / scaler.scale_
    c_vec = -scaler.min_ / scaler.scale_

    for i, name in enumerate(feature_names):
        if i < len(m_vec):
            unscalers[name] = (m_vec[i], c_vec[i])

    return unscalers


def _get_unscaler_map(feature_names, X_raw, X_scaled):
    unscalers = {}

    def get_col(X, idx, name):
        if isinstance(X, pd.DataFrame):
            return X[name].values if name in X.columns else None
        else:
            return X[:, idx] if idx < X.shape[1] else None

    for i, name in enumerate(feature_names):
        r_vals = get_col(X_raw, i, name)
        s_vals = get_col(X_scaled, i, name)
        if r_vals is not None and s_vals is not None and len(s_vals) > 1:
            try:
                if np.max(s_vals) != np.min(s_vals):
                    m, c = np.polyfit(s_vals, r_vals, 1)
                    unscalers[name] = (m, c)
                else:
                    unscalers[name] = (0, r_vals[0])
            except Exception:
                unscalers[name] = (1, 0)
    return unscalers


def _format_rule_bounds(path_tuples, feature_mins, feature_maxs):
    feature_bounds = {}
    for name, op, threshold in path_tuples:
        if name not in feature_bounds:
            mn = feature_mins.get(name, -np.inf)
            mx = feature_maxs.get(name, np.inf)
            feature_bounds[name] = [mn, mx]
        if op == "<=":
            feature_bounds[name][1] = min(feature_bounds[name][1], threshold)
        elif op == ">":
            feature_bounds[name][0] = max(feature_bounds[name][0], threshold)

    range_strings = []
    for name, bounds in feature_bounds.items():
        lower, upper = bounds
        lower_str = "(-∞" if lower == -np.inf else f"[{lower:.2f}"
        upper_str = "+∞)" if upper == np.inf else f"{upper:.2f}]"
        range_strings.append(f"{name} ∈ {lower_str}, {upper_str}")
    # A tree without splits (the classifiers disagree everywhere) yields one rule with no conditions
    return ",  ".join(range_strings) or "All points"


def _recurse_rules(
    node,
    path_tuples,
    tree_,
    tree_classes,
    feature_names,
    rules_data,
    drift_leaves,
    feature_mins,
    feature_maxs,
    unscalers,
    leaf_metrics,
):
    # Check if leaf node
    if tree_.feature[node] == -2:
        # The class-weighted leaf value selects drift regions: with balanced class weights it exceeds
        # 0.5 exactly when the leaf's disagreement rate is above the overall rate.
        val_weighted = tree_.value[node][0]
        node_weight_total = val_weighted.sum()

        target_class = 1
        class_prob = 0.0
        if len(tree_classes) > 1:
            if target_class in tree_classes:
                idx_1 = np.where(tree_classes == target_class)[0][0]
                class_prob = val_weighted[idx_1] / node_weight_total
        else:
            class_prob = 1.0 if tree_classes[0] == target_class else 0.0

        # FILTER: Only show leaves that are predominantly Drift (>50%)
        if class_prob > 0.5:
            readable_rule = _format_rule_bounds(path_tuples, feature_mins, feature_maxs)

            n_real, cov_real, disagree_rate, visual_area = leaf_metrics(node, class_prob)

            rules_data.append(
                {
                    "Rule": readable_rule,
                    "Drift Conf.": f"{disagree_rate:.1%}",  # Share of real points in the rule that disagree
                    "Samples": int(n_real),  # Actual Data Points
                    "Coverage": f"{cov_real:.1%}",  # Real Data Coverage
                    "Visual Area": f"{visual_area:.1%}",  # Size on the plot
                    "Leaf_ID": node,
                }
            )
            drift_leaves.append(node)
        return

    # Extract feature name and threshold
    feat_idx = tree_.feature[node]
    name = feature_names[feat_idx]
    threshold_scaled = tree_.threshold[node]

    # Unscale threshold
    threshold_raw = threshold_scaled
    if name in unscalers:
        m, c = unscalers[name]
        threshold_raw = threshold_scaled * m + c

    _recurse_rules(
        tree_.children_left[node],
        path_tuples + [(name, "<=", threshold_raw)],
        tree_,
        tree_classes,
        feature_names,
        rules_data,
        drift_leaves,
        feature_mins,
        feature_maxs,
        unscalers,
        leaf_metrics,
    )
    _recurse_rules(
        tree_.children_right[node],
        path_tuples + [(name, ">", threshold_raw)],
        tree_,
        tree_classes,
        feature_names,
        rules_data,
        drift_leaves,
        feature_mins,
        feature_maxs,
        unscalers,
        leaf_metrics,
    )


def get_disagreement_table(tree, feature_names, X_raw=None, X_scaled=None, scaler=None, y_disagree=None, X_grid=None):
    tree_ = tree.tree_
    tree_classes = tree.classes_
    rules_data = []
    drift_leaves = []

    feature_mins, feature_maxs = _get_feature_bounds(feature_names, X_raw)
    unscalers = {}
    # Priority 1: Use the exact Scaler object if provided
    if scaler is not None:
        unscalers = _get_unscaler_map_from_scaler(feature_names, scaler)
    # Priority 2: Fallback to empirical estimation (Polyfit)
    elif X_raw is not None and X_scaled is not None:
        unscalers = _get_unscaler_map(feature_names, X_raw, X_scaled)

    # Real data per leaf: point count and, given disagreement labels, the disagreement rate
    real_leaves = tree.apply(X_scaled) if X_scaled is not None else None
    # Plot area per leaf: share of the decoded grid; without a grid, share of the tree's training data
    grid_leaves = tree.apply(X_grid) if X_grid is not None else None

    def leaf_metrics(node, weighted_prob):
        n_real, cov_real, disagree_rate = 0, 0.0, weighted_prob
        if real_leaves is not None:
            in_leaf = real_leaves == node
            n_real = int(in_leaf.sum())
            cov_real = n_real / len(real_leaves) if len(real_leaves) else 0.0
            if y_disagree is not None and n_real:
                disagree_rate = float(np.asarray(y_disagree)[in_leaf].mean())
        if grid_leaves is not None:
            visual_area = float(np.mean(grid_leaves == node))
        else:
            visual_area = tree_.n_node_samples[node] / tree_.n_node_samples[0]
        return n_real, cov_real, disagree_rate, visual_area

    _recurse_rules(
        0,
        [],
        tree_,
        tree_classes,
        feature_names,
        rules_data,
        drift_leaves,
        feature_mins,
        feature_maxs,
        unscalers,
        leaf_metrics,
    )

    if not rules_data:
        return pd.DataFrame(columns=["No Drift Regions Found"]), []

    df = pd.DataFrame(rules_data)

    # 1. Sort by Real Data Coverage (descending), then Visual Area
    df["_sort_cov"] = df["Coverage"].apply(lambda x: float(x.strip("%")))
    df = df.sort_values(by=["_sort_cov", "Visual Area"], ascending=False).drop(columns=["_sort_cov"])

    # 2. Assign Rule Labels based on this sorted order
    # The top row becomes "Rule 1", second is "Rule 2", etc.
    df["Rule Label"] = [f"Rule {i + 1}" for i in range(len(df))]

    # Reorder columns
    cols = ["Rule Label", "Rule", "Drift Conf.", "Samples", "Coverage", "Visual Area", "Leaf_ID"]
    df = df[cols]

    # Return the leaf IDs in the EXACT same order as the table
    # This ensures the Visualization (which iterates this list) matches the Table
    return df, df["Leaf_ID"].tolist()


def compute_disagreement_analysis(
    clf_pre, clf_post, X_eval_raw, X_eval_scaled, X_grid_high_scaled=None, feature_names=None, scaler=None
):
    """Find where the pre-drift and post-drift classifiers disagree and describe it as rules.

    The rule tree is trained on the real evaluation points, not on the decoded grid: a projection's
    decoder can tie features to map position (a noise feature then looks informative on the grid),
    so grid-trained rules describe the projection rather than the data. The grid, when given, only
    measures each rule's area on the plot.
    """
    if feature_names is None:
        if hasattr(X_eval_raw, "columns"):
            feature_names = list(X_eval_raw.columns)
        else:
            feature_names = [f"Feature {i}" for i in range(X_eval_raw.shape[1])]

    # 1. Real Drift Rate (based on actual data samples)
    pred_pre = clf_pre.predict(X_eval_scaled)
    pred_post = clf_post.predict(X_eval_scaled)
    y_delta_real = (pred_pre != pred_post).astype(int)
    drift_rate_real = np.mean(y_delta_real)

    if drift_rate_real == 0:
        return {"drift_rate": 0.0, "disagreement_table": None, "drift_leaf_ids": [], "viz_tree": None}

    # 2. Train the rule tree on the real points
    viz_tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.01, class_weight="balanced", random_state=42)
    viz_tree.fit(X_eval_scaled, y_delta_real)

    # 3. Generate Rules
    df_rules, drift_leaf_ids = get_disagreement_table(
        viz_tree,
        feature_names,
        X_raw=X_eval_raw,
        X_scaled=X_eval_scaled,
        scaler=scaler,
        y_disagree=y_delta_real,
        X_grid=X_grid_high_scaled,
    )

    return {
        "drift_rate": drift_rate_real,  # Metric from real data
        "disagreement_table": df_rules,
        "drift_leaf_ids": drift_leaf_ids,
        "viz_tree": viz_tree,
    }
