import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def visualize_decision_boundary(results, title="Decision Boundary Analysis"):
    """
    Visualize the decision boundary analysis results using HSV style.
    Hue = Class, Value = Confidence.

    Parameters
    ----------
    results : dict
        Output from compute_decision_boundary_analysis
    title : str
        Main title for the figure

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    res_pre = results['pre']
    res_post = results['post']
    is_2d = results.get('is_2d', False)
    
    # Define labels based on dimensionality
    x_label = "Feature 1" if is_2d else "SSNP Component 1"
    y_label = "Feature 2" if is_2d else "SSNP Component 2"

    # Identify all unique classes to ensure consistent coloring
    y_all = np.concatenate([res_pre['y_train'], res_post['y_train']])
    classes = np.unique(y_all)
    n_classes = len(classes)

    # Create color map
    # Using tab20 or tab10 depending on n_classes
    if n_classes <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')

    class_to_idx = {c: i for i, c in enumerate(classes)}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    def plot_window(ax, res, window_name):
        # 1. Prepare Grid Image from HSV Logic
        grid_labels = res['grid_labels']
        grid_probs = res['grid_probs']

        # Determine the RGB baselines for each pixel based on class
        idx_grid = np.zeros_like(grid_labels, dtype=int)
        for c, idx in class_to_idx.items():
            idx_grid[grid_labels == c] = idx

        # Get RGB colors (ignore alpha from cmap)
        # normalize index for cmap
        norm_indices = idx_grid / max(1, n_classes - 1) if n_classes > 1 else np.zeros_like(idx_grid, dtype=float)
        rgba_grid = cmap(norm_indices)  # Shape: (H, W, 4)
        rgb_grid = rgba_grid[..., :3]   # Shape: (H, W, 3)

        # Convert to HSV
        hsv_grid = mcolors.rgb_to_hsv(rgb_grid)

        # Modulate Value (brightness) by probability
        # prob is 0..1.
        # In original code: data_hsv[:, :, 2] = prob_matrix
        # This means high confidence = bright, low confidence = dark.
        hsv_grid[..., 2] = grid_probs

        # Convert back to RGB
        final_rgb = mcolors.hsv_to_rgb(hsv_grid)

        # 2. Plot Image
        extent = res['grid_bounds']  # (xmin, xmax, ymin, ymax)

        ax.imshow(final_rgb, extent=extent, origin='lower', aspect='auto')

        # 3. Plot Data Points (Optional but requested to show them on plot)
        X_2d = res['X_2d']
        y_train = res['y_train']

        for c in classes:
            mask = (y_train == c)
            if not np.any(mask):
                continue

            idx = class_to_idx[c]
            # Solid color for points
            c_color = cmap(idx / max(1, n_classes - 1) if n_classes > 1 else 0)

            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[c_color], label=f"Class {c}",
                       edgecolor='white', s=30, alpha=0.9, linewidth=0.5)

        ax.set_title(f"{window_name} Drift")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plot_window(axes[0], res_pre, "Pre")
    plot_window(axes[1], res_post, "Post")

    # Add legend (using handles from one of the plots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(n_classes, 5), bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend
    return fig


def plot_categorical_drift_map(ssnp_model, viz_tree, drift_leaf_ids, grid_bounds, grid_size=300, is_2d=False):
    """
    Plots the Categorical Drift Map showing regions of disagreement in the SSNP latent space.

    Parameters
    ----------
    ssnp_model : SSNP
        The trained SSNP model (for inverse transform).
    viz_tree : DecisionTreeClassifier
        The tree trained to predict disagreement (y_delta) from scaled features.
    drift_leaf_ids : list
        List of leaf IDs from viz_tree that correspond to drift (disagreement).
    grid_bounds : tuple
        (xmin, xmax, ymin, ymax) defining the extent of the plot.
    grid_size : int
        Resolution of the grid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    xmin, xmax, ymin, ymax = grid_bounds

    # Create Grid
    x_intrvls = np.linspace(xmin, xmax, num=grid_size)
    y_intrvls = np.linspace(ymin, ymax, num=grid_size)

    # We use meshgrid to generate points
    # Note: meshgrid(x, y) returns shape (len(y), len(x))
    xx, yy = np.meshgrid(x_intrvls, y_intrvls)
    pts = np.c_[xx.ravel(), yy.ravel()]

    # Initialize grid for leaf IDs
    # We will map: 0 -> Stable, 1..N -> Drift Rules
    # So we need to compute leaves for all points

    # Process in batches
    batch_size = 50000
    n_pts = len(pts)
    leaves_list = []

    for i in range(0, n_pts, batch_size):
        batch_pts = pts[i:i+batch_size]
        # Inverse transform: 2D -> High Dim Scaled
        batch_high_dim = ssnp_model.inverse_transform(batch_pts)
        # Apply tree to get leaf IDs
        batch_leaves = viz_tree.apply(batch_high_dim)
        leaves_list.append(batch_leaves)

    leaves_flat = np.concatenate(leaves_list)
    leaves_grid = leaves_flat.reshape(grid_size, grid_size)

    # Flip to match imshow origin='lower' / 'upper' convention?
    # standard meshgrid with origin='lower' matches x,y axes
    # sdbm did a flipud because of how they constructed grid vs imshow.
    # In `visualize_decision_boundary`, we used origin='lower' and standard meshgrid.
    # We will stick to origin='lower' and NO flip, assuming consistency.

    # Map raw leaves to plot indices
    # 0 = No Drift / Other
    # 1..K = Drift Rules (indexed 1 to K)

    final_grid = np.zeros_like(leaves_grid, dtype=int)

    # Map drift leaves specific indices
    leaf_to_plot_idx = {leaf: i+1 for i, leaf in enumerate(drift_leaf_ids)}

    # Use numpy vectorize or mask for speed
    # But leaf IDs are sparse, so iteration over unique leaves in grid is okay
    unique_leaves = np.unique(leaves_grid)
    for leaf in unique_leaves:
        if leaf in leaf_to_plot_idx:
            final_grid[leaves_grid == leaf] = leaf_to_plot_idx[leaf]
        else:
            final_grid[leaves_grid == leaf] = 0

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    num_rules = len(drift_leaf_ids)
    if num_rules > 0:
        # Create Colormap
        # 0: Light Grey (Stable)
        # 1..N: Colors from tab10/tab20
        cmap_base = plt.get_cmap('tab10' if num_rules <= 10 else 'tab20', num_rules)
        colors = ['#f0f0f0'] + [mcolors.rgb2hex(cmap_base(i)) for i in range(num_rules)]
        cmap = mcolors.ListedColormap(colors)

        bounds_norm = np.arange(-0.5, num_rules + 1.5, 1)
        norm = mcolors.BoundaryNorm(bounds_norm, cmap.N)

        img = ax.imshow(final_grid, cmap=cmap, norm=norm,
                        extent=[xmin, xmax, ymin, ymax],
                        origin='lower', aspect='auto', interpolation='nearest')

        cbar = plt.colorbar(img, ticks=np.arange(0, num_rules + 1), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(['Stable'] + [f'Rule {i+1}' for i in range(num_rules)])
    else:
        # No drift
        ax.text(0.5, 0.5, "No Drift / Disagreement Detected",
                ha='center', va='center', transform=ax.transAxes)
        # Show empty grid
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_title("Categorical Drift Map (Disagreement)")
    ax.set_xlabel("Feature 1" if is_2d else "SSNP Component 1")
    ax.set_ylabel("Feature 2" if is_2d else "SSNP Component 2")

    return fig
