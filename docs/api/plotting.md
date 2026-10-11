# Visualizations & Plotting

The `stride.plotting` subpackage provides headless visualization functions for data streams, class distributions, and feature manifolds.

---

## Overview

All visualization utilities generate and return `matplotlib.figure.Figure` instances without invoking `plt.show()`. This guarantees non-blocking execution across headless CI servers, automated scripts, and Streamlit dashboard containers (`st.pyplot(fig)`).

```python
from stride.plotting import (
    visualize_data_stream,
    plot_class_distribution,
    plot_feature_space,
    plot_feature_target_relationship,
)
```

### Key Components

| Function | Return Type | Description |
| :--- | :--- | :--- |
| `visualize_data_stream` | `Figure` | Multi-panel stream trajectory overview plotting feature values and target labels across the time index. |
| `plot_class_distribution` | `Figure` | Class balance evolution chart displaying categorical target proportions across sliding windows. |
| `plot_feature_space` | `Figure` | 2D scatter visualization of feature coordinates color-coded by class labels. |
| `plot_feature_target_relationship` | `Figure` | Density and violin plots visualizing feature-target relationships pre- and post-drift. |

---

## API Specification

::: stride.plotting
