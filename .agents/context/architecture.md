# STRIDE Architectural Overview

## 1. System Components

The STRIDE framework is architected into three distinct layers:
1. **Core Algorithmic Library (`src/stride/`)**: A clean, modular, PyPA-standard Python package (`stride-xai`) installable via pip/flit. It contains zero GUI/Streamlit code, uses typed exceptions, and returns pure data structures (`DataFrame`, `ndarray`, `dict`) or standard visualization handles (`Figure`, `Axes`).
2. **Dashboard Configuration Layer (`dashboard/config/`)**: Decoupled UI schemas, widget state dictionaries, and interactive metadata that configure Streamlit controls without polluting the core API.
3. **Interactive Streamlit Application (`dashboard/`)**: The multi-tab web application (`app.py`, `components/`) consuming `stride.*` and `dashboard.config.*`.

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                  │
│                     (dashboard/)                       │
│  app.py ──► components/ (sidebar, data_gen, tabs...)   │
└──────────────┬───────────────────────────┬─────────────┘
               │ imports                   │ imports
┌──────────────▼─────────────┐ ┌───────────▼─────────────┐
│ Dashboard UI Config        │ │ STRIDE Core Package     │
│ (dashboard/config/)        │ │ (src/stride/)           │
│ ├─ model_schemas.py        │ │ ├─ common/              │
│ ├─ dataset_schemas.py      │ │ ├─ datasets/            │
│ └─ __init__.py             │ │ ├─ drift/               │
│                            │ │ ├─ models/              │
│                            │ │ ├─ plotting/            │
│                            │ │ ├─ xai/                 │
│                            │ │ │  ├─ boundary/         │
│                            │ │ │  ├─ importance/       │
│                            │ │ │  ├─ clustering/       │
│                            │ │ │  ├─ recurrence/       │
│                            │ │ │  └─ stats/            │
│                            │ │ └─ exceptions.py        │
└────────────────────────────┘ └─────────────────────────┘
```

## 2. Core Library Modules (`src/stride/`)

### 2.1 Synthetic Datasets (`stride.datasets`)
- **SEA Drift**: Simulates abrupt concept drift where the threshold boundary on $f_1 + f_2$ shifts.
- **Hyperplane Drift**: Simulates continuous gradual drift via rotating decision hyperplanes in $d$-dimensional space.
- **RBF Drift**: Non-linear drift shifting Gaussian cluster centroids.
- **Linear Weight Inversion (LWI)**: Shifts feature attribution signs to benchmark explainer sensitivity.
- **Registry**: Stream registry providing standardized instantiation and metadata.

### 2.2 Drift Detection (`stride.drift`)
- Statistical tests and sequential error-rate monitors that track classification error over rolling windows and trigger warning/drift flags.

### 2.3 Explainability (xAI) Modules (`stride.xai`)
- **Decision Boundary (`stride.xai.boundary`)**: Estimates and projects multidimensional decision boundaries (via grid sampling or SSNP) to visually depict how the separator evolves between reference and drift windows. SSNP and TensorFlow dependencies are lazily loaded.
- **Feature Importance (`stride.xai.importance`)**: Quantifies per-feature attribution deltas before and after drift using permutation importance and SHAP values.
- **Clustering Dynamics (`stride.xai.clustering`)**: Analyzes feature-space topology, centroid movements, and cluster dispersion shifts across stream windows.
- **Recurring Concept Analysis (`stride.xai.recurrence`)**: Extracts window prototypes (ProTree), computes pairwise concept distance matrices, and performs clustering to recognize reappearing concepts.
- **Descriptive Statistics (`stride.xai.stats`)**: Computes univariate and multivariate distribution divergence metrics across windows.

### 2.4 Classifiers under Drift (`stride.models`)
- Lightweight Scikit-learn wrappers (`MLPModel`, `RandomForestModel`) with unified signatures, standard scikit-learn estimator compatibility, and model serialization support.

## 3. Packaging & Dependencies (`pyproject.toml`)

The package is packaged as `stride-xai` with minimal core dependencies (`numpy<2.0`, `pandas`, `scipy`, `scikit-learn`) and optional dependency extras:
- `vis`: Matplotlib, Seaborn, Plotly
- `drift`: River
- `clustering`: Pyclustering, HDBSCAN, UMAP-learn
- `xai`: SHAP, LIME
- `deeplearning`: TensorFlow >= 2.15.0 (for SSNP boundary reduction)
- `dashboard`: Streamlit, Icecream
- `dev`: Ruff, Pytest
- `all`: All combined extras
