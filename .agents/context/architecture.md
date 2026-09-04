# STRIDE Architectural Overview

## 1. System Components

The STRIDE framework is architected into two primary layers: core algorithmic libraries (`src/`) and the interactive visualization dashboard (`dashboard/`).

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                  │
│                     (dashboard/)                       │
│  app.py ──► components/ (sidebar, data_gen, tabs...)   │
└───────────────────────────┬────────────────────────────┘
                            │ imports & orchestrates
┌───────────────────────────▼────────────────────────────┐
│                    STRIDE Core (src/)                  │
│ ├─ datasets/             : Synthetic stream generation │
│ ├─ DDM/                  : Drift detection methods     │
│ ├─ decision_boundary/    : Decision boundary analysis  │
│ ├─ feature_importance/   : SHAP & permutation metrics  │
│ ├─ clustering/           : Density & cluster shifts    │
│ ├─ recurrence/           : Prototype recurring concept │
│ ├─ descriptive_statistics: Distribution divergence     │
│ └─ models/               : Classifiers under drift     │
└────────────────────────────────────────────────────────┘
```

## 2. Core Modules in `src/`

### 2.1 Synthetic Datasets (`src/datasets/`)
- **SEA Drift**: Simulates abrupt concept drift where the threshold boundary on $f_1 + f_2$ shifts.
- **Hyperplane Drift**: Simulates continuous gradual drift via rotating decision hyperplanes in $d$-dimensional space.
- **RBF Drift**: Non-linear drift shifting Gaussian cluster centroids.
- **Linear Weight Inversion (LWI)**: Shifts feature attribution signs to benchmark explainer sensitivity.

### 2.2 Drift Detection (`src/DDM/`)
- Implements statistical tests and sequential error-rate monitors (e.g. DDM, EDDM) that track classification error over rolling windows and trigger detection/warning flags.

### 2.3 Explainability (xAI) Modules
- **Decision Boundary (`src/decision_boundary/`)**: Estimates and projects multidimensional decision boundaries (via grid sampling or SDBM) to visually depict how the separator evolves between reference and drift windows.
- **Feature Importance (`src/feature_importance/`)**: Quantifies per-feature attribution deltas before and after drift using permutation importance and SHAP values.
- **Clustering Dynamics (`src/clustering/`)**: Analyzes feature-space topology, centroid movements, and cluster dispersion shifts across stream windows.
- **Recurring Concept Analysis (`src/recurrence/`)**: Extracts window prototypes, computes pairwise concept distance matrices, and performs HDBSCAN clustering to recognize reappearing concepts.

## 3. Dashboard Structure (`dashboard/`)

- `app.py`: Main entry point initializing session state, layout containers, and tab routing.
- `components/`: Modular UI sections:
  - Sidebar: Dataset generators, drift parameters, model selection, and hyperparameter tuning.
  - Tabs: Data stream visualization, decision boundary comparisons, feature importance tracking, and clustering views.
  - State Management: Uses isolated `st.session_state` keys partitioned per model and dataset to avoid cross-contamination.
