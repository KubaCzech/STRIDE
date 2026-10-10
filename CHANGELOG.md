# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-10-07

### Added
- **Shaker Semi-Synthetic Concept Drift Protocol**:
    - Implemented the empirical dataset splicing protocol (*Shaker & Hüllermeier, Neurocomputing 2015*).
    - `stride.datasets.FeatureMatcher`: Statistical distribution alignment (Wasserstein distance with optimal bipartite assignment via Hungarian algorithm), exact column mapping, manual translation dictionary, shared PCA latent subspace projection, and feature scaling (`StandardScaler` / `MinMaxScaler`).
    - `stride.datasets.DriftBlender`: Deterministic stream blending supporting abrupt, gradual (Shaker sigmoidal Bernoulli trials), incremental (nearest-neighbor linear interpolation), and recurring (harmonic periodic oscillation) transition schedules.
    - `stride.datasets.SemiSyntheticDriftDataset`: Full `BaseDataset` compliance carrying ground-truth change-point timestamps, transition intervals, binary concept indicators, and statistical drifting feature diagnostics (KS test, Wasserstein distance).
    - Canonical UCI Wine Quality benchmark (Red $\to$ White wine) offline loader and generator (`load_wine_quality_drift`).
- **Interactive Streamlit Dataset Stitcher**:
    - Modal dialog `open_dataset_stitcher_modal()` allowing visual concept selection, feature mapping inspection, schedule tuning, interactive transition curve preview, and instant registry persistence.
    - Ground-truth change-point markers, shaded transition regions, and detector latency metrics in the Drift Detection tab.
    - Ground-truth drifting features banner in the Feature Importance Analysis tab.
- **Agentic Release Governance**:
    - Release management standards ([`.agents/rules/release_standards.md`](.agents/rules/release_standards.md)) and autonomous deployment skill ([`.agents/skills/release-management/SKILL.md`](.agents/skills/release-management/SKILL.md)).

## [0.1.0] - 2026-10-05

### Added
- **Core Package Architecture**: PyPA `src/stride/` modular package layout with PEP 621 metadata under `stride-xai`.
- **Synthetic Stream Generators**:
    - Rotating Hyperplane drift (`HyperplaneDriftDataset`).
    - SEA benchmark generator with abrupt shift (`SeaDriftDataset`).
    - Random Basis Function cluster movement (`RbfDriftDataset`).
    - Linear Weight Inversion drift (`LinearWeightInversionDriftDataset`).
    - Multi-window generators (Mixed, Sine, Plane, STAGGER, Random Tree).
    - Registry adapters for external and streaming datasets (`river`, CSV).
- **Sequential Drift Detection**:
    - Online drift monitoring and error characterization (`BinaryErrorDriftDescriptor`) leveraging River's Drift Detection Method (DDM).
- **Explainable AI (xAI) Modules**:
    - Supervised Decision Boundary Maps (SDBM) and Self-Supervised Neural Projection (SSNP) for 2D boundary shift forensics (`stride.xai.boundary`).
    - Feature importance attribution via model-agnostic SHAP and Permutation Feature Importance (`stride.xai.importance`).
    - Selective drift discriminator with conformal p-value thresholding and abstention for localized feature attribution.
    - X-means clustering dynamics and centroid tracking via Hungarian bipartite assignment (`stride.xai.clustering`).
    - Prototype-based recurring concept detection using ProTree and HDBSCAN density clustering (`stride.xai.recurrence`).
    - Non-parametric hypothesis testing (Kolmogorov-Smirnov, Anderson-Darling) and 1D Wasserstein distance tracking (`stride.xai.stats`).
- **Headless Visualization Pipeline**:
    - Plotting routines in `stride.plotting` returning `matplotlib.figure.Figure` without blocking execution.
- **Interactive Streamlit Dashboard**:
    - Multi-tab forensic application (`dashboard/app.py`) spanning Data, Model, and Explanation analytical layers.
    - Decoupled UI configuration schemas under `dashboard/config/`.
- **Package Release Engineering**:
    - Automated CI/CD release workflow using PyPI OpenID Connect (OIDC) Trusted Publishing (`.github/workflows/release.yml`).
    - Dynamic version extraction from `src/stride/__init__.py`.
    - PyPI discoverability metadata and navigation URLs.
