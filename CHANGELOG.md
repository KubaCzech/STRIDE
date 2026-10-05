# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
