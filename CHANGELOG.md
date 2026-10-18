# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Cluster-Based Drift Detection** (`ClusterBasedDriftDetector`):
    - New defaults `matching="reference"` and `n_permutations=50`: the detection window is assigned to the X-Means clusters of the reference window, so both windows are compared on the same partition, and drift is decided by a permutation test (`p_value < permutation_alpha`) instead of the fixed `decision_thr`. On synthetic benchmarks (SEA, Hyperplane, LWI, RBF; 20 seeds) the false-alarm rate drops from up to 0.20 to 0.00 and SEA detection rate rises from 0.85 to 1.00. Detection takes about 1-2 s; pass `matching="independent", n_permutations=0` for the previous behaviour.
    - Per-cluster statistic changes are absolute differences in standard deviations of the standardised feature instead of relative changes, which were unstable for values near zero.
    - Appeared and disappeared clusters count only towards the cluster-count criterion; missing statistics are no longer counted as drift; criteria without evaluations are excluded from the weighted score (`effective_weights`).
- **Datasets**: multi-window datasets use string feature names (`x0, x1, ...`) instead of integer column names.

### Added
- `ClusterBasedDriftDetector`: `p_value`, `null_strengths`, `alpha` (optional size-aware significance tests), `xmeans_repeat`, `novelty_quantile` and `n_jobs` (parallel permutations).
- `stride.xai.clustering`: `assign_to_reference`, `calculate_shift_significance`, `bic_score`; `run_xmeans(repeat=...)` keeps the run with the best BIC.
- Dashboard clustering tab: drift decision, strength and permutation p-value as metrics, method description, results cached per pair of windows.

### Fixed
- `run_xmeans` was not reproducible for a given `random_state` and changed the global `random` state (#31).
- Cluster labels were mixed up between classes when class labels were not sorted in set order (e.g. text labels), and the detector crashed on classes missing from a window or smaller than `k_init` (#32).
- Dashboard clustering tab: the "average distance to center change" table and the per-class contribution table (3+ classes) raised errors; the centroid-shift plot failed with more than 20 clusters; class-wise plots failed for text labels.
- Feature Importance tab failed with LIME on RBF, Mixed and SEA multi-window datasets (integer feature names).
- Prototype Analysis failed on NumPy 2.4 (`np.in1d` removed).
- `stride.xai.boundary` did not export `visualize_decision_boundary`.
- Documentation: code examples that did not run with the current API.

## [0.4.0] - 2026-10-17

### Added
- **Core Visualization & Dependency Restructuring**:
    - Promoted `matplotlib` and `seaborn` to core `dependencies` in `pyproject.toml`, ensuring out-of-the-box stream visualization and feature distribution plotting upon `pip install stride-xai` without requiring optional extras (#29).
    - Introduced the `xai-full` optional dependency extra (`pip install "stride-xai[xai-full]"`) packaging all analytical engines (`shap`, `lime`, `pyclustering`, `hdbscan`, `umap-learn`, `tensorflow>=2.15.0`, `plotly`) for headless pipelines and automated notebooks without installing Streamlit web server dependencies (#29).
    - Refined the `vis` optional extra to focus exclusively on `plotly` for interactive web plotting (#29).
    - Updated minimal base installation guardrail tests in `tests/test_minimal_install_import.py` to assert native execution of stream visualizations in the core library (#29).
    - Updated installation guides across `README.md` and documentation with the tiered installation model (#29).

### Fixed
- **Plotting**:
    - Resolved multiclass colormap indexing by adopting `plt.get_cmap` for categorical stream visualizations (#28).

## [0.3.0] - 2026-10-16

### Added
- **ADWIN Drift Detection & Adaptive Characterization**:
    - `stride.drift.DualADWIN`: Dual-threshold ADWIN wrapper providing simultaneous warning ($\delta_{\text{warn}}$) and confirmed drift ($\delta_{\text{drift}}$) signals with Hoeffding confidence bounds and $O(\log W)$ memory.
    - Directional degradation filtering in `BinaryErrorDriftDescriptor` ensuring drift and warning alarms only fire upon performance deterioration.
    - Adaptive lookback window calibration bounded dynamically by ADWIN subwindow width ($W$).
    - Streamlit dashboard integration: added ADWIN to detector selection dropdown with dedicated confidence and window tuning widgets.
- **Documentation Infrastructure**:
    - Hosted documentation framework powered by Material for MkDocs (`mkdocs-material`), `mkdocstrings[python]`, and KaTeX math rendering (`pymdownx.arithmatex`).
    - Narrative documentation guides across all framework analytical layers: Getting Started, User Guide, and Theoretical Drift Taxonomy.
    - Autogenerated API reference manual covering all public subpackages (`stride.datasets`, `stride.drift`, `stride.models`, `stride.plotting`, `stride.xai.*`, and `stride.exceptions`).
    - GitHub Actions documentation workflow (`.github/workflows/docs.yml`) with automated strict PR build validation and continuous deployment to GitHub Pages.
- **Codebase Docstring Standardization**:
    - Systematic standardization across all public modules in `src/stride/` to Google style.
    - Dedicated parameter types in Python 3.10+ signatures and deduplicated docstrings.
    - Formal LaTeX mathematical formulations for statistical tests, loss functions, divergences, and clustering dynamics.
    - Ruff `D417` docstring quality gate and `convention = "google"`.
- **Deterministic AI Agent Documentation Standards**:
    - Codified rules in `.agents/rules/documentation_standards.md` governing all future agent-generated code.

### Changed
- **Documentation Refinement & Professionalization**:
    - Eliminated promotional tropes, conversational filler, and decorative emojis across the documentation suite.
    - Expanded user guides, mathematical derivations, and architecture overviews into publication-grade technical prose.
    - Enriched all API reference pages with module architectures, exported component matrices, and standard import workflows.

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
