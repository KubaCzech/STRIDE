# STRIDE: STReam Insight and Drift Explanation

A Python Toolkit for Concept Drift Detection, Characterization, and Explanation

[![ECML PKDD 2026](https://img.shields.io/badge/ECML--PKDD_2026-Demo_Track-1E88E5.svg)](https://michalredm.github.io/stride-website/assets/pdf/paper.pdf)
[![DOI: 10.1007/978-3-032-37685-5_32](https://img.shields.io/badge/DOI-10.1007%2F978--3--032--37685--5__32-blue.svg)](https://link.springer.com/chapter/10.1007/978-3-032-37685-5_32)
[![Website](https://img.shields.io/badge/Website-STRIDE-0A66C2?logo=googlechrome&logoColor=white)](https://michalredm.github.io/stride-website/)
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stream-insight-and-drift-explanation.streamlit.app/)
[![CI](https://github.com/KubaCzech/STRIDE/actions/workflows/ci.yml/badge.svg)](https://github.com/KubaCzech/STRIDE/actions/workflows/ci.yml)
![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-3776AB?logo=python&logoColor=white)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Website](https://michalredm.github.io/stride-website/) &bull; [Live Dashboard](https://stream-insight-and-drift-explanation.streamlit.app/) &bull; [Springer Chapter](https://link.springer.com/chapter/10.1007/978-3-032-37685-5_32) &bull; [Paper (PDF)](https://michalredm.github.io/stride-website/assets/pdf/paper.pdf) &bull; [Architecture](#three-layer-architecture) &bull; [Quickstart](#quickstart) &bull; [Citation](#citation)

---

## Overview

Concept drift in streaming environments degrades machine learning models when underlying data distributions evolve over time. Traditional drift detectors (e.g., DDM, EDDM, ADWIN) signal **when** classifier accuracy drops, but treat the model and data as black boxes—failing to explain **how** the distribution shifted or **where** in the feature space decision boundaries deteriorated.

**STRIDE** (*STReam Insight and Drift Explanation*) bridges this diagnostic gap. Presented at **ECML PKDD 2026** (Demo Track), STRIDE is an open-source Python framework and interactive dashboard that orchestrates analytical signals across three concurrent layers:

1. **Data Layer** (*How*): Monitors raw input distributions and cluster geometry independently of the model.
2. **Model Layer** (*When*): Tracks predictive performance metrics and fires sequential drift alerts.
3. **Explanation Layer** (*Where*): Adapts Explainable AI (xAI) methods to locate local decision boundary shifts, identify drift-driving features, and detect recurring concepts.

Instead of triggering post-hoc explanations only after significant performance degradation, STRIDE executes a **synchronous pipeline** that processes incoming windows across all three layers in parallel. This design enables analysts to trace the emergence of drift and observe the evolution of model logic in real time.

---

## Three-Layer Architecture

![STRIDE Synchronous Processing Pipeline](assets/pipeline.png)

### 1. Data Layer (Statistical & Geometric Shifts)
Evaluates covariate shifts independent of the predictive model:
* **Descriptive Statistics & Divergences**: Computes class-conditional moments, variances, and 1D Wasserstein distances to evaluate distribution movement.
* **Non-Parametric Hypothesis Tests**: Two-sample Kolmogorov-Smirnov (KS) and Anderson-Darling tests detect significant univariate distribution differences between sliding windows.
* **Clustering Dynamics**: Evaluates density shifts using X-means clustering, applying the Hungarian algorithm to track cluster splitting, migration, and centroid displacement across consecutive batches.

### 2. Model Layer (Performance Triggers)
Tracks streaming model health and online error rates:
* **Streaming Detectors**: Integrates the [River](https://riverml.xyz/) stream learning ecosystem, supporting Drift Detection Method (DDM) and related sequential algorithms.
* **Continuous Error Profiling**: Real-time evaluation of error rate trajectories, warning bands, and predictive confidence across sliding windows.

### 3. Explanation Layer (Model-Agnostic Interpretability)
Explains the mechanics of detected drift within the classifier's feature space:
* **Supervised Decision Boundary Maps (SDBM)**: Projects high-dimensional feature spaces into 2D to visualize how class separation boundaries rotate, compress, or deform between pre-drift and post-drift states.
* **Feature Importance Attribution**: Evaluates both model-centric (SHAP, Permutation Feature Importance) and data-centric importance shifts to differentiate features driving model degradation from those preserving predictive utility.
* **Recurring Concept Analysis**: Extracts prototype representations per window and applies HDBSCAN clustering to distance matrices, distinguishing re-emerging historical concepts from novel anomalies.

---

## Interactive Dashboard

STRIDE includes a reactive [Streamlit](https://streamlit.io/) dashboard designed for interactive forensics and exploratory research.

* **Replay Streams**: Step through continuous streams or jump directly to detected drift points.
* **Comparative Window Analysis**: Compare pre-drift reference windows with post-drift detection windows across all three analytical layers side by side.
* **Dynamic Configuration**: Adjust sliding window sizes, statistical test significance thresholds ($\alpha$), and classifier architectures on the fly.
* **Hosted Demo**: An interactive demo is deployed at [stream-insight-and-drift-explanation.streamlit.app](https://stream-insight-and-drift-explanation.streamlit.app/).

---

## Quickstart

### Running the Interactive Dashboard

Launch the dashboard locally:

```bash
streamlit run dashboard/app.py
```

The application opens at `http://localhost:8501`.

### Python API Usage

STRIDE exposes a clean programmatic API under the `stride` namespace. After installation (see below), no `sys.path` manipulation is required:

```python
from stride.datasets import HyperplaneDriftDataset
from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType
from stride.models import RandomForestModel

# 1. Generate a synthetic streaming dataset with rotating hyperplane drift
dataset = HyperplaneDriftDataset()
X, y = dataset.generate(
    n_samples_before=1000,
    n_samples_after=1000,
    n_features=5,
    n_drift_features=2,
    drift_width=100,
    random_seed=42,
)

# 2. Partition into reference (pre-drift) and detection (post-drift) windows
X_ref, y_ref = X.iloc[:1000], y.iloc[:1000]
X_det, y_det = X.iloc[1000:], y.iloc[1000:]

# 3. Train a streaming classifier on the reference window
model = RandomForestModel()
model.fit(X_ref, y_ref)

# 4. Detect and characterize distribution shifts via non-parametric statistical tests
detector = StatisticalTestsDriftDetector(X_ref, y_ref, X_det, y_det)
has_drift = detector.detect(StatisticalTestType.KolmogorovSmirnov)

print(f"Drift detected: {has_drift}")
print(f"Per-feature test outcomes: {detector.drift_flags}")
```

Top-level convenience imports are also available directly from `stride`:

```python
import stride

model = stride.RandomForestModel()
detector = stride.DescriptiveStatisticsDriftDetector(...)
```

---

## Installation

### Prerequisites

* Python 3.10, 3.11, or 3.12
* Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KubaCzech/STRIDE.git
   cd STRIDE
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install the package and its dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

   The editable install (`-e .`) registers the `stride` package in your environment so that `from stride.datasets import ...` works without any `sys.path` manipulation.

   **Optional extras** can be appended in brackets to install additional capability groups:

   | Extra | Installs |
   |---|---|
   | `pip install -e ".[vis]"` | `matplotlib`, `seaborn`, `plotly` |
   | `pip install -e ".[drift]"` | `river` (online stream learning) |
   | `pip install -e ".[clustering]"` | `pyclustering`, `hdbscan`, `umap-learn` |
   | `pip install -e ".[xai]"` | `shap`, `lime` |
   | `pip install -e ".[deeplearning]"` | `tensorflow>=2.15` (for SSNP boundary projection) |
   | `pip install -e ".[dashboard]"` | `streamlit`, `icecream` |
   | `pip install -e ".[all]"` | All of the above |

4. **Verify the installation**:
   ```bash
   python -m unittest discover tests
   ```

---

## Supported Drift Scenarios & Models

### Synthetic Drift Generators
* **Hyperplane Drift**: Continuously rotating hyperplane in $d$-dimensional space simulating gradual concept drift.
* **SEA Drift**: Abrupt decision threshold displacement with noise features.
* **RBF Drift**: Non-linear cluster movement and centroid translation in continuous feature space.
* **Linear Weight Inversion (LWI)**: Correlation inversion testing model sensitivity to feature attribution flips.
* **Multi-Window Generators**: Mixed, Sine, STAGGER, and Random Tree streams for multi-concept scenarios.
* **Real-World Benchmarks**: Pre-configured support for benchmark datasets (e.g., Electricity, Covertype, NOAA weather).

### Model Implementations
* Multi-Layer Perceptrons (MLP / Neural Networks)
* Random Forests
* Support Vector Machines (SVM)
* Logistic Regression & incremental online models via River

---

## Project Structure

```text
STRIDE/
├── assets/                         # Architecture diagrams and visual documentation
│   └── pipeline.png
├── dashboard/                      # Streamlit interactive application
│   ├── app.py                      # Application entry point
│   ├── assets/                     # Dashboard styles and custom CSS
│   ├── components/                 # UI tabs (Data, Model, Explanation layers)
│   └── config/                     # Dashboard-only widget schemas & presets
│       ├── dataset_schemas.py      # Dataset UI configuration (decoupled from core)
│       └── model_schemas.py        # Model UI configuration (decoupled from core)
├── src/                            # PyPA src-layout root
│   ├── stride/                     # Canonical Python package (import as `stride`)
│   │   ├── __init__.py             # Top-level public API & convenience re-exports
│   │   ├── exceptions.py           # Domain exception hierarchy (StrideError, …)
│   │   ├── py.typed                # PEP 561 marker for static type checkers
│   │   ├── common/                 # Shared utilities and window helpers
│   │   ├── datasets/               # Streaming data generators and real datasets
│   │   │   ├── hyperplane_drift.py
│   │   │   ├── sea_drift.py
│   │   │   ├── rbf_drift.py
│   │   │   ├── linear_weight_inversion_drift.py
│   │   │   ├── *_multi_window.py   # Multi-concept stream generators
│   │   │   ├── csv_dataset.py
│   │   │   └── river_dataset.py
│   │   ├── drift/                  # Sequential drift detectors
│   │   │   └── binary_descriptor.py  # BinaryErrorDriftDescriptor (DDM-based)
│   │   ├── models/                 # Model wrappers with fit/predict API
│   │   │   ├── mlp.py              # Multi-Layer Perceptron
│   │   │   └── random_forest.py    # Random Forest
│   │   ├── plotting/               # Stream-level matplotlib visualizations
│   │   └── xai/                    # Explainable AI analyzers
│   │       ├── boundary/           # Decision boundary maps (SDBM / SSNP)
│   │       ├── clustering/         # X-means clustering dynamics
│   │       ├── importance/         # SHAP & Permutation Feature Importance
│   │       ├── recurrence/         # ProTree prototype & HDBSCAN concept detection
│   │       └── stats/              # Statistical tests (KS, AD, Wasserstein)
│   └── DDM/                        # Backward-compatibility bridge → stride.drift
├── tests/                          # Automated test suite
├── pyproject.toml                  # PEP 621 package metadata, extras, and ruff config
└── requirements.txt                # Pinned runtime dependencies
```

---

## Citation

If you use STRIDE in your research, please cite our paper from the **ECML PKDD 2026 Demo Track**:

```bibtex
@inproceedings{nagorka2026stride,
  title     = {{STReam Insight and Drift Explanation (STRIDE): A Python Toolkit for Concept Drift Detection and Explanation}},
  author    = {Nag{\'o}rka, Wojciech and Redmer, Micha{\l} and Czech, Kuba and Aksoy, Deniz and Stefanowski, Jerzy},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Applied Data Science Track, Demo Track and Industrial Track (ECML PKDD)},
  series    = {Lecture Notes in Computer Science},
  pages     = {348--352},
  year      = {2026},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-032-37685-5_32},
  url       = {https://link.springer.com/chapter/10.1007/978-3-032-37685-5_32}
}
```

---

## Authors & Acknowledgments

* **Wojciech Nagórka** &bull; Poznań University of Technology
* **Michał Redmer** &bull; Poznań University of Technology
* **Kuba Czech** &bull; Poznań University of Technology
* **Deniz Aksoy** &bull; Poznań University of Technology
* **Jerzy Stefanowski** &bull; Poznań University of Technology

**Funding**: The research by Jerzy Stefanowski was funded by the National Science Centre, Poland, under OPUS grant no. `2023/51/B/ST6/00545`.

Project Website: [https://michalredm.github.io/stride-website/](https://michalredm.github.io/stride-website/)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
