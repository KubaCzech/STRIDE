# Installation

STRIDE is published on the Python Package Index as [`stride-xai`](https://pypi.org/project/stride-xai/). It supports Python 3.10, 3.11, and 3.12.

---

## Prerequisites

Before installing STRIDE, ensure that your environment meets the following requirements:

* **Python**: Version 3.10, 3.11, or 3.12.
* **Package Manager**: `pip` (version 21.0 or higher is recommended).
* **Virtual Environment**: Using an isolated virtual environment (`venv` or `conda`) is strongly advised to prevent dependency conflicts.

---

## PyPI Installation

### Core Package

To install STRIDE with its core algorithmic suite (synthetic stream generators, statistical testing, model wrappers, and sequential error tracking):

```bash
pip install stride-xai
```

The core installation includes:
* `numpy<2.0`
* `pandas`
* `scipy`
* `scikit-learn`
* `river`
* `matplotlib`
* `seaborn`

With these core dependencies, synthetic stream generators, statistical testing, model wrappers, 2D decision boundary analysis, Permutation Feature Importance (PFI), and stream visualization are immediately functional.

### Optional Extras

STRIDE uses a modular dependency architecture. Install additional capabilities as needed for your specific deployment:

| Extra | Command | Included Dependencies & Capabilities |
| :--- | :--- | :--- |
| **Full XAI Suite (Headless)** | `pip install "stride-xai[xai-full]"` | Complete analytical horsepower (`shap`, `lime`, `pyclustering`, `hdbscan`, `umap-learn`, `tensorflow>=2.15`, `plotly`) for pipelines and notebooks without the web dashboard. |
| **Interactive Dashboard** | `pip install "stride-xai[dashboard]"` | `streamlit` and `icecream` alongside all analytical dependencies for the interactive multi-tab application. |
| **Explainable AI** | `pip install "stride-xai[xai]"` | `shap` and `lime` for model-agnostic feature attributions. |
| **Clustering** | `pip install "stride-xai[clustering]"` | `pyclustering`, `hdbscan`, and `umap-learn` for X-means and recurrence clustering. |
| **Deep Learning** | `pip install "stride-xai[deeplearning]"` | `tensorflow>=2.15` for SSNP neural 2D boundary projections. |
| **Interactive Visualizations** | `pip install "stride-xai[vis]"` | `plotly` for interactive web charts. |
| **Documentation** | `pip install "stride-xai[docs]"` | `mkdocs-material`, `mkdocstrings`, and KaTeX rendering extensions. |
| **Full Suite** | `pip install "stride-xai[all]"` | Installs all optional dependencies (dashboard, complete XAI, and developer tools). |

To install multiple extras simultaneously, provide a comma-separated list:

```bash
pip install "stride-xai[xai,clustering,dashboard]"
```

---

## Source Installation

For local development or running experimental branches, clone the repository and install in editable mode:

### 1. Clone the Repository

```bash
git clone https://github.com/KubaCzech/STRIDE.git
cd STRIDE
```

### 2. Create and Activate a Virtual Environment

**Linux / macOS**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[all]"
```

The editable flag (`-e`) installs the package in place, allowing source edits in `src/stride/` to take effect immediately without reinstallation.

---

## Verifying Installation

Verify that STRIDE is correctly registered in your Python environment:

```bash
python -c "import stride; print('STRIDE Version:', stride.__version__)"
```

To run the automated test suite locally:

```bash
python -m unittest discover tests
```
