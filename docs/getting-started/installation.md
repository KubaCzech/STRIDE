# Installation

STRIDE is published on the Python Package Index (**PyPI**) as `stride-xai`. It supports Python 3.10, 3.11, and 3.12.

---

## Minimal Base Installation

To install STRIDE with core algorithms (data stream generators, statistical testing, basic models, sequential error tracking):

```bash
pip install stride-xai
```

The core installation includes:
- `numpy<2.0`
- `pandas`
- `scipy`
- `scikit-learn`
- `river`

---

## Optional Dependency Extras

STRIDE is architected with modular dependency extras so users only install what their target pipelines require:

```bash
# Data stream and metric plotting (matplotlib, seaborn, plotly)
pip install stride-xai[vis]

# Advanced clustering (pyclustering, hdbscan, umap-learn)
pip install stride-xai[clustering]

# Advanced xAI explainers (SHAP, LIME)
pip install stride-xai[xai]

# Deep learning boundary projection (TensorFlow 2.15+)
pip install stride-xai[deeplearning]

# Interactive Streamlit dashboard
pip install stride-xai[dashboard]

# Documentation building (MkDocs Material, mkdocstrings)
pip install stride-xai[docs]

# Complete installation with all dependencies
pip install stride-xai[all]
```

---

## Development Installation

To install from source for development or contributions:

```bash
git clone https://github.com/KubaCzech/STRIDE.git
cd STRIDE
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[all,docs,dev]"
```

Verify your installation:

```bash
python -c "import stride; print('STRIDE Version:', stride.__version__)"
```
