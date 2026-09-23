# Project Context - STRIDE

## Master Entry Point
See [AGENTS.md](../AGENTS.md) at the repository root for immediate orientation and the Rule Routing Matrix.

## Current Goal
Develop, benchmark, and visualize Explainable AI (xAI) techniques for characterizing concept drift in data streams, integrated with an interactive Streamlit dashboard.

## Implementation Details
- **Architecture**: PyPA-standard Python package `stride-xai` (`src/stride/`) implementing data stream generation, drift detection, decision boundary shift analysis, feature importance shift (SHAP/permutation), clustering dynamics, and prototype-based recurring concept analysis + decoupled dashboard config (`dashboard/config/`) and interactive Streamlit UI (`dashboard/`).
- **Key Technologies**: Python 3.10-3.12, Flit / PEP 621 packaging, Streamlit, Scikit-learn, NumPy (<2.0), Pandas, SciPy, Matplotlib, Plotly, SHAP, HDBSCAN.

## Repository Status
- [x] Initial repository setup and agent context initialization (`AGENTS.md`, `.agents/`).
- [x] Core drift generation algorithms (SEA, Hyperplane, RBF, LWI).
- [x] Interactive Streamlit dashboard with multi-tab analysis and sidebar controls.
- [x] Isolated model parameters and dataset feature reduction sanitization.
- [x] Official Streamlit AI agent skills integration (`.agents/skills/developing-with-streamlit/`).
- [x] Publication-grade README.md and documentation alignment for ECML PKDD 2026 Demo Track.
- [x] PEP 621 / PyPA standard packaging migration (`src/stride/` layout with `stride-xai`).
- [x] Decouple UI widget schemas from core algorithmic engine (`dashboard/config/`).
- [x] Modular optional dependency extras (`vis`, `drift`, `clustering`, `xai`, `deeplearning`, `dashboard`, `dev`, `all`).
- [x] Single Responsibility Principle (SRP) cleanup, decomposition of monolithic God Classes (`ClusterBasedDriftDetector`), and headless plotting execution without `plt.show()`.
- [ ] Expand automated test coverage for core xAI algorithms in `src/stride/`.
- [ ] Implement additional statistical drift detectors and recurring concept benchmarks.

## Critical Requirements & Developer Guidelines
1. **Local Setup**: Python 3.10-3.12 with `.venv`. Run dashboard via `streamlit run dashboard/app.py`.
2. **Deterministic Rules**: Always consult `AGENTS.md` and read the matching `.agents/rules/*.md` before modifying code.
3. **Quality & Verification**: Execute local `ruff check .`, `ruff format --check .`, and `python -m unittest discover tests` before committing.
4. **Self-Maintenance**: Update `.agents/` when completing features or changing dependencies using `.agents/skills/agent-maintenance/SKILL.md`.
