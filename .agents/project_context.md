# Project Context - STRIDE

## Master Entry Point
See [AGENTS.md](../AGENTS.md) at the repository root for immediate orientation and the Rule Routing Matrix.

## Current Goal
Develop, benchmark, and visualize Explainable AI (xAI) techniques for characterizing concept drift in data streams, integrated with an interactive Streamlit dashboard.

## Implementation Details
- **Architecture**: Modular Python framework (`src/`) implementing data stream generation, drift detection (DDM), decision boundary shift analysis, feature importance shift (SHAP/permutation), clustering dynamics, and prototype-based recurring concept analysis + interactive Streamlit dashboard (`dashboard/`).
- **Key Technologies**: Python 3.10+, Streamlit, Scikit-learn, NumPy, Pandas, Altair, Matplotlib, Plotly, SHAP, HDBSCAN.

## Repository Status
- [x] Initial repository setup and agent context initialization (`AGENTS.md`, `.agents/`).
- [x] Core drift generation algorithms (SEA, Hyperplane, RBF, LWI).
- [x] Interactive Streamlit dashboard with multi-tab analysis and sidebar controls.
- [x] Isolated model parameters and dataset feature reduction sanitization.
- [x] Official Streamlit AI agent skills integration (`.agents/skills/developing-with-streamlit/`).
- [ ] Expand automated test coverage for core xAI algorithms in `src/`.
- [ ] Implement additional statistical drift detectors and recurring concept benchmarks.

## Critical Requirements & Developer Guidelines
1. **Local Setup**: Python 3.10-3.12 with `.venv`. Run dashboard via `streamlit run dashboard/app.py`.
2. **Deterministic Rules**: Always consult `AGENTS.md` and read the matching `.agents/rules/*.md` before modifying code.
3. **Quality & Verification**: Execute local flake8 and `python -m unittest discover tests` before committing.
4. **Self-Maintenance**: Update `.agents/` when completing features or changing dependencies using `.agents/skills/agent-maintenance/SKILL.md`.
