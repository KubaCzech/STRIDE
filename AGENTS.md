# AGENTS.md - Master Instructions & Context for AI Assistants

Welcome to the **STRIDE** (Explainable AI & Drift Detection Framework) repository! This file serves as the primary entry point, high-level context map, and deterministic rule routing matrix for AI coding assistants.

---

## 🏛️ Project Architecture & Overview

STRIDE is a research framework and interactive **Streamlit dashboard** for detecting, characterizing, and explaining concept drift in machine learning pipelines using Explainable AI (xAI). It combines synthetic stream generation, statistical drift detection, decision boundary shifts, feature importance shifts (SHAP/permutation), clustering dynamics, and prototype-based recurring concept analysis.

The codebase is split into core algorithmic modules under `src/` and an interactive multi-tab application under `dashboard/`.

---

## 🚦 Mandatory Rule Routing Matrix

Before writing or modifying any code, identify your target area and **read the corresponding rule file FIRST**:

| When working on / modifying... | Target Paths / Globs | Mandatory File to Read FIRST | Key Invariants & Pitfalls to Check |
| :--- | :--- | :--- | :--- |
| **Streamlit Dashboard & UI** | `dashboard/**`, `streamlit_app.py` | [`.agents/rules/dashboard_standards.md`](.agents/rules/dashboard_standards.md)<br>[`.agents/skills/developing-with-streamlit/SKILL.md`](.agents/skills/developing-with-streamlit/SKILL.md) | • Isolate model/dataset state per key<br>• Use `width="stretch"` (no `use_container_width`)<br>• Gating expensive computation with `st.fragment`/`st.form` |
| **ML Models, Drift & xAI** | `src/**` | [`.agents/rules/ml_and_drift_standards.md`](.agents/rules/ml_and_drift_standards.md)<br>[`.agents/context/architecture.md`](.agents/context/architecture.md) | • Fixed random seeds for stream reproducibility<br>• Consistent interface for data generators (`X, y`)<br>• Handle high-dimensional projections cleanly |
| **CI, Linting & Testing** | `.github/workflows/**`, `tests/**` | [`.agents/rules/ci_standards.md`](.agents/rules/ci_standards.md) | • Pass ruff check and ruff format with zero errors<br>• All tests pass via `unittest discover tests` |
| **Agent Context & Config** | `.agents/**`, `AGENTS.md` | [`.agents/rules/agent_maintenance_standards.md`](.agents/rules/agent_maintenance_standards.md)<br>[`.agents/skills/agent-maintenance/SKILL.md`](.agents/skills/agent-maintenance/SKILL.md) | • Update `acs.yaml` triggers on new paths<br>• Maintain roadmap status in `project_context.md` |
| **Git & Version Control** | Repository root / Git | [`.agents/rules/ci_standards.md`](.agents/rules/ci_standards.md) | • Atomic Conventional Commits (`feat`, `fix`, `test`, `chore`)<br>• Mandatory pre-push local CI validation |

---

## 🔄 Operational Phase Gates

Every task must progress sequentially through these 5 lifecycle gates:

```
[ Gate 1: Rule & Contract Intake ] ➔ [ Gate 2: Implementation ] ➔ [ Gate 3: Local CI Verification ] ➔ [ Gate 4: Context Maintenance ] ➔ [ Gate 5: Git & PR Protocol ]
```

1. **Gate 1: Rule & Contract Intake (MANDATORY)**: Identify target files. Read required rule and reference files from the *Rule Routing Matrix* using `view_file`. Inspect underlying interfaces before invocation.
2. **Gate 2: Implementation**: Write clean, modular Python adhering strictly to golden patterns in `.agents/rules/`.
3. **Gate 3: Local CI Verification**: Execute all local verification commands (ruff and unittest) to verify clean status.
4. **Gate 4: Context Self-Maintenance**: Update `.agents/context/` or `.agents/project_context.md` if components, models, or dependencies evolved.
5. **Gate 5: Git & PR Protocol**: Follow atomic Conventional Commits; verify branch ancestry before pushing.

---

## ⚙️ Core CLI Tools & Build Commands

Always run these commands with the project virtual environment activated (`.venv`):

| Purpose | Working Directory | Command |
| :--- | :--- | :--- |
| **Run Dashboard** | Root | `streamlit run dashboard/app.py` |
| **Lint (Ruff)** | Root | `ruff check .` |
| **Format Check (Ruff)** | Root | `ruff format --check .` |
| **Run Tests** | Root | `python -m unittest discover tests` |

---

## 🚨 Operational Boundaries & Escalation

- **Always**:
  - Consult the *Mandatory Rule Routing Matrix* before editing code.
  - Run full local verification commands (ruff check/format, tests) prior to committing or pushing.
  - Keep commits atomic with standard Conventional Commits.
- **Ask First (Human Escalation Gateways)**:
  - Adding heavy machine learning dependencies (e.g. PyTorch, TensorFlow) or changing `requirements.txt`.
  - Refactoring core data generator signatures or shared session state keys in the dashboard.
  - Destructive filesystem actions or modifying Git branches without explicit instructions.
- **Never (Safety & Workflow Anti-Patterns)**:
  - Never commit `.env` credentials, raw data caches, or transient runtime files.
  - Never silence errors using unconditional `# noqa` or bare `except:` to bypass CI.
  - Never push failing code to remote branches.

---

## 📁 Repository Layout & Navigation Map

- `AGENTS.md`: Master entry point & rule routing matrix (this file)
- `README.md`: Public project documentation & research overview
- `dashboard/`: Streamlit dashboard (`app.py`, `components/`, `assets/`)
- `src/`: Core framework algorithms (`datasets`, `DDM`, `decision_boundary`, `feature_importance`, `clustering`, `recurrence`, `models`)
- `tests/`: Automated test suite (`test_dashboard_regression.py`, etc.)
- `.agents/`: Agent configuration, rules, context, and workspace skills
  - `acs.yaml`: Machine-readable agent configuration and path triggers
  - `project_context.md`: Living project state and active roadmap
  - `rules/`: Modular declarative rules (`dashboard_standards.md`, `ml_and_drift_standards.md`, `ci_standards.md`, `agent_maintenance_standards.md`)
  - `context/`: Deep architectural specifications (`architecture.md`)
  - `skills/`: Project-specific skills (`developing-with-streamlit`, `agent-maintenance`)
