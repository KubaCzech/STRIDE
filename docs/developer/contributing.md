# Contributing to STRIDE

This guide outlines standards, engineering practices, and verification protocols for contributing to the STRIDE framework.

---

## Development Setup

### 1. Repository Setup

Fork the repository on GitHub and clone your fork locally:

```bash
git clone https://github.com/<your-username>/STRIDE.git
cd STRIDE
```

Configure the upstream remote:

```bash
git remote add upstream https://github.com/KubaCzech/STRIDE.git
git fetch upstream
```

### 2. Virtual Environment Configuration

Create and activate an isolated Python 3.10+ virtual environment:

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

### 3. Install Dependencies in Editable Mode

Install the framework along with all development and documentation dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[all,docs,dev]"
```

---

## Architectural Boundaries & Invariants

All contributions must adhere to the following architectural invariants:

1. **Strict Core / UI Decoupling**:
   - The core framework under `src/stride/` must contain zero Streamlit code. Never import `streamlit` or rely on UI session state in algorithmic modules.
   - Streamlit components, state management, and dashboard widgets belong exclusively under `dashboard/`.
2. **Headless Visualizations**:
   - Functions in `stride.plotting` must generate and return `matplotlib.figure.Figure` or `matplotlib.axes.Axes` objects.
   - Never call `plt.show()` within library modules, as this blocks execution in non-interactive environments and CI runners.
3. **Reproducibility & Random Seeds**:
   - Stochastic operations (data generation, train/test splits, neural autoencoder initialization) must accept a `random_state` or `random_seed` integer parameter to ensure deterministic execution across benchmark runs.
4. **Exception Handling**:
   - Library code must raise specific exceptions derived from `StrideError` (e.g. `DimensionalityError`, `OptionalDependencyError`, `DriftDetectionError`) rather than generic uncaught runtime exceptions.

---

## Git Workflow & Commit Protocol

### Branch Invariant

Always branch directly from the latest `origin/main`:

```bash
git fetch origin main
git checkout -b <type>/<short-description> origin/main
```

Allowed branch prefixes:
* `feat/` for new features or algorithmic components
* `fix/` for bug patches
* `refactor/` for structural refactorings without behavioral changes
* `docs/` for documentation updates
* `test/` for test suite additions

### Conventional Commits

Commit messages must strictly follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>[optional scope]: <imperative description>
```

* **Types**: `feat`, `fix`, `refactor`, `style`, `test`, `docs`, `chore`.
* **Scope**: Noun identifying the affected subsystem (e.g. `api`, `drift`, `xai`, `dashboard`, `models`, `datasets`, `clustering`, `recurrence`).
* **Format**: Imperative, present tense (e.g., `feat(xai): add conformal abstention to discriminator`).
* **Atomicity**: Each commit should solve a single logical problem and maintain passing tests.

---

## Local CI Verification Gates

Before submitting code or pushing branches, all local quality gates must pass cleanly:

```bash
# 1. Lint code with Ruff (Google docstrings, D417 parameter completeness)
ruff check .

# 2. Check code formatting with Ruff
ruff format --check .

# 3. Execute automated test suite
python -m unittest discover tests

# 4. Validate documentation builds in strict mode (zero warnings)
mkdocs build --strict
```

Ensure there are **0 errors, 0 warnings, and 0 test failures** before creating a pull request.

---

## Submitting Pull Requests

1. Push your feature branch to your GitHub fork:
   ```bash
   git push -u origin <branch-name>
   ```
2. Open a Pull Request against `KubaCzech/STRIDE:main`.
3. Fill out the Pull Request template completely, providing:
   * Summary of changes.
   * Motivation and context (including issue references like `Closes #12`).
   * Explicit confirmation that local verification gates (`ruff`, `unittest`, `mkdocs`) passed.
4. Monitor the remote GitHub Actions CI workflows to verify clean execution across all matrix configurations.
