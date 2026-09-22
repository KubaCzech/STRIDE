# Python Package & API Architecture Standards

> [!IMPORTANT]
> **Trigger Paths**: `src/stride/**`, `src/**`, `pyproject.toml`
> **When to Read**: MUST be read before implementing, modifying, or refactoring package modules, API interfaces, estimators, or algorithmic pipelines.

---

## 1. Package Architecture & Layout

The STRIDE framework follows the standard PyPA `src`-layout:

```
STRIDE/
├── pyproject.toml              # Build backend, metadata, core & optional dependencies (PEP 621)
├── src/
│   └── stride/                 # The ONLY installable package root ('import stride')
│       ├── __init__.py         # Curated public API exports (__all__) and __version__
│       ├── py.typed            # PEP 561 typing marker
│       ├── exceptions.py       # Domain exception hierarchy
│       ├── common/             # Preprocessing, scalers, dimensionality reducers
│       ├── datasets/           # Synthetic stream generators (SEA, Hyperplane, RBF, LWI)
│       ├── drift/              # Drift detection methods and descriptors (DDM, etc.)
│       ├── xai/                # Explainable AI modules (boundary, importance, clustering, recurrence)
│       ├── models/             # Estimator wrappers (MLP, RandomForest)
│       └── plotting/           # Visualization generators (return Figure/Axes, NO plt.show)
├── dashboard/                  # Interactive Streamlit application (consumer of 'stride')
│   ├── app.py
│   ├── config/                 # UI widget schemas and dataset storage adapters
│   └── components/             # Dashboard tabs, modals, and sidebar
└── tests/                      # Automated test suite importing 'stride'
```

### Architectural Tiers & Separation Invariants
1. **Zero UI Code in Core Engine**: Core algorithms, estimators, and stream generators must have ZERO knowledge of Streamlit, HTML, or UI schemas (`get_settings_schema()`). All UI widget configurations belong in `dashboard/config/`.
2. **Headless Execution**: Library visualization functions in `stride.plotting` or `stride.xai.*.visualization` must NEVER call `plt.show()` or block execution. They must return `matplotlib.figure.Figure` or `Axes` objects to the caller.
3. **No Brittle Deep Imports**: Public entities must be importable from the top-level package or subpackages (e.g. `from stride.drift import BinaryErrorDriftDescriptor`, `from stride.datasets import SeaDriftDataset`).

---

## 2. Typing & Interface Conventions

1. **Python 3.10+ Modern Type Annotations**:
   - Use built-in generics (`list[str]`, `dict[str, Any]`, `tuple[int, int]`) instead of `typing.List`, `typing.Dict`, `typing.Tuple`.
   - Use union operator syntax (`int | None`, `float | np.ndarray`) instead of `Optional` or `Union`.
2. **Explicit Return Types**: All public functions and methods must declare explicit return types (e.g. `-> np.ndarray:`, `-> tuple[pd.DataFrame, pd.Series]:`).
3. **Interface Contracts**:
   - **Data Generators**: Must accept `random_state: int | None = 42` and return `(X, y)` tuples where `X` is a 2D DataFrame/array and `y` is a 1D Series/array.
   - **Estimators**: Must implement Scikit-learn estimator interface (`fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)`).
   - **Transformers**: Must implement `fit(X)`, `transform(X)`, `fit_transform(X)`.

---

## 3. Exception Hierarchy

All custom domain exceptions must inherit from `StrideError` defined in `stride.exceptions`:

```python
class StrideError(Exception):
    """Base exception for all STRIDE errors."""


class OptionalDependencyError(StrideError):
    """Raised when an optional dependency (e.g. tensorflow, shap) is missing."""


class DriftDetectionError(StrideError):
    """Raised when drift detection calculation fails or inputs are invalid."""


class DimensionalityError(StrideError):
    """Raised when input feature dimensions violate method constraints (e.g. n_features < 2)."""
```

---

## 4. Documentation Standards

All public classes, methods, and functions must maintain NumPy/Google-style docstrings with explicit sections:
- `Parameters`: Parameter name, type, default, and semantic description.
- `Returns`: Return type and description of the output.
- `Raises`: Specific exceptions raised and under what conditions.
- `Examples` (optional for complex routines): Doctrinal code snippet.

---

## 5. Anti-Pattern & Pitfall Traps

| Anti-Pattern Trap | Why It Fails | Golden Pattern |
| :--- | :--- | :--- |
| **`import src.*`** | Breaks when installed as a package; causes global namespace collisions. | Use `import stride.*` with standard `src/stride/` layout. |
| **Calling `plt.show()` in library** | Blocks script execution, opens desktop windows, and crashes web/cloud runtimes. | Return `fig` (`matplotlib.figure.Figure`) and let caller render. |
| **Streamlit schemas in domain models** | Tightly couples ML engine to Streamlit; pollutes API with UI metadata. | Isolate UI widget definitions in `dashboard/config/`. |
| **Hardcoding file paths** | Fails when invoked from different working directories or installed via pip. | Use caller-provided paths or configurable parameters. |
| **Unseeded random generators** | Non-reproducible stream data breaks benchmarks and regression tests. | Pass and respect `random_state: int \| None` using `np.random.RandomState`. |
| **In-place array mutation** | Corrupts data when multiple explainers analyze the same stream window. | Always operate on `.copy()` of input matrices. |
| **Unconditional heavy imports** | Installing or importing `stride` fails if optional heavy packages (TensorFlow) aren't present. | Use lazy imports or guard with `try/except ImportError` raising `OptionalDependencyError`. |
