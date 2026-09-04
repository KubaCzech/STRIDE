# ML & Drift Simulation Standards & Guidelines

> [!IMPORTANT]
> **Trigger Paths**: `src/**`
> **When to Read**: MUST be read before implementing, modifying, or refactoring data stream generators, drift detectors, machine learning classifiers, or xAI explanation tools.

## 1. Core Principles & Stack

1. **Deterministic Reproducibility**: All synthetic generators (`SEA`, `Hyperplane`, `RBF`, `LWI`) and model training routines must accept and respect an explicit `random_state` seed.
2. **Interface Consistency**: Data generators must return standard tuples `(X, y)` where `X` is a 2D NumPy array or Pandas DataFrame of shape `(n_samples, n_features)` and `y` is a 1D array of shape `(n_samples,)`.
3. **Drift Metadata**: Stream generators must supply ground-truth drift transition indices or timestamps alongside the generated data for automated evaluation.
4. **Model Invariants**: Classifiers must conform to the Scikit-learn estimator interface (`fit(X, y)`, `predict(X)`, `predict_proba(X)`).
5. **Dimensionality Gating**: Boundary estimators and dense grid evaluators must validate feature dimension; project down or raise informative errors if `n_features > 2` and 2D visualization is requested.

## 2. Declarative Code Standards (Golden Patterns)

```python
# Standard generator signature and seeding
def generate_sea_drift(n_samples: int = 1000, random_state: int | None = 42) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rng = np.random.RandomState(random_state)
    ...
    return X, y, drift_indices

# Safe dimensionality projection for boundary visualizers
def prepare_2d_projection(X: np.ndarray, selected_features: tuple[int, int]) -> np.ndarray:
    if X.shape[1] < 2:
        raise ValueError("At least 2 features are required for 2D boundary analysis.")
    return X[:, [selected_features[0], selected_features[1]]]
```

---

## 3. Anti-Pattern & Pitfall Traps

| Anti-Pattern Trap | Why It Fails | Golden Pattern |
| :--- | :--- | :--- |
| **Unseeded `np.random` calls** in stream generation | Results vary across runs, breaking automated regression tests and dashboard reproducibility. | Always initialize and pass `np.random.RandomState(seed)` or `np.random.default_rng(seed)`. |
| **In-place Mutation of Input Feature Matrices** | Corrupts data when multiple explainers (SHAP, Permutation) analyze the same stream slice. | Always operate on `.copy()` of slices or read-only views. |
| **Assuming Target `y` is 0/1 integers** | Multi-class or string labels cause silent failure in decision boundary thresholds. | Assert or encode targets using `LabelEncoder` before boundary mesh computation. |
| **Direct Full-Mesh Sampling on >3 Features** | Combinatorial explosion causes memory exhaustion (`OutOfMemoryError`). | Restrict dense grid generation to 2 selected feature axes while holding remaining features constant. |
| **Unbounded Iterations in Classifiers** | Optimization loops hang without convergence warnings. | Explicitly set `max_iter` and handle `ConvergenceWarning` gracefully. |
