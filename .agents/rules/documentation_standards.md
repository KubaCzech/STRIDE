# AI Agent Documentation Standards & Docstring Invariants

> [!IMPORTANT]
> **Trigger Paths**: `src/stride/**`, `docs/**`, `pyproject.toml`
> **When to Read**: MUST be read before authoring, modifying, or refactoring public Python modules, classes, methods, functions, or documentation pages.

This document establishes the mandatory standards for API documentation, docstring structure, mathematical formulations, and documentation website synchronization across the **STRIDE** framework.

---

## 1. Core Documentation Invariants for AI Coding Assistants

All AI agents generating or modifying code in STRIDE must strictly adhere to these six non-negotiable invariants:

1. **Zero Undocumented Public APIs**:
   - Every public module, class, public method, public function, and exception must have a comprehensive Google-style docstring.
   - Private helpers (`_leading_underscore`) do not require full docstrings unless they implement non-trivial algorithmic mathematics.
2. **Single Convention (Google Style Only)**:
   - Google style is mandatory across the entire repository.
   - Mixing NumPy style (e.g. `Parameters \n ----------`) or freeform markdown docstrings is strictly prohibited.
3. **Signature vs Docstring Typing (No Type Duplication)**:
   - Types must be declared exclusively in the Python signature using modern Python 3.10+ syntax (`list[str]`, `dict[str, Any]`, `float | None`, `tuple[int, int]`).
   - Do NOT duplicate types in docstrings. Use `param: description`, NEVER `param (int): description` or `param : int`.
4. **Mandatory `Raises:` Block**:
   - Every raised custom exception (`StrideError`, `OptionalDependencyError`, `DimensionalityError`, `DriftDetectionError`) and standard exception (`ValueError`, `KeyError`) must be explicitly documented with its exact trigger condition.
5. **Formal Mathematical Formulations (LaTeX)**:
   - Statistical formulations, hypotheses, divergences, and loss functions must be written in standard LaTeX math syntax (`$...$` for inline, `$$...$$` for block display).
   - Use standard symbols: $P(X)$ for feature distributions, $P(Y \mid X)$ for posterior posteriors / real concept drift, $\mathcal{W}_1$ for 1D Wasserstein distance, $\alpha$ for significance levels.
6. **Synchronized API Reference (`docs/api/*.md`)**:
   - Whenever a new public module or estimator is introduced, the corresponding API reference file in `docs/api/` must be created or updated using the `::: stride.<subpackage>.<module>` mkdocstrings directive.

---

## 2. Canonical Google-Style Docstring Templates

### Class Template

```python
class DecisionBoundaryDriftAnalyzer:
    """Analyze decision boundary shifts between data stream windows.

    Projects high-dimensional feature spaces into a 2D manifold using Self-Supervised
    Neighbor Projection (SSNP) or passthrough projections, fits classifiers on pre-drift
    and post-drift windows, and characterizes geometric boundary changes.

    Attributes:
        random_state: Pseudo-random seed used for stochastic operations.
        X_before: Pre-drift feature matrix of shape `(n_samples, n_features)`.
        y_before: Pre-drift target labels.
        X_after: Post-drift feature matrix.
        y_after: Post-drift target labels.

    Examples:
        >>> analyzer = DecisionBoundaryDriftAnalyzer(X_ref, y_ref, X_det, y_det)
        >>> report = analyzer.analyze(grid_size=200)
    """
```

### Method / Function Template

```python
def compute_drift_importance(
    self,
    importance_method: str = "permutation",
    include_target: bool = True,
    model_class: type[BaseModel] | None = None,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze data drift or concept drift feature attributions.

    When `include_target=True`, analyzes Concept Drift ($P(Y \mid X)$ shifts) by
    concatenating features and labels $[X, Y]$ to classify the temporal window.
    When `include_target=False`, analyzes Covariate Shift ($P(X)$ shifts) using
    only feature representations $X$.

    Args:
        importance_method: Explainability method to calculate importance
            ("permutation", "shap", or "lime").
        include_target: Whether to append the target label $Y$ into the
            discriminator input matrix.
        model_class: Classifier wrapper class used for temporal discrimination.
            Defaults to `MLPModel`.
        model_params: Optional initialization dictionary passed to `model_class`.

    Returns:
        Dictionary containing:
            - 'model': Trained classifier instance.
            - 'accuracy': Discriminator classification accuracy.
            - 'importance_result': Detailed attribution metrics dictionary.
            - 'importance_mean': Mean importance array across evaluated features.
            - 'importance_std': Standard deviation array across permutations.
            - 'feature_names': List of feature names corresponding to the scores.

    Raises:
        OptionalDependencyError: If `importance_method="shap"` and `shap`
            is not installed.
        ValueError: If `importance_method` is not one of {"permutation", "shap", "lime"}.

    Examples:
        >>> analyzer = FeatureImportanceDriftAnalyzer(X_ref, y_ref, X_det, y_det)
        >>> res = analyzer.compute_drift_importance(importance_method="permutation")
    """
```

---

## 3. Formatting Rules & Style Details

| Element | Correct Pattern | Prohibited Anti-Pattern |
| :--- | :--- | :--- |
| **Section Names** | `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Examples:` | `Parameters:`, `Arguments:`, `Return:`, `Throws:` |
| **Parameter Line** | `param_name: Description of the parameter.` | `param_name (type): Description` or `param_name : type` |
| **Return Type** | Defined in signature `-> ReturnType:`. Docstring describes semantics. | `Returns: ReturnType: Description` duplicating type |
| **Optional Params** | Annotate in signature `x: int \| None = None`. Note defaults in prose if needed. | `x (int, optional): ...` |
| **Exceptions** | `Raises:\n    StrideError: If input is invalid.` | Omitting `Raises:` section when errors are thrown |
| **Math in Docs** | `$P(Y \mid X)$`, `$\mathcal{W}_1(P, Q)$` | `P(Y|X)`, `W1(P, Q)` in plain ASCII |

---

## 4. Ruff Quality Gate & Local Validation

Docstring formatting and completeness are enforced via Ruff and MkDocs:

```bash
# 1. Validate docstrings with Ruff (Google convention, D417 parameter completeness)
ruff check .

# 2. Strict documentation build (validates all references and docstring AST parsing)
mkdocs build --strict
```
