# AI Agent Documentation Standards

This page documents the engineering standards and docstring invariants enforced across the STRIDE codebase, reflecting the governance rules defined in `.agents/rules/documentation_standards.md`.

---

## Purpose & Scope

To ensure long-term codebase consistency across contributions from human developers and autonomous AI coding agents, all code modifying public interfaces under `src/stride/` must comply with the following invariants.

---

## Core Invariants

### 1. Zero Undocumented Public APIs
Every public module, class, public method, function, and custom exception must include a comprehensive Google-style docstring. Internal helper functions (`_leading_underscore`) are exempt unless implementing non-trivial algorithmic mathematics.

### 2. Single Convention (Google Style Exclusively)
The repository strictly enforces the Google docstring convention. Mixing NumPy style (e.g. `Parameters \n ----------`) or unstructured markdown blocks is prohibited.

### 3. Type Deduplication (Signatures vs Docstrings)
Types must be declared exclusively within the Python type signature using Python 3.10+ syntax:
* Use `list[str]`, `dict[str, Any]`, `tuple[int, int]`.
* Use union syntax `float | None` instead of `Optional[float]`.

**Do not duplicate types in docstrings**. Use `param_name: Description of parameter.`, never `param_name (int): Description` or `param_name : int`.

### 4. Mandatory `Raises:` Block
Every function or method that explicitly raises an exception (such as `StrideError`, `OptionalDependencyError`, `DimensionalityError`, `ValueError`, or `KeyError`) must document the exception and its trigger condition in a `Raises:` block.

### 5. Formal Mathematical Formulations (LaTeX)
Mathematical symbols, hypotheses, divergences, and loss objectives must be formatted in standard LaTeX math syntax (`$...$` inline, `$$...$$` display).

### 6. Synchronized API Reference
Whenever a new module or public estimator is introduced, the corresponding API reference document under `docs/api/` must be updated using the `::: stride.<subpackage>.<module>` mkdocstrings directive.

---

## Canonical Docstring Templates

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

### Function / Method Template

```python
def compute_drift_importance(
    self,
    importance_method: str = "permutation",
    include_target: bool = True,
    model_class: type[BaseModel] | None = None,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze data drift or concept drift feature attributions.

    When `include_target=True`, analyzes Concept Drift ($P(Y \\mid X)$ shifts) by
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

## Pattern Comparison Matrix

| Element | Correct Pattern | Prohibited Anti-Pattern |
| :--- | :--- | :--- |
| **Section Header** | `Args:`, `Returns:`, `Raises:`, `Attributes:`, `Examples:` | `Parameters:`, `Arguments:`, `Return:`, `Throws:` |
| **Parameter Line** | `param_name: Description of the parameter.` | `param_name (type): Description` or `param_name : type` |
| **Return Semantics** | Defined in signature `-> ReturnType:`. Docstring describes semantics. | `Returns: ReturnType: Description` duplicating type |
| **Optional Arguments**| Annotate in signature `x: int \| None = None`. Note defaults in prose. | `x (int, optional): ...` |
| **Exceptions** | `Raises:\n    StrideError: If input is invalid.` | Omitting `Raises:` section when errors are thrown |
| **Math Formatting** | `$P(Y \mid X)$`, `$\mathcal{W}_1(P, Q)$` | `P(Y|X)`, `W1(P, Q)` in plain ASCII text |

---

## Automated Verification

Docstring compliance is checked during local and remote CI workflows:

```bash
# 1. Check docstring parameter completeness and Google style with Ruff
ruff check .

# 2. Strict documentation build validating mkdocstrings AST parsing and links
mkdocs build --strict
```
