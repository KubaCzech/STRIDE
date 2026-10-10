# AI Agent Documentation Standards

This page mirrors the internal AI agent documentation standards codified in `.agents/rules/documentation_standards.md`.

---

## Non-Negotiable Invariants

1. **Zero Undocumented Public APIs**: Every public class, method, function, and exception in `src/stride/` must have a complete Google-style docstring.
2. **Single Convention**: Google docstrings are mandatory across the entire repository. Mixing NumPy style or freeform markdown is prohibited.
3. **Type Deduplication**: Types are declared exclusively in Python 3.10+ signatures (`list[str]`, `dict[str, Any]`, `X | None`). They must NOT be duplicated in docstrings.
4. **Mandatory `Raises:` Block**: Explicitly document every raised `StrideError` subclass and standard exception.
5. **Formal Math**: Use LaTeX math syntax (`$...$` inline, `$$...$$` display) for all mathematical concepts.
6. **API Sync**: When adding new modules, update `docs/api/*.md` with the corresponding `::: stride.<module>` block.

---

## Canonical Docstring Schema

```python
def compute_drift_importance(
    self,
    importance_method: str = "permutation",
    include_target: bool = True,
    model_class: type[BaseModel] | None = None,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze data drift or concept drift feature attributions.

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
    """
```
