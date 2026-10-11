# xAI: Feature Importance Attribution

The `stride.xai.importance` subpackage provides tools for attributing drift to specific input features using temporal discriminators, distinguishing Covariate Shift ($P(X)$) from Real Concept Drift ($P(Y \mid X)$), and applying conformal selective prediction.

---

## Overview

Feature importance attribution transforms drift characterization into a classification problem: an auxiliary temporal discriminator is trained to differentiate reference window samples from detection window samples. Explainability methods (PFI, SHAP, LIME) are then applied to the discriminator.

```python
from stride.xai.importance import (
    FeatureImportanceDriftAnalyzer,
    AbstainStrategy,
    FeatureImportanceMethod,
)
```

### Key Components

| Class / Enum | Description |
| :--- | :--- |
| `FeatureImportanceDriftAnalyzer` | High-level orchestrator that prepares discriminator datasets, trains the temporal classifier, applies explainability backends, and handles conformal abstention. |
| `AbstainStrategy` | Enumeration specifying selective classification strategies (`NONE`, `CONFORMAL`). |
| `FeatureImportanceMethod` | Enumeration of supported attribution algorithms (`PERMUTATION`, `SHAP`, `LIME`). |
| `calculate_feature_importance` | Low-level routine evaluating permutation scores or Shapley values on trained estimators. |

---

## API Specification

::: stride.xai.importance
