# Feature Importance Attribution

Feature attribution in STRIDE answers which input features are responsible for the detected distribution change.

---

## Covariate Shift vs Real Concept Drift

STRIDE trains an auxiliary **temporal discriminator** $g(Z)$ that predicts whether an observation originated from the reference window ($t=0$) or the detection window ($t=1$):

1. **Covariate Shift Analysis ($P(X)$)**:
   - Input: $Z = X$.
   - Measures change in the marginal data distribution independent of target values.
2. **Concept Drift Analysis ($P(Y \mid X)$)**:
   - Input: $Z = [X, Y]$.
   - Measures shifts in the joint distribution, pinpointing features whose relationship to the label altered.

```python
from stride.xai.importance import FeatureImportanceDriftAnalyzer

analyzer = FeatureImportanceDriftAnalyzer(X_before, y_before, X_after, y_after)

# Analyze with Permutation Feature Importance
res = analyzer.compute_drift_importance(
    importance_method="permutation",
    include_target=True,
)

print("Drift Importance Means:", res["importance_mean"])
print("Discriminator Accuracy:", res["accuracy"])
```

---

## Selective Discriminator & Conformal Abstention

When distributions overlap significantly, standard discriminators may overfit or make uncertain predictions. STRIDE supports **selective classification** with conformal prediction:

```python
from stride.xai.importance import AbstainStrategy

res = analyzer.compute_drift_importance(
    importance_method="permutation",
    include_target=True,
    abstain_strategy=AbstainStrategy.CONFORMAL,
    conformal_significance=0.10,
)
```
