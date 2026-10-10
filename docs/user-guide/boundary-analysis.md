# Decision Boundary Analysis

Concept drift frequently manifests as spatial deformations in the decision boundary separating classes. STRIDE projects high-dimensional data streams into an interpretable 2D landscape and characterizes boundary shifts.

---

## Self-Supervised Neighbor Projection (SSNP)

STRIDE incorporates **SSNP**, a bidirectional neural autoencoder mapping data from high-dimensional space $\mathbb{R}^D$ to a 2D projection $\mathbb{R}^2$ while preserving local neighborhood structures:

```python
from stride.xai.boundary import DecisionBoundaryDriftAnalyzer
from stride.models import MLPModel

analyzer = DecisionBoundaryDriftAnalyzer(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    random_state=42,
)

results = analyzer.analyze(
    model_class=MLPModel,
    grid_size=300,
    ssnp_epochs=10,
)

print("Pre-drift Accuracy:", results["accuracy_before"])
print("Post-drift Accuracy:", results["accuracy_after"])
```

---

## Disagreement Analysis

To explain *where* in feature space the decision logic diverged, STRIDE samples a synthetic grid over the 2D latent plane, evaluates pre-drift model $f_{pre}(x)$ and post-drift model $f_{post}(x)$, and fits a **Disagreement Decision Tree**:

$$D(x) = \mathbb{I}(f_{pre}(x) \neq f_{post}(x))$$

The rules extracted from the disagreement tree provide human-interpretable bounding boxes defining the drift boundary.
