# xAI: Decision Boundary Analysis

The `stride.xai.boundary` subpackage provides tools for projecting high-dimensional feature spaces into 2D manifolds, rendering Supervised Decision Boundary Maps (SDBM), and characterizing boundary shifts using surrogate disagreement decision trees.

---

## Overview

Boundary analysis tracks geometric deformations in classifier decision frontiers between reference and detection windows. It uses Self-Supervised Neighbor Projection (SSNP) for bi-directional dimensionality reduction and samples continuous latent grids to detect regions where models disagree.

```python
from stride.xai.boundary import (
    DecisionBoundaryDriftAnalyzer,
    compute_disagreement_analysis,
)
```

### Key Components

| Class / Function | Description |
| :--- | :--- |
| `DecisionBoundaryDriftAnalyzer` | High-level orchestrator that projects windows, fits pre- and post-drift classifiers, computes grid predictions, and evaluates disagreement regions. |
| `compute_disagreement_analysis` | Identifies disagreement regions where $f_{\text{pre}}(x) \neq f_{\text{post}}(x)$ and fits surrogate decision trees to extract bounding decision rules. |
| `SSNP` | Neural autoencoder mapping $\mathbb{R}^D \leftrightarrow \mathbb{R}^2$ while preserving neighborhood distances (requires `tensorflow>=2.15`). |

---

## API Specification

::: stride.xai.boundary
