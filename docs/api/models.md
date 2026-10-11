# Estimator Models

The `stride.models` subpackage provides Scikit-learn compatible classifier wrappers used for stream evaluation, temporal discrimination, and decision boundary analysis.

---

## Overview

All model wrappers inherit from `BaseModel`, providing unified `fit()`, `predict()`, and `predict_proba()` interfaces. These estimators are used throughout STRIDE to fit baseline and detection classifiers and to power auxiliary temporal discriminators.

```python
from stride.models import BaseModel, MLPModel, RandomForestModel, MODELS
```

### Key Components

| Class / Registry | Base Class | Description |
| :--- | :--- | :--- |
| `BaseModel` | `ABC` | Abstract base class establishing the estimator contract (`fit`, `predict`, `predict_proba`, `get_params`). |
| `MLPModel` | `BaseModel` | Multi-Layer Perceptron neural network wrapper with configurable hidden layers and activations. |
| `RandomForestModel` | `BaseModel` | Random Forest ensemble classifier wrapper with configurable estimator counts and depths. |
| `MODELS` | `dict` | Global dictionary mapping model aliases (`"mlp"`, `"random_forest"`) to wrapper classes. |

---

## API Specification

::: stride.models
