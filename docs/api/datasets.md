# Datasets & Stream Generators

The `stride.datasets` subpackage provides streaming data generators, semi-synthetic stream blending tools (Shaker protocol), and external benchmark loaders.

---

## Overview

All generators inherit from `BaseDataset` and implement a unified `generate()` interface returning pandas `DataFrame` feature matrices $X$ and target series $y$.

```python
from stride.datasets import (
    HyperplaneDriftDataset,
    SeaDriftDataset,
    RbfDriftDataset,
    LinearWeightInversionDriftDataset,
    SemiSyntheticDriftDataset,
)
```

### Key Components

| Class / Function | Description |
| :--- | :--- |
| `BaseDataset` | Abstract base class defining the generator contract and windowing parameters. |
| `HyperplaneDriftDataset` | Continuous rotating hyperplane in $d$-dimensional feature space simulating gradual drift. |
| `SeaDriftDataset` | Abrupt threshold displacement benchmark with noise features. |
| `RbfDriftDataset` | Moving Gaussian radial basis centroids simulating non-linear cluster displacement. |
| `LinearWeightInversionDriftDataset` | Feature correlation inversion benchmark testing model attribution sensitivity. |
| `SemiSyntheticDriftDataset` | Empirical stream splicing under the Shaker protocol (*Shaker & Hüllermeier, 2015*). |
| `FeatureMatcher` | Statistical feature alignment via Hungarian Wasserstein mapping or PCA subspace. |
| `DriftBlender` | Deterministic stream blending with abrupt, gradual, incremental, or harmonic schedules. |
| `RiverDataset` | Adapter wrapping River streaming benchmark datasets. |
| `CsvDataset` | Offline loader for tabular CSV streaming datasets with sliding window slicing. |

---

## API Specification

::: stride.datasets
