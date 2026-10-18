# xAI: Statistical Tests & Divergences

The `stride.xai.stats` subpackage provides statistical two-sample hypothesis tests, summary moment displacement detectors, and distribution divergence metrics.

---

## Overview

Statistical testing characterizes univariate and multivariate covariate shifts $P(X)$ between sliding windows without requiring classifier predictions or ground-truth class labels.

```python
from stride.xai.stats import (
    StatisticalTestsDriftDetector,
    StatisticalTestType,
    DescriptiveStatisticsDriftDetector,
    StatisticsType,
)
```

### Key Components

| Class / Enum | Description |
| :--- | :--- |
| `StatisticalTestsDriftDetector` | Evaluates two-sample hypothesis tests across continuous features between reference and detection windows. |
| `StatisticalTestType` | Enumeration of supported statistical tests (`KolmogorovSmirnov`, `AndersonDarling`, `Wasserstein`). |
| `DescriptiveStatisticsDriftDetector` | Detects class-conditional relative shifts of summary statistics (mean, std, min, max, median) and of the class imbalance ratio. |
| `StatisticsType` | Enumeration of summary statistics evaluated (`Mean`, `StandardDeviation`, `Min`, `Max`, `Median`, `ImbalanceRatio`, `All`). |

---

## API Specification

::: stride.xai.stats
