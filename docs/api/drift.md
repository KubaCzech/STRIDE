# Drift Detection & Descriptors

The `stride.drift` subpackage provides sequential drift monitoring algorithms, error stream tracking, and post-detection characterization descriptors.

---

## Overview

Streaming drift detection in STRIDE wraps River sequential detectors to monitor binary classification error streams $e_t \in \{0, 1\}$. When an alarm is triggered, the descriptor reconstructs the transition onset, duration, and error trajectory.

```python
from stride.drift import BinaryErrorDriftDescriptor, DriftDescription
```

### Key Components

| Class / Structure | Description |
| :--- | :--- |
| `BinaryErrorDriftDescriptor` | Wraps River drift detectors (e.g. DDM, EDDM) and applies CUSUM, threshold, or gradient lookback to estimate drift onset and duration. |
| `DriftDescription` | Dataclass storing reconstructed drift event parameters (`detected_at`, `drift_start_index`, `drift_end_index`, `duration`, `error_rate_at_start`, `error_rate_at_peak`, `severity`). |

---

## API Specification

::: stride.drift
