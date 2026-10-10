# Model Layer & Sequential Drift Monitoring

The model layer monitors the streaming performance of classification pipelines using error-based sequential detectors.

---

## Binary Error Drift Descriptor

The `BinaryErrorDriftDescriptor` wraps river detectors (such as DDM, EDDM, and ADWIN) and estimates drift characteristics (duration, onset point, peak error rate):

```python
from stride.drift import BinaryErrorDriftDescriptor
from river import drift

descriptor = BinaryErrorDriftDescriptor(
    warning_grace_period=3,
    rate_calculation_sample_size=100,
    ddm=drift.binary.DDM(),
    lookback_method="cusum",
    lookforward_method="peak",
    degradation_only=True,
)

# In streaming loop:
for y_true, y_pred in stream_evaluation:
    error = 1 if y_true != y_pred else 0
    descriptor.update(error)
    if descriptor.drift_detected:
        description = descriptor.describe_drift()
        print(f"Drift started at sample {description.drift_start_index}")
```

### Onset Localization Algorithms
1. **CUSUM (`lookback_method="cusum"`)**: Computes cumulative sums of deviations to pinpoint the earliest sample where error rates began to diverge upwards.
2. **Threshold (`lookback_method="threshold"`)**: Traces backwards to the sample where the error rate was strictly below the baseline warning threshold.
3. **Gradient (`lookback_method="gradient"`)**: Identifies the steepest acceleration point of error accumulation.

---

## Adaptive Windowing with DualADWIN

Standard detectors like DDM assume a static distribution model and rely on fixed warning levels. In contrast, **ADWIN (Adaptive Windowing)** automatically adjusts its window length $W$ based on statistical significance without needing prior knowledge of drift rate.

STRIDE provides `DualADWIN`, a two-threshold wrapper that maintains two internal ADWIN estimators with distinct confidence bounds ($\delta_{\text{warn}} > \delta_{\text{drift}}$):

- **Early Warning ($\delta_{\text{warn}}$)**: Triggers early warning signals, prompting the descriptor to cache the error stream for lookback analysis.
- **Confirmed Drift ($\delta_{\text{drift}}$)**: Triggers the confirmed drift alert when subwindow divergence is statistically proven.

```python
from stride.drift import BinaryErrorDriftDescriptor, DualADWIN

# Configure dual-threshold ADWIN
adwin_detector = DualADWIN(
    delta_warn=0.05,
    delta_drift=0.002,
    clock=32,
    max_buckets=5,
)

descriptor = BinaryErrorDriftDescriptor(
    ddm=adwin_detector,
    lookback_method="cusum",
    lookforward_method="peak",
    degradation_only=True,
)
```

### Key ADWIN Features in STRIDE

1. **Dynamic Lookback Window**: When paired with `DualADWIN`, the descriptor extracts `detector.width` (the actual adaptive subwindow size) to bound the lookback search horizon, avoiding fixed-length lookback heuristic mismatches.
2. **Directional Degradation Filter (`degradation_only=True`)**: Because ADWIN naturally detects any significant distribution shift (including performance *improvements* where error rates fall), STRIDE applies directional degradation filtering to ensure drift alarms only fire when error rates deteriorate.
