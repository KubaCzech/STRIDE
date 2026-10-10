# Model Layer & Sequential Drift Monitoring

The model layer monitors the streaming performance of classification pipelines using error-based sequential detectors.

---

## Binary Error Drift Descriptor

The `BinaryErrorDriftDescriptor` wraps river detectors (such as DDM and EDDM) and estimates drift characteristics (duration, onset point, peak error rate):

```python
from stride.drift import BinaryErrorDriftDescriptor
from river import drift

descriptor = BinaryErrorDriftDescriptor(
    warning_grace_period=3,
    rate_calculation_sample_size=100,
    ddm=drift.binary.DDM(),
    lookback_method="cusum",
    lookforward_method="peak",
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
