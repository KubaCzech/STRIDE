# Model Layer & Sequential Drift Monitoring

The model layer evaluates classifier performance over time by monitoring sequential prediction error streams and reconstructing the onset, duration, and magnitude of detected distribution shifts.

---

## Overview

In production streaming systems, raw statistical differences in input features do not always harm downstream task performance. A model may remain accurate if shifts occur along dimensions orthogonal to the true decision boundary. Conversely, small perturbations directly on the decision boundary can cause rapid classification failure.

The model layer operates directly on the binary error sequence:

$$e_t = \mathbb{I}\left( \hat{y}_t \neq y_t \right) \in \{0, 1\}$$

where $\hat{y}_t$ is the model's prediction at step $t$ and $y_t$ is the true target label.

---

## Drift Detection Method (DDM) Principles

STRIDE integrates with the [River](https://riverml.xyz/) stream learning ecosystem, leveraging the **Drift Detection Method (DDM)** (*Gama et al., 2004*) and related sequential algorithms.

DDM models the error rate $p_t$ as a Bernoulli process. For $t$ sequential predictions, the estimated error rate and standard deviation are:

$$p_t = \frac{1}{t} \sum_{i=1}^t e_i, \qquad s_t = \sqrt{\frac{p_t (1 - p_t)}{t}}$$

The detector tracks the minimum observed error parameters $p_{\min}$ and $s_{\min}$ over the current stationary regime:

1. **Warning Level**: Triggered when the current error rate satisfies:
   $$p_t + s_t \ge p_{\min} + 2 s_{\min}$$
   At this point, a new candidate model may begin pre-training in the background on incoming samples.
2. **Drift Level**: Triggered when the error rate breaches:
   $$p_t + s_t \ge p_{\min} + 3 s_{\min}$$
   At this point, a permanent concept shift is declared, the old model is replaced, and the detector statistics are reset.

---

## Binary Error Drift Descriptor

Standard drift detectors only fire a binary alarm at the timestamp of statistical significance ($t_{\text{detected}}$). By the time an alert triggers, the drift process may have been active for hundreds of time steps.

`BinaryErrorDriftDescriptor` wraps River detectors to reconstruct complete drift event profiles:

```python
import numpy as np
from stride.drift import BinaryErrorDriftDescriptor
from river import drift

# Initialize descriptor wrapping River's DDM
descriptor = BinaryErrorDriftDescriptor(
    warning_grace_period=5,
    rate_calculation_sample_size=100,
    ddm=drift.binary.DDM(),
    lookback_method="cusum",
    lookforward_method="peak",
    degradation_only=True,
)

# Simulated streaming loop:
# baseline error rate 5%, transitioning to 35% error rate at sample 500
rng = np.random.default_rng(42)
for step in range(1, 1000):
    true_error_prob = 0.05 if step < 500 else 0.35
    error = int(rng.random() < true_error_prob)

    descriptor.update(error)

    if descriptor.drift_detected:
        description = descriptor.describe_drift()
        print(f"Drift alert triggered at step: {description.detected_at}")
        print(f"Estimated onset sample:        {description.drift_start_index}")
        print(f"Estimated duration:            {description.drift_duration} steps")
        print(f"Error rate around onset:       {description.error_rate_at_warning:.2%}")
        print(f"Error rate at detection:       {description.error_rate_at_detection:.2%}")
        break

# The peak error rate needs samples after the alarm: once the stream has advanced,
# descriptor.post_process_drift_ends([description]) locates the peak (lookforward_method)
# and updates drift_end_index and error_rate_at_peak.
```

---

## Onset Localization Algorithms

When an alert fires, `BinaryErrorDriftDescriptor` executes one of three lookback algorithms to locate the true transition start index ($t_{\text{start}} \le t_{\text{detected}}$):

### 1. Cumulative Sum (`lookback_method="cusum"`)
Computes the cumulative sum of negative deviations from the baseline mean error rate:

$$S_k = \sum_{i=1}^k (e_i - \bar{e}_{\text{baseline}})$$

The onset sample corresponds to the minimum value of $S_k$ prior to the persistent upward trend. This method is resilient to stochastic noise spikes in the stream.

### 2. Threshold Lookback (`lookback_method="threshold"`)
Traces backwards from $t_{\text{detected}}$ until the rolling error rate falls below the warning threshold. This provides a conservative estimate suitable for abrupt concept shifts.

### 3. Gradient Acceleration (`lookback_method="gradient"`)
Evaluates the second numerical derivative of the smoothed error curve to locate the point of maximum error acceleration. This is particularly effective for gradual, sigmoidal drift transitions.

---

## Output Data Structure

The `describe_drift()` method returns a structured `DriftDescription` dataclass containing:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `detected_at` | `int` | Stream sample index where the detector fired its alarm. |
| `drift_start_index` | `int` | Reconstructed sample index where distribution divergence initiated. |
| `drift_end_index` | `int` | Sample index where error rates reached peak displacement or stabilized. |
| `duration` | `int` | Estimated span of the transition window ($t_{\text{end}} - t_{\text{start}}$). |
| `error_rate_at_start` | `float` | Estimated baseline error rate immediately prior to onset. |
| `error_rate_at_peak` | `float` | Maximum error rate observed during the transition phase. |
| `severity` | `float` | Relative elevation between peak and baseline error rates. |

This rich description provides downstream xAI modules with exact temporal bounds for partitioning reference and detection windows.

---

## Adaptive Windowing with DualADWIN

While sequential algorithms such as DDM rely on tracking stationary minimum variance baselines, **ADWIN (Adaptive Windowing)** (*Bifet & Gavaldà, 2007*) adaptively resizes its sliding observation window based on rigorous Hoeffding divergence bounds without prior knowledge of the underlying drift rate.

STRIDE implements `DualADWIN`, a two-threshold wrapper running coupled ADWIN estimators configured with distinct confidence parameters ($\delta_{\text{warn}} > \delta_{\text{drift}}$):

* **Early Warning ($\delta_{\text{warn}}$)**: Triggers sensitive warning alerts, directing the descriptor to cache error history across the preceding subwindow.
* **Confirmed Drift ($\delta_{\text{drift}}$)**: Triggers the confirmed drift alert once subwindow divergence satisfies statistical significance.

```python
from stride.drift import BinaryErrorDriftDescriptor, DualADWIN

# Initialize coupled dual-threshold ADWIN detector
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

### Algorithmic Mechanics in STRIDE

1. **Dynamic Lookback Horizon**: When coupled with `DualADWIN`, `BinaryErrorDriftDescriptor` dynamically retrieves the current window width (`detector.width`) to bound the lookback horizon. This eliminates arbitrary fixed window heuristics and scales the search interval to the empirical regime size.
2. **Directional Degradation Filtering (`degradation_only=True`)**: ADWIN triggers on any statistically significant shift in subwindow expectations, including scenarios where classification accuracy improves (error rate decreases). STRIDE evaluates rolling window estimations (`detector.estimation`) prior to raising alarms, suppressing false alarms caused by positive model adaptation and firing exclusively on performance deterioration.
