# Data Layer & Statistical Tests

The data layer in STRIDE characterizes covariate shifts in the marginal feature distribution $P(X)$ across sliding stream windows without relying on classifier predictions or ground-truth class labels.

---

## Overview

In streaming environments, input features frequently drift due to external environmental changes, sensor degradation, or seasonal behavior even before labels become available. Monitoring the data layer provides an early warning mechanism independent of classifier accuracy.

STRIDE provides two complementary detector classes in `stride.xai.stats`:

1. **`StatisticalTestsDriftDetector`**: Non-parametric two-sample hypothesis tests that evaluate whether feature values in a detection window are drawn from the same continuous distribution as a reference window.
2. **`DescriptiveStatisticsDriftDetector`**: Computes class-conditional relative shifts of summary statistics (mean, standard deviation, minimum, maximum, median) and of the class imbalance ratio to quantify the magnitude of distributional displacement.

---

## Two-Sample Hypothesis Testing

The `StatisticalTestsDriftDetector` applies univariate hypothesis tests across all continuous features between a reference window $W_{\text{ref}} = (X_{\text{ref}}, y_{\text{ref}})$ and a detection window $W_{\text{det}} = (X_{\text{det}}, y_{\text{det}})$.

```python
from stride.datasets import HyperplaneDriftDataset
from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType

# 1. Generate streaming data with rotating hyperplane drift
dataset = HyperplaneDriftDataset()
X, y = dataset.generate(
    n_samples_before=1000,
    n_samples_after=1000,
    n_features=5,
    n_drift_features=2,
    drift_width=50,
    random_seed=42,
)

X_ref, y_ref = X.iloc[:1000], y.iloc[:1000]
X_det, y_det = X.iloc[1000:], y.iloc[1000:]

# 2. Initialize detector with significance level alpha = 0.05
detector = StatisticalTestsDriftDetector(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    alpha=0.05,
)

# 3. Execute the Kolmogorov-Smirnov test for every feature within every class
drift_flag = detector.detect(StatisticalTestType.KolmogorovSmirnov)

print(f"Overall drift detected: {drift_flag}")
print("Per-class, per-feature test statistics:")
ks_results = detector.drift_details[StatisticalTestType.KolmogorovSmirnov.value]
for label, features in ks_results.items():
    for feat, result in features.items():
        print(
            f"  class {label} | {feat:4s} -> D = {result['stat']:.4f}, p = {result['p_value']:.4e}, drifting = {result['drift']}"
        )
```

### Supported Statistical Tests

`StatisticalTestType` supports three distinct test criteria:

#### 1. Kolmogorov-Smirnov Test (`StatisticalTestType.KolmogorovSmirnov`)
Quantifies the supremum distance between the empirical cumulative distribution functions (ECDF) of the reference and detection windows:

$$D_{KS} = \sup_{x} \left| F_{\text{ref}}(x) - F_{\text{det}}(x) \right|$$

* **Null Hypothesis ($H_0$)**: Both samples are drawn from the same continuous distribution.
* **Interpretation**: Drift is flagged when the calculated $p$-value satisfies $p < \alpha$. KS is distribution-free and non-parametric, making it widely applicable across arbitrary continuous streams.

#### 2. Anderson-Darling Test (`StatisticalTestType.AndersonDarling`)
A two-sample variant that places quadratic weight on the distribution tails:

$$A^2 = \frac{n m}{N} \int_{-\infty}^{+\infty} \frac{\left( F_{\text{ref}}(x) - F_{\text{det}}(x) \right)^2}{H_N(x)(1 - H_N(x))} \, dH_N(x)$$

where $H_N(x)$ is the pooled empirical distribution function.

* **Advantage**: Detects subtle boundary shifts and outlier density accumulations that Kolmogorov-Smirnov tests may overlook when differences occur predominantly in extreme values.

#### 3. Wasserstein Distance (`StatisticalTestType.Wasserstein`)
Computes the first Wasserstein metric ($\mathcal{W}_1$), also known as the Earth Mover's Distance:

$$\mathcal{W}_1(u, v) = \int_{-\infty}^{+\infty} \left| U(x) - V(x) \right| \, dx$$

* **Advantage**: Unlike $p$-value thresholding, $\mathcal{W}_1$ is metric-scaled, providing an interpretable physical distance indicating the minimal work required to transform the reference distribution into the detection distribution.

---

## Descriptive Statistics Shift

The `DescriptiveStatisticsDriftDetector` computes, within every class, the relative shift of summary statistics of each feature:

```python
from stride.xai.stats import DescriptiveStatisticsDriftDetector, StatisticsType

detector = DescriptiveStatisticsDriftDetector(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    decision_thr=0.4,  # share of statistics that must signal drift to raise the overall alarm
)

# Evaluate all statistics; a statistic signals drift if it shifts by more than 25%
drift_flag, details = detector.detect(StatisticsType.All, thr=0.25)

print(f"Descriptive drift detected: {drift_flag}")
drifting = sorted({feature for features in details.values() for feature, stats in features.items() if any(stats.values())})
print(f"Drifting features: {drifting}")
```

### Evaluation Metric

For each class, each feature $j$ and evaluated statistic $S \in \{\text{mean}, \text{std}, \text{min}, \text{max}, \text{median}\}$:

$$\Delta_{\text{rel}}(S_j) = \frac{\left| S_j(W_{\text{det}}) - S_j(W_{\text{ref}}) \right|}{\left| S_j(W_{\text{ref}}) \right| + \epsilon}$$

If $\Delta_{\text{rel}}(S_j) > \tau$ (where $\tau$ is the `thr` argument of `detect`), feature $j$ is flagged as drifting for that statistic. The overall alarm is raised when the share of statistics that signal drift exceeds `decision_thr`.

---

## Implementation Best Practices

1. **Window Size Selection**:
   - Small window sizes ($N < 100$) reduce statistical power and increase false positive rates under natural streaming variance.
   - For continuous feature streams, window sizes between $250$ and $1000$ samples balance detector responsiveness with test sensitivity.
2. **Multiple Hypothesis Testing**:
   - When monitoring high-dimensional data streams ($d \gg 10$), individual false-positive rates accumulate under standard significance levels ($\alpha = 0.05$).
   - STRIDE surfaces individual feature $p$-values, allowing downstream pipelines to apply Bonferroni ($\alpha / d$) or Benjamini-Hochberg False Discovery Rate (FDR) adjustments when strict family-wise error control is required.
3. **Integration with Downstream xAI**:
   - Features flagged as drifting at the data layer should be compared against the temporal discriminator attributions computed by `FeatureImportanceDriftAnalyzer` to distinguish harmless covariate shifts from performance-degrading concept drift.
