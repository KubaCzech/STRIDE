# Data Layer & Statistical Tests

The data layer in STRIDE characterizes feature distribution shifts across stream blocks without relying on classifier predictions.

---

## Statistical Two-Sample Tests

The `StatisticalTestsDriftDetector` evaluates univariate feature shifts between a baseline reference window and a current detection window:

```python
from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType

detector = StatisticalTestsDriftDetector(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    significance_level=0.05,
)

# Run Kolmogorov-Smirnov test across all features
results_ks = detector.detect(StatisticalTestType.KolmogorovSmirnov)
print("KS-test Drift Flag:", detector.drift_flag)
print("P-values per feature:", results_ks["p_values"])
```

### Supported Tests
- **Kolmogorov-Smirnov Test (`KolmogorovSmirnov`)**: Quantifies the supremum distance between empirical cumulative distribution functions:
  $$D_{KS} = \sup_x |F_{ref}(x) - F_{det}(x)|$$
- **Anderson-Darling Test (`AndersonDarling`)**: Sensitive to shifts in the tails of the distributions.
- **Wasserstein Distance (`Wasserstein`)**: Measures the Earth Mover's Distance between 1D continuous marginal distributions:
  $$\mathcal{W}_1(u, v) = \int_{-\infty}^{+\infty} |U(x) - V(x)| dx$$

---

## Descriptive Statistics Shift

The `DescriptiveStatisticsDriftDetector` computes absolute relative shifts for fundamental summary statistics:

```python
from stride.xai.stats import DescriptiveStatisticsDriftDetector, StatisticsType

detector = DescriptiveStatisticsDriftDetector(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    decision_thr=0.3,
)

summary = detector.detect(StatisticsType.All)
print("Significant shifts:", detector.drift_details)
```
