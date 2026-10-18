# Quickstart Tutorial

This guide demonstrates an end-to-end workflow in STRIDE: generating a synthetic data stream with concept drift, evaluating distribution shifts at the data layer, monitoring model error sequentially, and attributing the root causes of drift using Explainable AI (xAI).

---

## 1. Stream Generation & Window Partitioning

First, generate a synthetic stream using the rotating hyperplane generator (`HyperplaneDriftDataset`). The stream simulates gradual concept drift where feature weights continuously rotate across a specified transition window:

```python
from stride.datasets import HyperplaneDriftDataset

# Initialize synthetic stream generator
dataset = HyperplaneDriftDataset()
X, y = dataset.generate(
    n_samples_before=1000,
    n_samples_after=1000,
    n_features=5,
    n_drift_features=2,
    drift_width=100,
    random_seed=42,
)

# Partition into pre-drift (reference) and post-drift (detection) windows
X_ref, y_ref = X.iloc[:1000], y.iloc[:1000]
X_det, y_det = X.iloc[1000:], y.iloc[1000:]

print(f"Stream generated: {X.shape[0]} total samples across {X.shape[1]} features.")
print(f"Reference window: {X_ref.shape[0]} samples | Detection window: {X_det.shape[0]} samples.")
```

---

## 2. Statistical Drift Detection (Data Layer)

Before evaluating predictive models, assess whether the marginal feature distribution $P(X)$ shifted between the two windows using non-parametric two-sample hypothesis testing:

```python
from stride.xai.stats import StatisticalTestsDriftDetector, StatisticalTestType

detector = StatisticalTestsDriftDetector(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
    alpha=0.05,
)

# Execute two-sample Kolmogorov-Smirnov tests for every feature within every class
drift_flag = detector.detect(StatisticalTestType.KolmogorovSmirnov)

print(f"Data layer drift detected: {drift_flag}")
ks_results = detector.drift_details[StatisticalTestType.KolmogorovSmirnov.value]
for label, features in ks_results.items():
    for feature, result in features.items():
        print(f"  Class {label} | {feature:4s} -> p-value: {result['p_value']:.4e} | Drifting: {result['drift']}")
```

The detector reports whether the null hypothesis (identical distributions across windows) is rejected for each individual feature at the chosen significance level $\alpha = 0.05$.

---

## 3. Sequential Error Monitoring (Model Layer)

To monitor classification health over time, fit a classifier on the reference window, simulate an online prediction stream, and pipe the binary error stream into `BinaryErrorDriftDescriptor`:

```python
from stride.models import RandomForestModel
from stride.drift import BinaryErrorDriftDescriptor
from river import drift

# Fit classifier on pre-drift baseline
model = RandomForestModel(n_estimators=50, random_state=42)
model.fit(X_ref, y_ref)

# Initialize sequential detector wrapping River's DDM
descriptor = BinaryErrorDriftDescriptor(
    warning_grace_period=5,
    lookback_method="cusum",
    lookforward_method="peak",
    ddm=drift.binary.DDM(),
)

# Stream post-drift instances sequentially
for idx, (x_row, y_true) in enumerate(zip(X_det.to_dict(orient="records"), y_det)):
    y_pred = model.predict([list(x_row.values())])[0]
    error = int(y_true != y_pred)
    descriptor.update(error)

    if descriptor.drift_detected:
        desc = descriptor.describe_drift()
        print(f"Model alert triggered at sample {desc.detected_at} in detection window.")
        print(f"Estimated drift onset index: sample {desc.drift_start_index}")
        print(f"Peak error rate reached: {desc.error_rate_at_peak:.2%}")
        break
```

`BinaryErrorDriftDescriptor` goes beyond binary alert flags by applying CUSUM lookback analysis to estimate the exact time sample where the error rate diverged.

---

## 4. Feature Importance Attribution (Explanation Layer)

When drift is detected, determine which features drove the shift. STRIDE uses an auxiliary temporal discriminator that classifies whether instances belong to the reference window ($t=0$) or detection window ($t=1$):

```python
from stride.xai.importance import FeatureImportanceDriftAnalyzer

analyzer = FeatureImportanceDriftAnalyzer(
    X_before=X_ref,
    y_before=y_ref,
    X_after=X_det,
    y_after=y_det,
)

# 1. Analyze Covariate Shift (P(X)) - data drift without labels
covariate_shift = analyzer.compute_drift_importance(
    importance_method="permutation",
    include_target=False,
    random_state=42,
)

# 2. Analyze Real Concept Drift (P(Y | X)) - joint distribution with labels
concept_drift = analyzer.compute_drift_importance(
    importance_method="permutation",
    include_target=True,
    random_state=42,
)

print("\nCovariate Shift Attributions (P(X)):")
for name, score in zip(covariate_shift["feature_names"], covariate_shift["importance_mean"]):
    print(f"  {name:10s}: {score:.4f}")

print("\nConcept Drift Attributions (P(Y | X)):")
for name, score in zip(concept_drift["feature_names"], concept_drift["importance_mean"]):
    print(f"  {name:10s}: {score:.4f}")
```

By comparing attributions with and without target concatenation, analysts determine whether distribution changes stem from feature displacement alone or true modifications to the posterior decision boundary $P(Y \mid X)$.

---

## 5. Visual Exploration via Dashboard

To inspect stream behavior interactively, launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The application opens at `http://localhost:8501`, allowing visual exploration of sliding windows, 2D decision boundary projections, feature importance rankings, and cluster migration dynamics.
