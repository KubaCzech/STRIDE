# Quickstart Tutorial

This 5-minute tutorial walks through generating a synthetic concept drift stream, training an online classifier, detecting the drift point, and explaining why and where the drift occurred.

---

## 1. Generate a Concept Drift Data Stream

Let us generate a stream from the classic **SEA Concepts** benchmark with a drift transition between variant 0 and variant 3:

```python
from stride.datasets import SeaDriftDataset

# Initialize SEA generator
dataset = SeaDriftDataset()
X, y = dataset.generate(
    n_samples_before=1000,
    n_samples_after=1000,
    random_seed=42,
    drift_width=50,  # Gradual transition over 50 samples
)

print("Stream shape:", X.shape)
print("Features:", list(X.columns))
```

---

## 2. Detect Drift in the Error Stream

We use `BinaryErrorDriftDescriptor` to evaluate stream performance sequentially and pinpoint the drift's onset:

```python
from stride.drift import BinaryErrorDriftDescriptor
from river import drift

descriptor = BinaryErrorDriftDescriptor(
    warning_grace_period=3,
    lookback_method="cusum",
    lookforward_method="peak",
    ddm=drift.binary.DDM(),
)

# Simulate error stream where error rises near sample 1000
for i, (idx, row) in enumerate(X.iterrows()):
    error = 1 if (i >= 1000 and (i % 3 != 0)) else (1 if (i % 10 == 0) else 0)
    descriptor.update(error)
    if descriptor.drift_detected:
        desc = descriptor.describe_drift()
        print(f"Drift detected at sample {desc.detected_at}!")
        print(f"Estimated onset: sample {desc.drift_start_index}")
        print(f"Peak error rate: {desc.error_rate_at_peak:.2%}")
        break
```

---

## 3. Explain Drift with Feature Importance

To understand which features drove the drift, use `FeatureImportanceDriftAnalyzer`:

```python
from stride.xai import FeatureImportanceDriftAnalyzer

X_before = X.iloc[:1000]
y_before = y.iloc[:1000]
X_after = X.iloc[1000:2000]
y_after = y.iloc[1000:2000]

analyzer = FeatureImportanceDriftAnalyzer(X_before, y_before, X_after, y_after)

# Analyze Covariate Shift (P(X)) without label concatenation
p_x_results = analyzer.compute_drift_importance(
    importance_method="permutation",
    include_target=False,
)
print("Data Drift Importance (P(X)):", p_x_results["importance_mean"])

# Analyze Real Concept Drift (P(Y|X)) by including target labels
p_yx_results = analyzer.compute_drift_importance(
    importance_method="permutation",
    include_target=True,
)
print("Concept Drift Importance (P(Y|X)):", p_yx_results["importance_mean"])
```

---

## 4. Launch the Interactive Dashboard

To visually explore the entire stream, run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```
