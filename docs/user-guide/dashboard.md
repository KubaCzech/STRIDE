# Interactive Streamlit Dashboard

STRIDE features an interactive multi-tab Streamlit dashboard designed for live exploration and demonstration.

---

## Launching the Application

Ensure the `[dashboard]` extra is installed, then run from the repository root:

```bash
streamlit run dashboard/app.py
```

The app will start on `http://localhost:8501`.

---

## Dashboard Architecture & Tabs

The dashboard organizes drift analysis into six dedicated tabs:

1. **Stream Overview & Data Layer**: View raw stream distributions, inspect class balance over time, and run statistical two-sample tests.
2. **Model Performance & Drift Detection**: Real-time error rate curves, DDM warning and drift threshold indicators, and CUSUM change-point overlays.
3. **Decision Boundary Shift**: Interactive 2D SSNP projections comparing decision surfaces pre- and post-drift with disagreement heatmaps.
4. **Feature Importance Shift**: Bar charts comparing PFI and SHAP values for data drift ($P(X)$) versus real concept drift ($P(Y \mid X)$).
5. **Clustering Dynamics**: Centroid migration vectors, cluster assignment changes, and displacement statistics.
6. **Recurring Concepts**: Prototype comparisons across windows and recurrence distance heatmaps.
