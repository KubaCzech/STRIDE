# xAI and Data Analysis Tools for Drift Detection, Characterization, and Explanation

[![CI](https://github.com/KubaCzech/DriftDetectionWithExplainableAI/actions/workflows/ci.yml/badge.svg)](https://github.com/KubaCzech/DriftDetectionWithExplainableAI/actions/workflows/ci.yml)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code and notebooks for the Bachelor Thesis by **Deniz Aksoy, Kuba Czech, Wojciech Nagórka, and Michał Redmer**.

---

## 📖 About The Project

Modern machine learning systems often face performance degradation due to changes in the underlying data distribution over time, a phenomenon known as **concept drift**. While numerous methods exist to detect such drift, few provide meaningful explanations for *why* it occurs.

This project aims to develop a novel framework that not only **detects** but also **explains** concept drift using **Explainable Artificial Intelligence (xAI)** techniques. The proposed approach integrates statistical drift detection methods with model-agnostic explainability tools and prototype-based analysis.

The outcome is a comprehensive Python-based toolset and an interactive dashboard that generates interpretable explanations for drift phenomena, providing data scientists and machine learning practitioners with actionable insights and improved trust in model monitoring processes.

---

## ✨ Key Features

The framework consists of several key modules designed to handle different aspects of drift analysis:

### 1. Data Generation & Simulation
Generate synthetic datasets with known ground truth to benchmark detection and explanation methods.
*   **SEA Drift**: Simulates abrupt concept drift where the decision threshold on sum of two relevant features changes, while a third feature remains irrelevant noise.
*   **Hyperplane Drift**: Simulates gradual concept drift via a continuously rotating decision hyperplane in $d$-dimensional space, creating a dynamic environment where feature influence shifts over time.
*   **RBF Drift**: Simulates non-linear drift by moving the centroids of class-specific clusters, effectively swapping class regions in the feature space while maintaining overall data density.
*   **Linear Weight Inversion (LWI) Drift**: A specialized scenario where the correlation of specific features with the target class is inverted (weight sign flip), testing the model's ability to detect changes in feature attribution.

### 2. Drift Detection
Implements statistical methods to monitor data streams and trigger alerts when significant distribution changes are detected.

### 3. Explainable AI (xAI) Modules
Once drift is detected, these modules help characterize usage:
*   **Decision Boundary Analysis**: Visualizes how the model's decision boundary shifts between two time windows (reference vs. detection).
*   **Feature Importance Analysis**: Tracks changes in feature relevance (e.g., using SHAP or Permutation Importance) to identify which features are driving the drift.
*   **Clustering Analysis**: Uses clustering algorithms to visualize and quantify changes in the data structure and density in the feature space.
*   **Recurring Concept Analysis**: Utilizes a window-based approach to store and compare concepts over time. It employs **prototype-based explanations** to characterize each window and uses **HDBSCAN clustering** on the window distance matrix to identify and group recurring concepts, distinguishing them from novel anomalies.

---

## 📊 Interactive Dashboard

We provide a **Streamlit-based Dashboard** to visualize data streams, generate synthetic drift scenarios, and interactively apply the explanation methods.

**Key Capabilities:**
*   **Real-time Visualization**: Watch the data stream evolve and see drift points dynamically.
*   **Synthetic Data Generator**: Configure and generate datasets (SEA, Hyperplane, etc.) directly from the UI.
*   **Drift Analysis Tabs**: Switch between different analysis views (Decision Boundary, Feature Importance, Clustering) for the same data stream.
*   **Interactive Controls**: Adjust window sizes, drift parameters, and model settings on the fly.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Python 3.10-3.12
*   pip

### Installation

1.  **Clone the repository**
    ```sh
    git clone https://github.com/KubaCzech/STRIDE.git
    cd STRIDE
    ```

2.  **Create a virtual environment (Optional but Recommended)**
    ```sh
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

---

## 💻 Usage

### Running the Dashboard
The primary interface for this project is the Streamlit dashboard.

```sh
streamlit run dashboard/app.py
```

Once running, navigate to the URL provided in the terminal (usually `http://localhost:8501`).

### Jupyter Notebooks
For research experiments and step-by-step implementation details, explore the `notebooks/` directory:
*   `notebooks/data_drift.ipynb`: Experiments with data generation, visualization, and basic drift detection.

---

## 📂 Project Structure

A brief overview of the key directories:

*   `src/`: Core library code.
    *   `clustering/`: Clustering algorithms and visualization.
    *   `datasets/`: Data generators (SEA, Hyperplane, etc.).
    *   `DDM/`: Drift Detection Methods.
    *   `decision_boundary/`: Tools for analyzing decision boundaries.
    *   `descriptive_statistics/`: Statistical analysis tools.
    *   `feature_importance/`: SHAP and other importance metrics.
    *   `models/`: Wrapper classes for ML models.
    *   `recurrence/`: Recurring concept analysis.
*   `dashboard/`: Streamlit application code.
    *   `app.py`: Main entry point.
    *   `components/`: UI components (sidebar, tabs, charts).
    *   `assets/`: CSS and static assets.
*   `notebooks/`: Experimental notebooks.
*   `tests/`: Unit tests (if applicable).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👥 Authors

*   **Deniz Aksoy**
*   **Kuba Czech**
*   **Wojciech Nagórka**
*   **Michał Redmer**
