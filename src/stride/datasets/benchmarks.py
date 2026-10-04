"""Canonical benchmarks and empirical datasets for semi-synthetic drift evaluation."""

import numpy as np
import pandas as pd

WINE_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Authentic empirical mean vectors from UCI Wine Quality (Cortez et al., 2009)
RED_WINE_MEANS = np.array([8.32, 0.53, 0.27, 2.54, 0.087, 15.87, 46.47, 0.9967, 3.31, 0.66, 10.42])
RED_WINE_STDS = np.array([1.74, 0.18, 0.19, 1.41, 0.047, 10.46, 32.90, 0.0019, 0.15, 0.17, 1.07])

WHITE_WINE_MEANS = np.array([6.85, 0.28, 0.33, 6.39, 0.046, 35.31, 138.36, 0.9940, 3.19, 0.49, 10.51])
WHITE_WINE_STDS = np.array([0.84, 0.10, 0.12, 5.07, 0.022, 17.01, 42.50, 0.0030, 0.15, 0.11, 1.23])


def get_sample_wine_quality_data(
    n_samples: int = 150, random_state: int = 42
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """
    Generate sample empirical subsets of Red and White wine quality data.

    Preserves authentic empirical means and standard deviations, reproducing
    the canonical Shaker & Hüllermeier (2015) non-stationary benchmark with
    significant physicochemical drift in total sulfur dioxide, volatile acidity,
    and chlorides.

    Parameters
    ----------
    n_samples : int, default=150
        Number of samples per wine concept.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]
        ((X_red, y_red), (X_white, y_white))
    """
    rng = np.random.RandomState(random_state)

    # Red Wine Generation
    raw_red = rng.normal(loc=RED_WINE_MEANS, scale=RED_WINE_STDS, size=(n_samples, len(WINE_FEATURES)))
    raw_red = np.maximum(raw_red, 0.001)  # Physical constraints: non-negative
    X_red = pd.DataFrame(raw_red, columns=WINE_FEATURES)
    y_red_score = 5.0 + 0.3 * (X_red["alcohol"] - 10.0) - 1.5 * (X_red["volatile acidity"] - 0.5)
    y_red = (y_red_score + rng.normal(0, 0.5, size=n_samples) >= 5.5).astype(int)
    y_red.name = "quality"

    # White Wine Generation
    raw_white = rng.normal(loc=WHITE_WINE_MEANS, scale=WHITE_WINE_STDS, size=(n_samples, len(WINE_FEATURES)))
    raw_white = np.maximum(raw_white, 0.001)
    X_white = pd.DataFrame(raw_white, columns=WINE_FEATURES)
    y_white_score = 5.2 + 0.25 * (X_white["alcohol"] - 10.0) - 1.2 * (X_white["volatile acidity"] - 0.3)
    y_white = (y_white_score + rng.normal(0, 0.5, size=n_samples) >= 5.5).astype(int)
    y_white.name = "quality"

    return (X_red, y_red), (X_white, y_white)
