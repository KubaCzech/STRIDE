import numpy as np
import itertools
import pandas as pd


def generate_river_data(river_stream, total_samples, n_features=2):
    # TODO: return feature names
    """
    Helper function to generate data from a river stream.

    Parameters
    ----------
    river_stream : river stream object
        The stream generator
    total_samples : int
        Total number of samples to generate
    n_features : int, default=2
        Number of features to extract

    Returns
    -------
    tuple
        (X, y)
    """
    X_all = []
    y_all = []

    try:
        for x, y_val in itertools.islice(river_stream, total_samples):
            # Extract the specified number of features
            features = [x[i] for i in range(n_features)]
            X_all.append(features)
            y_all.append(y_val)

    except (StopIteration, KeyError) as e:
        print(f"Warning: Stream generation failed: {e}")
        return np.array([]), np.array([])

    X = pd.DataFrame(X_all, columns=[f"X{i + 1}" for i in range(n_features)])
    y = pd.Series(y_all, name="Y")

    return X, y


def generate_river_data_with_selection(river_stream, total_samples, feature_names):
    """
    Helper function to generate data from a river stream with selected features.

    Parameters
    ----------
    river_stream : river stream object
        The stream generator
    total_samples : int
        Total number of samples to generate
    feature_names : list of str
        List of feature names to extract

    Returns
    -------
    tuple
        (X, y)
    """
    X_all = []
    y_all = []

    try:
        for x, y_val in itertools.islice(river_stream, total_samples):
            # Extract the specified features by name
            features = [x[name] for name in feature_names if name in x]
            # Check if all features were found
            if len(features) == len(feature_names):
                X_all.append(features)
                y_all.append(y_val)
            else:
                # Handle missing features if necessary, for now skip or warn
                # Printing warning might be too verbose for large streams
                pass

    except (StopIteration, KeyError) as e:
        print(f"Warning: Stream generation failed: {e}")
        return pd.DataFrame(), pd.Series()

    X = pd.DataFrame(X_all, columns=feature_names)
    y = pd.Series(y_all, name="Y")

    return X, y


def apply_sigmoid_drift(arr1, arr2, drift_point, drift_width):
    """
    Apply probabilistic sigmoid drift mixing between two arrays (streams).

    Parameters
    ----------
    arr1 : np.ndarray
        Array representing the first concept (Pre-drift).
    arr2 : np.ndarray
        Array representing the second concept (Post-drift).
    drift_point : int
        The sample index where the drift is centered.
    drift_width : int
        The width of the drift window.

    Returns
    -------
    np.ndarray
        Mixed array where elements are chosen from arr1 or arr2 based on sigmoid probability.
    """
    if arr1.shape != arr2.shape:
        raise ValueError(f"Arrays must have the same shape. Got {arr1.shape} and {arr2.shape}")

    total_samples = arr1.shape[0]
    indices = np.arange(total_samples)
    w = max(1, drift_width)

    if drift_width < 2:
        # Strict sudden drift
        mask = indices >= drift_point
    else:
        # Sigmoid function centered at drift_point
        # v = -4 * (x - p) / w
        v = -4.0 * (indices - drift_point) / w

        # Clip v to avoid overflow in exp
        v = np.clip(v, -500, 500)

        p_concept_2 = 1.0 / (1.0 + np.exp(v))

        random_probs = np.random.random(total_samples)
        mask = random_probs < p_concept_2

    # Handle broadcasting for multidimensional arrays (e.g., X)
    if arr1.ndim > 1:
        # Reshape mask to (N, 1, 1, ...) to match arr1's dimensions
        shape = [1] * arr1.ndim
        shape[0] = total_samples
        mask = mask.reshape(shape)

    return np.where(mask, arr2, arr1)
