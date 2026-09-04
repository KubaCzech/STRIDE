import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import pandas as pd
import numpy as np


from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

from src.recurrence.full_window_storage import FullWindowStorage


def visualize_distance_matrix(matrix: pd.DataFrame, drift_positions: list[int] = None, title: str = "Window Distance Matrix"):
    """Visualize the distance matrix as a heatmap.

    Args:
        matrix: Distance matrix DataFrame
        drift_positions: List of iterations where drift was detected
        title: Plot title
    """

    fig, ax = plt.subplots(figsize=(14, 12))

    # Create heatmap
    sns.heatmap(matrix, cmap="RdYlGn_r", square=True, linewidths=0, cbar_kws={"label": "Distance"}, ax=ax)

    # Mark drift positions if provided
    if drift_positions:
        for drift_iter in drift_positions:
            if drift_iter in matrix.index:
                idx = list(matrix.index).index(drift_iter)
                # Draw lines to mark drifts
                ax.axhline(y=idx, color="blue", linewidth=3, alpha=0.8)
                ax.axvline(x=idx, color="blue", linewidth=3, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Iteration")

    plt.tight_layout()
    plt.show()


def median_mask(arr, k=3):
    arr = np.asarray(arr)
    pad = k // 2

    # Same/edge padding
    padded = np.pad(arr, pad_width=pad, mode="edge")

    # Sliding window view
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)

    return np.median(windows, axis=1).astype(arr.dtype)


def show_distance_median(storage: FullWindowStorage, window_nr, k=3, measure="centroid_displacement"):
    data_to_plot = storage.compare_window_to_all(window_nr, measure=measure)
    data_to_plot = [float(x) for x in data_to_plot]
    data_to_plot = median_mask(data_to_plot, k)

    plt.plot(data_to_plot)
    plt.title("distance from window nr " + str(window_nr))
    plt.show()


def cluster_windows(matrix: pd.DataFrame, fix_outliers=True, median_mask_width=1):
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=3,
    )

    if median_mask_width != 1:
        matrix = pd.DataFrame(median_mask_2d(matrix, kx=median_mask_width, ky=median_mask_width))

    labels = clusterer.fit_predict(matrix.values)

    if fix_outliers:
        # If both left and right neighbours are the same, replace the middle label with their value
        # in order to denoise
        for i, label in enumerate(labels[1:-1]):
            if labels[i] == labels[i + 2] and labels[i] != -1:
                labels[i + 1] = labels[i]

    return labels


def get_drift_from_clusters(labels):
    drift_locations = []
    last_label = labels[0]
    for i, label in enumerate(labels):
        if label == -1:
            continue

        if label != last_label:
            last_label = label
            drift_locations.append(i)

    return drift_locations


def clustered_labels_accuracy(labels, true_concept, outliers_included=True):
    mask = labels != -1
    pred = labels[mask]
    true_non_out = true_concept[mask]

    # confusion matrix between predicted cluster IDs and true cluster IDs
    cm = confusion_matrix(true_non_out, pred)

    # Hungarian algorithm finds best 1-to-1 mapping
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)

    # Build mapping: pred_label → true_label
    # The clustering algorighms labels and true labels are different, but they still can be correct.
    # All that is important is taht the same windows are grouped together in both system. The label itself is not imoprtant
    # However they need to be the same during evaluation, so we are re-maping
    mapping = dict(zip(col_ind, row_ind))

    pred_mapped = np.array([mapping[p] for p in pred])
    accuracy_no_outliers = np.mean(pred_mapped == true_non_out)

    n_outliers = sum([x == -1 for x in labels])
    # assume outliers are always incorrectly classified
    accuracy_with_outliers = accuracy_no_outliers * (1 - (n_outliers / len(labels)))

    if outliers_included:
        return accuracy_with_outliers
    else:
        return accuracy_no_outliers


def median_mask_2d(arr, ky=3, kx=3):
    """
    2D median filter with SAME (edge) padding and independent kernel sizes.

    arr : 2D array-like
    ky  : kernel height (odd)
    kx  : kernel width  (odd)
    """
    arr = np.asarray(arr, dtype=np.float64)

    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("Both ky and kx must be odd.")

    py = ky // 2
    px = kx // 2

    padded = np.pad(arr, pad_width=((py, py), (px, px)), mode="edge")

    windows = np.lib.stride_tricks.sliding_window_view(padded, (ky, kx))
    return np.median(windows, axis=(2, 3))


def evaluate_threshold(concepts_same, distance, threshold):
    if concepts_same:
        return distance < threshold  # predict similar
    else:
        return distance > threshold  # predict different


def threshold_test(M: pd.DataFrame, true_concept: list, median_mask_dimensions=(1, 1)):
    # Test the accuracy of separation of 'similar' and 'different' concept for various threshold values
    M = M.to_numpy()
    M = median_mask_2d(M, ky=median_mask_dimensions[0], kx=median_mask_dimensions[1])

    thresholds = [x * 0.005 for x in range(200)]

    results = []

    for threshold in thresholds:
        TP = FP = TN = FN = 0

        for i in range(len(M)):
            for j in range(i - 1):
                concepts_same = true_concept[i] == true_concept[j]
                distance = M[i][j]

                if distance is None:
                    continue

                prediction_correct = evaluate_threshold(concepts_same, distance, threshold)

                if concepts_same and prediction_correct:
                    TP += 1
                elif not concepts_same and prediction_correct:
                    TN += 1
                elif concepts_same and not prediction_correct:
                    FN += 1
                else:
                    FP += 1

        total = TP + FP + TN + FN

        accuracy = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "fnr": fnr,
                "f1": f1,
            }
        )

    return results


def plot_threshold_analysis_results(results):
    # Plot All Metrics
    thresholds = sorted([x["threshold"] for x in results])

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [r["accuracy"] for r in results], label="Accuracy")
    plt.plot(thresholds, [r["precision"] for r in results], label="Precision")
    plt.plot(thresholds, [r["recall"] for r in results], label="Recall")
    plt.plot(thresholds, [r["f1"] for r in results], label="F1 Score")
    plt.plot(thresholds, [r["fpr"] for r in results], label="FPR")
    plt.plot(thresholds, [r["fnr"] for r in results], label="FNR")

    plt.title("Threshold Performance Metrics (Median Distance)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Best Threshold by F1
    best = max(results, key=lambda x: x["f1"])
    print("Best Threshold by F1:")
    print(best)


def create_prototypes_for_stream(model, storage, dataset, verbose=True):
    """Process stream and create prototypes for each window.

    Args:
        model: The classification model (e.g., ARFClassifier)
        storage: FullWindowStorage instance to store results
        dataset: List of [x_block, y_block] tuples
        verbose: Whether to print progress
    """

    for i in range(len(dataset)):
        if verbose and i % 10 == 0:
            print(f"{i}/{len(dataset)}")

        x_block, y_block = dataset[i]

        # Learn from data
        for x, y in zip(x_block, y_block):
            model.learn_one(x, y)

        # Create fresh prototype selector with current model state
        from src.recurrence.protree.explainers import APete

        current_explainer = APete(model=model, alpha=0.01)

        # Select prototypes independently for this window only
        current_prototypes = current_explainer.select_prototypes(x_block, y_block)

        # In case a class has 0 or 1 prototypes, run prototype selection just for this class to fix this
        for class_name in set(y_block):
            class_prototypes = current_prototypes[class_name]
            if len(class_prototypes) in [0, 1, 2]:
                current_explainer = APete(model=model, alpha=0.05)  # change alpha threshold to make less prototypes
                if verbose:
                    print("Anomaly detected. Window:", i, "class:", class_name)

                # create prototypes for the class with missing prototypes
                x_block_missing_class = []
                for x, y in zip(x_block, y_block):
                    if y == class_name:
                        x_block_missing_class.append(x)

                y_block_missing_class = tuple([class_name] * len(x_block_missing_class))
                x_block_missing_class = tuple(x_block_missing_class)

                missing_prototypes = current_explainer.select_prototypes(x_block_missing_class, y_block_missing_class)
                current_prototypes[class_name] = missing_prototypes[class_name]

        # Store window data
        storage.store_window(
            iteration=i,
            x=x_block,
            y=y_block,
            prototypes=current_prototypes,
            explainer=None,
            drift=False,  # No drift detection being used
        )

    if verbose:
        print(f"\nStored {len(storage.get_all_iterations())} windows")


def prepare_dataset_from_generator(X, y, window_size):
    """Convert numpy arrays into windowed dataset format.

    Args:
        X: Feature array or DataFrame
        y: Target array or Series
        window_size: Number of samples per window

    Returns:
        List of [x_block, y_block] tuples
    """
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else X
    y_np = y.to_numpy() if hasattr(y, "to_numpy") else y

    total_samples = len(X)
    num_windows = total_samples // window_size

    dataset = []
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        if end_idx <= total_samples:
            x_block = X_np[start_idx:end_idx]
            y_block = y_np[start_idx:end_idx]

            # Convert to list of dicts for River compatibility
            feature_names = list(X.columns) if hasattr(X, "columns") else [f"f{k}" for k in range(X.shape[1])]
            x_dicts = []
            for row in x_block:
                if hasattr(row, "tolist"):
                    row = row.tolist()
                row_dict = {feature_names[k]: v for k, v in enumerate(row)}
                x_dicts.append(row_dict)

            y_list = y_block.tolist() if hasattr(y_block, "tolist") else list(y_block)

            dataset.append([tuple(x_dicts), tuple(y_list)])

    return dataset
