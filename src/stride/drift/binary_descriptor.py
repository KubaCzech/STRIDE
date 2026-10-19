"""Sequential binary error drift descriptor and change-point characterization."""

from typing import Any
import numpy as np
from river import drift as river_drift

from ._types import DriftDescription


class BinaryErrorDriftDescriptor:
    r"""Track and characterize concept drifts identified by sequential error detectors.

    Wraps River detectors (such as DDM and EDDM) to continuously monitor the binary
    error stream of an online model ($e_t \in \{0, 1\}$). When a warning or drift signal
    is triggered, it applies change-point lookback algorithms (CUSUM, threshold, gradient)
    to estimate the actual onset timestamp and lookforward methods to locate the drift peak.

    Attributes:
        warning_grace_period: Allowed consecutive non-warning steps before warning chain resets.
        rate_calculation_sample_size: Number of samples used to calculate local error rates.
        ddm: Underlying River binary drift and warning detector instance.
        lookback_method: Change-point algorithm used to locate onset ("cusum", "threshold", "gradient", "none").
        lookforward_method: Algorithm used to estimate stabilization ("peak", "recovery", "none").
        error_history: Window of recent error signals.
        complete_error_history: Full recorded error stream sequence.
        drift_detected: Flag indicating whether drift occurred at current timestep.
        last_detected_drift: Most recent `DriftDescription` object, or `None`.
        current_index: Current stream iteration index.
        degradation_only: Whether to restrict detections to error degradation rather than improvement.
    """

    def __init__(
        self,
        warning_grace_period: int = 3,
        rate_calculation_sample_size: int = 100,
        ddm: Any = river_drift.binary.DDM(),
        lookback_method: str = "cusum",
        lookforward_method: str = "peak",
        degradation_only: bool = True,
    ) -> None:
        """Initialize the binary error drift descriptor.

        Args:
            warning_grace_period: Allowed non-warning steps within a warning chain.
            rate_calculation_sample_size: Window size for moving error rate averages.
            ddm: River drift detector instance with `warning_detected` and `drift_detected`.
            lookback_method: Method for onset estimation ("cusum", "threshold", "gradient", "none").
            lookforward_method: Method for end/peak estimation ("peak", "recovery", "none").
            degradation_only: Whether to restrict detections to error degradation rather than improvement.
        """
        self.warning_grace_period = warning_grace_period
        self.rate_calculation_sample_size = rate_calculation_sample_size
        self.ddm = ddm
        self.lookback_method = lookback_method
        self.lookforward_method = lookforward_method
        self.degradation_only = degradation_only

        self.warning_grace_period_left = warning_grace_period
        self.error_history: list[int | float] = []
        self.complete_error_history: list[int | float] = []
        self.previous_was_warning = False
        self.assume_warning = False
        self.last_detected_drift: DriftDescription | None = None
        self.drift_detected = False
        self.current_index = 0

    def find_drift_start_cusum(self, detection_idx: int, lookback_window: int = 300) -> int:
        """Estimate drift onset using CUSUM change-point detection.

        Args:
            detection_idx: Stream index where drift alarm was raised.
            lookback_window: Maximum history window examined prior to detection.

        Returns:
            Estimated sample index representing drift onset.
        """
        if detection_idx < 50:
            return 0

        start_idx = max(0, detection_idx - lookback_window)
        history_segment = self.complete_error_history[start_idx:detection_idx]

        if len(history_segment) < 20:
            return start_idx

        baseline_size = min(50, len(history_segment) // 4)
        baseline_mean = np.mean(history_segment[:baseline_size])

        cusum = 0.0
        threshold = 1.5

        for i in range(baseline_size, len(history_segment)):
            cusum = max(0.0, cusum + (history_segment[i] - baseline_mean - 0.1))
            if cusum > threshold:
                return start_idx + i

        return start_idx + baseline_size

    def find_drift_start_threshold(self, detection_idx: int, lookback_window: int = 300) -> int:
        """Estimate drift onset by searching for sustained error rate elevation.

        Args:
            detection_idx: Stream index where drift alarm was raised.
            lookback_window: Maximum history window examined prior to detection.

        Returns:
            Estimated sample index representing drift onset.
        """
        if detection_idx < 50:
            return 0

        window_size = self.rate_calculation_sample_size
        start_idx = max(0, detection_idx - lookback_window)

        baseline_end = min(start_idx + window_size, detection_idx - 50)
        if baseline_end <= start_idx:
            return start_idx

        baseline = self.complete_error_history[start_idx:baseline_end]
        baseline_rate = float(np.mean(baseline))

        increase_threshold = 1.5
        sustained_count = 0
        sustained_threshold = 20

        for i in range(baseline_end, detection_idx):
            if i + window_size > detection_idx:
                window_size_local = detection_idx - i
            else:
                window_size_local = window_size

            if window_size_local < 10:
                continue

            window = self.complete_error_history[i : i + window_size_local]
            current_rate = float(np.mean(window))

            if current_rate > baseline_rate * increase_threshold:
                sustained_count += 1
                if sustained_count >= sustained_threshold:
                    return max(start_idx, i - sustained_threshold)
            else:
                sustained_count = 0

        return baseline_end

    def find_drift_start_gradient(self, detection_idx: int, lookback_window: int = 300) -> int:
        """Estimate drift onset using error rate gradient acceleration.

        Args:
            detection_idx: Stream index where drift alarm was raised.
            lookback_window: Maximum history window examined prior to detection.

        Returns:
            Estimated sample index representing drift onset.
        """
        if detection_idx < 50:
            return 0

        window_size = 20
        start_idx = max(0, detection_idx - lookback_window)

        error_rates = []
        for i in range(start_idx, detection_idx - window_size):
            window = self.complete_error_history[i : i + window_size]
            error_rates.append(float(np.mean(window)))

        if len(error_rates) < 2:
            return start_idx

        gradients = np.diff(error_rates)
        gradient_threshold = 0.01
        for i, grad in enumerate(gradients):
            if grad > gradient_threshold:
                return start_idx + i

        return start_idx

    def find_drift_end_peak(self, detection_idx: int, lookforward_window: int = 200) -> int:
        """Estimate peak error rate timestamp following drift alarm.

        Args:
            detection_idx: Stream index where drift alarm was raised.
            lookforward_window: Maximum subsequent samples evaluated.

        Returns:
            Sample index corresponding to maximum error rate window.
        """
        max_idx = len(self.complete_error_history)
        end_idx = min(detection_idx + lookforward_window, max_idx)

        if detection_idx >= max_idx:
            return detection_idx

        window_size = self.rate_calculation_sample_size
        current_idx = detection_idx

        if current_idx + window_size <= max_idx:
            current_window = self.complete_error_history[current_idx : current_idx + window_size]
            previous_error_rate = float(np.mean(current_window))
        else:
            return detection_idx

        best_idx = detection_idx
        step_size = window_size
        current_idx += step_size

        while current_idx + window_size <= end_idx:
            window = self.complete_error_history[current_idx : current_idx + window_size]
            current_error_rate = float(np.mean(window))

            if current_error_rate < previous_error_rate:
                return best_idx

            best_idx = current_idx
            previous_error_rate = current_error_rate
            current_idx += step_size

        return best_idx

    def find_drift_end_recovery(self, detection_idx: int, lookforward_window: int = 200) -> int:
        """Estimate stabilization point where error rates return toward baseline.

        Args:
            detection_idx: Stream index where drift alarm was raised.
            lookforward_window: Maximum subsequent samples evaluated.

        Returns:
            Sample index representing error recovery point.
        """
        max_idx = len(self.complete_error_history)
        end_idx = min(detection_idx + lookforward_window, max_idx)

        if detection_idx >= max_idx or end_idx - detection_idx < 20:
            return detection_idx

        window_size = min(20, end_idx - detection_idx)
        baseline_window = self.complete_error_history[detection_idx : detection_idx + window_size]
        baseline_high = float(np.mean(baseline_window))

        cusum = 0.0
        threshold = 1.5

        for i in range(detection_idx + window_size, end_idx):
            cusum = max(0.0, cusum + (baseline_high - self.complete_error_history[i] - 0.1))
            if cusum > threshold:
                return i

        return end_idx - 1

    def update(self, x: int | float) -> None:
        """Update detector state with a binary error signal ($0$ for correct, $1$ for error).

        Args:
            x: Binary prediction error indicator.
        """
        prev_estimation = getattr(self.ddm, "estimation", None)
        prev_warn_est = None
        if hasattr(self.ddm, "adwin_warn") and hasattr(self.ddm.adwin_warn, "estimation"):
            prev_warn_est = self.ddm.adwin_warn.estimation

        self.ddm.update(x)
        self.complete_error_history.append(x)
        self.drift_detected = False

        warning_detected = getattr(self.ddm, "warning_detected", False)
        if self.degradation_only and warning_detected and prev_warn_est is not None:
            if hasattr(self.ddm, "adwin_warn") and self.ddm.adwin_warn.estimation < prev_warn_est:
                warning_detected = False

        if warning_detected:
            self.warning_grace_period_left = self.warning_grace_period
        elif self.previous_was_warning is True:
            self.warning_grace_period_left -= 1

        if self.previous_was_warning and self.warning_grace_period_left > 0:
            self.assume_warning = True
        elif warning_detected:
            self.assume_warning = True
        else:
            self.assume_warning = False

        self.error_history.append(x)

        drift_detected = getattr(self.ddm, "drift_detected", getattr(self.ddm, "change_detected", False))

        # Directional filter: only flag if error rate degraded (worsened)
        if self.degradation_only and drift_detected and prev_estimation is not None:
            curr_estimation = getattr(self.ddm, "estimation", None)
            if curr_estimation is not None and curr_estimation < prev_estimation:
                drift_detected = False

        if not self.assume_warning and not drift_detected:
            self.error_history = self.error_history[-self.rate_calculation_sample_size :]

        if drift_detected:
            detection_idx = self.current_index

            # Determine dynamic lookback window
            if hasattr(self.ddm, "width"):
                lookback_window = int(self.ddm.width)
            else:
                lookback_window = len(self.error_history)

            # Find actual drift start using selected method
            if self.lookback_method == "cusum":
                drift_start_idx = self.find_drift_start_cusum(detection_idx, lookback_window=lookback_window)
            elif self.lookback_method == "threshold":
                drift_start_idx = self.find_drift_start_threshold(detection_idx, lookback_window=lookback_window)
            elif self.lookback_method == "gradient":
                drift_start_idx = self.find_drift_start_gradient(detection_idx, lookback_window=lookback_window)
            else:  # 'none'
                drift_start_idx = max(0, detection_idx - lookback_window)

            # Ensure the found starting point is within valid bounds [detection_idx - lookback_window, detection_idx]
            min_lookback_bound = max(0, detection_idx - lookback_window)
            drift_start_idx = max(min_lookback_bound, min(drift_start_idx, detection_idx))

            window_size = self.rate_calculation_sample_size
            start_window_begin = max(0, drift_start_idx - window_size // 2)
            start_window_end = min(drift_start_idx + window_size // 2, len(self.complete_error_history))
            start_window = self.complete_error_history[start_window_begin:start_window_end]
            error_rate_at_warning = float(np.mean(start_window)) if len(start_window) > 0 else 0.0

            detection_window_begin = max(0, detection_idx - window_size)
            detection_window = self.complete_error_history[detection_window_begin:detection_idx]
            error_rate_at_detection = float(np.mean(detection_window)) if len(detection_window) > 0 else 0.0

            drift_duration = detection_idx - drift_start_idx

            self.last_detected_drift = DriftDescription(
                error_rate_at_detection=error_rate_at_detection,
                error_rate_at_warning=error_rate_at_warning,
                drift_duration=drift_duration,
                drift_start_index=drift_start_idx,
                drift_end_index=detection_idx,
                error_rate_at_peak=error_rate_at_detection,
                detected_at=detection_idx,
            )

            self.drift_detected = True

            self.warning_grace_period_left = self.warning_grace_period
            self.error_history = []
            self.previous_was_warning = False
            self.assume_warning = False

        self.previous_was_warning = self.assume_warning
        self.current_index += 1

    def describe_drift(self) -> DriftDescription | None:
        """Return the `DriftDescription` object for the most recent drift event.

        Returns:
            Description object containing duration, onset, and error rates, or `None`.
        """
        return self.last_detected_drift

    def post_process_drift_ends(self, drift_descriptions: list[DriftDescription]) -> list[DriftDescription]:
        """Post-process a collection of detected drifts to compute peak or recovery timestamps.

        Args:
            drift_descriptions: Sequence of `DriftDescription` objects to update.

        Returns:
            Updated sequence of descriptions with resolved end indices and peak rates.
        """
        if self.lookforward_method == "none":
            return drift_descriptions

        for drift_description in drift_descriptions:
            detection_idx = drift_description.detected_at or 0

            if self.lookforward_method == "peak":
                drift_end_idx = self.find_drift_end_peak(detection_idx)
            elif self.lookforward_method == "recovery":
                drift_end_idx = self.find_drift_end_recovery(detection_idx)
            else:
                drift_end_idx = detection_idx

            drift_description.drift_end_index = drift_end_idx

            window_size = self.rate_calculation_sample_size
            end_window_begin = max(0, drift_end_idx - window_size // 2)
            end_window_end = min(drift_end_idx + window_size // 2, len(self.complete_error_history))
            end_window = self.complete_error_history[end_window_begin:end_window_end]
            drift_description.error_rate_at_peak = float(np.mean(end_window)) if len(end_window) > 0 else 0.0

            if drift_description.drift_start_index is not None:
                drift_description.drift_duration = drift_end_idx - drift_description.drift_start_index

        return drift_descriptions

def run_drift_detection(
    error_stream,
    detector_type,
    warning_grace_period,
    rate_calculation_sample_size,
    lookback_method="gradient",
    lookforward_method="peak",
    confidence_level=None,
    adwin_params=None,
    degradation_only=True,
    burn_in=0,
):
    if lookback_method not in ['cusum', 'threshold', 'gradient', 'none']:
        raise Exception("No such lookback method") 

    if lookforward_method not in ['peak', 'recovery', 'none']:
        raise Exception("No such lookforward method") 

    if detector_type not in ["DDM", "EDDM", "FHDDM", "HDDM_A", "HDDM_W", "ADWIN"]:
        raise Exception("No such detector") 
    
    # Create detector
    if detector_type == "DDM":
        detector = river_drift.binary.DDM()
    elif detector_type == "EDDM":
        detector = river_drift.binary.EDDM()
    elif detector_type == "FHDDM":
        detector = river_drift.binary.FHDDM(confidence_level=confidence_level)
    elif detector_type == "HDDM_A":
        detector = river_drift.binary.HDDM_A()
    elif detector_type == "HDDM_W":
        detector = river_drift.binary.HDDM_W()
    elif detector_type == "ADWIN":
        if adwin_params is None:
            adwin_params = {}
        detector = river_drift.ADWIN(**adwin_params)

    drift_descriptor = BinaryErrorDriftDescriptor(
        warning_grace_period=warning_grace_period,
        rate_calculation_sample_size=rate_calculation_sample_size,
        ddm=detector,
        lookback_method=lookback_method,
        lookforward_method=lookforward_method,
        degradation_only=degradation_only,
    )

    drift_descriptions = []

    for i, error in enumerate(error_stream):
        if i < burn_in:
            drift_descriptor.complete_error_history.append(error)
            drift_descriptor.current_index += 1
            continue

        drift_descriptor.update(error)

        if drift_descriptor.drift_detected:
            drift = drift_descriptor.last_detected_drift
            drift.detected_at = i
            drift_descriptions.append(drift)

    # Post-process to find actual drift ends
    drift_descriptions = drift_descriptor.post_process_drift_ends(drift_descriptions)

    return drift_descriptions

def generate_error_stream(X_np, y_np, model):
    """
    Runs test-then-train once and returns the error stream and predictions.
    """
    classes = np.unique(y_np)

    error_stream = []
    predictions = []

    # Initialize model
    model.partial_fit(X_np[0].reshape(1, -1), [y_np[0]], classes=classes)
    error_stream.append(1)  # Add the first sample as if it was an error

    for i in range(1, len(X_np)):
        x_i = X_np[i].reshape(1, -1)
        y_true = y_np[i]

        # Predict
        y_pred = model.predict(x_i)[0]
        predictions.append(y_pred)

        # Binary error
        error_stream.append(int(y_pred != y_true))

        # Learn
        model.partial_fit(x_i, [y_true])


    return error_stream, predictions

import plotly.graph_objects as go

def visualization_and_correction(drift_descriptions, error_rate, rate_calculation_sample_size, visualize = True, take_max_for_end = True, start_index = None, end_index = None):
    corrected_descriptions = []

    if visualize:
        fig = go.Figure()

        # Determine the range to visualize
        if start_index is None:
            start_index = rate_calculation_sample_size
        if end_index is None:
            end_index = rate_calculation_sample_size + len(error_rate)
        
        # Ensure indices are within valid bounds
        start_index = max(rate_calculation_sample_size, start_index)
        end_index = min(rate_calculation_sample_size + len(error_rate), end_index)
        
        # Calculate slice indices for error_rate array
        error_start_idx = start_index - rate_calculation_sample_size
        error_end_idx = end_index - rate_calculation_sample_size
        
        # Slice the error_rate data
        error_rate_slice = error_rate[error_start_idx:error_end_idx]
        x_values = list(range(start_index, start_index + len(error_rate_slice)))

        # Add error rate line
        fig.add_trace(go.Scatter(
            x=x_values,
            y=error_rate_slice,
            mode='lines',
            name='Error Rate',
            line=dict(color='royalblue', width=2),
            hovertemplate='Index: %{x}<br>Error Rate: %{y:.3f}<extra></extra>'
        ))

    # Add drift annotations using ACTUAL drift start and end
    for idx, drift in enumerate(drift_descriptions):
        # Use the actual drift start index
        if hasattr(drift, 'drift_start_index') and drift.drift_start_index is not None:
            start = drift.drift_start_index
        else:
            start = max(0, drift.detected_at - drift.drift_duration)

        # Use the actual drift end index (peak or recovery point)
        if hasattr(drift, 'drift_end_index') and drift.drift_end_index is not None:
            end = drift.drift_end_index
        else:
            end = drift.detected_at

        start += rate_calculation_sample_size
        end += rate_calculation_sample_size

        # Update end to the index with maximum error rate
        start_idx = start - rate_calculation_sample_size
        end_idx = end - rate_calculation_sample_size
        start_idx = max(0, min(start_idx, len(error_rate) - 1))
        end_idx = max(0, min(end_idx, len(error_rate) - 1))

        if start_idx <= end_idx:
            if take_max_for_end:
                max_idx_in_range = start_idx + np.argmax(error_rate[start_idx:end_idx + 1])
                end = max_idx_in_range + rate_calculation_sample_size
            else:
                end = end_idx + rate_calculation_sample_size
            drift.drift_end_index = end

        # correct the duration based
        drift.drift_duration = end - start

        detection = drift.detected_at

        # Make sure they're within the error_rate bounds
        start_plot = max(rate_calculation_sample_size,
                            min(start, rate_calculation_sample_size + len(error_rate) - 1))
        end_plot = max(rate_calculation_sample_size,
                        min(end, rate_calculation_sample_size + len(error_rate) - 1))

        # Calculate error_rate array indices
        start_idx = start - rate_calculation_sample_size
        end_idx = end - rate_calculation_sample_size
        detection_idx = detection - rate_calculation_sample_size
        start_idx = max(0, min(start_idx, len(error_rate) - 1))
        end_idx = max(0, min(end_idx, len(error_rate) - 1))
        detection_idx = max(0, min(detection_idx, len(error_rate) - 1))

        description = {
            'start': start_plot,
            'end': end_plot,
            'duration': end_plot - start_plot,
            'start_error_rate': error_rate[start_idx],
            'end_error_rate': error_rate[end_idx],
            'error_rate_increase': error_rate[end_idx] - error_rate[start_idx]
        }
        corrected_descriptions.append(description)

        if visualize:
            # Skip drifts outside the visualization range
            if end_plot < start_index or start_plot > end_index:
                continue
            
            # Clip drift boundaries to visualization range
            drift_start_plot = max(start_plot, start_index)
            drift_end_plot = min(end_plot, end_index)
            
            # # Add drift region as shaded area (from start to end)
            # fig.add_vrect(
            #     x0=drift_start_plot,
            #     x1=drift_end_plot,
            #     fillcolor="rgba(255, 107, 53, 0.2)",
            #     layer="below",
            #     line_width=0,
            # )

            # Add drift line from start to end
            fig.add_trace(go.Scatter(
                x=[drift_start_plot, drift_end_plot],
                y=[error_rate[start_idx], error_rate[end_idx]],
                mode='lines+markers',
                name=f'Drift' if len(drift_descriptions)==1 else f'Drift {idx+1}',
                line=dict(color='#FF6B35', width=3, dash='dash'),
                marker=dict(size=10, color='#FF6B35', symbol=['circle', 'x']),
                hovertemplate=(
                    f'<b>Drift {idx+1}</b><br>'
                    f'Start Index: {start}<br>'
                    f'Peak/End Index: {end}<br>'
                    f'Duration: {end - start}<br>'
                    f'Error Rate at Start: {error_rate[start_idx]:.3f}<br>'
                    f'Error Rate at Peak: {error_rate[end_idx]:.3f}<br>'
                    f'<extra></extra>'
                )
            ))

    if visualize:
        # Update layout
        fig.update_layout(
            title=f'Drift Detection',
            xaxis_title='Data Point Index',
            yaxis_title='Error Rate',
            hovermode='closest',
            height=600,
            width= 1000,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                title=dict(
                    font=dict(size=14, color='black', family='Arial, sans-serif')
                ),
                tickfont=dict(size=12, color='black', family='Arial, sans-serif')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='lightgray',
                title=dict(
                    font=dict(size=14, color='black', family='Arial, sans-serif')
                ),
                tickfont=dict(size=12, color='black', family='Arial, sans-serif')
            ),
            font=dict(size=12, color='black', family='Arial, sans-serif')
        )
        fig.show()
        
    return corrected_descriptions