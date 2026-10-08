"""Sequential binary error drift descriptor and change-point characterization."""

from typing import Any
import numpy as np
from river import drift

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
    """

    def __init__(
        self,
        warning_grace_period: int = 3,
        rate_calculation_sample_size: int = 100,
        ddm: Any = drift.binary.DDM(),
        lookback_method: str = "cusum",
        lookforward_method: str = "peak",
    ) -> None:
        """Initialize the binary error drift descriptor.

        Args:
            warning_grace_period: Allowed non-warning steps within a warning chain.
            rate_calculation_sample_size: Window size for moving error rate averages.
            ddm: River drift detector instance with `warning_detected` and `drift_detected`.
            lookback_method: Method for onset estimation ("cusum", "threshold", "gradient", "none").
            lookforward_method: Method for end/peak estimation ("peak", "recovery", "none").
        """
        self.warning_grace_period = warning_grace_period
        self.rate_calculation_sample_size = rate_calculation_sample_size
        self.ddm = ddm
        self.lookback_method = lookback_method
        self.lookforward_method = lookforward_method

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
        self.ddm.update(x)
        self.complete_error_history.append(x)
        self.drift_detected = False

        if self.ddm.warning_detected:
            self.warning_grace_period_left = self.warning_grace_period
        elif self.previous_was_warning is True:
            self.warning_grace_period_left -= 1

        if self.previous_was_warning and self.warning_grace_period_left > 0:
            self.assume_warning = True
        elif self.ddm.warning_detected:
            self.assume_warning = True
        else:
            self.assume_warning = False

        self.error_history.append(x)

        if not self.assume_warning and not self.ddm.drift_detected:
            self.error_history = self.error_history[-self.rate_calculation_sample_size :]

        if self.ddm.drift_detected:
            detection_idx = self.current_index

            if self.lookback_method == "cusum":
                drift_start_idx = self.find_drift_start_cusum(detection_idx)
            elif self.lookback_method == "threshold":
                drift_start_idx = self.find_drift_start_threshold(detection_idx)
            elif self.lookback_method == "gradient":
                drift_start_idx = self.find_drift_start_gradient(detection_idx)
            else:
                drift_start_idx = max(0, detection_idx - len(self.error_history))

            drift_start_idx = min(drift_start_idx, detection_idx - len(self.error_history))

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
