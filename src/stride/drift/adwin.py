"""Adaptive Windowing (ADWIN) sequential drift detection wrappers."""

from river.drift import ADWIN


class DualADWIN:
    r"""Dual-threshold ADWIN wrapper providing explicit warning and drift signals.

    Maintains two River `ADWIN` instances internally configured with different
    confidence bounds ($\delta_{\text{warn}} > \delta_{\text{drift}}$). The more
    sensitive instance ($\delta_{\text{warn}}$) triggers early warning alerts to
    commence lookback caching, while the conservative instance ($\delta_{\text{drift}}$)
    confirms actual concept drift.

    Attributes:
        adwin_warn: Highly sensitive ADWIN instance tracking warning state.
        adwin_drift: Conservative ADWIN instance tracking confirmed drift state.

    Examples:
        >>> detector = DualADWIN(delta_warn=0.05, delta_drift=0.002)
        >>> detector.update(1.0)
        >>> detector.drift_detected
        False
    """

    def __init__(
        self,
        delta_warn: float = 0.05,
        delta_drift: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ) -> None:
        """Initialize dual-threshold ADWIN detector.

        Args:
            delta_warn: Confidence value for warning threshold. Higher delta implies higher sensitivity.
            delta_drift: Confidence value for confirmed drift threshold.
            clock: Number of incoming samples between subwindow split checks.
            max_buckets: Maximum number of buckets maintained at each exponential compression tier.
            min_window_length: Minimum subwindow size required before testing for statistical difference.
            grace_period: Initial observations processed before drift detection begins.
        """
        self.adwin_warn = ADWIN(
            delta=delta_warn,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        self.adwin_drift = ADWIN(
            delta=delta_drift,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )

    def update(self, x: float | int) -> None:
        """Update both warning and drift ADWIN instances with a new observation.

        Args:
            x: Input observation value (e.g., binary classification error $0$ or $1$).
        """
        self.adwin_warn.update(x)
        self.adwin_drift.update(x)

    @property
    def warning_detected(self) -> bool:
        """Flag indicating whether early warning threshold was exceeded at current step."""
        return bool(self.adwin_warn.drift_detected)

    @property
    def drift_detected(self) -> bool:
        """Flag indicating whether confirmed drift threshold was exceeded at current step."""
        return bool(self.adwin_drift.drift_detected)

    @property
    def width(self) -> int:
        """Current window width of the main drift tracking instance."""
        return int(self.adwin_drift.width)

    @property
    def estimation(self) -> float:
        """Estimated mean error rate of the main drift tracking instance window."""
        return float(self.adwin_drift.estimation)
