from river.drift import ADWIN


class DualADWIN:
    """
    Dual-threshold ADWIN wrapper providing explicit warning and drift signals.
    Maintains two ADWIN instances internally: one for early warning and one for definite drift.
    """

    def __init__(
        self,
        delta_warn: float = 0.05,
        delta_drift: float = 0.002,
        clock: int = 32,
        max_buckets: int = 5,
        min_window_length: int = 5,
        grace_period: int = 10,
    ):
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

    def update(self, x: float) -> None:
        """Update both ADWIN instances with a new value."""
        self.adwin_warn.update(x)
        self.adwin_drift.update(x)

    @property
    def warning_detected(self) -> bool:
        """Returns True if the warning instance has detected a drift."""
        return self.adwin_warn.drift_detected

    @property
    def drift_detected(self) -> bool:
        """Returns True if the drift instance has detected a drift."""
        return self.adwin_drift.drift_detected

    @property
    def width(self) -> int:
        """Returns the width of the main drift instance window."""
        return self.adwin_drift.width
