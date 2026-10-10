"""Domain exceptions for the STRIDE framework."""


class StrideError(Exception):
    """Base exception for all STRIDE errors."""


class OptionalDependencyError(StrideError):
    """Raised when an optional dependency (e.g. tensorflow, shap) is missing."""

    def __init__(
        self,
        package_name: str,
        feature_name: str | None = None,
        extra_name: str | None = None,
    ) -> None:
        """Initialize the missing dependency error.

        Args:
            package_name: Name of the missing PyPI dependency package.
            feature_name: High-level feature requiring the dependency.
            extra_name: Optional install extra target (e.g. `stride-xai[xai]`).
        """
        if feature_name is None:
            super().__init__(package_name)
            self.package_name = package_name
            self.feature_name = ""
            self.extra_name = ""
        else:
            self.package_name = package_name
            self.feature_name = feature_name
            self.extra_name = extra_name or package_name
            super().__init__(
                f"Feature '{feature_name}' requires optional dependency '{package_name}'. "
                f"Install it using: pip install stride-xai[{self.extra_name}] or pip install {package_name}"
            )


class DriftDetectionError(StrideError):
    """Raised when drift detection calculation fails or inputs are invalid."""


class DimensionalityError(StrideError):
    """Raised when input feature dimensions violate method constraints (e.g. n_features < 2)."""


class ConfigurationError(StrideError):
    """Raised when invalid parameters or configurations are provided."""
