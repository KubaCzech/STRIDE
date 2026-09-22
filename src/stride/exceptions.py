"""Domain exceptions for the STRIDE framework."""


class StrideError(Exception):
    """Base exception for all STRIDE errors."""


class OptionalDependencyError(StrideError):
    """Raised when an optional dependency (e.g. tensorflow, shap) is missing."""

    def __init__(self, package_name: str, feature_name: str):
        super().__init__(
            f"Feature '{feature_name}' requires optional dependency '{package_name}'. "
            f"Install it using: pip install stride-xai[{package_name}] or pip install {package_name}"
        )
        self.package_name = package_name
        self.feature_name = feature_name


class DriftDetectionError(StrideError):
    """Raised when drift detection calculation fails or inputs are invalid."""


class DimensionalityError(StrideError):
    """Raised when input feature dimensions violate method constraints (e.g. n_features < 2)."""


class ConfigurationError(StrideError):
    """Raised when invalid parameters or configurations are provided."""
