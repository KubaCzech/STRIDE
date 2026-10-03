class FeatureImportanceMethod:
    """Enum-like class for feature importance methods."""

    PFI = "permutation"
    SHAP = "shap"
    LIME = "lime"

    @classmethod
    def all_available(cls) -> list[str]:
        """Return list of all available methods."""
        return [cls.PFI, cls.SHAP, cls.LIME]


class AbstainStrategy:
    """Enum-like class for abstention strategies in drift localization."""

    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CONFORMAL = "conformal"

    @classmethod
    def all_available(cls) -> list[str]:
        """Return list of all available abstention strategies."""
        return [cls.CONFIDENCE_THRESHOLD, cls.CONFORMAL]
