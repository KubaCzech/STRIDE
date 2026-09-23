class FeatureImportanceMethod:
    """Enum-like class for feature importance methods."""

    PFI = "permutation"
    SHAP = "shap"
    LIME = "lime"

    @classmethod
    def all_available(cls):
        """Return list of all available methods."""
        return [cls.PFI, cls.SHAP, cls.LIME]
