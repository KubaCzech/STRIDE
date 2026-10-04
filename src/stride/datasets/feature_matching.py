"""Feature alignment and matching engine for semi-synthetic concept drift."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class FeatureAlignmentResult:
    """Result of feature alignment between two datasets."""

    X_a_aligned: pd.DataFrame
    X_b_aligned: pd.DataFrame
    feature_mapping: dict[str, str]
    aligned_feature_names: list[str]
    drifting_features: list[str]
    drift_diagnostics: pd.DataFrame


class FeatureMatcher:
    """
    Aligns and matches feature spaces between two distinct datasets or cohorts.

    Supports exact column matching, user-specified manual dictionary mapping,
    statistical distribution matching (Wasserstein/KS bipartite assignment),
    and latent subspace projection via PCA. Also provides distribution
    normalization (joint or per-concept) and drifting feature identification.
    """

    SUPPORTED_STRATEGIES = ("exact", "manual", "statistical", "subspace")
    SUPPORTED_SCALERS = ("standard", "minmax", None)
    SUPPORTED_SCALER_MODES = ("joint", "per_concept")

    def __init__(
        self,
        match_strategy: str = "exact",
        manual_mapping: dict[str, str] | None = None,
        align_distributions: bool = False,
        scaler_type: str | None = "standard",
        scaler_mode: str = "joint",
        subspace_components: int | None = None,
        drift_significance_level: float = 0.05,
        random_state: int | None = 42,
    ) -> None:
        """
        Initialize FeatureMatcher.

        Parameters
        ----------
        match_strategy : str, default="exact"
            Strategy to align features: "exact", "manual", "statistical", or "subspace".
        manual_mapping : dict[str, str] | None, default=None
            Mapping dictionary {col_A: col_B} for "manual" strategy.
        align_distributions : bool, default=False
            Whether to apply feature scaling/normalization after alignment.
        scaler_type : str | None, default="standard"
            Type of scaler: "standard" (StandardScaler) or "minmax" (MinMaxScaler).
        scaler_mode : str, default="joint"
            "joint" to fit scaler across pooled [X_A; X_B] (preserves relative shifts),
            or "per_concept" to fit independently on X_A and X_B (isolates P(y|x) shift).
        subspace_components : int | None, default=None
            Number of latent dimensions for "subspace" strategy. If None, min(d_A, d_B).
        drift_significance_level : float, default=0.05
            p-value threshold below which a feature is categorized as drifting under KS test.
        random_state : int | None, default=42
            Random seed for stochastic operations (e.g. PCA).
        """
        if match_strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported match_strategy '{match_strategy}'. Supported strategies are: {self.SUPPORTED_STRATEGIES}"
            )
        if scaler_type not in self.SUPPORTED_SCALERS:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Supported scalers are: {self.SUPPORTED_SCALERS}")
        if scaler_mode not in self.SUPPORTED_SCALER_MODES:
            raise ValueError(
                f"Unsupported scaler_mode '{scaler_mode}'. Supported scaler modes are: {self.SUPPORTED_SCALER_MODES}"
            )

        self.match_strategy = match_strategy
        self.manual_mapping = manual_mapping or {}
        self.align_distributions = align_distributions
        self.scaler_type = scaler_type
        self.scaler_mode = scaler_mode
        self.subspace_components = subspace_components
        self.drift_significance_level = drift_significance_level
        self.random_state = random_state

    def align(self, X_a: pd.DataFrame, X_b: pd.DataFrame) -> FeatureAlignmentResult:
        """
        Align feature representations between dataset A and dataset B.

        Parameters
        ----------
        X_a : pd.DataFrame
            Feature matrix of Concept A (pre-drift).
        X_b : pd.DataFrame
            Feature matrix of Concept B (post-drift).

        Returns
        -------
        FeatureAlignmentResult
            Dataclass containing aligned DataFrames, column mapping, and drift diagnostics.
        """
        if not isinstance(X_a, pd.DataFrame) or not isinstance(X_b, pd.DataFrame):
            raise TypeError("Both X_a and X_b must be pandas DataFrames.")
        if X_a.empty or X_b.empty:
            raise ValueError("Input feature DataFrames must not be empty.")

        X_a_work = X_a.copy()
        X_b_work = X_b.copy()

        # Step 1: Feature Matching
        if self.match_strategy == "exact":
            X_a_matched, X_b_matched, mapping, aligned_names = self._match_exact(X_a_work, X_b_work)
        elif self.match_strategy == "manual":
            X_a_matched, X_b_matched, mapping, aligned_names = self._match_manual(X_a_work, X_b_work)
        elif self.match_strategy == "statistical":
            X_a_matched, X_b_matched, mapping, aligned_names = self._match_statistical(X_a_work, X_b_work)
        elif self.match_strategy == "subspace":
            X_a_matched, X_b_matched, mapping, aligned_names = self._match_subspace(X_a_work, X_b_work)
        else:
            raise ValueError(f"Unknown match strategy: {self.match_strategy}")

        # Step 2: Distribution Normalization
        if self.align_distributions and self.scaler_type is not None:
            X_a_norm, X_b_norm = self._normalize(X_a_matched, X_b_matched)
        else:
            X_a_norm, X_b_norm = X_a_matched, X_b_matched

        # Step 3: Drift Diagnostics & Identification
        diagnostics, drifting_features = self.identify_drifting_features(X_a_norm, X_b_norm)

        return FeatureAlignmentResult(
            X_a_aligned=X_a_norm,
            X_b_aligned=X_b_norm,
            feature_mapping=mapping,
            aligned_feature_names=aligned_names,
            drifting_features=drifting_features,
            drift_diagnostics=diagnostics,
        )

    def _match_exact(
        self, X_a: pd.DataFrame, X_b: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
        common_cols = [c for c in X_a.columns if c in X_b.columns]
        if not common_cols:
            raise ValueError(
                "No overlapping column names found between X_a and X_b for exact matching. "
                f"X_a columns: {list(X_a.columns)}, X_b columns: {list(X_b.columns)}"
            )

        mapping = {c: c for c in common_cols}
        X_a_matched = X_a[common_cols].copy()
        X_b_matched = X_b[common_cols].copy()
        return X_a_matched, X_b_matched, mapping, common_cols

    def _match_manual(
        self, X_a: pd.DataFrame, X_b: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
        if not self.manual_mapping:
            raise ValueError("manual_mapping dictionary must be provided when match_strategy='manual'.")

        for col_a, col_b in self.manual_mapping.items():
            if col_a not in X_a.columns:
                raise ValueError(f"Column '{col_a}' not found in X_a columns: {list(X_a.columns)}")
            if col_b not in X_b.columns:
                raise ValueError(f"Column '{col_b}' not found in X_b columns: {list(X_b.columns)}")

        cols_a = list(self.manual_mapping.keys())
        cols_b = [self.manual_mapping[c] for c in cols_a]

        # Standardize target names in aligned frames to cols_a
        X_a_matched = X_a[cols_a].copy()
        X_b_matched = X_b[cols_b].copy()
        X_b_matched.columns = cols_a

        return X_a_matched, X_b_matched, self.manual_mapping.copy(), cols_a

    def _match_statistical(
        self, X_a: pd.DataFrame, X_b: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
        # Select numeric columns
        num_a = X_a.select_dtypes(include=[np.number]).columns.tolist()
        num_b = X_b.select_dtypes(include=[np.number]).columns.tolist()

        if not num_a or not num_b:
            raise ValueError("Statistical matching requires numeric columns in both datasets.")

        # Compute cost matrix using standardized Wasserstein distance
        # Standardizing per column ensures scale-invariant distribution distance
        scaler_a = StandardScaler().fit_transform(X_a[num_a].fillna(0))
        scaler_b = StandardScaler().fit_transform(X_b[num_b].fillna(0))

        cost_matrix = np.zeros((len(num_a), len(num_b)))
        for i, _ in enumerate(num_a):
            col_data_a = scaler_a[:, i]
            for j, _ in enumerate(num_b):
                col_data_b = scaler_b[:, j]
                cost_matrix[i, j] = wasserstein_distance(col_data_a, col_data_b)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        mapping = {}
        aligned_cols_a = []
        aligned_cols_b = []
        for r, c in zip(row_ind, col_ind, strict=False):
            col_a = num_a[r]
            col_b = num_b[c]
            mapping[col_a] = col_b
            aligned_cols_a.append(col_a)
            aligned_cols_b.append(col_b)

        X_a_matched = X_a[aligned_cols_a].copy()
        X_b_matched = X_b[aligned_cols_b].copy()
        X_b_matched.columns = aligned_cols_a

        return X_a_matched, X_b_matched, mapping, aligned_cols_a

    def _match_subspace(
        self, X_a: pd.DataFrame, X_b: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
        num_a = X_a.select_dtypes(include=[np.number]).fillna(0)
        num_b = X_b.select_dtypes(include=[np.number]).fillna(0)

        d_a = num_a.shape[1]
        d_b = num_b.shape[1]
        k = self.subspace_components or min(d_a, d_b)
        k = max(1, min(k, d_a, d_b))

        pca_a = PCA(n_components=k, random_state=self.random_state)
        pca_b = PCA(n_components=k, random_state=self.random_state)

        proj_a = pca_a.fit_transform(StandardScaler().fit_transform(num_a))
        proj_b = pca_b.fit_transform(StandardScaler().fit_transform(num_b))

        feature_names = [f"PC{i + 1}" for i in range(k)]
        X_a_matched = pd.DataFrame(proj_a, columns=feature_names, index=X_a.index)
        X_b_matched = pd.DataFrame(proj_b, columns=feature_names, index=X_b.index)

        mapping = {name: name for name in feature_names}
        return X_a_matched, X_b_matched, mapping, feature_names

    def _normalize(self, X_a: pd.DataFrame, X_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        scaler_cls = StandardScaler if self.scaler_type == "standard" else MinMaxScaler

        if self.scaler_mode == "joint":
            # Fit on pooled data
            scaler = scaler_cls()
            pooled = pd.concat([X_a, X_b], axis=0, ignore_index=True)
            scaler.fit(pooled)

            arr_a = scaler.transform(X_a)
            arr_b = scaler.transform(X_b)
        else:
            # Per-concept normalization: aligns marginals to standard scale
            scaler_a = scaler_cls()
            scaler_b = scaler_cls()
            arr_a = scaler_a.fit_transform(X_a)
            arr_b = scaler_b.fit_transform(X_b)

        df_a = pd.DataFrame(arr_a, columns=X_a.columns, index=X_a.index)
        df_b = pd.DataFrame(arr_b, columns=X_b.columns, index=X_b.index)
        return df_a, df_b

    def identify_drifting_features(self, X_a: pd.DataFrame, X_b: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Evaluate statistical drift between aligned feature distributions.

        Parameters
        ----------
        X_a : pd.DataFrame
            Pre-drift aligned feature matrix.
        X_b : pd.DataFrame
            Post-drift aligned feature matrix.

        Returns
        -------
        tuple[pd.DataFrame, list[str]]
            Diagnostics DataFrame and list of feature names flagged as drifting.
        """
        diagnostics: list[dict[str, Any]] = []
        drifting_features: list[str] = []

        for col in X_a.columns:
            vals_a = np.asarray(X_a[col].dropna())
            vals_b = np.asarray(X_b[col].dropna())

            # KS test
            ks_res = ks_2samp(vals_a, vals_b)
            ks_stat = float(ks_res.statistic)
            p_val = float(ks_res.pvalue)

            # Wasserstein distance
            wd = float(wasserstein_distance(vals_a, vals_b))

            mean_a = float(np.mean(vals_a)) if len(vals_a) else 0.0
            mean_b = float(np.mean(vals_b)) if len(vals_b) else 0.0
            std_a = float(np.std(vals_a)) if len(vals_a) else 0.0
            std_b = float(np.std(vals_b)) if len(vals_b) else 0.0

            is_drifting = bool(p_val < self.drift_significance_level)
            if is_drifting:
                drifting_features.append(col)

            diagnostics.append(
                {
                    "feature": col,
                    "ks_statistic": ks_stat,
                    "p_value": p_val,
                    "wasserstein_distance": wd,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff": abs(mean_b - mean_a),
                    "std_a": std_a,
                    "std_b": std_b,
                    "is_drifting": is_drifting,
                }
            )

        diag_df = (
            pd.DataFrame(diagnostics)
            .sort_values(by=["is_drifting", "wasserstein_distance"], ascending=[False, False])
            .reset_index(drop=True)
        )

        return diag_df, drifting_features
