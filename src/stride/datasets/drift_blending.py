"""Stream blending and non-stationary schedule generation for concept drift."""

from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class DriftSchedule(str, Enum):
    """Supported drift transition schedules."""

    ABRUPT = "abrupt"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


@dataclass
class DriftBlendResult:
    """Result of stream blending between two concepts."""

    X: pd.DataFrame
    y: pd.Series
    concept_stream: pd.Series
    ground_truth_drift_points: list[int]
    drift_intervals: list[tuple[int, int]]
    probabilities: np.ndarray


class DriftBlender:
    """
    Blends two concept datasets into a continuous non-stationary stream.

    Supports abrupt, gradual (Shaker sigmoidal Bernoulli schedule),
    incremental (nearest-neighbor linear interpolation), and recurring
    (periodic harmonic modulation) drift schedules.
    """

    def __init__(
        self,
        schedule: str = "gradual",
        n_samples: int | None = None,
        t_0: int | None = None,
        w: int = 200,
        period: int = 500,
        random_state: int | None = 42,
    ) -> None:
        """
        Initialize DriftBlender.

        Parameters
        ----------
        schedule : str, default="gradual"
            Transition schedule: "abrupt", "gradual", "incremental", or "recurring".
        n_samples : int | None, default=None
            Total stream length. If None, defaults to len(X_a) + len(X_b).
        t_0 : int | None, default=None
            Inflection point (midpoint) of drift. If None, defaults to n_samples // 2.
        w : int, default=200
            Transition window width in samples.
        period : int, default=500
            Period length T for recurring schedule.
        random_state : int | None, default=42
            Random seed for deterministic stream generation.
        """
        valid_schedules = [s.value for s in DriftSchedule]
        if schedule not in valid_schedules:
            raise ValueError(f"Unsupported schedule '{schedule}'. Supported schedules are: {valid_schedules}")

        self.schedule = schedule
        self.n_samples = n_samples
        self.t_0 = t_0
        self.w = max(1, int(w))
        self.period = max(2, int(period))
        self.random_state = random_state

    def blend(
        self,
        X_a: pd.DataFrame,
        y_a: pd.Series,
        X_b: pd.DataFrame,
        y_b: pd.Series,
    ) -> DriftBlendResult:
        """
        Blend Concept A and Concept B according to the configured schedule.

        Parameters
        ----------
        X_a : pd.DataFrame
            Aligned feature matrix for Concept A.
        y_a : pd.Series
            Target series for Concept A.
        X_b : pd.DataFrame
            Aligned feature matrix for Concept B.
        y_b : pd.Series
            Target series for Concept B.

        Returns
        -------
        DriftBlendResult
            Synthesized stream DataFrame, target Series, concept indicators,
            and ground-truth change-point metadata.
        """
        if len(X_a) == 0 or len(X_b) == 0:
            raise ValueError("Input concept datasets must not be empty.")
        if list(X_a.columns) != list(X_b.columns):
            raise ValueError("Feature columns in X_a and X_b must match. Please align them first.")

        rng = np.random.RandomState(self.random_state)
        n_total = self.n_samples or (len(X_a) + len(X_b))
        t_mid = self.t_0 if self.t_0 is not None else n_total // 2

        # Permute indices for drawing instances without artificial ordering
        idx_a = rng.permutation(len(X_a))
        idx_b = rng.permutation(len(X_b))

        if self.schedule == DriftSchedule.ABRUPT.value:
            return self._blend_abrupt(X_a, y_a, X_b, y_b, n_total, t_mid, idx_a, idx_b)
        elif self.schedule == DriftSchedule.GRADUAL.value:
            return self._blend_gradual(X_a, y_a, X_b, y_b, n_total, t_mid, idx_a, idx_b, rng)
        elif self.schedule == DriftSchedule.INCREMENTAL.value:
            return self._blend_incremental(X_a, y_a, X_b, y_b, n_total, t_mid, idx_a, idx_b, rng)
        elif self.schedule == DriftSchedule.RECURRING.value:
            return self._blend_recurring(X_a, y_a, X_b, y_b, n_total, idx_a, idx_b, rng)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _blend_abrupt(
        self,
        X_a: pd.DataFrame,
        y_a: pd.Series,
        X_b: pd.DataFrame,
        y_b: pd.Series,
        n_total: int,
        t_mid: int,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
    ) -> DriftBlendResult:
        n_a_needed = min(t_mid, n_total)
        n_b_needed = max(0, n_total - n_a_needed)

        chosen_a = [idx_a[i % len(idx_a)] for i in range(n_a_needed)]
        chosen_b = [idx_b[i % len(idx_b)] for i in range(n_b_needed)]

        part_x_a = X_a.iloc[chosen_a]
        part_y_a = y_a.iloc[chosen_a]
        part_x_b = X_b.iloc[chosen_b]
        part_y_b = y_b.iloc[chosen_b]

        X_stream = pd.concat([part_x_a, part_x_b], ignore_index=True)
        y_stream = pd.concat([part_y_a, part_y_b], ignore_index=True)

        concepts = np.zeros(n_total, dtype=int)
        concepts[n_a_needed:] = 1

        probs = np.zeros(n_total, dtype=float)
        probs[n_a_needed:] = 1.0

        return DriftBlendResult(
            X=X_stream,
            y=y_stream,
            concept_stream=pd.Series(concepts, name="concept"),
            ground_truth_drift_points=[t_mid],
            drift_intervals=[(t_mid, t_mid)],
            probabilities=probs,
        )

    def _blend_gradual(
        self,
        X_a: pd.DataFrame,
        y_a: pd.Series,
        X_b: pd.DataFrame,
        y_b: pd.Series,
        n_total: int,
        t_mid: int,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        rng: np.random.RandomState,
    ) -> DriftBlendResult:
        t = np.arange(n_total)
        # Shaker's sigmoidal probability: P(t) = 1 / (1 + exp(-4 * (t - t_0) / w))
        exponent = np.clip(-4.0 * (t - t_mid) / self.w, -100.0, 100.0)
        probs = 1.0 / (1.0 + np.exp(exponent))

        # Bernoulli coin flip at each time step
        bernoulli = rng.random_sample(n_total) < probs
        concepts = bernoulli.astype(int)

        ptr_a = 0
        ptr_b = 0
        chosen_indices_a = []
        chosen_indices_b = []
        is_concept_b_list = []

        for b in bernoulli:
            if b:  # concept B
                chosen_indices_b.append(idx_b[ptr_b % len(idx_b)])
                ptr_b += 1
                is_concept_b_list.append(True)
            else:  # concept A
                chosen_indices_a.append(idx_a[ptr_a % len(idx_a)])
                ptr_a += 1
                is_concept_b_list.append(False)

        # Assemble stream
        rows_x = []
        rows_y = []
        idx_a_iter = iter(chosen_indices_a)
        idx_b_iter = iter(chosen_indices_b)

        for is_b in is_concept_b_list:
            if is_b:
                i = next(idx_b_iter)
                rows_x.append(X_b.iloc[i].values)
                rows_y.append(y_b.iloc[i])
            else:
                i = next(idx_a_iter)
                rows_x.append(X_a.iloc[i].values)
                rows_y.append(y_a.iloc[i])

        X_stream = pd.DataFrame(rows_x, columns=X_a.columns)
        y_stream = pd.Series(rows_y, name=y_a.name or "target")

        interval_start = max(0, int(t_mid - self.w / 2))
        interval_end = min(n_total - 1, int(t_mid + self.w / 2))

        return DriftBlendResult(
            X=X_stream,
            y=y_stream,
            concept_stream=pd.Series(concepts, name="concept"),
            ground_truth_drift_points=[t_mid],
            drift_intervals=[(interval_start, interval_end)],
            probabilities=probs,
        )

    def _blend_incremental(
        self,
        X_a: pd.DataFrame,
        y_a: pd.Series,
        X_b: pd.DataFrame,
        y_b: pd.Series,
        n_total: int,
        t_mid: int,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        rng: np.random.RandomState,
    ) -> DriftBlendResult:
        # Fit NearestNeighbors on numeric features of Concept B
        num_cols = X_a.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            num_cols = list(X_a.columns)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_b[num_cols].fillna(0).values)

        t_start = max(0, int(t_mid - self.w / 2))
        t_end = min(n_total - 1, int(t_mid + self.w / 2))

        probs = np.zeros(n_total, dtype=float)
        for t in range(n_total):
            if t < t_start:
                probs[t] = 0.0
            elif t > t_end:
                probs[t] = 1.0
            else:
                probs[t] = (t - t_start) / max(1, (t_end - t_start))

        rows_x = []
        rows_y = []
        concepts = []

        ptr_a = 0
        ptr_b = 0

        # Check if target is discrete or continuous
        y_is_numeric = np.issubdtype(y_a.dtype, np.number) and np.issubdtype(y_b.dtype, np.number)

        for t in range(n_total):
            lam = probs[t]
            if lam == 0.0:
                i_a = idx_a[ptr_a % len(idx_a)]
                ptr_a += 1
                rows_x.append(X_a.iloc[i_a].values)
                rows_y.append(y_a.iloc[i_a])
                concepts.append(0)
            elif lam == 1.0:
                i_b = idx_b[ptr_b % len(idx_b)]
                ptr_b += 1
                rows_x.append(X_b.iloc[i_b].values)
                rows_y.append(y_b.iloc[i_b])
                concepts.append(1)
            else:
                # Interpolation window: sample x_A, find matched nearest neighbor x_B
                i_a = idx_a[ptr_a % len(idx_a)]
                ptr_a += 1
                val_a = X_a.iloc[i_a]
                y_val_a = y_a.iloc[i_a]

                # Find NN in B
                query = val_a[num_cols].fillna(0).values.reshape(1, -1)
                nn_idx = nn.kneighbors(query, return_distance=False)[0, 0]
                val_b = X_b.iloc[nn_idx]
                y_val_b = y_b.iloc[nn_idx]

                # Linear interpolation: x_t = (1 - lam) * x_A + lam * x_B
                interp_x = (1.0 - lam) * val_a.values.astype(float) + lam * val_b.values.astype(float)
                rows_x.append(interp_x)

                if y_is_numeric:
                    interp_y = (1.0 - lam) * float(y_val_a) + lam * float(y_val_b)
                    # Round if both original targets were integers
                    if np.issubdtype(y_a.dtype, np.integer) and np.issubdtype(y_b.dtype, np.integer):
                        interp_y = int(round(interp_y))
                    rows_y.append(interp_y)
                else:
                    rows_y.append(y_val_b if rng.random_sample() < lam else y_val_a)

                concepts.append(1 if lam >= 0.5 else 0)

        X_stream = pd.DataFrame(rows_x, columns=X_a.columns)
        y_stream = pd.Series(rows_y, name=y_a.name or "target")

        return DriftBlendResult(
            X=X_stream,
            y=y_stream,
            concept_stream=pd.Series(concepts, name="concept"),
            ground_truth_drift_points=[t_mid],
            drift_intervals=[(t_start, t_end)],
            probabilities=probs,
        )

    def _blend_recurring(
        self,
        X_a: pd.DataFrame,
        y_a: pd.Series,
        X_b: pd.DataFrame,
        y_b: pd.Series,
        n_total: int,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        rng: np.random.RandomState,
    ) -> DriftBlendResult:
        t = np.arange(n_total)
        # Periodic modulation: P(t) = 0.5 * (1 + sin(2 * pi * t / T - pi / 2))
        probs = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / self.period - (np.pi / 2.0)))

        bernoulli = rng.random_sample(n_total) < probs
        concepts = bernoulli.astype(int)

        # Inflection points where P(t) = 0.5 (sin term crosses 0)
        # 2 * pi * t / T - pi / 2 = k * pi -> 2 * t / T = k + 0.5 -> t = (k + 0.5) * T / 2
        drift_points: list[int] = []
        drift_intervals: list[tuple[int, int]] = []

        half_w = self.w // 2
        k = 0
        while True:
            t_cross = int(round((k + 0.5) * self.period / 2.0))
            if t_cross >= n_total:
                break
            if t_cross >= 0:
                drift_points.append(t_cross)
                drift_intervals.append((max(0, t_cross - half_w), min(n_total - 1, t_cross + half_w)))
            k += 1

        ptr_a = 0
        ptr_b = 0
        rows_x = []
        rows_y = []

        for b in bernoulli:
            if b:
                i = idx_b[ptr_b % len(idx_b)]
                ptr_b += 1
                rows_x.append(X_b.iloc[i].values)
                rows_y.append(y_b.iloc[i])
            else:
                i = idx_a[ptr_a % len(idx_a)]
                ptr_a += 1
                rows_x.append(X_a.iloc[i].values)
                rows_y.append(y_a.iloc[i])

        X_stream = pd.DataFrame(rows_x, columns=X_a.columns)
        y_stream = pd.Series(rows_y, name=y_a.name or "target")

        return DriftBlendResult(
            X=X_stream,
            y=y_stream,
            concept_stream=pd.Series(concepts, name="concept"),
            ground_truth_drift_points=drift_points,
            drift_intervals=drift_intervals,
            probabilities=probs,
        )
