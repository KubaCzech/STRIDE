import itertools
from river.datasets import synth
from .base import BaseDataset
from .utils import generate_river_data


class HyperplaneDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "hyperplane_drift"

    @property
    def display_name(self) -> str:
        return "Hyperplane Drift"

    def get_params(self) -> dict:
        params = super().get_params()
        params.update(
            {
                "n_windows_before": 1,
                "n_windows_after": 1,
                "n_features": 2,
                "n_drift_features": 2,
                "noise_percentage": 0.05,
                "drift_noise_percentage": 0.1,
                "mag_change": 0.2,
                "drift_width": 1,
            }
        )
        return params

    # TODO: Seems that the parameter mag_change has no effect on the data stream.
    # Remove it, or fix it.
    def generate(
        self,
        n_samples_before=1000,
        n_samples_after=1000,
        n_features=2,
        n_drift_features=2,
        noise_percentage=0.05,
        drift_noise_percentage=0.1,
        mag_change=0.2,
        drift_width=1,
        random_seed=42,
        **kwargs,
    ):
        """
        Generate synthetic data stream using River's Hyperplane generator.
        """
        # Validation
        if n_features < 2:
            raise ValueError("n_features must be at least 2")
        if n_drift_features < 2:
            raise ValueError("n_drift_features must be at least 2")
        if n_drift_features > n_features:
            raise ValueError("n_drift_features cannot exceed n_features")

        # FIX: Avoid math range error in River's ConceptDriftStream for small width
        # The sigmoid function 1/(1+exp(-4*(i-p)/w)) overflows if w is very small.
        # If drift_width is small, we treat it as abrupt drift (chaining streams).
        SAFE_MIN_WIDTH = 50

        if drift_width < SAFE_MIN_WIDTH:
            # Manual Abrupt Drift
            stream1 = synth.Hyperplane(
                n_features=n_features, n_drift_features=n_drift_features, seed=random_seed, noise_percentage=noise_percentage
            )
            stream2 = synth.Hyperplane(
                n_features=n_features,
                n_drift_features=n_drift_features,
                seed=random_seed,
                mag_change=mag_change,
                noise_percentage=drift_noise_percentage,
            )

            # Chain the streams: First n_before from stream1, then n_after from stream2
            # Note: stream1 and stream2 are infinite generators
            stream_HP = itertools.chain(
                itertools.islice(stream1, n_samples_before), itertools.islice(stream2, n_samples_after)
            )
        else:
            stream_HP = synth.ConceptDriftStream(
                stream=synth.Hyperplane(
                    n_features=n_features,
                    n_drift_features=n_drift_features,
                    seed=random_seed,
                    noise_percentage=noise_percentage,
                ),
                drift_stream=synth.Hyperplane(
                    n_features=n_features,
                    n_drift_features=n_drift_features,
                    seed=random_seed,
                    mag_change=mag_change,
                    noise_percentage=drift_noise_percentage,
                ),
                position=n_samples_before,
                width=drift_width,  # Gradual drift
                seed=random_seed,
            )
        return generate_river_data(stream_HP, n_samples_before + n_samples_after, n_features)
