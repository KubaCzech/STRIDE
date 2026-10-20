from river.datasets import synth
import itertools
from .base import BaseDataset
from .utils import generate_river_data


class SeaDriftDataset(BaseDataset):
    @property
    def name(self) -> str:
        return "sea_drift"

    @property
    def display_name(self) -> str:
        return "SEA Drift"

    def get_params(self) -> dict:
        return {
            "n_windows_before": 10,
            "n_windows_after": 10,
            "random_seed": 42,
            "n_features": 3,
            "drift_width": 1,
        }

    def generate(self, n_samples_before=1000, n_samples_after=1000, random_seed=42, n_features=3, drift_width=1, **kwargs):
        """
        Generate synthetic data stream using River's SEA generator.
        Drifts from variant 0 to variant 3 at the drift point.
        The SEA generator has 3 features, and we use all of them.

        Both concepts are generated with the same ``random_seed``. For abrupt drift
        the post-drift stream therefore starts from the same feature sequence: the
        first ``n_samples_after`` samples after the drift have exactly the same
        features as the first samples of the stream and only their labels differ.
        This is intentional and isolates pure concept drift (identical inputs,
        changed decision boundary). To compare two independent feature samples,
        use windows of at most ``n_samples_before / 2`` samples on each side of
        the drift point.
        """
        if drift_width < 5:
            # Sudden drift workaround for River overflow bug at small widths
            stream_SEA = itertools.chain(
                itertools.islice(synth.SEA(seed=random_seed, variant=0), n_samples_before),
                itertools.islice(synth.SEA(seed=random_seed, variant=3), n_samples_after),
            )
        else:
            stream_SEA = synth.ConceptDriftStream(
                stream=synth.SEA(seed=random_seed, variant=0),
                drift_stream=synth.SEA(seed=random_seed, variant=3),
                position=n_samples_before,
                width=drift_width,  # Gradual drift
                seed=random_seed,
            )
        return generate_river_data(stream_SEA, n_samples_before + n_samples_after, n_features=n_features)
