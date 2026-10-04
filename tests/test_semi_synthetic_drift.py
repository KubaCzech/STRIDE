"""Tests for semi-synthetic concept drift generation and feature matching (Shaker Protocol)."""

import tempfile
import unittest
import numpy as np
import pandas as pd
from river import drift as river_drift
from stride.datasets import (
    DatasetRegistry,
    DriftBlender,
    DriftSchedule,
    FeatureMatcher,
    SemiSyntheticDriftDataset,
    get_sample_wine_quality_data,
)
from stride.drift import BinaryErrorDriftDescriptor


class TestFeatureMatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.RandomState(42)
        self.df_a = pd.DataFrame(
            {
                "feat_1": self.rng.normal(0, 1, 100),
                "feat_2": self.rng.normal(5, 2, 100),
                "feat_extra_a": self.rng.uniform(0, 1, 100),
            }
        )
        self.df_b = pd.DataFrame(
            {
                "feat_1": self.rng.normal(3, 1, 100),  # Shifting mean
                "feat_2": self.rng.normal(5, 2, 100),  # Invariant
                "feat_extra_b": self.rng.exponential(1, 100),
            }
        )

    def test_exact_matching(self) -> None:
        matcher = FeatureMatcher(match_strategy="exact")
        result = matcher.align(self.df_a, self.df_b)

        self.assertEqual(result.aligned_feature_names, ["feat_1", "feat_2"])
        self.assertEqual(list(result.X_a_aligned.columns), ["feat_1", "feat_2"])
        self.assertEqual(list(result.X_b_aligned.columns), ["feat_1", "feat_2"])
        self.assertIn("feat_1", result.drifting_features)
        self.assertNotIn("feat_2", result.drifting_features)

    def test_manual_matching(self) -> None:
        mapping = {"feat_extra_a": "feat_extra_b", "feat_1": "feat_1"}
        matcher = FeatureMatcher(match_strategy="manual", manual_mapping=mapping)
        result = matcher.align(self.df_a, self.df_b)

        self.assertEqual(result.aligned_feature_names, ["feat_extra_a", "feat_1"])
        self.assertEqual(result.feature_mapping, mapping)
        self.assertEqual(result.X_a_aligned.shape, (100, 2))
        self.assertEqual(result.X_b_aligned.shape, (100, 2))

    def test_statistical_matching(self) -> None:
        matcher = FeatureMatcher(match_strategy="statistical")
        result = matcher.align(self.df_a, self.df_b)

        self.assertEqual(len(result.aligned_feature_names), 3)
        self.assertEqual(result.X_a_aligned.shape, (100, 3))
        self.assertEqual(result.X_b_aligned.shape, (100, 3))

    def test_subspace_matching(self) -> None:
        matcher = FeatureMatcher(match_strategy="subspace", subspace_components=2)
        result = matcher.align(self.df_a, self.df_b)

        self.assertEqual(result.aligned_feature_names, ["PC1", "PC2"])
        self.assertEqual(result.X_a_aligned.shape, (100, 2))
        self.assertEqual(result.X_b_aligned.shape, (100, 2))

    def test_distribution_alignment_joint(self) -> None:
        matcher = FeatureMatcher(
            match_strategy="exact",
            align_distributions=True,
            scaler_type="standard",
            scaler_mode="joint",
        )
        result = matcher.align(self.df_a, self.df_b)

        pooled = pd.concat([result.X_a_aligned, result.X_b_aligned], axis=0)
        np.testing.assert_almost_equal(pooled["feat_1"].mean(), 0.0, decimal=1)
        np.testing.assert_almost_equal(pooled["feat_1"].std(), 1.0, decimal=1)

    def test_distribution_alignment_per_concept(self) -> None:
        matcher = FeatureMatcher(
            match_strategy="exact",
            align_distributions=True,
            scaler_type="standard",
            scaler_mode="per_concept",
        )
        result = matcher.align(self.df_a, self.df_b)

        np.testing.assert_almost_equal(result.X_a_aligned["feat_1"].mean(), 0.0, decimal=1)
        np.testing.assert_almost_equal(result.X_b_aligned["feat_1"].mean(), 0.0, decimal=1)


class TestDriftBlender(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.RandomState(42)
        self.X_a = pd.DataFrame({"x": self.rng.normal(0, 1, 100), "y": self.rng.normal(0, 1, 100)})
        self.y_a = pd.Series(np.zeros(100, dtype=int), name="target")

        self.X_b = pd.DataFrame({"x": self.rng.normal(10, 1, 100), "y": self.rng.normal(10, 1, 100)})
        self.y_b = pd.Series(np.ones(100, dtype=int), name="target")

    def test_abrupt_schedule(self) -> None:
        blender = DriftBlender(schedule="abrupt", n_samples=200, t_0=100)
        res = blender.blend(self.X_a, self.y_a, self.X_b, self.y_b)

        self.assertEqual(len(res.X), 200)
        self.assertEqual(res.ground_truth_drift_points, [100])
        self.assertEqual(res.drift_intervals, [(100, 100)])
        self.assertTrue((res.concept_stream.iloc[:100] == 0).all())
        self.assertTrue((res.concept_stream.iloc[100:] == 1).all())

    def test_gradual_schedule_probabilities(self) -> None:
        blender = DriftBlender(schedule="gradual", n_samples=400, t_0=200, w=100, random_state=42)
        res = blender.blend(self.X_a, self.y_a, self.X_b, self.y_b)

        self.assertEqual(len(res.X), 400)
        self.assertEqual(res.ground_truth_drift_points, [200])
        self.assertEqual(res.drift_intervals, [(150, 250)])

        # Probability checks
        self.assertAlmostEqual(res.probabilities[200], 0.5, places=2)
        self.assertLess(res.probabilities[50], 0.01)
        self.assertGreater(res.probabilities[350], 0.99)

        # Monotonicity of concept B proportions before vs after
        prop_before = res.concept_stream.iloc[:150].mean()
        prop_after = res.concept_stream.iloc[250:].mean()
        self.assertLess(prop_before, 0.1)
        self.assertGreater(prop_after, 0.9)

    def test_incremental_schedule(self) -> None:
        blender = DriftBlender(schedule="incremental", n_samples=200, t_0=100, w=40, random_state=42)
        res = blender.blend(self.X_a, self.y_a, self.X_b, self.y_b)

        self.assertEqual(len(res.X), 200)
        self.assertEqual(res.drift_intervals, [(80, 120)])

        # Feature interpolation inside transition window
        # Pre-drift x mean is ~0, post-drift x mean is ~10
        mid_x = res.X.iloc[100]["x"]
        self.assertGreater(mid_x, 1.0)
        self.assertLess(mid_x, 9.0)

    def test_recurring_schedule(self) -> None:
        blender = DriftBlender(schedule="recurring", n_samples=600, period=200, w=50, random_state=42)
        res = blender.blend(self.X_a, self.y_a, self.X_b, self.y_b)

        self.assertEqual(len(res.X), 600)
        self.assertGreaterEqual(len(res.ground_truth_drift_points), 3)

    def test_blender_reproducibility(self) -> None:
        b1 = DriftBlender(schedule="gradual", n_samples=200, random_state=123)
        b2 = DriftBlender(schedule="gradual", n_samples=200, random_state=123)

        res1 = b1.blend(self.X_a, self.y_a, self.X_b, self.y_b)
        res2 = b2.blend(self.X_a, self.y_a, self.X_b, self.y_b)

        pd.testing.assert_frame_equal(res1.X, res2.X)
        pd.testing.assert_series_equal(res1.y, res2.y)
        pd.testing.assert_series_equal(res1.concept_stream, res2.concept_stream)


class TestSemiSyntheticDriftDataset(unittest.TestCase):
    def setUp(self) -> None:
        (self.X_red, self.y_red), (self.X_white, self.y_white) = get_sample_wine_quality_data(n_samples=100, random_state=42)

    def test_dataset_generation_and_metadata(self) -> None:
        ds = SemiSyntheticDriftDataset(
            name="wine_red_white",
            display_name="Red Wine to White Wine",
            X_a=self.X_red,
            y_a=self.y_red,
            X_b=self.X_white,
            y_b=self.y_white,
            schedule=DriftSchedule.GRADUAL.value,
            n_samples=200,
            t_0=100,
            drift_width=40,
            random_seed=42,
        )

        self.assertEqual(ds.name, "wine_red_white")
        self.assertEqual(ds.display_name, "Red Wine to White Wine")

        X, y = ds.generate()
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape, (200, 11))
        self.assertEqual(y.shape, (200,))

        self.assertEqual(ds.ground_truth_drift_points, [100])
        self.assertEqual(ds.drift_intervals, [(80, 120)])
        self.assertEqual(len(ds.concept_stream), 200)

        # Total sulfur dioxide and volatile acidity must be flagged as drifting
        self.assertIn("total sulfur dioxide", ds.drifting_features)
        self.assertIn("volatile acidity", ds.drifting_features)

    def test_registry_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = DatasetRegistry(data_dir=temp_dir)

            ds = SemiSyntheticDriftDataset(
                name="synth_test",
                X_a=self.X_red,
                y_a=self.y_red,
                X_b=self.X_white,
                y_b=self.y_white,
                n_samples=100,
                t_0=50,
                drift_width=20,
            )
            X, y = ds.generate()

            combined_df = X.copy()
            combined_df["target"] = y
            combined_df["_concept"] = ds.concept_stream

            recipe = {
                "match_strategy": "exact",
                "schedule": "gradual",
                "n_samples": 100,
                "t_0": 50,
                "drift_width": 20,
                "ground_truth_drift_points": ds.ground_truth_drift_points,
                "drift_intervals": ds.drift_intervals,
                "drifting_features": ds.drifting_features,
            }

            registry.save_semi_synthetic_dataset(
                name="synth_test",
                df=combined_df,
                target_column="target",
                recipe=recipe,
            )

            # Reload
            info = registry.get_dataset_info("synth_test")
            self.assertIsNotNone(info)
            self.assertEqual(info["type"], "semi_synthetic")

            reloaded_ds = SemiSyntheticDriftDataset.from_registry_info("synth_test", info, registry)
            X_rel, y_rel = reloaded_ds.generate()

            self.assertEqual(X_rel.shape, (100, 11))
            self.assertEqual(len(y_rel), 100)
            self.assertEqual(reloaded_ds.ground_truth_drift_points, [50])
            self.assertEqual(reloaded_ds.drift_intervals, [(40, 60)])


class TestWineBenchmarkReplication(unittest.TestCase):
    def test_wine_drift_benchmark_detection(self) -> None:
        """
        Replicate Shaker & Hüllermeier canonical benchmark using Red Wine -> White Wine.
        Verifies that online binary drift detector flags the concept transition.
        """
        (X_red, y_red), (X_white, y_white) = get_sample_wine_quality_data(n_samples=250, random_state=42)

        ds = SemiSyntheticDriftDataset(
            name="wine_benchmark",
            X_a=X_red,
            y_a=y_red,
            X_b=X_white,
            y_b=y_white,
            schedule="gradual",
            n_samples=500,
            t_0=250,
            drift_width=60,
            random_seed=42,
        )
        X, y = ds.generate()

        # Run online learning with River DDM
        detector = river_drift.binary.DDM()
        descriptor = BinaryErrorDriftDescriptor(
            warning_grace_period=20,
            rate_calculation_sample_size=30,
            ddm=detector,
        )

        # Online evaluation with Naive Bayes / standard online model
        from river.naive_bayes import GaussianNB

        model = GaussianNB()

        detected_points = []
        for i in range(len(X)):
            x_dict = X.iloc[i].to_dict()
            y_val = int(y.iloc[i])

            y_pred = model.predict_one(x_dict)
            error = int(y_pred != y_val) if y_pred is not None else 0

            descriptor.update(error)
            if descriptor.drift_detected:
                detected_points.append(i)

            model.learn_one(x_dict, y_val)

        # Drift midpoint is 250 with window 220-280.
        # Detector should detect drift within the transition or shortly following recovery.
        self.assertGreater(len(detected_points), 0, "DDM should flag drift on Wine Quality benchmark")
        first_detection = detected_points[0]
        # First detection should occur around or after t_0 - w/2
        self.assertGreater(first_detection, 100)


if __name__ == "__main__":
    unittest.main()
