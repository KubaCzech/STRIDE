import unittest
from river.drift import ADWIN as RiverADWIN

from stride.drift import ADWIN, DualADWIN, BinaryErrorDriftDescriptor
from DDM import ADWIN as DDM_ADWIN, DualADWIN as DDM_DualADWIN
from stride.drift.binary_descriptor import DriftDescription


class TestADWINUnit(unittest.TestCase):
    """Deterministic unit tests for ADWIN and DualADWIN implementations."""

    def test_imports_and_aliases(self):
        """Test public API exposure and backwards compatibility bridges."""
        self.assertIs(ADWIN, RiverADWIN)
        self.assertIs(DDM_ADWIN, RiverADWIN)
        self.assertIs(DualADWIN, DDM_DualADWIN)

    def test_dual_adwin_initialization(self):
        """Test DualADWIN initialization and default parameter assignment."""
        dual = DualADWIN(delta_warn=0.05, delta_drift=0.002, clock=16)
        self.assertEqual(dual.adwin_warn.delta, 0.05)
        self.assertEqual(dual.adwin_drift.delta, 0.002)
        self.assertEqual(dual.adwin_warn.clock, 16)
        self.assertEqual(dual.adwin_drift.clock, 16)
        self.assertEqual(dual.width, 0)
        self.assertFalse(dual.warning_detected)
        self.assertFalse(dual.drift_detected)

    def test_dual_adwin_warning_and_drift_triggers(self):
        """Test that DualADWIN triggers warning earlier or independently from drift."""
        # Warning has a higher delta (0.2), drift has a lower delta (0.001)
        dual = DualADWIN(delta_warn=0.2, delta_drift=0.001, clock=8)

        # Baseline stationary stream of 0s
        for _ in range(100):
            dual.update(0)

        self.assertFalse(dual.warning_detected)
        self.assertFalse(dual.drift_detected)
        self.assertGreater(dual.width, 0)

        # Inject sudden shift to 1s
        warning_seen = False
        drift_seen = False
        warning_step = None
        drift_step = None

        for step in range(101, 250):
            dual.update(1)
            if dual.warning_detected and not warning_seen:
                warning_seen = True
                warning_step = step
            if dual.drift_detected and not drift_seen:
                drift_seen = True
                drift_step = step

        self.assertTrue(warning_seen, "Warning detector should have triggered on sustained shift")
        self.assertTrue(drift_seen, "Drift detector should have triggered on sustained shift")
        self.assertLessEqual(
            warning_step,
            drift_step,
            f"Warning step ({warning_step}) should be <= drift step ({drift_step})",
        )

    def test_dual_adwin_estimation(self):
        """Test that DualADWIN exposes estimation property."""
        dual = DualADWIN(delta_warn=0.05, delta_drift=0.002)
        self.assertEqual(dual.estimation, 0.0)
        for _ in range(50):
            dual.update(1)
        self.assertAlmostEqual(dual.estimation, 1.0)


class TestBinaryErrorDriftDescriptorADWIN(unittest.TestCase):
    """Unit tests for BinaryErrorDriftDescriptor handling ADWIN and DualADWIN."""

    def test_directional_filtering_ignores_error_drop(self):
        """Test that degradation_only=True suppresses drift detection when error rate improves."""
        # 1. With degradation_only=True: dropping error should NOT trigger drift
        adwin_suppressed = RiverADWIN(delta=0.002, clock=8)
        desc_suppressed = BinaryErrorDriftDescriptor(
            ddm=adwin_suppressed,
            degradation_only=True,
            rate_calculation_sample_size=20,
        )

        for _ in range(200):
            desc_suppressed.update(1 if _ % 2 == 0 else 0)  # 50% error rate
        for _ in range(200):
            desc_suppressed.update(0)  # Drops to 0% error rate (model improves)

        self.assertFalse(
            desc_suppressed.drift_detected,
            "Drop in error rate should not be flagged as concept drift when degradation_only=True",
        )

        # 2. With degradation_only=False: dropping error triggers raw ADWIN two-sided shift
        adwin_raw = RiverADWIN(delta=0.002, clock=8)
        desc_raw = BinaryErrorDriftDescriptor(
            ddm=adwin_raw,
            degradation_only=False,
            rate_calculation_sample_size=20,
        )
        for _ in range(200):
            desc_raw.update(1 if _ % 2 == 0 else 0)
        raw_drifts = 0
        for _ in range(200):
            desc_raw.update(0)
            if desc_raw.drift_detected:
                raw_drifts += 1

        self.assertGreaterEqual(raw_drifts, 1, "Raw ADWIN without directional filter triggers on error drops")

    def test_safe_handling_of_detector_without_warning(self):
        """Verify that BinaryErrorDriftDescriptor does not raise AttributeError for detectors without warning_detected."""
        detector = RiverADWIN()
        descriptor = BinaryErrorDriftDescriptor(
            ddm=detector,
            lookback_method="none",
            lookforward_method="none",
            rate_calculation_sample_size=10,
        )

        # Should not raise AttributeError when updating
        try:
            for val in [0, 0, 1, 0, 1]:
                descriptor.update(val)
        except AttributeError as e:
            self.fail(f"Updating descriptor with ADWIN raised AttributeError: {e}")

        self.assertEqual(len(descriptor.complete_error_history), 5)

    def test_adaptive_lookback_using_detector_width(self):
        """Verify that descriptor dynamically extracts lookback_window from detector.width."""
        detector = RiverADWIN(delta=0.002, clock=8)
        descriptor = BinaryErrorDriftDescriptor(
            ddm=detector,
            lookback_method="none",
            lookforward_method="none",
            rate_calculation_sample_size=20,
        )

        # Feed stationary stream then a strong drift
        for _ in range(150):
            descriptor.update(0)

        initial_width = detector.width
        self.assertGreater(initial_width, 50)

        # Induce drift
        drift_found = False
        for _ in range(100):
            descriptor.update(1)
            if descriptor.drift_detected:
                drift_found = True
                drift = descriptor.last_detected_drift
                drift.detected_at = descriptor.current_index
                self.assertIsInstance(drift, DriftDescription)
                self.assertIsNotNone(drift.drift_start_index)
                self.assertLessEqual(drift.drift_start_index, descriptor.current_index)
                break

        self.assertTrue(drift_found, "ADWIN should have detected the abrupt transition from 0 to 1")

    def test_deterministic_change_point_post_processing(self):
        """Test CUSUM and threshold lookback methods on deterministic error stream."""
        for method in ["cusum", "gradient", "threshold", "none"]:
            with self.subTest(lookback_method=method):
                detector = RiverADWIN(delta=0.01, clock=8)
                descriptor = BinaryErrorDriftDescriptor(
                    ddm=detector,
                    lookback_method=method,
                    lookforward_method="peak",
                    rate_calculation_sample_size=15,
                )

                drifts = []
                # 120 zeros followed by 60 ones
                for x in [0] * 120 + [1] * 60:
                    descriptor.update(x)
                    if descriptor.drift_detected:
                        drift = descriptor.last_detected_drift
                        drift.detected_at = descriptor.current_index
                        drifts.append(drift)

                processed = descriptor.post_process_drift_ends(drifts)
                self.assertGreaterEqual(len(processed), 1)
                for d in processed:
                    self.assertGreaterEqual(d.drift_start_index, 0)
                    self.assertLessEqual(d.drift_start_index, d.detected_at)

    def test_lookback_cusum_inflection_point_localization(self):
        """Test that CUSUM accurately localizes the inflection point within the lookback window."""
        detector = RiverADWIN(delta=0.01, clock=8)
        descriptor = BinaryErrorDriftDescriptor(
            ddm=detector,
            lookback_method="cusum",
            lookforward_method="none",
            rate_calculation_sample_size=15,
        )

        drift = None
        # Injected step increase at sample 100
        for i in range(160):
            x = 0 if i < 100 else 1
            descriptor.update(x)
            if descriptor.drift_detected:
                drift = descriptor.last_detected_drift
                drift.detected_at = descriptor.current_index
                break

        self.assertIsNotNone(drift)
        # Drift occurred at 100, detected shortly after (e.g. ~130)
        # CUSUM should identify start close to 100, NOT clamp all the way to the window boundary
        self.assertGreaterEqual(drift.drift_start_index, 80)
        self.assertLessEqual(drift.drift_start_index, 115)


if __name__ == "__main__":
    unittest.main()
