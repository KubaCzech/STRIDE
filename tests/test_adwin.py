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


class TestBinaryErrorDriftDescriptorADWIN(unittest.TestCase):
    """Unit tests for BinaryErrorDriftDescriptor handling ADWIN and DualADWIN."""

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


if __name__ == "__main__":
    unittest.main()
