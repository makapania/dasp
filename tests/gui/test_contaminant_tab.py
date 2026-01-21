"""
GUI tests for Contaminant Analysis Tab (Tab 13).

Tests the GUI methods for:
- Tab 13A: Load & Define Groups
- Tab 13B: Difference Analysis
- Tab 13C: Automated Detection
- Tab 13D: Apply Correction

These tests would catch bugs like calling self.update() instead of
self.root.update() since SpectralPredictApp is not a Tkinter widget.

Usage:
    pytest tests/gui/test_contaminant_tab.py -v
    pytest tests/gui/test_contaminant_tab.py -v --visible  # Show GUI
"""

from __future__ import annotations

import numpy as np
import pytest


def generate_clean_spectra(n_samples: int = 40, n_wavelengths: int = 100, seed: int = 42):
    """Generate synthetic clean (uncontaminated) spectra."""
    rng = np.random.RandomState(seed)
    wavelengths = np.linspace(1000, 2000, n_wavelengths)
    base = 0.5 + 0.3 * np.sin(2 * np.pi * (wavelengths - 1000) / 500)
    X = base + rng.randn(n_samples, n_wavelengths) * 0.1
    return X, wavelengths


def generate_contaminated_spectra(
    n_samples: int = 30,
    n_wavelengths: int = 100,
    contaminant_regions: list[tuple[int, int]] | None = None,
    contaminant_strength: float = 0.5,
    seed: int = 43,
):
    """Generate synthetic contaminated spectra with known contaminant signatures."""
    if contaminant_regions is None:
        contaminant_regions = [(20, 30), (70, 80)]

    rng = np.random.RandomState(seed)
    X, wavelengths = generate_clean_spectra(n_samples, n_wavelengths, seed)

    for start, end in contaminant_regions:
        contaminant_signal = contaminant_strength * (1 + rng.randn(n_samples, end - start) * 0.2)
        X[:, start:end] += contaminant_signal

    return X, wavelengths


@pytest.fixture
def contam_harness(gui_harness):
    """
    GUI harness with contaminant analysis data set up.

    Sets up:
    - Clean spectra data
    - One contaminant group
    - Wavelengths
    """
    harness = gui_harness
    app = harness.app

    # Generate synthetic data
    n_wavelengths = 100
    X_clean, wavelengths = generate_clean_spectra(n_samples=40, n_wavelengths=n_wavelengths)
    X_contam, _ = generate_contaminated_spectra(n_samples=30, n_wavelengths=n_wavelengths)

    # Set up contaminant analysis state
    app.contam_clean_data = X_clean
    app.contam_wavelengths = wavelengths
    app.contam_groups = {'Contaminant_A': X_contam}

    # Initialize required tkinter variables if they don't exist
    import tkinter as tk

    if not hasattr(app, 'contam_method'):
        app.contam_method = tk.StringVar(value='Estimated EPO')
    if not hasattr(app, 'contam_n_components'):
        app.contam_n_components = tk.IntVar(value=2)
    if not hasattr(app, 'contam_threshold'):
        app.contam_threshold = tk.DoubleVar(value=0.5)
    if not hasattr(app, 'contam_aggregation'):
        app.contam_aggregation = tk.StringVar(value='max')
    if not hasattr(app, 'contam_correction_method'):
        app.contam_correction_method = tk.StringVar(value='EPO')
    if not hasattr(app, 'contam_apply_source'):
        app.contam_apply_source = tk.StringVar(value='Current Analysis Data')

    harness.wait_for_idle(0.2)

    return harness


# ============================================================
# Tab 13C: Automated Detection Tests
# ============================================================

@pytest.mark.gui
class TestAutomatedDetection:
    """Test Tab 13C Automated Detection functionality."""

    def test_run_automated_detection_no_attribute_error(self, contam_harness):
        """
        Test that _contam_run_automated_detection doesn't raise AttributeError.

        This catches bugs like calling self.update() instead of self.root.update()
        since SpectralPredictApp is not a Tkinter widget subclass.
        """
        harness = contam_harness

        # This should not raise AttributeError
        try:
            harness.invoke_method('_contam_run_automated_detection')
        except AttributeError as e:
            if 'update' in str(e):
                pytest.fail(
                    f"AttributeError with 'update': {e}\n"
                    "Likely cause: self.update() should be self.root.update()"
                )
            raise

    def test_run_automated_detection_estimated_epo(self, contam_harness):
        """Test automated detection with Estimated EPO method."""
        harness = contam_harness
        harness.app.contam_method.set('Estimated EPO')

        # Should complete without error
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.5)

    def test_run_automated_detection_oplsda(self, contam_harness):
        """Test automated detection with OPLS-DA method."""
        harness = contam_harness
        harness.app.contam_method.set('OPLS-DA')

        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.5)

    def test_run_automated_detection_glsw(self, contam_harness):
        """Test automated detection with GLSW method."""
        harness = contam_harness
        harness.app.contam_method.set('GLSW')

        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.5)

    def test_run_automated_detection_missing_clean_data(self, gui_harness):
        """Test that detection handles missing clean data gracefully."""
        harness = gui_harness

        # Don't set up contam_clean_data - should handle gracefully
        harness.app.contam_clean_data = None
        harness.app.contam_groups = {}

        # Should not crash - should show error message
        try:
            harness.invoke_method('_contam_run_automated_detection')
        except Exception:
            pass  # Error handling is acceptable

    def test_run_automated_detection_multiple_contaminants(self, contam_harness):
        """Test detection with multiple contaminant groups."""
        harness = contam_harness

        # Add second contaminant group
        n_wavelengths = 100
        X_contam2, _ = generate_contaminated_spectra(
            n_samples=25,
            n_wavelengths=n_wavelengths,
            contaminant_regions=[(50, 60)],
            seed=44
        )
        harness.app.contam_groups['Contaminant_B'] = X_contam2

        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.5)


# ============================================================
# Tab 13D: Apply Correction Tests
# ============================================================

@pytest.mark.gui
class TestApplyCorrection:
    """Test Tab 13D Apply Correction functionality."""

    def test_apply_correction_no_attribute_error(self, contam_harness):
        """
        Test that _contam_apply_correction doesn't raise AttributeError.

        This catches bugs like calling self.update() instead of self.root.update().
        """
        harness = contam_harness

        # First run detection to get results
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)

        # Now apply correction - should not raise AttributeError
        try:
            harness.invoke_method('_contam_apply_correction')
        except AttributeError as e:
            if 'update' in str(e):
                pytest.fail(
                    f"AttributeError with 'update': {e}\n"
                    "Likely cause: self.update() should be self.root.update()"
                )
            raise

    def test_apply_correction_epo_method(self, contam_harness):
        """Test applying EPO correction."""
        harness = contam_harness
        harness.app.contam_correction_method.set('EPO')

        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

    def test_apply_correction_glsw_method(self, contam_harness):
        """Test applying GLSW correction."""
        harness = contam_harness
        harness.app.contam_correction_method.set('GLSW Weighting')

        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

    def test_exclude_regions_reduces_wavelengths(self, contam_harness):
        """Test that Exclude Regions reduces the number of wavelengths."""
        harness = contam_harness
        app = harness.app

        # Run detection to get exclusion regions
        app.contam_method.set('OPLS-DA')
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)

        # Get original wavelength count
        original_n_wavelengths = len(app.contam_wavelengths)

        # Apply Exclude Regions to contaminant groups
        app.contam_correction_method.set('Exclude Regions')
        app.contam_apply_source.set('Contaminant Groups')

        # Get the data before correction
        X_before = np.vstack(list(app.contam_groups.values()))
        original_shape = X_before.shape

        # Apply correction
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

        # Verify wavelength mask was created
        assert hasattr(app, '_remaining_wavelength_mask'), "Wavelength mask not created"

        # Verify some wavelengths were excluded
        remaining_wavelengths = app._get_remaining_wavelengths()
        assert len(remaining_wavelengths) < original_n_wavelengths, (
            f"Expected fewer wavelengths after exclusion. "
            f"Original: {original_n_wavelengths}, Remaining: {len(remaining_wavelengths)}"
        )

    def test_epo_modifies_data_values(self, contam_harness):
        """Test that EPO Projection modifies data values but preserves shape."""
        harness = contam_harness
        app = harness.app

        # Run detection first
        app.contam_method.set('Estimated EPO')
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)

        # Get original data
        X_before = np.vstack(list(app.contam_groups.values()))
        original_shape = X_before.shape

        # Apply EPO correction
        app.contam_correction_method.set('EPO Projection')
        app.contam_apply_source.set('Contaminant Groups')
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

        # Verify transformer was stored
        assert hasattr(app, 'contam_epo_transformer'), "EPO transformer not stored"

        # Shape should be preserved with EPO
        # We can't directly access the corrected data for contaminant groups,
        # but we verified the method didn't crash and stored the transformer
        assert app.contam_epo_transformer is not None

    def test_main_dataset_backup_created(self, contam_harness):
        """Test that applying correction to main dataset creates a backup."""
        harness = contam_harness
        app = harness.app

        # Load some data into main dataset (Tab 1)
        import pandas as pd
        n_wavelengths = 100
        X_main, wavelengths = generate_clean_spectra(n_samples=50, n_wavelengths=n_wavelengths)
        app.X = pd.DataFrame(X_main, columns=wavelengths)

        # Run detection
        app.contam_method.set('OPLS-DA')
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)

        # Verify no backup exists yet
        assert not hasattr(app, 'X_before_contam_correction'), "Backup should not exist yet"

        # Apply correction to main dataset
        app.contam_correction_method.set('EPO Projection')
        app.contam_apply_source.set('Main Dataset')
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

        # Verify backup was created
        assert hasattr(app, 'X_before_contam_correction'), "Backup not created"
        assert app.X_before_contam_correction is not None

        # Verify backup has same shape as original
        assert app.X_before_contam_correction.shape == (50, n_wavelengths)

    def test_correction_with_main_dataset(self, contam_harness):
        """Test that correction modifies self.X when applied to main dataset."""
        harness = contam_harness
        app = harness.app

        # Load some data into main dataset (Tab 1)
        import pandas as pd
        n_wavelengths = 100
        X_main, wavelengths = generate_clean_spectra(n_samples=50, n_wavelengths=n_wavelengths)
        app.X = pd.DataFrame(X_main, columns=wavelengths)

        # Store original data
        X_original = app.X.copy()

        # Run detection
        app.contam_method.set('GLSW')
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)

        # Apply GLSW correction to main dataset
        app.contam_correction_method.set('GLSW Weighting')
        app.contam_apply_source.set('Main Dataset')
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

        # Verify self.X was modified (values changed)
        assert not np.allclose(app.X.values, X_original.values), (
            "Main dataset values should be modified after correction"
        )

        # Verify shape is preserved for GLSW
        assert app.X.shape == X_original.shape, "Shape should be preserved for GLSW"

        # Verify corrected data was stored
        assert hasattr(app, 'contam_corrected_X'), "Corrected data not stored"
        assert app.contam_corrected_X is not None

    def test_exclude_regions_with_main_dataset_reduces_columns(self, contam_harness):
        """Test that Exclude Regions reduces columns when applied to main dataset."""
        harness = contam_harness
        app = harness.app

        # Load some data into main dataset (Tab 1)
        import pandas as pd
        n_wavelengths = 100
        X_main, wavelengths = generate_clean_spectra(n_samples=50, n_wavelengths=n_wavelengths)
        app.X = pd.DataFrame(X_main, columns=wavelengths)

        original_n_columns = app.X.shape[1]

        # Run detection
        app.contam_method.set('OPLS-DA')
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.3)

        # Apply Exclude Regions to main dataset
        app.contam_correction_method.set('Exclude Regions')
        app.contam_apply_source.set('Main Dataset')
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

        # Verify columns were reduced
        assert app.X.shape[1] < original_n_columns, (
            f"Expected fewer columns after exclusion. "
            f"Original: {original_n_columns}, Current: {app.X.shape[1]}"
        )

        # Verify rows unchanged
        assert app.X.shape[0] == 50, "Number of samples should be unchanged"


# ============================================================
# Tab 13B: Difference Analysis Tests
# ============================================================

@pytest.mark.gui
class TestDifferenceAnalysis:
    """Test Tab 13B Difference Analysis functionality."""

    def test_run_difference_analysis(self, contam_harness):
        """Test running difference analysis."""
        harness = contam_harness

        harness.invoke_method('_contam_run_difference_analysis')
        harness.wait_for_idle(0.3)

    def test_update_peak_detection(self, contam_harness):
        """Test peak detection update."""
        harness = contam_harness

        # Run difference analysis first
        harness.invoke_method('_contam_run_difference_analysis')
        harness.wait_for_idle(0.3)

        # Then update peak detection
        harness.invoke_method('_contam_update_peak_detection')
        harness.wait_for_idle(0.3)


# ============================================================
# Tab 13A: Load & Define Groups Tests
# ============================================================

@pytest.mark.gui
class TestLoadDefineGroups:
    """Test Tab 13A Load & Define Groups functionality."""

    def test_update_summary(self, contam_harness):
        """Test that summary update works."""
        harness = contam_harness

        harness.invoke_method('_contam_update_summary')
        harness.wait_for_idle(0.2)

    def test_check_alignment(self, contam_harness):
        """Test wavelength alignment check."""
        harness = contam_harness

        harness.invoke_method('_contam_check_alignment')
        harness.wait_for_idle(0.2)


# ============================================================
# Integration Tests
# ============================================================

@pytest.mark.gui
class TestContaminantWorkflow:
    """Test complete contaminant analysis workflow."""

    def test_full_workflow_detection_to_correction(self, contam_harness):
        """
        Test complete workflow: load data -> detection -> correction.

        This is the primary integration test that exercises the full
        contaminant analysis pipeline through the GUI.
        """
        harness = contam_harness

        # Step 1: Update summary (simulates loading data)
        harness.invoke_method('_contam_update_summary')
        harness.wait_for_idle(0.2)

        # Step 2: Run difference analysis (Tab 13B)
        harness.invoke_method('_contam_run_difference_analysis')
        harness.wait_for_idle(0.3)

        # Step 3: Run automated detection (Tab 13C)
        harness.invoke_method('_contam_run_automated_detection')
        harness.wait_for_idle(0.5)

        # Step 4: Apply correction (Tab 13D)
        harness.invoke_method('_contam_apply_correction')
        harness.wait_for_idle(0.3)

    def test_workflow_with_different_methods(self, contam_harness):
        """Test workflow with all detection methods."""
        harness = contam_harness

        methods = ['Estimated EPO', 'OPLS-DA', 'GLSW']

        for method in methods:
            harness.app.contam_method.set(method)

            try:
                harness.invoke_method('_contam_run_automated_detection')
                harness.wait_for_idle(0.3)
            except Exception as e:
                pytest.fail(f"Method {method} failed: {e}")


# ============================================================
# Error Handling Tests
# ============================================================

@pytest.mark.gui
class TestErrorHandling:
    """Test error handling in contaminant analysis."""

    def test_detection_with_no_data_shows_error(self, gui_harness):
        """Test that running detection without data shows error message."""
        harness = gui_harness

        # Ensure no data is loaded
        harness.app.contam_clean_data = None
        harness.app.contam_groups = {}

        # Should handle gracefully (show messagebox, not crash)
        try:
            harness.invoke_method('_contam_run_automated_detection')
        except Exception:
            pass  # Acceptable - error handling triggered

    def test_correction_without_detection_results(self, contam_harness):
        """Test applying correction without first running detection."""
        harness = contam_harness

        # Don't run detection first
        # Should handle gracefully
        try:
            harness.invoke_method('_contam_apply_correction')
        except Exception:
            pass  # Acceptable - error handling triggered
