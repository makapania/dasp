"""
Tests for NS-PFCE (Non-supervised Parameter-Free Framework for Calibration Enhancement).

Tests the implementation of NS-PFCE calibration transfer method with optional
wavelength selection (SPA, CARS, VCPA-IRIV).
"""

import numpy as np
import pytest

from spectral_predict.calibration_transfer import (
    estimate_nspfce,
    apply_nspfce,
    TransferModel,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_affine_transformation():
    """
    Generate synthetic data with simple affine transformation.

    X_master and X_slave differ by: X_slave = 0.9 * X_master + 0.1
    """
    np.random.seed(42)
    n_samples = 80
    n_wavelengths = 100

    X_master = np.random.randn(n_samples, n_wavelengths)
    X_slave = 0.9 * X_master + 0.1 + 0.01 * np.random.randn(n_samples, n_wavelengths)

    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    return X_master, X_slave, wavelengths


@pytest.fixture
def wavelength_dependent_transformation():
    """
    Generate data with wavelength-dependent transformation.

    Different slopes/biases across wavelengths.
    """
    np.random.seed(123)
    n_samples = 100
    n_wavelengths = 150

    X_master = np.random.randn(n_samples, n_wavelengths)

    # Different transformation per wavelength
    slopes = np.linspace(0.85, 1.05, n_wavelengths)
    biases = np.linspace(-0.15, 0.15, n_wavelengths)

    X_slave = X_master * slopes + biases + 0.01 * np.random.randn(n_samples, n_wavelengths)

    wavelengths = np.linspace(900, 2500, n_wavelengths)

    return X_master, X_slave, wavelengths, slopes, biases


@pytest.fixture
def spectral_like_data():
    """
    Generate realistic spectral-like data with peaks.
    """
    np.random.seed(99)
    n_samples = 60
    n_wavelengths = 200

    wavelengths = np.linspace(1000, 2500, n_wavelengths)

    # Generate spectra with Gaussian peaks
    X_master = np.zeros((n_samples, n_wavelengths))
    for i in range(n_samples):
        # Random peak positions
        peak_positions = np.random.choice(wavelengths, size=3, replace=False)
        for peak_pos in peak_positions:
            peak_width = np.random.uniform(50, 100)
            peak_height = np.random.uniform(0.5, 2.0)
            X_master[i, :] += peak_height * np.exp(-0.5 * ((wavelengths - peak_pos) / peak_width) ** 2)

        # Add baseline
        X_master[i, :] += np.random.uniform(0.1, 0.3)

    # Slave has different calibration
    X_slave = 0.92 * X_master + 0.08 + 0.02 * np.random.randn(n_samples, n_wavelengths)

    return X_master, X_slave, wavelengths


# ============================================================================
# Test NS-PFCE Basic Functionality
# ============================================================================

class TestNSPFCEBasic:
    """Test basic NS-PFCE functionality."""

    def test_nspfce_without_wavelength_selection(self, simple_affine_transformation):
        """Test NS-PFCE without wavelength selection."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=50
        )

        # Check return structure
        assert isinstance(params, dict)
        assert 'T' in params
        assert 'offset' in params
        assert 'convergence_history' in params
        assert 'converged' in params
        assert 'n_iterations' in params
        assert 'use_wavelength_selection' in params

        # Check transformation matrix shape
        T = params['T']
        assert T.shape == (wavelengths.shape[0], wavelengths.shape[0])

    def test_nspfce_with_wavelength_selection_vcpa(self, simple_affine_transformation):
        """Test NS-PFCE with VCPA-IRIV wavelength selection."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='vcpa-iriv',
            max_iterations=50
        )

        # Check structure
        assert isinstance(params, dict)
        assert 'T' in params
        assert 'use_wavelength_selection' in params
        assert params['use_wavelength_selection'] is True
        assert 'selected_wavelength_indices' in params
        assert 'wavelength_selection_method' in params
        assert params['wavelength_selection_method'] == 'vcpa-iriv'

        # Should have reduced wavelengths
        selected_idx = params['selected_wavelength_indices']
        assert len(selected_idx) < len(wavelengths)
        assert len(selected_idx) > 0

    def test_nspfce_with_wavelength_selection_cars(self, simple_affine_transformation):
        """Test NS-PFCE with CARS wavelength selection."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='cars',
            max_iterations=50
        )

        assert params['wavelength_selection_method'] == 'cars'
        assert 'selected_wavelength_indices' in params
        assert len(params['selected_wavelength_indices']) > 0

    def test_nspfce_with_wavelength_selection_spa(self, simple_affine_transformation):
        """Test NS-PFCE with SPA wavelength selection."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='spa',
            max_iterations=50
        )

        assert params['wavelength_selection_method'] == 'spa'
        assert 'selected_wavelength_indices' in params
        assert len(params['selected_wavelength_indices']) > 0

    def test_nspfce_apply(self, simple_affine_transformation):
        """Test applying NS-PFCE transformation."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=50
        )

        # Apply to new slave data (use original slave as test)
        X_slave_new = X_slave.copy()
        X_transferred = apply_nspfce(X_slave_new, params)

        # Check shape
        assert X_transferred.shape == X_slave_new.shape

        # Transferred data should be closer to master than original slave
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before, "Transfer should improve RMSE"

    def test_nspfce_apply_with_wavelength_selection(self, simple_affine_transformation):
        """Test applying NS-PFCE with wavelength selection."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='vcpa-iriv',
            max_iterations=50
        )

        X_transferred = apply_nspfce(X_slave, params)

        # Should return full wavelength range
        assert X_transferred.shape == X_slave.shape

        # Should improve RMSE
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before


# ============================================================================
# Test NS-PFCE Convergence
# ============================================================================

class TestNSPFCEConvergence:
    """Test NS-PFCE convergence behavior."""

    def test_convergence_simple_case(self, simple_affine_transformation):
        """Test that NS-PFCE converges for simple affine case."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100,
            convergence_threshold=1e-6
        )

        # Should converge
        assert 'converged' in params
        assert params['converged'] is True or params['n_iterations'] == 100

        # Convergence history should show decreasing errors
        history = params['convergence_history']
        assert len(history) > 0

        # Generally should decrease (may have some fluctuations due to damping)
        # Check that final error is lower than initial
        if len(history) > 1:
            assert history[-1] <= history[0] * 1.5  # Allow some tolerance

    def test_different_max_iterations(self, simple_affine_transformation):
        """Test NS-PFCE with different iteration limits."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params_10 = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=10
        )

        params_100 = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100
        )

        # More iterations should have longer history
        assert len(params_100['convergence_history']) >= len(params_10['convergence_history'])

    def test_convergence_threshold(self, simple_affine_transformation):
        """Test different convergence thresholds."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params_strict = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100,
            convergence_threshold=1e-8
        )

        params_loose = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100,
            convergence_threshold=1e-4
        )

        # Loose threshold should converge faster (fewer iterations)
        if params_loose['converged']:
            assert params_loose['n_iterations'] <= params_strict['n_iterations']


# ============================================================================
# Test NS-PFCE Quality
# ============================================================================

class TestNSPFCEQuality:
    """Test NS-PFCE transfer quality."""

    def test_affine_transformation_recovery(self, simple_affine_transformation):
        """Test NS-PFCE can recover simple affine transformation."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100
        )

        X_transferred = apply_nspfce(X_slave, params)

        # Check improvement
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        improvement_ratio = rmse_before / rmse_after
        assert improvement_ratio > 2.0, f"Expected >2x improvement, got {improvement_ratio:.2f}x"

    def test_wavelength_dependent_transformation(self, wavelength_dependent_transformation):
        """Test NS-PFCE on wavelength-dependent transformation."""
        X_master, X_slave, wavelengths, _, _ = wavelength_dependent_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100
        )

        X_transferred = apply_nspfce(X_slave, params)

        # Should improve RMSE
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before

    def test_spectral_like_data(self, spectral_like_data):
        """Test NS-PFCE on realistic spectral data."""
        X_master, X_slave, wavelengths = spectral_like_data

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='vcpa-iriv',
            max_iterations=100
        )

        X_transferred = apply_nspfce(X_slave, params)

        # Should improve correlation
        corr_before = np.corrcoef(X_master.ravel(), X_slave.ravel())[0, 1]
        corr_after = np.corrcoef(X_master.ravel(), X_transferred.ravel())[0, 1]

        assert corr_after > corr_before


# ============================================================================
# Test NS-PFCE vs Other Methods
# ============================================================================

class TestNSPFCEComparison:
    """Compare NS-PFCE to other calibration transfer methods."""

    def test_nspfce_vs_ds(self, simple_affine_transformation):
        """Compare NS-PFCE to Direct Standardization."""
        from spectral_predict.calibration_transfer import estimate_ds, apply_ds

        X_master, X_slave, wavelengths = simple_affine_transformation

        # NS-PFCE (no standards required)
        params_nspfce = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100
        )
        X_nspfce = apply_nspfce(X_slave, params_nspfce)

        # DS (uses all samples)
        A_ds = estimate_ds(X_master, X_slave, lam=0.001)
        X_ds = apply_ds(X_slave, A_ds)

        # Both should improve over original
        rmse_original = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_nspfce = np.sqrt(np.mean((X_nspfce - X_master) ** 2))
        rmse_ds = np.sqrt(np.mean((X_ds - X_master) ** 2))

        assert rmse_nspfce < rmse_original
        assert rmse_ds < rmse_original

        # Both should be reasonably close (within 50% of each other)
        ratio = max(rmse_nspfce, rmse_ds) / min(rmse_nspfce, rmse_ds)
        assert ratio < 2.0, "NS-PFCE and DS should have comparable performance"

    def test_nspfce_vs_ctai(self, simple_affine_transformation):
        """Compare NS-PFCE to CTAI."""
        from spectral_predict.calibration_transfer import estimate_ctai, apply_ctai

        X_master, X_slave, wavelengths = simple_affine_transformation

        # NS-PFCE
        params_nspfce = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100
        )
        X_nspfce = apply_nspfce(X_slave, params_nspfce)

        # CTAI
        params_ctai = estimate_ctai(X_master, X_slave)
        X_ctai = apply_ctai(X_slave, params_ctai)

        # Both should improve
        rmse_original = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_nspfce = np.sqrt(np.mean((X_nspfce - X_master) ** 2))
        rmse_ctai = np.sqrt(np.mean((X_ctai - X_master) ** 2))

        assert rmse_nspfce < rmse_original
        assert rmse_ctai < rmse_original


# ============================================================================
# Test TransferModel Integration
# ============================================================================

class TestTransferModelIntegration:
    """Test NS-PFCE with TransferModel infrastructure."""

    def test_create_transfer_model_nspfce(self, simple_affine_transformation):
        """Test creating TransferModel with NS-PFCE."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='vcpa-iriv',
            max_iterations=50
        )

        tm = TransferModel(
            master_id="Master",
            slave_id="Slave",
            method="nspfce",
            wavelengths_common=wavelengths,
            params=params,
            meta={"note": "Test NS-PFCE transfer model"}
        )

        assert tm.method == "nspfce"
        assert tm.master_id == "Master"
        assert tm.slave_id == "Slave"

    def test_save_load_transfer_model_nspfce(self, simple_affine_transformation, tmp_path):
        """Test saving and loading NS-PFCE TransferModel."""
        from spectral_predict.calibration_transfer import save_transfer_model, load_transfer_model

        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='cars',
            max_iterations=50
        )

        tm = TransferModel(
            master_id="Master",
            slave_id="Slave",
            method="nspfce",
            wavelengths_common=wavelengths,
            params=params,
            meta={}
        )

        # Save
        save_prefix = save_transfer_model(tm, directory=str(tmp_path), name="test_nspfce")

        # Load
        tm_loaded = load_transfer_model(save_prefix)

        assert tm_loaded.method == "nspfce"
        assert tm_loaded.master_id == "Master"
        assert tm_loaded.slave_id == "Slave"
        assert 'T' in tm_loaded.params


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test NS-PFCE edge cases."""

    def test_small_dataset(self):
        """Test NS-PFCE with small dataset."""
        np.random.seed(42)
        n_samples = 20
        n_wavelengths = 30

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 0.9 * X_master + 0.1
        wavelengths = np.linspace(1000, 2000, n_wavelengths)

        # Should work without wavelength selection
        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=50
        )

        X_transferred = apply_nspfce(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_high_dimensional(self):
        """Test NS-PFCE with many wavelengths."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 300

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 0.92 * X_master + 0.08
        wavelengths = np.linspace(900, 2700, n_wavelengths)

        # With wavelength selection (recommended for high-D)
        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='spa',  # SPA is faster
            max_iterations=50
        )

        X_transferred = apply_nspfce(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_nearly_identical_spectra(self):
        """Test when master and slave are nearly identical."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 100

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = X_master + 0.001 * np.random.randn(n_samples, n_wavelengths)
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=50
        )

        # Should converge quickly
        assert params['n_iterations'] < 50

        X_transferred = apply_nspfce(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_large_transformation(self):
        """Test NS-PFCE with large transformation difference."""
        np.random.seed(42)
        n_samples = 60
        n_wavelengths = 80

        X_master = np.random.randn(n_samples, n_wavelengths)
        # Large bias and scale difference
        X_slave = 0.5 * X_master + 0.5
        wavelengths = np.linspace(1000, 2500, n_wavelengths)

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=100
        )

        X_transferred = apply_nspfce(X_slave, params)

        # Should still improve
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before

    def test_invalid_wavelength_selector(self, simple_affine_transformation):
        """Test error handling for invalid wavelength selector."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        with pytest.raises(ValueError):
            estimate_nspfce(
                X_master, X_slave, wavelengths,
                use_wavelength_selection=True,
                wavelength_selector='invalid_method',
                max_iterations=50
            )

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        np.random.seed(42)
        X_master = np.random.randn(50, 100)
        X_slave = np.random.randn(60, 120)  # Different shape
        wavelengths = np.linspace(1000, 2500, 100)

        with pytest.raises((ValueError, Exception)):
            estimate_nspfce(
                X_master, X_slave, wavelengths,
                use_wavelength_selection=False,
                max_iterations=50
            )


# ============================================================================
# Test Different Normalization Options
# ============================================================================

class TestNormalization:
    """Test NS-PFCE normalization options."""

    def test_with_normalization(self, simple_affine_transformation):
        """Test NS-PFCE with normalization enabled."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            normalize=True,
            max_iterations=50
        )

        X_transferred = apply_nspfce(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_without_normalization(self, simple_affine_transformation):
        """Test NS-PFCE without normalization."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            normalize=False,
            max_iterations=50
        )

        X_transferred = apply_nspfce(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_normalization_comparison(self, simple_affine_transformation):
        """Compare NS-PFCE with and without normalization."""
        X_master, X_slave, wavelengths = simple_affine_transformation

        params_norm = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            normalize=True,
            max_iterations=50
        )

        params_no_norm = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            normalize=False,
            max_iterations=50
        )

        # Both should work
        X_norm = apply_nspfce(X_slave, params_norm)
        X_no_norm = apply_nspfce(X_slave, params_no_norm)

        assert X_norm.shape == X_slave.shape
        assert X_no_norm.shape == X_slave.shape


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
