"""
Tests for advanced calibration transfer methods (TSR and CTAI).

Tests the newly implemented calibration transfer algorithms:
- TSR (Transfer Sample Regression / Shenk-Westerhaus)
- CTAI (Calibration Transfer based on Affine Invariance)
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from spectral_predict.calibration_transfer import (
    estimate_tsr,
    apply_tsr,
    estimate_ctai,
    apply_ctai,
    TransferModel
)


class TestTSR:
    """Test Transfer Sample Regression (TSR / Shenk-Westerhaus)."""

    def test_basic_tsr(self):
        """Test basic TSR estimation and application."""
        np.random.seed(42)

        # Generate master/slave spectra with known transformation
        n_samples, n_wavelengths = 100, 150
        X_master = np.random.randn(n_samples, n_wavelengths)

        # Slave has simple affine transformation
        true_slope = 0.95
        true_bias = 0.1
        X_slave = true_slope * X_master + true_bias

        # Select 12 transfer samples (literature-recommended)
        transfer_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])

        # Estimate TSR
        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Check parameters were estimated
        assert 'slope' in params
        assert 'bias' in params
        assert 'r_squared' in params
        assert 'mean_r_squared' in params

        assert params['slope'].shape == (n_wavelengths,)
        assert params['bias'].shape == (n_wavelengths,)

        # Should have high R² for this perfect linear case
        assert params['mean_r_squared'] > 0.95

        # Apply TSR to new data
        X_transferred = apply_tsr(X_slave, params)

        # Should closely reconstruct X_master
        rmse = np.sqrt(np.mean((X_transferred - X_master) ** 2))
        assert rmse < 0.1  # Should be very small for perfect affine transform

    def test_tsr_perfect_recovery(self):
        """Test TSR recovers perfect affine transformation."""
        np.random.seed(42)

        # Simple case: constant slope and bias
        n_samples = 50
        n_wavelengths = 100

        X_master = np.random.randn(n_samples, n_wavelengths)

        # Known transformation
        slope_true = 0.9
        bias_true = 0.2
        X_slave = slope_true * X_master + bias_true

        # Use all samples as transfer samples (ideal case)
        transfer_idx = np.arange(n_samples)

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Estimated slopes should be close to true_slope
        assert_allclose(params['slope'], slope_true, rtol=1e-2)

        # Estimated biases should be close to true_bias
        assert_allclose(params['bias'], bias_true, atol=1e-2)

        # R² should be near perfect
        assert params['mean_r_squared'] > 0.999

    def test_tsr_wavelength_dependent(self):
        """Test TSR with wavelength-dependent transformation."""
        np.random.seed(42)

        n_samples = 80
        n_wavelengths = 120

        X_master = np.random.randn(n_samples, n_wavelengths)

        # Wavelength-dependent slope and bias
        wavelength_slopes = np.linspace(0.85, 1.05, n_wavelengths)
        wavelength_biases = np.linspace(-0.1, 0.1, n_wavelengths)

        X_slave = X_master * wavelength_slopes + wavelength_biases

        transfer_idx = np.linspace(0, n_samples-1, 15, dtype=int)

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Should estimate different slopes per wavelength
        assert params['slope'].std() > 0.01  # Not all the same

        # Should still achieve good fit
        assert params['mean_r_squared'] > 0.9

        # Apply and check
        X_transferred = apply_tsr(X_slave, params)
        rmse = np.sqrt(np.mean((X_transferred - X_master) ** 2))
        assert rmse < 0.2

    def test_tsr_few_samples(self):
        """Test TSR with minimal transfer samples (12-13 recommended)."""
        np.random.seed(42)

        X_master = np.random.randn(100, 150)
        X_slave = 0.92 * X_master + 0.08

        # Use only 12 samples (literature recommendation)
        transfer_idx = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85, 90, 95, 98])

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        assert params['n_transfer_samples'] == 12

        # Should still work reasonably well
        X_transferred = apply_tsr(X_slave, params)
        rmse = np.sqrt(np.mean((X_transferred - X_master) ** 2))
        assert rmse < 0.3  # Slightly higher tolerance with fewer samples

    def test_tsr_bias_only_correction(self):
        """Test TSR with bias-only correction (slope=1)."""
        np.random.seed(42)

        X_master = np.random.randn(60, 100)
        # Only bias difference (no slope)
        X_slave = X_master + 0.15

        transfer_idx = np.arange(0, 60, 5)

        params = estimate_tsr(
            X_master, X_slave, transfer_idx,
            slope_bias_correction=False  # Bias only
        )

        # All slopes should be 1.0
        assert_allclose(params['slope'], 1.0)

        # Biases should be close to 0.15
        assert_allclose(params['bias'], 0.15, atol=1e-2)

    def test_tsr_with_regularization(self):
        """Test TSR with ridge regularization."""
        np.random.seed(42)

        X_master = np.random.randn(50, 80)
        X_slave = 0.9 * X_master + 0.1

        transfer_idx = np.array([0, 10, 20, 30, 40, 49])

        # Without regularization
        params_no_reg = estimate_tsr(X_master, X_slave, transfer_idx, regularization=0.0)

        # With regularization
        params_reg = estimate_tsr(X_master, X_slave, transfer_idx, regularization=0.01)

        # Both should work, regularized might be slightly different
        assert params_no_reg['mean_r_squared'] > 0.9
        assert params_reg['mean_r_squared'] > 0.85

    def test_tsr_quality_metrics(self):
        """Test that TSR provides quality metrics."""
        np.random.seed(42)

        X_master = np.random.randn(70, 100)
        X_slave = 0.95 * X_master + np.random.randn(70, 100) * 0.05  # Add noise

        transfer_idx = np.linspace(0, 69, 10, dtype=int)

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Check all quality metrics present
        assert 'r_squared' in params
        assert 'mean_r_squared' in params
        assert 'wavelength_quality' in params

        # R² should vary by wavelength
        assert params['r_squared'].std() > 0

        # Should indicate some wavelengths better than others
        assert params['r_squared'].min() < params['r_squared'].max()

    def test_tsr_error_dimension_mismatch(self):
        """Test TSR errors on dimension mismatch."""
        X_master = np.random.randn(50, 100)
        X_slave = np.random.randn(50, 90)  # Wrong number of wavelengths

        transfer_idx = np.array([0, 10, 20, 30, 40])

        with pytest.raises(ValueError, match="must have same shape"):
            estimate_tsr(X_master, X_slave, transfer_idx)

    def test_tsr_error_too_few_samples(self):
        """Test TSR errors with insufficient transfer samples."""
        X_master = np.random.randn(50, 100)
        X_slave = np.random.randn(50, 100)

        transfer_idx = np.array([0])  # Only 1 sample

        with pytest.raises(ValueError, match="at least 2 transfer samples"):
            estimate_tsr(X_master, X_slave, transfer_idx)

    def test_tsr_error_invalid_indices(self):
        """Test TSR errors with out-of-range indices."""
        X_master = np.random.randn(50, 100)
        X_slave = np.random.randn(50, 100)

        transfer_idx = np.array([0, 10, 20, 100])  # Index 100 out of range

        with pytest.raises(ValueError, match="only 50 samples available"):
            estimate_tsr(X_master, X_slave, transfer_idx)

    def test_apply_tsr_dimension_check(self):
        """Test apply_tsr validates dimensions."""
        np.random.seed(42)

        X_master = np.random.randn(50, 100)
        X_slave = 0.9 * X_master + 0.1

        transfer_idx = np.array([0, 10, 20, 30, 40])
        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Correct dimension
        X_new = np.random.randn(30, 100)
        X_transferred = apply_tsr(X_new, params)
        assert X_transferred.shape == (30, 100)

        # Wrong dimension
        X_wrong = np.random.randn(30, 80)
        with pytest.raises(ValueError, match="expects 100"):
            apply_tsr(X_wrong, params)


class TestCTAI:
    """Test CTAI (Calibration Transfer based on Affine Invariance)."""

    def test_basic_ctai(self):
        """Test basic CTAI estimation and application."""
        np.random.seed(42)

        # Generate master and slave datasets (DIFFERENT samples!)
        n_master, n_slave, n_wavelengths = 80, 100, 150

        X_master = np.random.randn(n_master, n_wavelengths)

        # Slave dataset with affine transformation
        X_slave_base = np.random.randn(n_slave, n_wavelengths)
        X_slave = 0.9 * X_slave_base + 0.15

        # Estimate CTAI (no transfer samples needed!)
        params = estimate_ctai(X_master, X_slave)

        # Check parameters
        assert 'M' in params
        assert 'T' in params
        assert 'n_components' in params
        assert 'explained_variance' in params
        assert 'reconstruction_error' in params

        assert params['M'].shape == (n_wavelengths, n_wavelengths)
        assert params['T'].shape == (n_wavelengths,)

        # Should have reasonable explained variance
        assert params['explained_variance'] > 0.8

    def test_ctai_no_transfer_samples_needed(self):
        """Test that CTAI works without paired samples."""
        np.random.seed(42)

        # Master: 70 samples
        X_master = np.random.randn(70, 120)

        # Slave: 90 samples (completely different!)
        X_slave = np.random.randn(90, 120)

        # Apply transformation to slave
        X_slave = 0.88 * X_slave + 0.12

        # CTAI should still work
        params = estimate_ctai(X_master, X_slave)

        assert params['M'].shape == (120, 120)
        assert params['reconstruction_error'] >= 0  # Should compute without error

    def test_ctai_component_selection(self):
        """Test CTAI automatic component selection."""
        np.random.seed(42)

        X_master = np.random.randn(60, 100)
        X_slave = np.random.randn(60, 100)

        # Auto-select components (default)
        params_auto = estimate_ctai(X_master, X_slave)

        # Manual component selection
        params_manual = estimate_ctai(X_master, X_slave, n_components=20)

        assert params_manual['n_components'] == 20
        assert params_auto['n_components'] > 0
        assert params_auto['n_components'] <= 100

    def test_ctai_explained_variance_threshold(self):
        """Test CTAI respects explained variance threshold."""
        np.random.seed(42)

        X_master = np.random.randn(80, 120)
        X_slave = np.random.randn(80, 120)

        # Higher threshold -> more components
        params_high = estimate_ctai(X_master, X_slave, explained_variance_threshold=0.999)

        # Lower threshold -> fewer components
        params_low = estimate_ctai(X_master, X_slave, explained_variance_threshold=0.95)

        assert params_high['n_components'] >= params_low['n_components']

    def test_ctai_affine_recovery(self):
        """Test CTAI approximately recovers affine transformation."""
        np.random.seed(42)

        n_samples, n_wavelengths = 100, 80

        # Master data
        X_master = np.random.randn(n_samples, n_wavelengths)

        # Slave with known affine transformation
        # Note: Same samples for this test to validate recovery
        true_M = np.eye(n_wavelengths) * 0.92  # Scaling matrix
        true_T = np.ones(n_wavelengths) * 0.08

        X_slave = X_master @ true_M + true_T

        # Estimate CTAI
        params = estimate_ctai(X_master, X_slave)

        # Apply transformation
        X_transferred = apply_ctai(X_slave, params)

        # Should closely reconstruct X_master
        rmse = np.sqrt(np.mean((X_transferred - X_master) ** 2))
        assert rmse < 0.5  # Reasonable tolerance for covariance-based method

    def test_ctai_different_sample_sizes(self):
        """Test CTAI with different master/slave sample sizes."""
        np.random.seed(42)

        n_wavelengths = 100

        # Master: 50 samples
        X_master = np.random.randn(50, n_wavelengths)

        # Slave: 150 samples
        X_slave = np.random.randn(150, n_wavelengths)

        params = estimate_ctai(X_master, X_slave)

        # Should work fine
        assert params['M'].shape == (n_wavelengths, n_wavelengths)

        # Try opposite
        params2 = estimate_ctai(X_slave, X_master)
        assert params2['M'].shape == (n_wavelengths, n_wavelengths)

    def test_ctai_reconstruction_error_metric(self):
        """Test that CTAI provides reconstruction error."""
        np.random.seed(42)

        X_master = np.random.randn(70, 100)
        X_slave = np.random.randn(70, 100)

        params = estimate_ctai(X_master, X_slave)

        assert 'reconstruction_error' in params
        assert params['reconstruction_error'] >= 0
        assert np.isfinite(params['reconstruction_error'])

    def test_apply_ctai_basic(self):
        """Test applying CTAI to new data."""
        np.random.seed(42)

        # Estimate CTAI
        X_master = np.random.randn(60, 100)
        X_slave = np.random.randn(60, 100)
        params = estimate_ctai(X_master, X_slave)

        # Apply to new slave data
        X_new_slave = np.random.randn(30, 100)
        X_transferred = apply_ctai(X_new_slave, params)

        assert X_transferred.shape == (30, 100)
        assert np.all(np.isfinite(X_transferred))

    def test_ctai_error_wavelength_mismatch(self):
        """Test CTAI errors on wavelength mismatch."""
        X_master = np.random.randn(50, 100)
        X_slave = np.random.randn(50, 90)  # Wrong wavelengths

        with pytest.raises(ValueError, match="same number of wavelengths"):
            estimate_ctai(X_master, X_slave)

    def test_ctai_error_too_few_samples(self):
        """Test CTAI errors with too few samples."""
        X_master = np.random.randn(1, 100)  # Only 1 sample
        X_slave = np.random.randn(50, 100)

        with pytest.raises(ValueError, match="at least 2 samples"):
            estimate_ctai(X_master, X_slave)

    def test_apply_ctai_dimension_check(self):
        """Test apply_ctai validates dimensions."""
        np.random.seed(42)

        X_master = np.random.randn(50, 100)
        X_slave = np.random.randn(50, 100)
        params = estimate_ctai(X_master, X_slave)

        # Correct dimension
        X_new = np.random.randn(30, 100)
        X_transferred = apply_ctai(X_new, params)
        assert X_transferred.shape == (30, 100)

        # Wrong dimension
        X_wrong = np.random.randn(30, 80)
        with pytest.raises(ValueError, match="expects 100"):
            apply_ctai(X_wrong, params)


class TestTransferModelIntegration:
    """Test integration of new methods with TransferModel."""

    def test_transfer_model_tsr(self):
        """Test creating TransferModel with TSR."""
        np.random.seed(42)

        X_master = np.random.randn(50, 100)
        X_slave = 0.9 * X_master + 0.1

        transfer_idx = np.array([0, 10, 20, 30, 40])
        wavelengths_common = np.arange(100)

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Create TransferModel
        model = TransferModel(
            master_id="Instrument_A",
            slave_id="Instrument_B",
            method="tsr",
            wavelengths_common=wavelengths_common,
            params=params
        )

        assert model.method == "tsr"
        assert 'slope' in model.params
        assert 'bias' in model.params

    def test_transfer_model_ctai(self):
        """Test creating TransferModel with CTAI."""
        np.random.seed(42)

        X_master = np.random.randn(60, 100)
        X_slave = np.random.randn(60, 100)

        wavelengths_common = np.arange(100)

        params = estimate_ctai(X_master, X_slave)

        # Create TransferModel
        model = TransferModel(
            master_id="Instrument_A",
            slave_id="Instrument_B",
            method="ctai",
            wavelengths_common=wavelengths_common,
            params=params
        )

        assert model.method == "ctai"
        assert 'M' in model.params
        assert 'T' in model.params


class TestMethodComparison:
    """Compare TSR and CTAI performance."""

    def test_tsr_vs_ctai_simple_affine(self):
        """Compare TSR and CTAI on simple affine transformation."""
        np.random.seed(42)

        # Same samples for both methods (fair comparison)
        n_samples, n_wavelengths = 100, 150
        X_master = np.random.randn(n_samples, n_wavelengths)

        # Simple affine transformation
        X_slave = 0.92 * X_master + 0.08

        # TSR with 12 samples
        transfer_idx = np.linspace(0, n_samples-1, 12, dtype=int)
        tsr_params = estimate_tsr(X_master, X_slave, transfer_idx)
        X_tsr = apply_tsr(X_slave, tsr_params)
        rmse_tsr = np.sqrt(np.mean((X_tsr - X_master) ** 2))

        # CTAI (no transfer samples)
        ctai_params = estimate_ctai(X_master, X_slave)
        X_ctai = apply_ctai(X_slave, ctai_params)
        rmse_ctai = np.sqrt(np.mean((X_ctai - X_master) ** 2))

        # Both should work well
        assert rmse_tsr < 0.2
        assert rmse_ctai < 0.5  # CTAI uses covariance, slightly less precise

        # TSR should be better with perfect samples
        # (This may not always hold, depends on data structure)
        print(f"\nTSR RMSE: {rmse_tsr:.6f}")
        print(f"CTAI RMSE: {rmse_ctai:.6f}")


class TestEdgeCases:
    """Test edge cases for both methods."""

    def test_identical_spectra(self):
        """Test when master and slave are identical."""
        np.random.seed(42)

        X = np.random.randn(50, 100)

        # TSR
        transfer_idx = np.array([0, 10, 20, 30, 40])
        tsr_params = estimate_tsr(X, X, transfer_idx)

        # Slopes should be ~1, biases ~0
        assert_allclose(tsr_params['slope'], 1.0, rtol=1e-2)
        assert_allclose(tsr_params['bias'], 0.0, atol=1e-2)

        # CTAI
        ctai_params = estimate_ctai(X, X)

        # Transformation should be near-identity
        X_ctai = apply_ctai(X, ctai_params)
        assert_allclose(X_ctai, X, rtol=0.1, atol=0.1)

    def test_high_dimensional_case(self):
        """Test with more features than samples."""
        np.random.seed(42)

        n_samples = 30
        n_wavelengths = 200  # More wavelengths than samples

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 0.9 * X_master + 0.1

        # TSR should still work
        transfer_idx = np.array([0, 5, 10, 15, 20, 25])
        tsr_params = estimate_tsr(X_master, X_slave, transfer_idx)
        assert tsr_params['slope'].shape == (n_wavelengths,)

        # CTAI should handle with regularization
        ctai_params = estimate_ctai(X_master, X_slave)
        assert ctai_params['M'].shape == (n_wavelengths, n_wavelengths)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
