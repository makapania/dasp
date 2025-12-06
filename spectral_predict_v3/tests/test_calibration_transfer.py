"""
Comprehensive tests for calibration transfer methods.

Tests all transfer methods (DS, PDS, TSR, CTAI, NS-PFCE, JYPLS-inv) with:
- Synthetic instrument shift
- Varying window sizes (PDS)
- Varying numbers of transfer samples
- Mismatched wavelength ranges
- Edge cases (identical instruments, extreme shifts)
- Transfer quality metrics
"""

import pytest
import numpy as np
from pathlib import Path

from spectral_predict_v3.core.calibration_transfer import (
    estimate_ds, apply_ds,
    estimate_pds, apply_pds,
    estimate_tsr, apply_tsr,
    estimate_ctai, apply_ctai,
    estimate_nspfce, apply_nspfce,
    estimate_jypls_inv, apply_jypls_inv,
    resample_to_grid,
    TransferModel,
    save_transfer_model,
    load_transfer_model,
)


@pytest.fixture
def synthetic_instruments():
    """Generate synthetic master/slave instrument pair."""
    np.random.seed(42)

    n_samples = 50
    n_wavelengths = 100
    wavelengths = np.linspace(1000, 2000, n_wavelengths)

    # Master instrument
    X_master = np.random.randn(n_samples, n_wavelengths) * 0.5 + 1.0

    # Slave instrument with bias and scale shift
    bias = 0.1
    scale = 0.95
    X_slave = scale * X_master + bias + np.random.randn(n_samples, n_wavelengths) * 0.05

    return wavelengths, X_master, X_slave


@pytest.fixture
def transfer_indices():
    """Get transfer sample indices."""
    return np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])


class TestDS:
    """Tests for Direct Standardization."""

    def test_ds_basic(self, synthetic_instruments, transfer_indices):
        """Test basic DS estimation and application."""
        wavelengths, X_master, X_slave = synthetic_instruments

        # Use transfer samples
        X_master_transfer = X_master[transfer_indices]
        X_slave_transfer = X_slave[transfer_indices]

        # Estimate DS matrix
        A = estimate_ds(X_master_transfer, X_slave_transfer)

        assert A.shape == (X_master.shape[1], X_master.shape[1])

        # Apply to new data
        X_test_slave = X_slave[30:40]
        X_transferred = apply_ds(X_test_slave, A)

        assert X_transferred.shape == X_test_slave.shape

        # Should improve alignment
        rmse_before = np.sqrt(np.mean((X_test_slave - X_master[30:40]) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master[30:40]) ** 2))

        assert rmse_after < rmse_before

    def test_ds_with_regularization(self, synthetic_instruments, transfer_indices):
        """Test DS with ridge regularization."""
        wavelengths, X_master, X_slave = synthetic_instruments

        X_master_transfer = X_master[transfer_indices]
        X_slave_transfer = X_slave[transfer_indices]

        A = estimate_ds(X_master_transfer, X_slave_transfer, lam=0.1)

        assert A.shape == (X_master.shape[1], X_master.shape[1])
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))


class TestPDS:
    """Tests for Piecewise Direct Standardization."""

    @pytest.mark.parametrize("window", [5, 11, 21])
    def test_pds_varying_windows(self, synthetic_instruments, transfer_indices, window):
        """Test PDS with different window sizes."""
        wavelengths, X_master, X_slave = synthetic_instruments

        X_master_transfer = X_master[transfer_indices]
        X_slave_transfer = X_slave[transfer_indices]

        B = estimate_pds(X_master_transfer, X_slave_transfer, window=window)

        assert B.shape == (X_master.shape[1], window)

        X_test_slave = X_slave[30:40]
        X_transferred = apply_pds(X_test_slave, B, window=window)

        assert X_transferred.shape == X_test_slave.shape

        # Should improve alignment
        rmse_before = np.sqrt(np.mean((X_test_slave - X_master[30:40]) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master[30:40]) ** 2))

        assert rmse_after < rmse_before

    def test_pds_edge_wavelengths(self, synthetic_instruments, transfer_indices):
        """Test PDS handles edge wavelengths correctly."""
        wavelengths, X_master, X_slave = synthetic_instruments

        X_master_transfer = X_master[transfer_indices]
        X_slave_transfer = X_slave[transfer_indices]

        B = estimate_pds(X_master_transfer, X_slave_transfer, window=11)

        # Check first and last wavelengths have valid coefficients
        assert np.any(B[0, :] != 0)
        assert np.any(B[-1, :] != 0)


class TestTSR:
    """Tests for Transfer Sample Regression."""

    @pytest.mark.parametrize("n_transfer", [5, 10, 20])
    def test_tsr_varying_transfer_samples(self, synthetic_instruments, n_transfer):
        """Test TSR with different numbers of transfer samples."""
        wavelengths, X_master, X_slave = synthetic_instruments

        transfer_idx = np.linspace(0, X_master.shape[0] - 1, n_transfer, dtype=int)

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        assert 'slope' in params
        assert 'bias' in params
        assert 'r_squared' in params
        assert 'mean_r_squared' in params

        assert params['slope'].shape == (X_master.shape[1],)
        assert params['bias'].shape == (X_master.shape[1],)

        # Apply transfer
        X_test_slave = X_slave[30:40]
        X_transferred = apply_tsr(X_test_slave, params)

        assert X_transferred.shape == X_test_slave.shape

        # Quality metric should be reasonable
        assert params['mean_r_squared'] > 0.5

    def test_tsr_slope_bias_correction(self, synthetic_instruments, transfer_indices):
        """Test TSR with slope and bias correction."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_tsr(X_master, X_slave, transfer_indices, slope_bias_correction=True)

        # Slopes should be close to 0.95 (known scale factor)
        assert 0.85 < np.mean(params['slope']) < 1.05

    def test_tsr_bias_only_correction(self, synthetic_instruments, transfer_indices):
        """Test TSR with bias-only correction."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_tsr(X_master, X_slave, transfer_indices, slope_bias_correction=False)

        # All slopes should be 1.0
        assert np.all(params['slope'] == 1.0)
        assert params['slope_bias_correction'] is False


class TestCTAI:
    """Tests for Calibration Transfer based on Affine Invariance."""

    def test_ctai_basic(self, synthetic_instruments):
        """Test basic CTAI estimation and application."""
        wavelengths, X_master, X_slave = synthetic_instruments

        # CTAI requires paired samples
        params = estimate_ctai(X_master, X_slave)

        assert 'M' in params
        assert 'T' in params
        assert 'n_components' in params
        assert 'explained_variance' in params
        assert 'reconstruction_error' in params

        # Apply transfer
        X_transferred = apply_ctai(X_slave, params)

        assert X_transferred.shape == X_slave.shape

        # Should reduce error
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before

    def test_ctai_component_selection(self, synthetic_instruments):
        """Test CTAI with different numbers of components."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_ctai(X_master, X_slave, n_components=10)

        assert params['n_components'] == 10

    def test_ctai_unpaired_samples_fails(self, synthetic_instruments):
        """Test CTAI fails gracefully with unpaired samples."""
        wavelengths, X_master, X_slave = synthetic_instruments

        # Use different number of samples
        X_master_subset = X_master[:30]

        with pytest.raises(ValueError, match="paired samples"):
            estimate_ctai(X_master_subset, X_slave)


class TestNSPFCE:
    """Tests for NS-PFCE."""

    def test_nspfce_without_wavelength_selection(self, synthetic_instruments):
        """Test NS-PFCE without wavelength selection."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=20
        )

        assert 'transformation_matrix' in params
        assert 'offset' in params
        assert 'convergence_iterations' in params
        assert 'final_objective' in params

        # Apply transfer
        X_transferred = apply_nspfce(X_slave, params)

        assert X_transferred.shape == X_slave.shape

    def test_nspfce_with_vcpa_iriv(self, synthetic_instruments):
        """Test NS-PFCE with VCPA-IRIV wavelength selection."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=True,
            wavelength_selector='vcpa-iriv',
            max_iterations=10
        )

        assert 'selected_wavelengths' in params
        assert params['use_wavelength_selection'] is True

        # Should have selected some wavelengths
        if params['selected_wavelengths'] is not None:
            assert len(params['selected_wavelengths']) < len(wavelengths)

    def test_nspfce_convergence(self, synthetic_instruments):
        """Test NS-PFCE convergence behavior."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_nspfce(
            X_master, X_slave, wavelengths,
            use_wavelength_selection=False,
            max_iterations=50,
            convergence_threshold=1e-6
        )

        # Should converge before max iterations
        assert params['convergence_iterations'] < 50
        assert len(params['objective_history']) > 0


class TestJYPLSInv:
    """Tests for JYPLS-inv."""

    def test_jypls_inv_basic(self, synthetic_instruments, transfer_indices):
        """Test basic JYPLS-inv estimation and application."""
        wavelengths, X_master, X_slave = synthetic_instruments

        # Generate reference values
        y_reference = 2.0 * X_master[:, 25] - X_master[:, 75] + np.random.randn(X_master.shape[0]) * 0.1
        y_transfer = y_reference[transfer_indices]

        params = estimate_jypls_inv(X_master, X_slave, y_transfer, transfer_indices, n_components=5)

        assert 'transformation_matrix' in params
        assert 'n_components' in params
        assert params['n_components'] == 5

        # Apply transfer
        X_transferred = apply_jypls_inv(X_slave, params)

        assert X_transferred.shape == X_slave.shape

    def test_jypls_inv_auto_components(self, synthetic_instruments, transfer_indices):
        """Test JYPLS-inv with automatic component selection."""
        wavelengths, X_master, X_slave = synthetic_instruments

        y_reference = 2.0 * X_master[:, 25] - X_master[:, 75] + np.random.randn(X_master.shape[0]) * 0.1
        y_transfer = y_reference[transfer_indices]

        params = estimate_jypls_inv(X_master, X_slave, y_transfer, transfer_indices)

        assert params['n_components'] > 0
        assert params['n_components'] <= len(transfer_indices) - 1


class TestResample:
    """Tests for wavelength resampling."""

    def test_resample_to_grid(self):
        """Test wavelength resampling."""
        wl_src = np.linspace(1000, 2000, 100)
        wl_target = np.linspace(1100, 1900, 80)

        X = np.random.randn(10, 100)

        X_resampled = resample_to_grid(X, wl_src, wl_target)

        assert X_resampled.shape == (10, 80)

    def test_resample_extrapolation(self):
        """Test resampling with extrapolation."""
        wl_src = np.linspace(1000, 2000, 100)
        wl_target = np.linspace(900, 2100, 120)

        X = np.random.randn(10, 100)

        X_resampled = resample_to_grid(X, wl_src, wl_target)

        assert X_resampled.shape == (10, 120)
        assert not np.any(np.isnan(X_resampled))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_identical_instruments(self):
        """Test transfer between identical instruments."""
        np.random.seed(42)
        n_samples, n_wavelengths = 50, 100

        X = np.random.randn(n_samples, n_wavelengths)
        transfer_idx = np.array([0, 10, 20, 30, 40])

        # DS should produce near-identity matrix
        A = estimate_ds(X[transfer_idx], X[transfer_idx])
        assert np.allclose(A, np.eye(n_wavelengths), atol=0.1)

    def test_extreme_shift(self):
        """Test transfer with extreme instrumental shift."""
        np.random.seed(42)
        n_samples, n_wavelengths = 50, 100

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 2.0 * X_master + 1.0  # Large shift

        transfer_idx = np.array([0, 10, 20, 30, 40])

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        # Should still produce valid parameters
        assert not np.any(np.isnan(params['slope']))
        assert not np.any(np.isnan(params['bias']))

    def test_minimal_transfer_samples(self):
        """Test with minimum number of transfer samples."""
        np.random.seed(42)
        n_samples, n_wavelengths = 50, 100

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 0.95 * X_master + 0.1

        # Only 2 transfer samples (minimum)
        transfer_idx = np.array([0, 25])

        params = estimate_tsr(X_master, X_slave, transfer_idx)

        assert params is not None
        assert params['n_transfer_samples'] == 2


class TestTransferModel:
    """Tests for TransferModel dataclass and save/load."""

    def test_save_load_ds_model(self, synthetic_instruments, transfer_indices, tmp_path):
        """Test saving and loading DS transfer model."""
        wavelengths, X_master, X_slave = synthetic_instruments

        A = estimate_ds(X_master[transfer_indices], X_slave[transfer_indices])

        model = TransferModel(
            master_id="master_inst",
            slave_id="slave_inst",
            method="ds",
            wavelengths_common=wavelengths,
            params={'A': A},
            meta={'test': 'metadata'}
        )

        # Save
        path_prefix = save_transfer_model(model, tmp_path)

        # Load
        loaded_model = load_transfer_model(path_prefix)

        assert loaded_model.master_id == model.master_id
        assert loaded_model.slave_id == model.slave_id
        assert loaded_model.method == model.method
        assert np.allclose(loaded_model.wavelengths_common, model.wavelengths_common)
        assert np.allclose(loaded_model.params['A'], model.params['A'])

    def test_save_load_tsr_model(self, synthetic_instruments, transfer_indices, tmp_path):
        """Test saving and loading TSR transfer model."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_tsr(X_master, X_slave, transfer_indices)

        model = TransferModel(
            master_id="master",
            slave_id="slave",
            method="tsr",
            wavelengths_common=wavelengths,
            params=params
        )

        path_prefix = save_transfer_model(model, tmp_path)
        loaded_model = load_transfer_model(path_prefix)

        assert loaded_model.method == "tsr"
        assert np.allclose(loaded_model.params['slope'], params['slope'])
        assert np.allclose(loaded_model.params['bias'], params['bias'])


class TestQualityMetrics:
    """Tests for transfer quality assessment."""

    def test_tsr_quality_metrics(self, synthetic_instruments, transfer_indices):
        """Test TSR produces quality metrics."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_tsr(X_master, X_slave, transfer_indices)

        assert 'mean_r_squared' in params
        assert 'r_squared' in params
        assert 0 <= params['mean_r_squared'] <= 1

    def test_ctai_quality_metrics(self, synthetic_instruments):
        """Test CTAI produces quality metrics."""
        wavelengths, X_master, X_slave = synthetic_instruments

        params = estimate_ctai(X_master, X_slave)

        assert 'reconstruction_error' in params
        assert 'explained_variance' in params
        assert params['reconstruction_error'] >= 0
        assert 0 <= params['explained_variance'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
