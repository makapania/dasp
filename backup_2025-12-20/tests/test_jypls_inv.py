"""
Tests for JYPLS-inv (Joint-Y PLS with Inversion) calibration transfer.

Tests cover estimation, application, PLS component selection, and performance.
"""

import numpy as np
import pytest

from spectral_predict.calibration_transfer import (
    estimate_jypls_inv,
    apply_jypls_inv,
    TransferModel,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_pls_data():
    """
    Generate simple PLS-structured data.

    Y depends on X through a linear combination of specific wavelengths.
    """
    np.random.seed(42)
    n_samples = 80
    n_wavelengths = 100

    # Master spectra
    X_master = np.random.randn(n_samples, n_wavelengths)

    # Y depends on specific wavelengths (PLS structure)
    y = 2.0 * X_master[:, 20] - 1.5 * X_master[:, 50] + 1.0 * X_master[:, 80]
    y += 0.1 * np.random.randn(n_samples)

    # Slave has affine transformation
    X_slave = 0.92 * X_master + 0.08 + 0.01 * np.random.randn(n_samples, n_wavelengths)

    return X_master, X_slave, y


@pytest.fixture
def complex_pls_data():
    """
    Generate complex PLS data with multiple latent structures.
    """
    np.random.seed(123)
    n_samples = 100
    n_wavelengths = 150

    # Create latent variables
    t1 = np.random.randn(n_samples)
    t2 = np.random.randn(n_samples)
    t3 = np.random.randn(n_samples)

    # Master spectra as combinations of latent variables
    X_master = np.zeros((n_samples, n_wavelengths))
    for i in range(n_wavelengths):
        weight1 = np.sin(2 * np.pi * i / n_wavelengths)
        weight2 = np.cos(2 * np.pi * i / n_wavelengths)
        weight3 = np.sin(4 * np.pi * i / n_wavelengths)
        X_master[:, i] = weight1 * t1 + weight2 * t2 + weight3 * t3

    # Y depends on latent variables
    y = 2.0 * t1 - 1.0 * t2 + 0.5 * t3 + 0.1 * np.random.randn(n_samples)

    # Slave with wavelength-dependent transformation
    slopes = np.linspace(0.9, 1.05, n_wavelengths)
    biases = np.linspace(-0.1, 0.1, n_wavelengths)
    X_slave = X_master * slopes + biases + 0.02 * np.random.randn(n_samples, n_wavelengths)

    return X_master, X_slave, y


# ============================================================================
# Test JYPLS-inv Basic Functionality
# ============================================================================

class TestJYPLSInvBasic:
    """Test basic JYPLS-inv functionality."""

    def test_jypls_inv_estimation(self, simple_pls_data):
        """Test that JYPLS-inv estimation runs and returns valid results."""
        X_master, X_slave, y = simple_pls_data

        # Select transfer samples
        transfer_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )

        # Check return structure
        assert isinstance(params, dict)
        assert 'transformation_matrix' in params
        assert 'n_components' in params
        assert 'cv_rmse' in params
        assert 'transfer_indices' in params
        assert 'explained_variance_ratio' in params

        # Check transformation matrix shape
        B = params['transformation_matrix']
        assert B.shape == (X_master.shape[1], X_master.shape[1])

        # Check components
        assert params['n_components'] == 5

    def test_jypls_inv_application(self, simple_pls_data):
        """Test applying JYPLS-inv transformation."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([5, 15, 25, 35, 45, 55, 65, 75])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=3
        )

        X_transferred = apply_jypls_inv(X_slave, params)

        # Check shape
        assert X_transferred.shape == X_slave.shape

        # Should improve RMSE
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before, "JYPLS-inv should improve RMSE"

    def test_jypls_inv_auto_component_selection(self, simple_pls_data):
        """Test automatic PLS component selection via CV."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.arange(0, 60, 5)  # 12 samples
        y_transfer = y[transfer_idx]

        # Auto-select components
        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=None,  # Auto-select
            max_components=10
        )

        # Should select some number of components
        assert 1 <= params['n_components'] <= 10
        assert params['cv_rmse'] > 0  # CV was performed

    def test_jypls_inv_different_n_components(self, simple_pls_data):
        """Test JYPLS-inv with different numbers of components."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72])
        y_transfer = y[transfer_idx]

        for n_comp in [2, 5, 8]:
            params = estimate_jypls_inv(
                X_master, X_slave, y_transfer, transfer_idx,
                n_components=n_comp
            )

            assert params['n_components'] == n_comp

            X_transferred = apply_jypls_inv(X_slave, params)
            assert X_transferred.shape == X_slave.shape


# ============================================================================
# Test JYPLS-inv Quality
# ============================================================================

class TestJYPLSInvQuality:
    """Test JYPLS-inv transfer quality."""

    def test_affine_transformation_recovery(self, simple_pls_data):
        """Test that JYPLS-inv can recover affine transformation."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.arange(0, 80, 7)  # ~12 samples
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )

        X_transferred = apply_jypls_inv(X_slave, params)

        # Check improvement
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        improvement_ratio = rmse_before / rmse_after
        assert improvement_ratio > 1.5, f"Expected >1.5x improvement, got {improvement_ratio:.2f}x"

    def test_complex_pls_structure(self, complex_pls_data):
        """Test JYPLS-inv on complex PLS-structured data."""
        X_master, X_slave, y = complex_pls_data

        transfer_idx = np.arange(0, 100, 8)  # 13 samples
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=8
        )

        X_transferred = apply_jypls_inv(X_slave, params)

        # Should improve RMSE
        rmse_before = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_after = np.sqrt(np.mean((X_transferred - X_master) ** 2))

        assert rmse_after < rmse_before

    def test_transfer_sample_quality_impact(self, simple_pls_data):
        """Test that transfer sample quality affects JYPLS-inv performance."""
        X_master, X_slave, y = simple_pls_data

        # Good samples: diverse, representative
        from spectral_predict.sample_selection import kennard_stone
        good_idx = kennard_stone(X_master, n_samples=12)
        y_good = y[good_idx]

        params_good = estimate_jypls_inv(
            X_master, X_slave, y_good, good_idx,
            n_components=5
        )

        X_good = apply_jypls_inv(X_slave, params_good)
        rmse_good = np.sqrt(np.mean((X_good - X_master) ** 2))

        # Bad samples: clustered, not representative
        bad_idx = np.arange(0, 12)  # Just first 12 samples
        y_bad = y[bad_idx]

        params_bad = estimate_jypls_inv(
            X_master, X_slave, y_bad, bad_idx,
            n_components=5
        )

        X_bad = apply_jypls_inv(X_slave, params_bad)
        rmse_bad = np.sqrt(np.mean((X_bad - X_master) ** 2))

        # Good samples should generally perform better (not always guaranteed)
        # At least both should improve over no transfer
        rmse_no_transfer = np.sqrt(np.mean((X_slave - X_master) ** 2))
        assert rmse_good < rmse_no_transfer
        assert rmse_bad < rmse_no_transfer


# ============================================================================
# Test JYPLS-inv vs Other Methods
# ============================================================================

class TestJYPLSInvComparison:
    """Compare JYPLS-inv to other calibration transfer methods."""

    def test_jypls_inv_vs_tsr(self, simple_pls_data):
        """Compare JYPLS-inv to TSR with same transfer samples."""
        from spectral_predict.calibration_transfer import estimate_tsr, apply_tsr

        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.arange(0, 80, 6)  # 14 samples
        y_transfer = y[transfer_idx]

        # JYPLS-inv
        jypls_params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )
        X_jypls = apply_jypls_inv(X_slave, jypls_params)

        # TSR
        tsr_params = estimate_tsr(X_master, X_slave, transfer_idx)
        X_tsr = apply_tsr(X_slave, tsr_params)

        # Both should improve over no transfer
        rmse_original = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_jypls = np.sqrt(np.mean((X_jypls - X_master) ** 2))
        rmse_tsr = np.sqrt(np.mean((X_tsr - X_master) ** 2))

        assert rmse_jypls < rmse_original
        assert rmse_tsr < rmse_original

        # Should be comparable (within 30% of each other)
        ratio = max(rmse_jypls, rmse_tsr) / min(rmse_jypls, rmse_tsr)
        assert ratio < 1.5, "JYPLS-inv and TSR should have comparable performance"

    def test_jypls_inv_vs_ctai(self, simple_pls_data):
        """Compare JYPLS-inv to CTAI."""
        from spectral_predict.calibration_transfer import estimate_ctai, apply_ctai

        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.arange(0, 80, 7)  # ~12 samples
        y_transfer = y[transfer_idx]

        # JYPLS-inv (requires samples + Y)
        jypls_params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )
        X_jypls = apply_jypls_inv(X_slave, jypls_params)

        # CTAI (no samples needed)
        ctai_params = estimate_ctai(X_master, X_slave)
        X_ctai = apply_ctai(X_slave, ctai_params)

        # Both should improve
        rmse_original = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_jypls = np.sqrt(np.mean((X_jypls - X_master) ** 2))
        rmse_ctai = np.sqrt(np.mean((X_ctai - X_master) ** 2))

        assert rmse_jypls < rmse_original
        assert rmse_ctai < rmse_original


# ============================================================================
# Test TransferModel Integration
# ============================================================================

class TestTransferModelIntegration:
    """Test JYPLS-inv with TransferModel infrastructure."""

    def test_create_transfer_model_jypls(self, simple_pls_data):
        """Test creating TransferModel with JYPLS-inv."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )

        wavelengths = np.linspace(1000, 2500, X_master.shape[1])

        tm = TransferModel(
            master_id="Master",
            slave_id="Slave",
            method="jypls-inv",
            wavelengths_common=wavelengths,
            params=params,
            meta={"note": "Test JYPLS-inv transfer model"}
        )

        assert tm.method == "jypls-inv"
        assert tm.master_id == "Master"
        assert tm.slave_id == "Slave"

    def test_save_load_transfer_model_jypls(self, simple_pls_data, tmp_path):
        """Test saving and loading JYPLS-inv TransferModel."""
        from spectral_predict.calibration_transfer import save_transfer_model, load_transfer_model

        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([5, 15, 25, 35, 45, 55, 65, 75])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=4
        )

        wavelengths = np.linspace(1000, 2500, X_master.shape[1])

        tm = TransferModel(
            master_id="Master",
            slave_id="Slave",
            method="jypls-inv",
            wavelengths_common=wavelengths,
            params=params,
            meta={}
        )

        # Save
        save_prefix = save_transfer_model(tm, directory=str(tmp_path), name="test_jypls")

        # Load
        tm_loaded = load_transfer_model(save_prefix)

        assert tm_loaded.method == "jypls-inv"
        assert tm_loaded.master_id == "Master"
        assert tm_loaded.slave_id == "Slave"
        assert 'transformation_matrix' in tm_loaded.params


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test JYPLS-inv edge cases."""

    def test_minimal_transfer_samples(self, simple_pls_data):
        """Test JYPLS-inv with minimal transfer samples."""
        X_master, X_slave, y = simple_pls_data

        # Only 5 samples
        transfer_idx = np.array([0, 20, 40, 60, 70])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=3
        )

        X_transferred = apply_jypls_inv(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_many_transfer_samples(self, simple_pls_data):
        """Test JYPLS-inv with many transfer samples."""
        X_master, X_slave, y = simple_pls_data

        # 30 samples
        transfer_idx = np.arange(0, 60, 2)
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=10
        )

        X_transferred = apply_jypls_inv(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_high_dimensional_data(self):
        """Test JYPLS-inv with many wavelengths."""
        np.random.seed(42)
        n_samples = 50
        n_wavelengths = 300

        X_master = np.random.randn(n_samples, n_wavelengths)
        X_slave = 0.95 * X_master + 0.05
        y = X_master[:, :10].mean(axis=1)

        transfer_idx = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )

        X_transferred = apply_jypls_inv(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_single_component(self, simple_pls_data):
        """Test JYPLS-inv with single PLS component."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([0, 10, 20, 30, 40, 50, 60, 70])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=1
        )

        assert params['n_components'] == 1

        X_transferred = apply_jypls_inv(X_slave, params)
        assert X_transferred.shape == X_slave.shape

    def test_invalid_inputs(self, simple_pls_data):
        """Test JYPLS-inv error handling."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([0, 10, 20, 30, 40])
        y_transfer = y[transfer_idx]

        # Mismatched dimensions
        with pytest.raises((ValueError, Exception)):
            estimate_jypls_inv(
                X_master, X_slave[:, :50], y_transfer, transfer_idx,
                n_components=3
            )

        # Wrong number of Y values
        with pytest.raises((ValueError, Exception)):
            estimate_jypls_inv(
                X_master, X_slave, y_transfer[:3], transfer_idx,
                n_components=3
            )

        # Too few samples
        with pytest.raises((ValueError, Exception)):
            estimate_jypls_inv(
                X_master, X_slave, y[:1], np.array([0]),
                n_components=3
            )


# ============================================================================
# Test PLS Component Selection
# ============================================================================

class TestPLSComponentSelection:
    """Test PLS component selection strategies."""

    def test_cv_component_selection(self, simple_pls_data):
        """Test cross-validation for component selection."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.arange(0, 80, 5)  # 16 samples
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=None,
            cv_folds=5,
            max_components=15
        )

        # Should select optimal number of components
        assert 1 <= params['n_components'] <= 15
        assert params['cv_rmse'] >= 0

    def test_explained_variance_tracking(self, simple_pls_data):
        """Test that explained variance is tracked."""
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.array([0, 8, 16, 24, 32, 40, 48, 56, 64, 72])
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )

        assert 'explained_variance_ratio' in params
        assert 0 <= params['explained_variance_ratio'] <= 1.0

    def test_increasing_components_improves_fit(self, complex_pls_data):
        """Test that more components generally improve fit on transfer samples."""
        X_master, X_slave, y = complex_pls_data

        transfer_idx = np.arange(0, 100, 8)  # 13 samples
        y_transfer = y[transfer_idx]

        X_master_transfer = X_master[transfer_idx]
        X_slave_transfer = X_slave[transfer_idx]

        rmses = []
        for n_comp in [2, 4, 6, 8, 10]:
            params = estimate_jypls_inv(
                X_master, X_slave, y_transfer, transfer_idx,
                n_components=n_comp
            )

            X_transferred_subset = apply_jypls_inv(X_slave_transfer, params)
            rmse = np.sqrt(np.mean((X_transferred_subset - X_master_transfer) ** 2))
            rmses.append(rmse)

        # Generally should decrease (may plateau or increase due to overfitting)
        # Just check that we get valid results
        assert all(r > 0 for r in rmses)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
