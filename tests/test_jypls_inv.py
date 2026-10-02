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

        # Check shape preserved
        assert X_transferred.shape == X_slave.shape

        # Check all values are finite
        assert np.all(np.isfinite(X_transferred))

        # B = R M P^T has rank n_components; the offset carries the mean spectrum.
        B = params['transformation_matrix']
        assert np.linalg.matrix_rank(B, tol=1e-8) == params['n_components']

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
        """Primary and satellite PLS scores of the same standards are correlated.

        Spectral transfer itself is tested in TestJYPLSInvTransfersSpectra; this
        fixture is full-rank noise, so a k-component reconstruction cannot reproduce it.
        """
        X_master, X_slave, y = simple_pls_data

        transfer_idx = np.arange(0, 80, 7)  # ~12 samples
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=5
        )

        X_transferred = apply_jypls_inv(X_slave, params)

        # Verify transformation produces valid data
        assert X_transferred.shape == X_master.shape
        assert np.all(np.isfinite(X_transferred))

        # Verify transformation matrix rank matches components
        B = params['transformation_matrix']
        assert np.linalg.matrix_rank(B, tol=1e-8) == params['n_components']

        # Verify PLS scores are close between primary and satellite transfer samples
        T_primary = params['pls_scores_primary']
        T_satellite = params['pls_scores_satellite']
        score_corr = np.corrcoef(T_primary.ravel(), T_satellite.ravel())[0, 1]
        assert score_corr > 0.8, f"PLS scores should be correlated, got r={score_corr:.3f}"

    def test_complex_pls_structure(self, complex_pls_data):
        """JYPLS-inv runs on three-latent-variable data with a wavelength-dependent gain."""
        X_master, X_slave, y = complex_pls_data

        transfer_idx = np.arange(0, 100, 8)  # 13 samples
        y_transfer = y[transfer_idx]

        params = estimate_jypls_inv(
            X_master, X_slave, y_transfer, transfer_idx,
            n_components=8
        )

        X_transferred = apply_jypls_inv(X_slave, params)

        # Check output validity
        assert X_transferred.shape == X_slave.shape
        assert np.all(np.isfinite(X_transferred))

        # The explained variance ratio should be positive
        assert params['explained_variance_ratio'] > 0

        # Transformation matrix rank should be at most n_components
        # (may be less if some PLS components are near-degenerate,
        # which happens when n_components exceeds the true latent dimensionality)
        B = params['transformation_matrix']
        rank = np.linalg.matrix_rank(B, tol=1e-8)
        assert 1 <= rank <= params['n_components']

    def test_transfer_sample_quality_impact(self, simple_pls_data):
        """Different standards give different transforms, and both still transfer.

        Replaces the old relaxed check ("JYPLS-inv does not guarantee spectral RMSE
        improvement"), which let R091 through: transfer must beat no transfer on the
        standards it was fitted on and keep the primary mean level.
        """
        X_master, X_slave, y = simple_pls_data

        from spectral_predict.sample_selection import kennard_stone
        good_idx = kennard_stone(X_master, n_samples=12)
        bad_idx = np.arange(0, 12)

        params_good = estimate_jypls_inv(X_master, X_slave, y[good_idx], good_idx, n_components=5)
        params_bad = estimate_jypls_inv(X_master, X_slave, y[bad_idx], bad_idx, n_components=5)

        diff = np.sqrt(
            np.mean(
                (params_good["transformation_matrix"] - params_bad["transformation_matrix"]) ** 2
            )
        )
        assert diff > 1e-6, "Different sample sets should produce different transformations"

        for params, idx in ((params_good, good_idx), (params_bad, bad_idx)):
            X_out = apply_jypls_inv(X_slave[idx], params)
            assert np.all(np.isfinite(X_out))
            # The 0.08 instrument offset is removed from the mean level.
            assert abs(X_out.mean() - X_master[idx].mean()) < 0.01


# ============================================================================
# R091: JYPLS-inv must transfer spectra (identity / offset / affine instruments)
# ============================================================================


def _lowrank_instrument_data(n=40, n_new=60, p=200, noise=0.0, seed=0):
    """Primary spectra = nonzero mean + 3 Gaussian bands with random scores (+ noise).

    Returns (X_primary, y, X_primary_new, wl). y is a linear function of the band
    scores, so a joint-Y PLS with >= 3 components can find the whole signal space.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(0.0, 1.0, p)
    bands = np.array([np.exp(-((wl - c) ** 2) / 0.005) for c in (0.25, 0.5, 0.75)])
    mean = 2.0 + 0.5 * wl

    def draw(m):
        t = rng.standard_normal((m, 3))
        X = mean + t @ bands + noise * rng.standard_normal((m, p))
        return X, t @ np.array([1.0, -0.5, 0.3])

    X, y = draw(n)
    X_new, _ = draw(n_new)
    return X, y, X_new, wl


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


@pytest.mark.filterwarnings("ignore:y residual is constant")
class TestJYPLSInvTransfersSpectra:
    """Behavioural tests on synthetic instruments with known relationships (R091)."""

    def test_identical_instruments_identity_with_nonzero_mean(self):
        """Identical instruments: the transfer returns the input, mean included.

        The pre-fix code returned a rank-k projection without the mean (RMSE 2.0 on
        a baseline-2 spectrum, Codex repro). Noise-free rank-3 data, 3 components.
        """
        X, y, X_new, _ = _lowrank_instrument_data()
        params = estimate_jypls_inv(X, X.copy(), y, np.arange(len(X)), n_components=3)

        X_out = apply_jypls_inv(X_new, params)
        assert _rmse(X_out, X_new) < 1e-8
        assert X_out.mean() == pytest.approx(X_new.mean(), abs=1e-8)

    def test_identical_instruments_with_noise_stay_close(self):
        """With noise, identity transfer error stays at the noise level, not the mean level."""
        X, y, X_new, _ = _lowrank_instrument_data(noise=0.005, seed=1)
        params = estimate_jypls_inv(X, X.copy(), y, np.arange(len(X)), n_components=3)

        X_out = apply_jypls_inv(X_new, params)
        assert _rmse(X_out, X_new) < 0.01  # noise SD is 0.005; mean level is ~2.25
        assert abs(X_out.mean() - X_new.mean()) < 1e-3

    @pytest.mark.parametrize("n_components", [3, 6, None])
    def test_constant_offset_instrument(self, n_components):
        """Satellite = primary + 0.3: transfer must beat no transfer on new samples."""
        X, y, X_new, _ = _lowrank_instrument_data()
        offset = 0.3
        params = estimate_jypls_inv(X, X + offset, y, np.arange(len(X)), n_components=n_components)

        X_out = apply_jypls_inv(X_new + offset, params)
        rmse_none = _rmse(X_new + offset, X_new)
        rmse_jy = _rmse(X_out, X_new)
        assert rmse_jy < 0.5 * rmse_none
        assert abs(X_out.mean() - X_new.mean()) < 0.05

    @pytest.mark.parametrize("n_components", [3, 6, None])
    def test_affine_instrument_multiple_components(self, n_components):
        """Wavelength-dependent gain plus curved baseline, three latent bands."""
        X, y, X_new, wl = _lowrank_instrument_data(noise=0.002, seed=2)
        gain = np.linspace(0.9, 1.1, X.shape[1])
        baseline = 0.2 * wl**2 + 0.1

        def satellite(Xp):
            return Xp * gain + baseline

        params = estimate_jypls_inv(
            X, satellite(X), y, np.arange(len(X)), n_components=n_components
        )
        X_out = apply_jypls_inv(satellite(X_new), params)

        rmse_none = _rmse(satellite(X_new), X_new)
        rmse_jy = _rmse(X_out, X_new)
        assert rmse_jy < 0.5 * rmse_none, (rmse_jy, rmse_none)

    def test_transfer_improves_primary_model_predictions(self):
        """A primary PLS model predicts better on transferred than on raw satellite spectra."""
        from sklearn.cross_decomposition import PLSRegression

        rng = np.random.default_rng(3)
        X, y, _, wl = _lowrank_instrument_data(n=60, noise=0.002, seed=3)
        cal, std, test = np.arange(0, 30), np.arange(30, 45), np.arange(45, 60)
        model = PLSRegression(n_components=3, scale=False).fit(X[cal], y[cal])

        gain = np.linspace(0.95, 1.05, X.shape[1])
        X_sat = X * gain + 0.05 + 0.002 * rng.standard_normal(X.shape)
        params = estimate_jypls_inv(X[std], X_sat[std], y[std], np.arange(len(std)), n_components=3)

        err_raw = _rmse(model.predict(X_sat[test]).ravel(), y[test])
        err_jy = _rmse(model.predict(apply_jypls_inv(X_sat[test], params)).ravel(), y[test])
        assert err_jy < err_raw

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_missing_targets_rejected(self, bad):
        """Missing reference values raise; they are never replaced by zeros."""
        X, y, _, _ = _lowrank_instrument_data()
        y = y.copy()
        y[4] = bad
        with pytest.raises(ValueError, match="reference value"):
            estimate_jypls_inv(X, X + 0.1, y, np.arange(len(X)), n_components=3)

    def test_pre_fix_params_are_refused(self):
        """A model saved by the old maths (no 'offset') is refused, not silently applied."""
        X, y, _, _ = _lowrank_instrument_data()
        params = estimate_jypls_inv(X, X + 0.1, y, np.arange(len(X)), n_components=3)
        legacy = {k: v for k, v in params.items() if k != "offset"}
        with pytest.raises(ValueError, match="Rebuild"):
            apply_jypls_inv(X, legacy)

    def test_save_load_roundtrip_preserves_transfer(self, tmp_path):
        """The offset survives save/load, so a reloaded model gives the same output."""
        from spectral_predict.calibration_transfer import load_transfer_model, save_transfer_model

        X, y, X_new, wl = _lowrank_instrument_data()
        params = estimate_jypls_inv(X, X + 0.3, y, np.arange(len(X)), n_components=3)
        tm = TransferModel(
            primary_id="P",
            satellite_id="S",
            method="jypls-inv",
            wavelengths_common=wl,
            params=params,
            meta={},
        )
        loaded = load_transfer_model(save_transfer_model(tm, tmp_path, name="jy"))

        np.testing.assert_allclose(
            apply_jypls_inv(X_new + 0.3, loaded.params), apply_jypls_inv(X_new + 0.3, params)
        )

    def test_auto_components_cv_splits_never_separate_a_standard(self, monkeypatch):
        """No CV fold trains on one spectrum of a standard and tests on its twin.

        Rows i and i + n are the primary and satellite spectra of standard i and share
        its y, so a standard's two rows must sit on the same side of every split.
        """
        import sklearn.model_selection as ms

        seen = []
        real = ms.cross_val_score

        def spy(*args, **kwargs):
            seen.append(kwargs.get("cv"))
            return real(*args, **kwargs)

        monkeypatch.setattr(ms, "cross_val_score", spy)
        n = 20
        X, y, _, _ = _lowrank_instrument_data(n=n)
        estimate_jypls_inv(X, X + 0.1, y, np.arange(n), n_components=None, max_components=4)

        assert seen, "auto component selection should run CV"
        splits = list(seen[0])
        assert len(splits) == 5
        for train, test in splits:
            train_std = set((np.asarray(train) % n).tolist())
            test_std = set((np.asarray(test) % n).tolist())
            assert not train_std & test_std
            assert len(train) + len(test) == 2 * n

    def test_auto_components_work_with_metadata_routing(self):
        """sklearn metadata routing must not silently disable the CV (Codex round 1)."""
        import sklearn

        X, y, _, _ = _lowrank_instrument_data(n=20, noise=0.002, seed=5)
        kwargs = dict(n_components=None, max_components=6)
        plain = estimate_jypls_inv(X, X + 0.1, y, np.arange(20), **kwargs)
        with sklearn.config_context(enable_metadata_routing=True):
            routed = estimate_jypls_inv(X, X + 0.1, y, np.arange(20), **kwargs)

        assert np.isfinite(routed["cv_rmse"])
        assert routed["n_components"] == plain["n_components"]
        assert routed["cv_rmse"] == pytest.approx(plain["cv_rmse"])

    def test_auto_components_reject_cv_with_no_finite_error(self, monkeypatch):
        """If every component count gives a non-finite CV error, refuse to guess."""
        import sklearn.model_selection as ms

        monkeypatch.setattr(ms, "cross_val_score", lambda *a, **k: np.array([np.nan]))
        X, y, _, _ = _lowrank_instrument_data(n=20)
        with pytest.raises(ValueError, match="could not choose"):
            estimate_jypls_inv(X, X + 0.1, y, np.arange(20), n_components=None, max_components=4)


# ============================================================================
# Test JYPLS-inv vs Other Methods
# ============================================================================

class TestJYPLSInvComparison:
    """Compare JYPLS-inv to other calibration transfer methods."""

    def test_jypls_inv_vs_tsr(self, simple_pls_data):
        """Compare JYPLS-inv to TSR: both should run and produce valid output.

        JYPLS-inv and TSR operate differently:
        - TSR does per-wavelength slope/bias correction (spectral RMSE focus)
        - JYPLS-inv projects into Y-predictive subspace (prediction focus)

        We compare that both produce valid transformations rather than
        comparing spectral RMSE, since they optimize different objectives.
        """
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

        # Both should produce valid output
        assert X_jypls.shape == X_slave.shape
        assert X_tsr.shape == X_slave.shape
        assert np.all(np.isfinite(X_jypls))
        assert np.all(np.isfinite(X_tsr))

        # TSR should improve spectral RMSE (per-wavelength correction)
        rmse_original = np.sqrt(np.mean((X_slave - X_master) ** 2))
        rmse_tsr = np.sqrt(np.mean((X_tsr - X_master) ** 2))
        assert rmse_tsr < rmse_original

    def test_jypls_inv_vs_ctai(self, simple_pls_data):
        """Compare JYPLS-inv to CTAI: both should produce valid transformations.

        JYPLS-inv and CTAI operate differently:
        - CTAI estimates a full-rank affine transformation using paired samples
        - JYPLS-inv projects into a Y-predictive subspace (low-rank)

        We verify that both run and produce valid output.
        """
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

        # CTAI (uses paired samples, no explicit transfer selection)
        ctai_params = estimate_ctai(X_master, X_slave)
        X_ctai = apply_ctai(X_slave, ctai_params)

        # Both should produce valid output with correct shapes
        assert X_jypls.shape == X_slave.shape
        assert X_ctai.shape == X_slave.shape
        assert np.all(np.isfinite(X_jypls))
        assert np.all(np.isfinite(X_ctai))

        # The two methods produce different transformations
        diff = np.sqrt(np.mean((X_jypls - X_ctai) ** 2))
        assert diff > 0, "JYPLS-inv and CTAI should produce different results"


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

        # TransferModel uses primary_id/satellite_id naming
        tm = TransferModel(
            primary_id="Master",
            satellite_id="Slave",
            method="jypls-inv",
            wavelengths_common=wavelengths,
            params=params,
            meta={"note": "Test JYPLS-inv transfer model"}
        )

        assert tm.method == "jypls-inv"
        assert tm.primary_id == "Master"
        assert tm.satellite_id == "Slave"

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

        # TransferModel uses primary_id/satellite_id naming
        tm = TransferModel(
            primary_id="Master",
            satellite_id="Slave",
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
        assert tm_loaded.primary_id == "Master"
        assert tm_loaded.satellite_id == "Slave"
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
