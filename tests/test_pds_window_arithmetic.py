"""
Unit tests for PDS window-arithmetic robustness.

Roadmap ticket T-07 — verify that estimate_pds rejects even window
sizes (which would otherwise overflow the B-coefficient allocation),
and that apply_pds derives its slice geometry from B.shape[1] rather
than a redundant `window` argument.

This implementation's window is the full centered width 2k+1, so it
must be odd.  The canonical centered-local-regression definition
(coefficients from channel i-k to i+k, total width 2k+1) is confirmed
in Wang, Veltkamp, & Kowalski (1991), "Multivariate Instrument
Standardization," Anal. Chem. 63(23), 2750-2756; see also the RNIR
package documentation and the specProc R package.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from spectral_predict.calibration_transfer import estimate_pds, apply_pds


@pytest.fixture
def synthetic_pair():
    rng = np.random.default_rng(seed=42)
    n_samples, p = 12, 20
    X_primary = rng.normal(size=(n_samples, p))
    X_satellite = X_primary + 0.05 * rng.normal(size=(n_samples, p))
    return X_primary, X_satellite


def test_estimate_pds_odd_window_baseline(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=5)
    assert B.shape == (X_satellite.shape[1], 5)
    X_transformed = apply_pds(X_satellite, B)
    pre = np.mean(np.abs(X_primary - X_satellite))
    post = np.mean(np.abs(X_primary - X_transformed))
    assert post < pre, f"PDS should reduce error: pre={pre:.4f}, post={post:.4f}"


def test_estimate_pds_rejects_even_window_4(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    with pytest.raises(ValueError, match="(?i)odd"):
        estimate_pds(X_primary, X_satellite, window=4)


def test_estimate_pds_rejects_even_window_10(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    with pytest.raises(ValueError, match="(?i)odd"):
        estimate_pds(X_primary, X_satellite, window=10)


def test_estimate_pds_error_message_cites_wang(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    with pytest.raises(ValueError) as excinfo:
        estimate_pds(X_primary, X_satellite, window=4)
    msg = str(excinfo.value)
    assert "1991" in msg or "Wang" in msg, (
        f"Error should reference Wang et al. (1991); got: {msg!r}"
    )


def test_estimate_pds_window_1_is_identity_like(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=1)
    assert B.shape == (X_satellite.shape[1], 1)


def test_estimate_pds_no_broadcast_error_at_interior(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=5)
    assert np.any(B[2, :] != 0), (
        "Interior B row should not be all zeros for non-degenerate input"
    )


def test_apply_pds_uses_B_shape_when_window_arg_disagrees(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    B = estimate_pds(X_primary, X_satellite, window=5)
    assert B.shape[1] == 5

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X_a = apply_pds(X_satellite, B, window=11)
    future_warns = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert future_warns, "Expected FutureWarning when window arg disagrees with B.shape[1]"

    X_b = apply_pds(X_satellite, B, window=5)
    np.testing.assert_allclose(X_a, X_b, rtol=1e-12, atol=1e-12)


def test_estimate_apply_pds_roundtrip_window_11():
    rng = np.random.default_rng(seed=7)
    n_samples, p = 30, 50
    X_primary = rng.normal(size=(n_samples, p))
    X_satellite = X_primary + 0.02 * rng.normal(size=(n_samples, p))
    B = estimate_pds(X_primary, X_satellite, window=11)
    assert B.shape == (p, 11)
    X_t = apply_pds(X_satellite, B, window=11)
    assert X_t.shape == X_satellite.shape
    pre = np.mean(np.abs(X_primary - X_satellite))
    post = np.mean(np.abs(X_primary - X_t))
    assert post < pre


def test_apply_pds_rejects_even_width_B(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    p = X_satellite.shape[1]
    B_even = np.zeros((p, 4))
    with pytest.raises(ValueError, match="even"):
        apply_pds(X_satellite, B_even)


def test_apply_pds_validates_shape_compatibility(synthetic_pair):
    X_primary, X_satellite = synthetic_pair
    B_wrong = np.zeros((30, 5))
    with pytest.raises(ValueError, match="shape"):
        apply_pds(X_satellite, B_wrong)
