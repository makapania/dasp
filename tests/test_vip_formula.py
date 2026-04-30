"""
Regression test for the canonical VIP formula in compute_vip().

Bug (pre-T-05): compute_vip() used np.var(y) as a per-component scalar weight
for all components, so the per-component Y-explained-variance term
collapsed to a constant times sum(T_a**2). This skewed VIP rankings on any
PLS problem where Y-explained variance was not proportional to X-score
energy (i.e. effectively all real PLS problems with > 1 component).

Canonical formula (Wold 2001; Mehmood et al. 2012, Eq. 1):

    SSY_a = q_a**2 * (T_a.T @ T_a)               # q = y_loadings_
    VIP_j = sqrt( p * sum_a [ SSY_a * (W_{j,a} / ||W_a||)**2 ] / SSY_total )

This test:
  1. Builds a small synthetic PLS regression problem where the OLD and NEW
     formulas disagree by > 1e-3 on at least one variable (proves the
     fixture is sensitive to the formula difference). Codex independently
     confirmed max_abs_diff ~= 1.39 on this fixture.
  2. [DISCRIMINATING] Compares compute_vip() output to an independent
     reference implementation of the canonical formula.
  3. [DISCRIMINATING] Asserts compute_vip() does NOT match the old buggy
     formula (catches the unfixed state).
  4. [SANITY] Verifies the canonical invariant: sum(VIP**2) / p ~= 1.
     NOTE: this invariant holds for BOTH the old and new formulas because
     sklearn's x_weights_ columns are unit-norm, so it cannot distinguish
     the bug from the fix. It is a guard against future regressions that
     break weight normalization, not a bug discriminator.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from spectral_predict.models import compute_vip


def _reference_vip_canonical(pls, X, y):
    """Independent VIP reference using y_loadings_ (Wold 2001)."""
    W = np.asarray(pls.x_weights_)
    T = np.asarray(pls.x_scores_)
    Q = np.asarray(pls.y_loadings_)
    q = Q.ravel() if Q.ndim == 1 else Q[0, :]
    p = W.shape[0]

    ssy_comp = (q ** 2) * np.sum(T ** 2, axis=0)
    ssy_total = ssy_comp.sum()

    w_norm_sq = (W ** 2) / (np.sum(W ** 2, axis=0, keepdims=True) + 1e-300)

    vip = np.sqrt(p * (w_norm_sq @ ssy_comp) / (ssy_total + 1e-300))
    return vip


def _reference_vip_OLD_BUGGY(pls, X, y):
    """The pre-T-05 formula. Used only to assert the test would catch it."""
    W = np.asarray(pls.x_weights_)
    T = np.asarray(pls.x_scores_)
    y_arr = np.asarray(y).reshape(-1, 1)
    ssy_comp = np.sum(T ** 2, axis=0) * np.var(y_arr, axis=0)
    ssy_total = np.sum(ssy_comp)
    p = W.shape[0]
    weight = np.sum((W ** 2) * ssy_comp, axis=1)
    return np.sqrt(p * weight / ssy_total)


@pytest.fixture
def synthetic_pls_problem():
    """
    Two latent factors with very different Y-loading magnitudes:
    - factor 1 contributes strongly to Y (q1 = 5.0)
    - factor 2 contributes weakly to Y (q2 = 0.5) but has comparable X energy
    The old (constant np.var(y)) formula will badly mis-weight factor 2.
    """
    rng = np.random.default_rng(0)
    n_samples = 60
    p = 20

    t1 = rng.standard_normal(n_samples)
    t2 = rng.standard_normal(n_samples)
    w1 = np.zeros(p); w1[2:6] = 1.0
    w2 = np.zeros(p); w2[12:16] = 1.0
    X = np.outer(t1, w1) + np.outer(t2, w2) + 0.05 * rng.standard_normal((n_samples, p))

    y = 5.0 * t1 + 0.5 * t2 + 0.05 * rng.standard_normal(n_samples)
    return X, y


@pytest.fixture
def fitted_pls(synthetic_pls_problem):
    X, y = synthetic_pls_problem
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X, y)
    return pls, X, y


class TestVIPCanonicalFormula:
    def test_old_and_new_formulas_disagree_on_this_problem(self, fitted_pls):
        pls, X, y = fitted_pls
        old = _reference_vip_OLD_BUGGY(pls, X, y)
        new = _reference_vip_canonical(pls, X, y)
        max_abs_diff = float(np.max(np.abs(old - new)))
        assert max_abs_diff > 1e-3, (
            "Synthetic problem is not sensitive enough — old and new VIP "
            f"agree to {max_abs_diff:.2e}. Adjust the fixture."
        )

    def test_compute_vip_matches_canonical_reference(self, fitted_pls):
        pls, X, y = fitted_pls
        got = compute_vip(pls, X, y)
        want = _reference_vip_canonical(pls, X, y)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)

    def test_compute_vip_does_not_match_old_buggy_formula(self, fitted_pls):
        pls, X, y = fitted_pls
        got = compute_vip(pls, X, y)
        old = _reference_vip_OLD_BUGGY(pls, X, y)
        max_abs_diff = float(np.max(np.abs(got - old)))
        assert max_abs_diff > 1e-3, (
            "compute_vip() still matches the old (buggy) formula — fix not applied."
        )

    def test_canonical_invariant_average_squared_vip_is_one(self, fitted_pls):
        pls, X, y = fitted_pls
        vip = compute_vip(pls, X, y)
        p = pls.x_weights_.shape[0]
        mean_sq = float(np.sum(vip ** 2) / p)
        assert mean_sq == pytest.approx(1.0, rel=1e-6, abs=1e-8)

    def test_output_shape_and_nonnegativity(self, fitted_pls):
        pls, X, y = fitted_pls
        vip = compute_vip(pls, X, y)
        assert vip.shape == (pls.x_weights_.shape[0],)
        assert np.all(vip >= 0)
        assert np.all(np.isfinite(vip))


class TestVIPEdgeCases:
    def test_compute_vip_n_components_1(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 12))
        y = X[:, 0] + 0.1 * rng.standard_normal(40)
        pls = PLSRegression(n_components=1, scale=False)
        pls.fit(X, y)

        vip = compute_vip(pls, X, y)
        assert vip.shape == (12,)
        assert np.all(np.isfinite(vip))
        assert np.all(vip >= 0)
        assert float(np.sum(vip ** 2) / 12) == pytest.approx(1.0, rel=1e-6, abs=1e-8)

    def test_compute_vip_zero_y_loadings_returns_zeros(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((30, 8))
        y = rng.standard_normal(30)
        pls = PLSRegression(n_components=2, scale=False)
        pls.fit(X, y)
        pls.y_loadings_ = np.zeros_like(pls.y_loadings_)

        vip = compute_vip(pls, X, y)
        assert vip.shape == (8,)
        assert np.all(vip == 0.0)
