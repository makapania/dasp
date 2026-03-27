"""Tests for integrating BaselineAdvanced into sklearn Pipeline workflows.

Verifies that BaselineAdvanced can be used as a preprocessing step inside a
standard sklearn Pipeline and produces correct output when combined with other
spectral transformers.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectra(n_samples: int = 10, n_features: int = 200, seed: int = 0) -> np.ndarray:
    """Create synthetic spectra with a quadratic baseline plus Gaussian peaks."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_features)
    peak = 0.9 * np.exp(-((x - 0.5) ** 2) / (2 * 0.02**2))
    baseline = 0.002 * np.arange(n_features) + 0.1
    noise = rng.normal(0, 0.005, (n_samples, n_features))
    return peak[np.newaxis, :] + baseline[np.newaxis, :] + noise


# ---------------------------------------------------------------------------
# Pipeline construction and correctness
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_pipeline_with_arpls_builds_and_transforms():
    """Pipeline with BaselineAdvanced(arpls) must fit and transform without error."""
    from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

    if not HAS_PYBASELINES:
        pytest.skip("pybaselines not installed")

    X = _make_spectra()
    pipe = Pipeline(
        [
            ("baseline", BaselineAdvanced(method="arpls", lam=1e5)),
            ("scaler", StandardScaler()),
        ]
    )
    X_out = pipe.fit_transform(X)
    assert X_out.shape == X.shape


@pytest.mark.integration
def test_pipeline_step_produces_correct_output_shape():
    """The baseline step alone should return the same shape as its input."""
    from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

    if not HAS_PYBASELINES:
        pytest.skip("pybaselines not installed")

    X = _make_spectra(n_samples=8, n_features=150)
    step = BaselineAdvanced(method="arpls", lam=1e5)
    X_out = step.fit_transform(X)
    assert X_out.shape == X.shape


@pytest.mark.integration
def test_pipeline_step_reduces_baseline():
    """After arPLS correction, MSE against the true signal must decrease."""
    from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

    if not HAS_PYBASELINES:
        pytest.skip("pybaselines not installed")

    rng = np.random.default_rng(7)
    n, p = 12, 200
    x = np.linspace(0, 1, p)
    true_signal = 0.9 * np.exp(-((x - 0.5) ** 2) / (2 * 0.02**2))
    quadratic_baseline = 0.001 * np.arange(p) ** 2 / p + 0.1
    noise = rng.normal(0, 0.003, (n, p))
    X = true_signal[np.newaxis, :] + quadratic_baseline[np.newaxis, :] + noise
    X_true = np.broadcast_to(true_signal, (n, p)).copy()

    step = BaselineAdvanced(method="arpls", lam=1e5)
    X_corrected = step.fit_transform(X)

    mse_before = float(np.mean((X - X_true) ** 2))
    mse_after = float(np.mean((X_corrected - X_true) ** 2))
    assert mse_after < mse_before, (
        f"arPLS step did not reduce baseline MSE (before={mse_before:.6f}, after={mse_after:.6f})"
    )


@pytest.mark.integration
def test_pipeline_with_baseline_params_dict():
    """Params passed as individual kwargs must be picked up during transform."""
    from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

    if not HAS_PYBASELINES:
        pytest.skip("pybaselines not installed")

    X = _make_spectra()
    # lam is an extra kwarg that gets set as an attribute via __init__ **params
    pipe = Pipeline(
        [
            ("baseline", BaselineAdvanced(method="arpls", lam=1e5)),
        ]
    )
    X_out = pipe.fit_transform(X)
    assert not np.any(np.isnan(X_out))
    assert X_out.shape == X.shape


@pytest.mark.integration
def test_pipeline_fit_transform_vs_fit_then_transform_are_equal():
    """fit_transform and fit + transform must produce identical outputs."""
    from spectral_predict.baseline_advanced import BaselineAdvanced, HAS_PYBASELINES

    if not HAS_PYBASELINES:
        pytest.skip("pybaselines not installed")

    X = _make_spectra(n_samples=6, n_features=100)

    step1 = BaselineAdvanced(method="snip", max_half_window=20)
    X_ft = step1.fit_transform(X)

    step2 = BaselineAdvanced(method="snip", max_half_window=20)
    step2.fit(X)
    X_t = step2.transform(X)

    np.testing.assert_array_equal(X_ft, X_t)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
