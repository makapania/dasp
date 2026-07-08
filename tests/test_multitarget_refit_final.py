"""T-17 W3-1: refit_multitarget_final -> save -> reload -> predict round-trip.

A search ``MultiTargetResult`` retains the metadata to *identify* a cell but not
the fitted estimator or the selected variable indices. ``refit_multitarget_final``
replays the exact deterministic cell pipeline (preprocessing + full-calibration
variable selection) on the full calibration set and fits the final model,
honoring JOINT (fit on Y-scaled block + persisted FoldYScaler) vs INDEPENDENT
(per-target fit on raw Y) semantics.

These tests pin that a refit model saved to ``.dasp`` and reloaded reproduces the
refit's own RAW-unit predictions to ~1e-9 for:
  * a JOINT result (PLS, full spectrum),
  * an INDEPENDENT result (Ridge, full spectrum),
  * a result WITH a varsel subset (uve Top-N) so subset bookkeeping is exercised,
    including preprocessed (SNV) + subset column reconstruction on reload.
Plus a guard that the single-Y model_io save/load/predict path is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from spectral_predict.model_io import load_model, predict_with_model, save_model
from spectral_predict.multitarget_grid import (
    refit_multitarget_final,
    run_multitarget_grid_search,
)


@pytest.fixture(scope="module")
def synthetic_multi_y():
    """Deterministic 3-target spectral regression set with divergent scales."""
    rng = np.random.default_rng(11)
    n_samples, n_features = 60, 30
    latent = rng.normal(size=(n_samples, 2))
    base = np.sin(np.linspace(0, 4 * np.pi, n_features))
    base2 = np.cos(np.linspace(0, 6 * np.pi, n_features))
    X = (
        np.outer(latent[:, 0], base)
        + np.outer(latent[:, 1], base2)
        + rng.normal(scale=0.05, size=(n_samples, n_features))
    )
    y0 = 0.1 * latent[:, 0] + 0.01 * rng.normal(size=n_samples)
    y1 = 1.0 * latent[:, 0] + 0.3 * latent[:, 1] + 0.1 * rng.normal(size=n_samples)
    y2 = 100.0 * latent[:, 0] + 5.0 * rng.normal(size=n_samples)
    Y = np.column_stack([y0, y1, y2])
    wl = np.linspace(1000.0, 2500.0, n_features)
    return X, Y, wl


# The single run config; refit is called with the SAME kwargs so reconstruction
# is exact. Kept in one place so search + refit cannot drift.
RUN_KW = dict(
    model_names=["PLS", "Ridge"],
    target_names=["yield", "irsf", "carbonate"],
    preprocessing_methods={"raw": True, "snv": True},
    autoscale=False,
    variable_selection_methods=["uve"],
    variable_counts=[10],
    tier="quick",
    max_n_components=8,
    cv="kfold",
    n_folds=5,
    n_repeats=1,
    random_state=42,
    coupling_mode="both",
)


@pytest.fixture(scope="module")
def search_output(synthetic_multi_y):
    X, Y, wl = synthetic_multi_y
    return run_multitarget_grid_search(X, Y, wavelengths=wl, **RUN_KW)


def _refit_kwargs():
    """The subset of RUN_KW that refit_multitarget_final consumes (drops search-
    only keys like coupling_mode / target_names, which are passed positionally)."""
    kw = dict(RUN_KW)
    kw.pop("target_names")
    kw.pop("coupling_mode")
    return kw


def _find(results, *, mode=None, model_name=None, varsel_tag=None, varsel_method=None):
    for r in results:
        if not np.isfinite(r.joint_q2):
            continue
        if mode is not None and r.mode != mode:
            continue
        if model_name is not None and r.model_name != model_name:
            continue
        if varsel_tag is not None and r.varsel_tag != varsel_tag:
            continue
        if varsel_method is not None and r.varsel_method != varsel_method:
            continue
        return r
    return None


def _roundtrip_check(result, synthetic_multi_y, tmp_path):
    X, Y, wl = synthetic_multi_y
    refit = refit_multitarget_final(
        result, X, Y, RUN_KW["target_names"], wavelengths=wl, **_refit_kwargs()
    )
    # Subset bookkeeping: reconstructed column count matches the stored result.
    assert len(refit.variable_indices) == result.n_variables

    pred_refit = refit.predict(refit.X_final)
    assert pred_refit.shape == (X.shape[0], len(RUN_KW["target_names"]))

    filepath = tmp_path / f"{result.model_name}_{result.mode}_{result.varsel_tag}.dasp"
    refit.save(filepath)

    loaded = load_model(filepath)
    # JOINT persists a y_scaler; INDEPENDENT must not.
    if result.mode == "JOINT":
        assert loaded["y_scaler"] is not None
    else:
        assert loaded["y_scaler"] is None
    assert loaded["metadata"]["multitarget_mode"] == result.mode
    assert loaded["metadata"]["n_targets"] == len(RUN_KW["target_names"])

    # Reload predicts on RAW full spectra and reproduces the refit's RAW-unit
    # predictions (preprocessing + subset reconstructed from full_wavelengths).
    pred_reload = predict_with_model(loaded, X, validate_wavelengths=False)
    np.testing.assert_allclose(pred_reload, pred_refit, atol=1e-9, rtol=0)
    return refit


class TestRefitRoundtrip:
    def test_joint_pls_full_spectrum(self, search_output, synthetic_multi_y, tmp_path):
        res = _find(search_output.results, mode="JOINT", model_name="PLS", varsel_tag="full")
        assert res is not None, "no finite JOINT PLS full-spectrum result to refit"
        self_refit = _roundtrip_check(res, synthetic_multi_y, tmp_path)
        # JOINT PLS fits on scaled Y -> raw predict != scaled predict (scaler active).
        assert self_refit.y_scaler is not None
        assert isinstance(self_refit.estimator, PLSRegression)

    def test_independent_ridge_full_spectrum(self, search_output, synthetic_multi_y, tmp_path):
        res = _find(
            search_output.results, mode="INDEPENDENT", model_name="Ridge", varsel_tag="full"
        )
        assert res is not None, "no finite INDEPENDENT Ridge full-spectrum result to refit"
        refit = _roundtrip_check(res, synthetic_multi_y, tmp_path)
        assert refit.y_scaler is None

    def test_result_with_varsel_subset(self, search_output, synthetic_multi_y, tmp_path):
        # Prefer a preprocessed (snv) + uve-subset cell so BOTH preprocessing and
        # subset-column reconstruction are exercised on reload.
        res = None
        for r in search_output.results:
            if (
                np.isfinite(r.joint_q2)
                and r.varsel_method == "uve"
                and r.varsel_tag == "uve_top10"
                and "snv" in r.preprocessing
            ):
                res = r
                break
        if res is None:  # fall back to any finite uve subset cell
            res = _find(search_output.results, varsel_method="uve", varsel_tag="uve_top10")
        assert res is not None, "no finite uve_top10 subset result to refit"
        refit = _roundtrip_check(res, synthetic_multi_y, tmp_path)
        assert len(refit.variable_indices) == 10
        assert refit.preprocessor is not None or "raw" in res.preprocessing


class TestSingleYGuard:
    """The single-Y model_io save/load/predict path stays byte-identical: no
    y_scaler, no multitarget metadata, predictions unchanged."""

    def test_single_y_save_load_predict_unchanged(self, tmp_path):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(30, 12))
        y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=30)
        model = PLSRegression(n_components=2)
        model.fit(X, y)
        metadata = {
            "model_name": "PLS", "task_type": "regression", "preprocessing": "raw",
            "wavelengths": [float(i) for i in range(12)], "n_vars": 12,
            "n_samples": 30, "performance": {"R2": 0.9},
        }
        filepath = tmp_path / "single_y.dasp"
        save_model(model=model, preprocessor=None, metadata=metadata, filepath=filepath)
        loaded = load_model(filepath)
        assert loaded["y_scaler"] is None
        assert "has_y_scaler" not in loaded["metadata"]
        assert "multitarget_mode" not in loaded["metadata"]
        pred = predict_with_model(loaded, X, validate_wavelengths=False)
        np.testing.assert_array_equal(pred, model.predict(X))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
