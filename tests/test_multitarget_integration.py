"""T-17 end-to-end integration: real ASD spectra + a constructed 2nd target.

Phase C = plan Task 14. Exercises the full ``run_multitarget_grid_search`` path
(preprocessing x varsel x model x hp grid -> joint multi-Y CV -> joint-Q2-ranked
leaderboard) on REAL ``read_asd_dir('example')`` spectra (49 x 2151) joined to
``example/BoneCollagen.csv`` on the ``File Number`` column, then pins the F7
save -> load -> predict round-trip reproducing RAW-unit predictions.

The dataset has only ONE continuous target (``%Collagen``); a second
spectrally-grounded target is CONSTRUCTED for the 2-target smoke (labelled
smoke-only below) so the multi-target path has a second, genuinely
spectrally-predictable column distinct from %Collagen. It is NOT real science.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def _load_example():
    """Load the real ASD spectra + join to BoneCollagen %Collagen.

    Stems in ``read_asd_dir`` are already space-stripped ("Spectrum00001") while
    the CSV ``File Number`` carries a space ("Spectrum 00001"); strip the space
    on the CSV side to align. Returns (X, Y, wl) with Y a 2-column matrix.
    """
    from spectral_predict.io import read_asd_dir

    X_df, _meta = read_asd_dir("example")          # (49, 2151), columns are float nm
    ref = pd.read_csv("example/BoneCollagen.csv")
    ref["_key"] = ref["File Number"].astype(str).str.replace(" ", "", regex=False)
    idx_key = [str(i).replace(" ", "") for i in X_df.index]
    y0 = pd.Series(ref.set_index("_key")["%Collagen"]).reindex(idx_key).to_numpy(float)
    mask = np.isfinite(y0)
    Xv = X_df.to_numpy(float)[mask]
    y0 = y0[mask]
    wl = np.asarray(X_df.columns, dtype=float)

    # SMOKE-ONLY synthetic 2nd target: weighted sum of mean reflectance in three
    # NIR/SWIR bands + small noise. NOT real science — exists only to give the
    # multi-target path a second, spectrally-predictable column distinct from
    # %Collagen so PLS-2 / Ridge-INDEP both have something to fit.
    rng = np.random.default_rng(20260702)
    band_idx = [int(np.argmin(np.abs(wl - b))) for b in (1400.0, 1700.0, 2100.0)]
    y1 = (
        Xv[:, band_idx[0]]
        + 0.5 * Xv[:, band_idx[1]]
        - 0.3 * Xv[:, band_idx[2]]
    )
    y1 = y1 + 0.02 * float(np.std(y1)) * rng.standard_normal(len(y1))

    Y = np.column_stack([y0, y1])
    return Xv, Y, wl


pytestmark = pytest.mark.skipif(
    not os.path.isdir("example"), reason="example data unavailable"
)


def test_multitarget_grid_end_to_end_real_data():
    """Full grid on real spectra: leaderboard is real, ranked, and covers the
    JOINT + INDEPENDENT + varsel(< full width) cells the feature promises."""
    from spectral_predict.multitarget_grid import run_multitarget_grid_search

    X, Y, wl = _load_example()
    full_width = X.shape[1]
    out = run_multitarget_grid_search(
        X, Y, model_names=["PLS", "Ridge"], target_names=["collagen", "synthetic"],
        wavelengths=wl, preprocessing_methods={"raw": True, "snv": True}, autoscale=False,
        variable_selection_methods=["ipls_forward"], variable_counts=[10],
        ipls_subset_limit="Top 3", tier="quick", cv="kfold", n_folds=3, n_repeats=1,
    )

    # A real leaderboard was produced.
    assert out.results, "no cells produced"

    # Ranking: best row is finite and is the max finite joint_q2; finite block
    # is non-increasing (NaN-safe sort pushes any non-finite to the bottom).
    finite_q2s = [r.joint_q2 for r in out.results if np.isfinite(r.joint_q2)]
    assert finite_q2s, "no finite joint_q2 rows in leaderboard"
    assert np.isfinite(out.results[0].joint_q2)
    assert out.results[0].joint_q2 == max(finite_q2s)
    assert all(finite_q2s[i] >= finite_q2s[i + 1] for i in range(len(finite_q2s) - 1))

    # JOINT (PLS-2) and INDEPENDENT (Ridge) couplings both appear.
    modes = {r.mode for r in out.results}
    assert "JOINT" in modes and "INDEPENDENT" in modes

    # A varsel cell that actually narrowed the spectrum (n_variables < full width)
    # is present, and it is the ipls_forward method we requested.
    assert any(
        r.varsel_method == "ipls_forward"
        and r.n_variables is not None
        and r.n_variables < full_width
        for r in out.results
    ), "no ipls_forward varsel cell narrower than full width"


def test_multitarget_save_reload_predict_roundtrip(tmp_path):
    """F7: fit a JOINT PLS-2 on FoldYScaler-scaled Y, persist with y_scaler=,
    reload, and reproduce RAW-unit predictions (max|diff| ~ 0.0)."""
    from sklearn.cross_decomposition import PLSRegression

    from spectral_predict.model_io import load_model, predict_with_model, save_model
    from spectral_predict.multi_y import FoldYScaler

    X, Y, wl = _load_example()

    # Mirror the in-app JOINT training path: scale Y on the full training set,
    # fit PLS-2 on scaled Y. RAW-unit preds come from inverse-transforming.
    y_scaler = FoldYScaler().fit(Y)
    model = PLSRegression(n_components=5, scale=False)
    model.fit(X, y_scaler.transform(Y))

    metadata = {
        "model_name": "PLS",
        "task_type": "regression",
        "preprocessing": "raw",
        "wavelengths": [float(w) for w in wl],
        "n_vars": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "performance": {},
        "multitarget_mode": "JOINT",
        "n_targets": 2,
        "target_names": ["collagen", "synthetic"],
        "prediction_columns": ["collagen_pred", "synthetic_pred"],
    }
    filepath = tmp_path / "mt_joint_pls2.dasp"
    save_model(
        model=model, preprocessor=None, metadata=metadata,
        filepath=filepath, y_scaler=y_scaler,
    )

    loaded = load_model(filepath)
    raw_pred = np.asarray(predict_with_model(loaded, X, validate_wavelengths=False))

    # RAW-unit predictions reproduce the in-app pooled prediction exactly.
    expected_raw = y_scaler.inverse_transform(model.predict(X))
    assert raw_pred.shape == (X.shape[0], 2)
    assert np.max(np.abs(raw_pred - expected_raw)) == pytest.approx(0.0, abs=1e-9)
