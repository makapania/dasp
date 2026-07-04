"""Generate tests/gold_standards/varsel_single_y.npz from CURRENT single-Y code.

Captures selected indices + score vectors for each performance-based varsel
method (iPLS fwd/bwd, SPA, MC-siPLS, MWPLS, GA-PLS) on a fixed 1-D synthetic
case with pinned RNG. Run this BEFORE the multi-Y refactor; the parity test in
tests/test_variable_selection.py then asserts np.array_equal / assert_allclose
after the refactor to prove single-Y stays byte-identical.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral_predict.ga_pls import ga_pls_selection
from spectral_predict.variable_selection import (
    ipls_backward,
    ipls_forward,
    mc_sipls,
    mwpls,
    spa_selection,
)

SEED = 20260701
N, P = 60, 100


def make_data():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N, P))
    # y depends on a handful of features so varsel has real signal to find
    coef = np.zeros(P)
    coef[10:18] = rng.standard_normal(8)
    coef[55:60] = rng.standard_normal(5)
    y = X @ coef + 0.1 * rng.standard_normal(N)
    wavelengths = np.linspace(1000.0, 2500.0, P)
    return X, y, wavelengths


def subsets_arrays(subsets):
    rmsecv = np.array([s["rmsecv"] for s in subsets], dtype=float)
    r2 = np.array([s["r2"] for s in subsets], dtype=float)
    if subsets:
        best = min(subsets, key=lambda s: s["rmsecv"])
        best_idx = np.asarray(best["indices"], dtype=int)
    else:
        best_idx = np.array([], dtype=int)
    return rmsecv, r2, best_idx


def main():
    X, y, wl = make_data()
    out = {}

    fwd = ipls_forward(X, y, wl, n_intervals=10, max_combine=4, cv_folds=5, random_state=42)
    out["ipls_fwd_rmsecv"], out["ipls_fwd_r2"], out["ipls_fwd_best_indices"] = subsets_arrays(fwd)

    bwd = ipls_backward(X, y, wl, n_intervals=10, cv_folds=5, random_state=42, min_intervals=1)
    out["ipls_bwd_rmsecv"], out["ipls_bwd_r2"], out["ipls_bwd_best_indices"] = subsets_arrays(bwd)

    mc = mc_sipls(X, y, wl, n_intervals=10, n_combinations=120, max_combine=4, cv_folds=5,
                  random_state=42)
    out["mc_sipls_rmsecv"], out["mc_sipls_r2"], out["mc_sipls_best_indices"] = subsets_arrays(mc)

    mw = mwpls(X, y, wl, window_sizes=[10, 20, 40], step_size=None, cv_folds=5)
    out["mwpls_rmsecv"], out["mwpls_r2"], out["mwpls_best_indices"] = subsets_arrays(mw)

    spa_imp = spa_selection(X, y, n_features=15, cv_folds=5)
    out["spa_importances"] = np.asarray(spa_imp, dtype=float)

    ga_freq = ga_pls_selection(
        X, y, task_type="regression", population_size=20, n_generations=8,
        n_runs=2, cv=5, random_state=42, n_jobs=1, verbose=0,
        min_wavelengths=5, n_components=5,
    )
    out["gapls_frequency"] = np.asarray(ga_freq, dtype=float)

    dest = os.path.join(os.path.dirname(__file__), "..", "tests", "gold_standards",
                        "varsel_single_y.npz")
    dest = os.path.abspath(dest)
    np.savez(dest, **out)
    print("WROTE", dest)
    for k, v in out.items():
        print(f"  {k}: shape={np.asarray(v).shape}")


if __name__ == "__main__":
    main()
