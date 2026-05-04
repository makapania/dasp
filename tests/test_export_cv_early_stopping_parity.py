"""Export-side CV parity for boosters with early stopping.

Pins the contract that the exported notebook/script CV reproduces the in-app
CV when the in-app uses early stopping on the held-out fold (the
``cv_utils._fit_with_early_stopping`` path called from
``search.py:_run_single_fold`` and from the GUI refined-model path).

Background
----------
Commit ``af6f4cf`` (Jan 2026) added in-app early stopping for boosters. The
exported templates were never updated — they emitted plain
``fold_model.fit(X_train, y_train)`` regardless. For LightGBM/XGBoost/CatBoost
this silently diverged: in-app stops trees when the held-out fold's loss
flattens, export grows the configured ``n_estimators`` to completion. On the
user's collagen-cat dataset this produced a one-sample flip in 5-fold CV
(in-app accuracy 1.0 vs notebook 0.976).

The existing ``test_t20_saved_model_export_parity`` does not catch this
because it only compares final-model predictions, not per-fold CV predictions.

What this pins
--------------
1. Generator threads ``early_stopping_rounds`` from ``model_config`` into
   the emitted ``EARLY_STOPPING_ROUNDS`` constant.
2. The emitted ``_fit_fold`` helper, when applied across CV folds, produces
   per-sample predictions identical to ``cv_utils._fit_with_early_stopping``
   applied across the same splits with the same model.
3. When ``early_stopping_rounds`` is None/0, ``_fit_fold`` falls through to
   plain ``.fit()`` — preserves prior export behavior for non-boosters and
   for boosters where the user disabled early stopping.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.cv_utils import _fit_with_early_stopping


def _make_data(seed: int = 42):
    """Three-class data sized like the user's collagen-cat case (41×20)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((41, 20))
    y = np.concatenate([np.zeros(21, int), np.ones(6, int), 2 * np.ones(14, int)])
    rng.shuffle(y)
    return X, y


def _lightgbm_params() -> dict:
    return {
        "n_estimators": 200, "learning_rate": 0.1, "num_leaves": 50,
        "colsample_bytree": 0.8, "subsample": 0.8, "min_child_samples": 5,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "random_state": 42,
        "n_jobs": -1, "verbosity": -1, "bagging_freq": 1,
    }


def _build_model_config(early_stopping_rounds, params=None) -> dict:
    return {
        "model_name": "LightGBM",
        "preprocessing": "raw",
        "task_type": "classification",
        "target_name": "target",
        "params": params or _lightgbm_params(),
        "metrics": {},
        "cv_folds": 5,
        "cv_strategy": "kfold",
        "cv_n_repeats": 5,
        "imbalance_method": None,
        "imbalance_params": {},
        "autoscale": False,
        "variable_indices": None,
        "variable_selection_method": None,
        "trim_derivative_edges": False,
        "inlier_class_label": "",
        "wavelengths": list(range(20)),
        "early_stopping_rounds": early_stopping_rounds,
    }


def _exec_generated_cv(model_config, X, y) -> np.ndarray:
    """Generate the notebook, exec the model+CV cell, return per-sample preds."""
    opts = ExportOptions(
        format="notebook", include_data=True, data_X=X, data_y=y,
        wavelengths=None, colab_ready=False, include_visualization=False,
    )
    gen = CodeGenerator(model_config, opts)
    nb = gen.generate_notebook()
    ns: dict = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        # Skip the install-deps cell — venv has everything; subprocess pip is slow.
        if "subprocess.check_call" in code:
            continue
        exec(code, ns)
    return ns["all_y_pred_arr"]


def _inapp_cv_preds(X, y, params, early_stopping_rounds) -> np.ndarray:
    """Reproduce in-app per-fold predictions using the same helper used in
    ``_run_single_fold`` / GUI refined-model path."""
    from lightgbm import LGBMClassifier

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.empty_like(y)
    for tr_idx, te_idx in cv.split(X, y):
        m = clone(LGBMClassifier(**params))
        if early_stopping_rounds:
            _fit_with_early_stopping(
                m, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx],
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            m.fit(X[tr_idx], y[tr_idx])
        preds[te_idx] = m.predict(X[te_idx])
    return preds


def test_lightgbm_early_stop_export_matches_inapp():
    """Export CV with early_stopping_rounds=40 must match in-app CV predictions."""
    X, y = _make_data()
    params = _lightgbm_params()

    inapp_preds = _inapp_cv_preds(X, y, params, early_stopping_rounds=40)
    export_preds = _exec_generated_cv(_build_model_config(40, params), X, y)

    assert np.array_equal(inapp_preds, export_preds), (
        f"Export CV diverges from in-app CV under early stopping.\n"
        f"  in-app preds:  {inapp_preds.tolist()}\n"
        f"  export preds:  {export_preds.tolist()}"
    )


def test_lightgbm_no_early_stop_export_matches_plain_fit():
    """early_stopping_rounds=None must fall through to plain .fit() — no
    behavior change for exports that didn't use early stopping."""
    X, y = _make_data()
    params = _lightgbm_params()

    inapp_preds = _inapp_cv_preds(X, y, params, early_stopping_rounds=None)
    export_preds = _exec_generated_cv(_build_model_config(None, params), X, y)

    assert np.array_equal(inapp_preds, export_preds), (
        f"Plain-fit export CV diverges from plain-fit in-app CV.\n"
        f"  in-app preds:  {inapp_preds.tolist()}\n"
        f"  export preds:  {export_preds.tolist()}"
    )


def test_zero_early_stop_treated_as_disabled():
    """early_stopping_rounds=0 must behave like None (disabled)."""
    X, y = _make_data()
    params = _lightgbm_params()

    plain_preds = _exec_generated_cv(_build_model_config(None, params), X, y)
    zero_preds = _exec_generated_cv(_build_model_config(0, params), X, y)

    assert np.array_equal(plain_preds, zero_preds)


@pytest.mark.parametrize("esr", [None, 40])
def test_emitted_constant_reflects_threaded_value(esr):
    """The notebook source must contain the threaded EARLY_STOPPING_ROUNDS
    value so reviewers can see what early-stopping setting produced the
    reported numbers."""
    X, y = _make_data()
    opts = ExportOptions(
        format="notebook", include_data=True, data_X=X, data_y=y,
        wavelengths=None, colab_ready=False, include_visualization=False,
    )
    gen = CodeGenerator(_build_model_config(esr), opts)
    nb = gen.generate_notebook()
    code = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    expected = 0 if esr is None else esr
    assert f"EARLY_STOPPING_ROUNDS = {expected}" in code
