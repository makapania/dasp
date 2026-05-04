"""T-20: saved-model ↔ exported-script reproducibility test.

T-19 fixed model-native imbalance handling and Auto-mode runtime resolution
across the export-code surface; T-32 fixed sample_weight threading through
the resampler. Both bugs lived in the parity contract between the runtime
(``model_io.save_model`` / ``load_model``) and the exported Python script
(``code_generator.CodeGenerator.generate_script``). Without an explicit
parity test pinning the contract, the next regression slips silently.

What this test pins
-------------------
For each (model, task_type, imbalance_method) combination:

1. Train a model end-to-end in the test process with a fixed seed.
2. Save it via ``save_model`` to a ``.dasp`` file.
3. Generate the equivalent exported Python script via
   ``CodeGenerator.generate_script`` with the training data embedded.
4. Append a small predict-and-save block to the script.
5. Execute the script in a subprocess (Python in ``.venv312``).
6. Load the saved ``.dasp`` model and predict on the same test set.
7. Assert the script's predictions match the saved model's predictions
   to within numerical tolerance.

Determinism comes from fixing ``random_state=42`` on every model that
exposes one and using ``preprocessing='raw'`` (which avoids the
embedded-data path's "preprocessing skipped because embedded data is
already preprocessed" branch — orthogonal to the parity surface).

Why the boosting + imbalance combinations matter
------------------------------------------------
T-19's user-facing fix was Auto-mode runtime resolution of
``imbalance_method`` for boosting classifiers (XGBoost, LightGBM,
CatBoost). The bug class was: the runtime did one thing with
``class_weight``-style imbalance; the exported script did another. So
the boosting + ``class_weight`` and boosting + ``auto`` rows below are
the highest-leverage rows in the matrix.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spectral_predict.code_generator import CodeGenerator, ExportOptions
from spectral_predict.model_io import load_model, save_model


# ---------------------------------------------------------------------------
# Synthetic data helpers (small enough to keep subprocess startup the bottleneck)
# ---------------------------------------------------------------------------


def _make_regression_data(
    n_train: int = 60, n_test: int = 20, n_features: int = 80, seed: int = 42
):
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(n_features)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = X_train @ coef + rng.standard_normal(n_train) * 0.1
    X_test = rng.standard_normal((n_test, n_features))
    return X_train, y_train, X_test


def _make_binary_classification_data(
    n_train: int = 60, n_test: int = 20, n_features: int = 80, seed: int = 42,
    imbalanced: bool = False,
):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    # Build y_train deterministically by stratification so a future reseed
    # cannot land a fold with a single class — the codegen uses
    # StratifiedKFold(n_splits=cv_folds=3) which would fail to split a
    # degenerate fold and the test would fail-for-the-wrong-reason. With a
    # deterministic count + shuffle we know cells per class up front.
    if imbalanced:
        # ~80/20 split so class_weight='balanced' has something to do.
        n_pos = max(n_train // 5, 1)
    else:
        n_pos = n_train // 2
    y_train = np.concatenate(
        [np.ones(n_pos, dtype=int), np.zeros(n_train - n_pos, dtype=int)]
    )
    rng.shuffle(y_train)
    X_test = rng.standard_normal((n_test, n_features))
    return X_train, y_train, X_test


def _make_multiclass_classification_data(
    n_train: int = 60, n_test: int = 20, n_features: int = 80,
    n_classes: int = 3, seed: int = 42,
):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = rng.integers(low=0, high=n_classes, size=n_train).astype(int)
    # Guarantee every class appears at least once so stratified CV in the
    # exported script doesn't choke.
    for cls in range(n_classes):
        if (y_train == cls).sum() == 0:
            y_train[cls] = cls
    X_test = rng.standard_normal((n_test, n_features))
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# Core parity runner
# ---------------------------------------------------------------------------


def _build_metadata(model_name: str, task_type: str, n_vars: int, params: dict) -> dict:
    return {
        "model_name": model_name,
        "task_type": task_type,
        "preprocessing": "raw",
        "wavelengths": list(range(n_vars)),
        "n_vars": n_vars,
        "performance": {"R2": 0.0, "RMSE": 0.0},
        "params": params,
    }


def _build_export_config(
    model_name: str, task_type: str, n_vars: int, params: dict,
    imbalance_method: str | None = None,
) -> dict:
    return {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": task_type,
        "target_name": "target",
        "params": params,
        "metrics": {"RMSE": 0.0, "R2": 0.0},
        "variable_indices": None,
        "wavelengths": list(range(n_vars)),
        "cv_folds": 3,  # small so subprocess CV is fast
        "imbalance_method": imbalance_method,
    }


def _run_parity(
    *,
    model: Any,
    model_name: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    imbalance_method: str | None,
    tmp_path: Path,
    rtol: float | None = None,
    atol: float | None = None,
    fit_kwargs: dict | None = None,
    expect_stdout_marker: str | None = None,
) -> None:
    """Run save→export→exec→compare parity for one (model, config) row.

    For classification, predictions are compared via ``predict_proba`` (full
    probability vectors), not ``predict`` (hard labels). Hard labels mask
    probability drift up to the nearest class boundary — empirically a
    weighting-mechanism mismatch can produce 0.3 probability drift while
    hard labels remain identical. ``predict_proba`` makes the parity
    assertion sensitive to the kind of regressions T-19/T-32 actually fixed.

    For regression, ``predict`` is the right comparison (continuous output,
    no quantisation).

    ``fit_kwargs`` is forwarded to the in-process ``model.fit(...)`` call.
    For XGBoost + ``class_weight`` (or auto-resolves-to-class_weight), pass
    ``{'sample_weight': compute_sample_weight('balanced', y_train)}`` to
    match what the codegen emits in
    ``code_generator._render_final_model_with_imbalance``.

    ``expect_stdout_marker``: if set, asserts the substring appears in the
    subprocess's stdout. Use to confirm a code path actually executed —
    e.g. the auto-resolution branch in
    ``code_generator._render_imbalance_handling`` prints
    ``"[Auto imbalance] ratio ..."`` when it fires; without this assertion
    the test would silently pass even if auto-resolution was disabled.
    """
    n_vars = X_train.shape[1]

    if rtol is None:
        # Looser tolerance for classification: predict_proba is a learned
        # function and tiny float-arithmetic-order divergences are
        # legitimate. Tight tolerance for regression: it should match
        # essentially exactly.
        rtol = 1e-3 if task_type == "classification" else 1e-5
    if atol is None:
        atol = 1e-6 if task_type == "classification" else 1e-8

    # 1. Train in test process.
    model.fit(X_train, y_train, **(fit_kwargs or {}))

    # 2. Save via model_io.
    save_path = tmp_path / "model.dasp"
    save_model(
        model=model,
        preprocessor=None,
        metadata=_build_metadata(model_name, task_type, n_vars, params),
        filepath=str(save_path),
    )

    # 3. Generate exported script with training data embedded.
    config = _build_export_config(
        model_name, task_type, n_vars, params, imbalance_method=imbalance_method
    )
    options = ExportOptions(
        format="script",
        include_data=True,
        data_X=X_train,
        data_y=y_train,
        wavelengths=np.arange(n_vars, dtype=float),
        include_visualization=False,
        include_prediction_template=False,
    )
    script = CodeGenerator(config, options).generate_script()

    # 4. Append our own predict-and-dump block. ``model`` is module-level in
    # the generated script (see FINAL_MODEL_TEMPLATE in
    # spectral_predict/templates/validation.py: ``model.fit(X_final, y)``).
    # Classification compares predict_proba (full probability vectors) so
    # that drift smaller than the class-boundary quantisation is detectable.
    X_test_path = tmp_path / "X_test.npy"
    pred_out_path = tmp_path / "script_predictions.npy"
    np.save(X_test_path, X_test)
    if task_type == "classification":
        predict_call = "model.predict_proba(_X_test_t20)"
    else:
        predict_call = "model.predict(_X_test_t20).ravel()"
    appendix = textwrap.dedent(
        f"""

        # ===== T-20 parity-test appendix =====
        import numpy as _np_t20
        _X_test_t20 = _np_t20.load(r"{X_test_path}")
        _predictions_t20 = {predict_call}
        _np_t20.save(r"{pred_out_path}", _predictions_t20)
        """
    )
    script_path = tmp_path / "exported.py"
    script_path.write_text(script + appendix, encoding="utf-8")

    # 5. Run in subprocess. sys.executable points at .venv312 python under
    # `python -m pytest`, so the subprocess gets the same sklearn / xgboost
    # / lightgbm versions as the test process — avoiding cross-version drift
    # that would defeat the parity assertion. UTF-8 encoding is forced both
    # ways: the codegen emits non-ASCII chars (e.g. `≥` in the auto-mode
    # comment block); without this a Windows shell's cp1252 default would
    # raise UnicodeDecodeError on any traceback that quotes the source line,
    # masking the real failure with a decode error.
    import os as _os
    env = {**_os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert result.returncode == 0, (
        f"exported script failed (returncode={result.returncode})\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
    )

    if expect_stdout_marker is not None:
        assert expect_stdout_marker in result.stdout, (
            f"expected stdout marker {expect_stdout_marker!r} not found; "
            f"the code path it gates probably did not execute.\n"
            f"--- STDOUT ---\n{result.stdout}"
        )

    # 6. Compare loaded saved model's predictions to script's predictions.
    # Use predict_proba for classification (sensitive to drift below class
    # boundary), predict for regression (continuous output).
    script_predictions = np.load(pred_out_path)
    loaded = load_model(str(save_path))
    if task_type == "classification":
        saved_predictions = loaded["model"].predict_proba(X_test)
    else:
        saved_predictions = loaded["model"].predict(X_test).ravel()

    np.testing.assert_allclose(
        script_predictions,
        saved_predictions,
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"exported-script predictions diverge from saved-model predictions "
            f"for {model_name}/{task_type}/imbalance={imbalance_method!r}; "
            "the runtime↔export parity contract has regressed (T-20 surface)"
        ),
    )


# ---------------------------------------------------------------------------
# Regression — covers the simplest baseline before we layer on imbalance
# ---------------------------------------------------------------------------


def test_pls_regression_parity(tmp_path):
    """PLS is deterministic and is the canonical chemometrics baseline.

    ``scale=False`` is mandatory: the exported script's ``DEFAULT_PARAMS``
    merge injects ``scale=False`` (templates/models.DEFAULT_PARAMS) because the
    chemometrics pipeline assumes SNV/derivative preprocessing has already
    centered/standardised the spectra — letting sklearn auto-scale on top
    would double-process. The whole runtime hardcodes ``scale=False`` on
    every PLSRegression instantiation (24+ sites). To mirror what the
    runtime would have saved, the in-test model uses the same kwarg."""
    from sklearn.cross_decomposition import PLSRegression

    X_train, y_train, X_test = _make_regression_data()
    params = {"n_components": 5, "scale": False}
    _run_parity(
        model=PLSRegression(**params),
        model_name="PLS",
        task_type="regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_random_forest_regression_parity(tmp_path):
    """RandomForest with fixed random_state. Pins the bagging-randomness
    determinism contract end-to-end."""
    from sklearn.ensemble import RandomForestRegressor

    X_train, y_train, X_test = _make_regression_data()
    params = {"n_estimators": 30, "max_depth": 6, "random_state": 42, "n_jobs": 1}
    _run_parity(
        model=RandomForestRegressor(**params),
        model_name="RandomForest",
        task_type="regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# Binary classification — the imbalance surface T-19 actually fixed
# ---------------------------------------------------------------------------


def test_random_forest_binary_classification_parity_no_imbalance(tmp_path):
    """Binary classification baseline with no imbalance handling."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, y_train, X_test = _make_binary_classification_data()
    params = {"n_estimators": 30, "max_depth": 6, "random_state": 42, "n_jobs": 1}
    _run_parity(
        model=RandomForestClassifier(**params),
        model_name="RandomForest",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_random_forest_binary_classification_parity_class_weight(tmp_path):
    """T-19 surface: imbalance_method='class_weight' on a class_weight-aware
    sklearn classifier. The exported script must instantiate
    RandomForestClassifier(class_weight='balanced', ...) the same way the
    runtime did.

    Param-split shape (pr-test-analyzer 2026-05-07 follow-on review): the
    in-process model gets ``class_weight='balanced'`` added at the
    constructor call site, BUT the params dict passed to
    ``_build_export_config`` and ``_build_metadata`` is the BARE base
    params (without ``class_weight``). This makes the codegen's runtime
    conditional the ONLY source of ``class_weight`` in the exported
    script. Same rationale as the auto-with-correction rows: pre-injecting
    the kwarg in BOTH paths makes the conditional dead code; adversarial
    deletion of ``code_generator._render_model``'s ``if IMBALANCE_METHOD ==
    'class_weight': model_params['class_weight'] = 'balanced'`` block would
    have silently passed under the previous shape."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    base_params = {
        "n_estimators": 30,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": 1,
    }
    _run_parity(
        # In-process model gets class_weight; codegen receives bare base_params
        # and must emit class_weight via the runtime conditional to match.
        model=RandomForestClassifier(**base_params, class_weight="balanced"),
        model_name="RandomForest",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=base_params,
        imbalance_method="class_weight",
        tmp_path=tmp_path,
    )


def test_random_forest_multiclass_classification_parity(tmp_path):
    """Multi-class baseline — verifies the parity contract holds for
    multi-output prediction (classes > 2)."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, y_train, X_test = _make_multiclass_classification_data(n_classes=3)
    params = {"n_estimators": 30, "max_depth": 6, "random_state": 42, "n_jobs": 1}
    _run_parity(
        model=RandomForestClassifier(**params),
        model_name="RandomForest",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# Boosting — the T-19 sample_weight + Auto-mode surface
# ---------------------------------------------------------------------------


# Common XGBoost params: tree_method='hist' + n_jobs=1 makes both the
# in-process fit and the subprocess fit deterministic and version-stable.
# The codegen path injects tree_method='hist' / n_jobs=-1 via setdefault
# (code_generator._render_model setdefault block), so test params explicitly setting
# both keeps the merged params identical in both paths and removes a
# latent fragility (multi-threaded determinism depends on XGBoost version
# and CPU). use_label_encoder is intentionally absent — XGBoost 3.x emits
# a deprecation warning otherwise.
_XGB_BASE_PARAMS = {
    "n_estimators": 30,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": 1,
    "eval_metric": "logloss",
}


def test_xgboost_binary_classification_parity_no_imbalance(tmp_path):
    """Boosting baseline. XGBoost is the headline model for the T-19
    sample_weight fix, so we want to confirm the no-imbalance path is clean
    before testing the imbalance variants."""
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier

    X_train, y_train, X_test = _make_binary_classification_data()
    params = dict(_XGB_BASE_PARAMS)
    _run_parity(
        model=XGBClassifier(**params),
        model_name="XGBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_xgboost_binary_classification_parity_class_weight(tmp_path):
    """T-19 highest-leverage row: XGBoost + class_weight imbalance.

    The codegen emits ``model.fit(X, y, sample_weight=
    compute_sample_weight('balanced', y))`` for XGBoost +
    imbalance_method='class_weight' in
    ``code_generator._render_final_model_with_imbalance``. The
    in-process model must apply the same sample_weight at fit time;
    using ``scale_pos_weight`` instead would mean the two paths use
    different weighting mechanisms — the test would only pass on
    datasets where both produce trivially identical predictions."""
    pytest.importorskip("xgboost")
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    params = dict(_XGB_BASE_PARAMS)
    sample_weight = compute_sample_weight("balanced", y_train)
    _run_parity(
        model=XGBClassifier(**params),
        model_name="XGBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method="class_weight",
        tmp_path=tmp_path,
        fit_kwargs={"sample_weight": sample_weight},
    )


def test_xgboost_binary_classification_parity_auto_no_correction(tmp_path):
    """T-19 Auto-mode runtime resolution — the auto-resolves-to-None path.

    With BALANCED data (ratio < 3:1), the script's auto-resolution block
    in ``code_generator._render_imbalance_handling`` mutates
    IMBALANCE_METHOD to None. Downstream gates do not fire; the script
    fits without sample_weight. The in-process model must do the same.

    The ``expect_stdout_marker`` assertion below pins the auto-resolution
    branch actually executed — without it, this test would silently pass
    even if auto-resolution were broken (since the in-process model also
    has no sample_weight, the predictions would match for the wrong
    reason). The marker text is emitted by
    ``_render_imbalance_handling``'s auto-resolution print."""
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier

    # Balanced ~50/50 — auto-resolution → None
    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=False)
    params = dict(_XGB_BASE_PARAMS)
    _run_parity(
        model=XGBClassifier(**params),
        model_name="XGBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method="auto",
        tmp_path=tmp_path,
        expect_stdout_marker="[Auto imbalance] ratio",
    )


def test_xgboost_binary_classification_parity_auto_with_correction(tmp_path):
    """T-19 Auto-mode runtime resolution — the auto-resolves-to-class_weight path.

    With IMBALANCED data (ratio ≥ 3:1), the script's auto-resolution block
    in ``code_generator._render_imbalance_handling`` mutates IMBALANCE_METHOD
    to "class_weight". The XGBoost branch then emits ``model.fit(X, y,
    sample_weight=compute_sample_weight('balanced', y))`` exactly as the
    explicit ``imbalance_method='class_weight'`` row does. The in-process
    model must mirror that — same sample_weight at fit time.

    The ``expect_stdout_marker="applying class_weight"`` pins the
    class_weight branch of the auto-resolution print specifically; the
    other branch emits ``"no correction"`` so the marker is unique to
    the with-correction path. Without this marker, a regression that
    flipped the threshold or short-circuited to None would still pass
    here (predictions would diverge but the wrong-branch print is the
    real signal)."""
    pytest.importorskip("xgboost")
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    params = dict(_XGB_BASE_PARAMS)
    sample_weight = compute_sample_weight("balanced", y_train)
    _run_parity(
        model=XGBClassifier(**params),
        model_name="XGBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method="auto",
        tmp_path=tmp_path,
        fit_kwargs={"sample_weight": sample_weight},
        expect_stdout_marker="applying class_weight",
    )


def test_xgboost_multiclass_classification_parity(tmp_path):
    """Multi-class boosting parity — covers the multi-output path on the
    T-19 surface. softprob/softmax mismatch between runtime and script
    would surface here. Multi-class predict_proba returns shape
    (n_samples, n_classes); assert_allclose handles the 2-D array."""
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier

    X_train, y_train, X_test = _make_multiclass_classification_data(n_classes=3)
    params = dict(_XGB_BASE_PARAMS)
    params.update({
        "eval_metric": "mlogloss",
        "objective": "multi:softprob",
        "num_class": 3,
    })
    _run_parity(
        model=XGBClassifier(**params),
        model_name="XGBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# LightGBM — sklearn-API native class_weight kwarg path (T-20b)
# ---------------------------------------------------------------------------

# Same shape as _XGB_BASE_PARAMS. Explicit n_jobs=1 so multi-threaded
# determinism isn't a worry across CI hosts; cross-family review confirmed
# the codegen's n_jobs=-1 setdefault at code_generator._render_model uses a
# startswith check that does not actually match LightGBM's resolved class
# name, so the injection is dead code there — passing n_jobs=1 explicitly
# pins both the in-process and subprocess paths regardless. verbose=-1
# suppresses the voluminous LightGBM training log so failure stdout stays
# diagnostic.
_LGBM_BASE_PARAMS = {
    "n_estimators": 30,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


def test_lightgbm_regression_parity(tmp_path):
    """LightGBM regression baseline — pins the LGBM parity contract for the
    no-imbalance path. The codegen's regressor branch resolves to
    ``LGBMRegressor`` via ``_resolve_model_ctor_class``."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMRegressor

    X_train, y_train, X_test = _make_regression_data()
    params = dict(_LGBM_BASE_PARAMS)
    _run_parity(
        model=LGBMRegressor(**params),
        model_name="LightGBM",
        task_type="regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_lightgbm_binary_classification_parity_no_imbalance(tmp_path):
    """LightGBM binary classification baseline — no imbalance handling."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    X_train, y_train, X_test = _make_binary_classification_data()
    params = dict(_LGBM_BASE_PARAMS)
    _run_parity(
        model=LGBMClassifier(**params),
        model_name="LightGBM",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_lightgbm_binary_classification_parity_class_weight(tmp_path):
    """T-19 surface for LightGBM: ``class_weight='balanced'`` is a native
    sklearn-API kwarg on ``LGBMClassifier``. Unlike XGBoost (which has no
    class_weight kwarg and falls back to sample_weight at fit time),
    LightGBM's class_weight is a constructor kwarg — the codegen's runtime
    conditional in ``_render_model`` injects it into ``model_params``
    when ``IMBALANCE_METHOD == 'class_weight'``.

    Param-split shape (pr-test-analyzer 2026-05-07 follow-on review): see
    test_random_forest_binary_classification_parity_class_weight for the
    full rationale. Bare base_params to codegen + class_weight added only
    to the in-process model makes the runtime conditional the unique
    source of class_weight in the exported script."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    base_params = dict(_LGBM_BASE_PARAMS)  # no class_weight — bare for codegen
    _run_parity(
        model=LGBMClassifier(**base_params, class_weight="balanced"),
        model_name="LightGBM",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=base_params,
        imbalance_method="class_weight",
        tmp_path=tmp_path,
    )


def test_lightgbm_binary_classification_parity_auto_with_correction(tmp_path):
    """T-19 Auto-mode runtime resolution — auto-resolves-to-class_weight for
    LightGBM. Mirror of test_xgboost_binary_classification_parity_auto_with_
    correction (the symmetric coverage flagged by pr-test-analyzer on PR #32).

    With IMBALANCED data (ratio >= 3:1), the codegen's auto-resolution block
    in ``_render_imbalance_handling`` mutates IMBALANCE_METHOD to "class_weight".
    The LightGBM branch then injects ``model_params['class_weight'] = 'balanced'``
    via the runtime conditional at the auto-resolution site. Unlike XGBoost
    (which routes through fit-time sample_weight because there is no
    class_weight kwarg), LightGBM's class_weight is a native sklearn-API
    constructor kwarg.

    Param-split shape (DeepSeek 2026-05-07 review of PR #48): the in-process
    model is constructed with ``class_weight='balanced'`` added at the
    constructor call site, BUT the params dict passed through to
    ``_build_export_config`` and ``_build_metadata`` is the BARE base params
    (without ``class_weight``). This makes the codegen's runtime conditional
    the ONLY source of ``class_weight`` in the exported script. If a future
    refactor deletes the conditional, the script's predictions diverge from
    the in-process model's, and this test fails. The earlier shape (passing
    ``class_weight='balanced'`` in BOTH the in-process params and the codegen
    params) made the conditional dead code — adversarial deletion would have
    silently passed the test.

    The ``expect_stdout_marker="applying class_weight"`` pins the
    class_weight branch of the auto-resolution print. Without it, a regression
    that flipped the threshold or short-circuited to None would still pass
    here (predictions would diverge but the wrong-branch print is the real
    signal)."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    base_params = dict(_LGBM_BASE_PARAMS)  # no class_weight — bare for codegen
    _run_parity(
        # In-process model gets class_weight added at construction; codegen
        # path receives bare params and must add class_weight via runtime
        # conditional to match.
        model=LGBMClassifier(**base_params, class_weight="balanced"),
        model_name="LightGBM",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=base_params,
        imbalance_method="auto",
        tmp_path=tmp_path,
        expect_stdout_marker="applying class_weight",
    )


def test_lightgbm_multiclass_classification_parity(tmp_path):
    """Multi-class LightGBM — pins the multi-output predict_proba shape
    parity. LGBMClassifier handles multi-class natively (objective inferred
    from y); no extra params needed."""
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    X_train, y_train, X_test = _make_multiclass_classification_data(n_classes=3)
    params = dict(_LGBM_BASE_PARAMS)
    _run_parity(
        model=LGBMClassifier(**params),
        model_name="LightGBM",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# CatBoost — auto_class_weights='Balanced' constructor kwarg path (T-20b)
# ---------------------------------------------------------------------------

# CatBoost uses different kwarg names from the rest of the boosting family:
# ``thread_count`` (not ``n_jobs``), and ``verbose=0`` (the codegen injects
# this automatically when missing — set it explicitly here so merged params
# are identical).
_CATBOOST_BASE_PARAMS = {
    "n_estimators": 30,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 42,
    "thread_count": 1,
    "verbose": 0,
}


def test_catboost_regression_parity(tmp_path):
    """CatBoost regression baseline."""
    pytest.importorskip("catboost")
    from catboost import CatBoostRegressor

    X_train, y_train, X_test = _make_regression_data()
    params = dict(_CATBOOST_BASE_PARAMS)
    _run_parity(
        model=CatBoostRegressor(**params),
        model_name="CatBoost",
        task_type="regression",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_catboost_binary_classification_parity_no_imbalance(tmp_path):
    """CatBoost binary classification baseline — no imbalance handling."""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X_train, y_train, X_test = _make_binary_classification_data()
    params = dict(_CATBOOST_BASE_PARAMS)
    _run_parity(
        model=CatBoostClassifier(**params),
        model_name="CatBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


def test_catboost_binary_classification_parity_class_weight(tmp_path):
    """T-19 surface for CatBoost: imbalance handled via the native
    ``auto_class_weights='Balanced'`` constructor kwarg (CatBoost-specific;
    the codegen emits it as a balanced_kwarg in ``_render_model`` when
    ``IMBALANCE_METHOD == 'class_weight'``).

    Param-split shape (pr-test-analyzer 2026-05-07 follow-on review): see
    test_random_forest_binary_classification_parity_class_weight for the
    full rationale. Bare base_params to codegen + auto_class_weights added
    only to the in-process model makes the runtime conditional the unique
    source of the balancing kwarg in the exported script."""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    base_params = dict(_CATBOOST_BASE_PARAMS)  # no auto_class_weights — bare
    _run_parity(
        model=CatBoostClassifier(**base_params, auto_class_weights="Balanced"),
        model_name="CatBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=base_params,
        imbalance_method="class_weight",
        tmp_path=tmp_path,
    )


def test_catboost_binary_classification_parity_auto_with_correction(tmp_path):
    """T-19 Auto-mode runtime resolution — auto-resolves-to-class_weight for
    CatBoost. Mirror of test_xgboost_binary_classification_parity_auto_with_
    correction (the symmetric coverage flagged by pr-test-analyzer on PR #32).

    With IMBALANCED data (ratio >= 3:1), the codegen's auto-resolution block
    in ``_render_imbalance_handling`` mutates IMBALANCE_METHOD to "class_weight".
    The CatBoost branch then injects ``model_params['auto_class_weights'] =
    'Balanced'`` via the runtime conditional — CatBoost-specific kwarg name;
    the codegen emits it as a balanced_kwarg in ``_render_model``.

    Param-split shape (DeepSeek 2026-05-07 review of PR #48): the in-process
    model is constructed with ``auto_class_weights='Balanced'`` added at the
    constructor call site, BUT the params dict passed through to
    ``_build_export_config`` and ``_build_metadata`` is the BARE base params
    (without ``auto_class_weights``). This makes the codegen's runtime
    conditional the ONLY source of the balanced kwarg in the exported
    script. If a future refactor deletes the conditional, the script's
    predictions diverge from the in-process model's and this test fails.

    Same expect_stdout_marker pin as the XGBoost / LightGBM auto-with-
    correction rows — the auto-resolution print emits "applying class_weight"
    regardless of which downstream kwarg the model uses, because the
    auto-resolution decision happens before the per-model dispatch."""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    base_params = dict(_CATBOOST_BASE_PARAMS)  # no auto_class_weights — bare
    _run_parity(
        model=CatBoostClassifier(**base_params, auto_class_weights="Balanced"),
        model_name="CatBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=base_params,
        imbalance_method="auto",
        tmp_path=tmp_path,
        expect_stdout_marker="applying class_weight",
    )


def test_catboost_multiclass_classification_parity(tmp_path):
    """Multi-class CatBoost — pins the multi-output predict_proba shape
    parity. CatBoost infers multi-class objective from the y label set.

    Was xfail in T-20b due to a codegen CV-pooling bug
    (``CatBoostClassifier.predict()`` returns shape ``(n, 1)`` for
    multiclass; the validation template's ``Counter(ndarray)`` call
    crashed before reaching the parity-test appendix). Fixed by
    scalarising at ``templates/validation.py``'s
    CROSS_VALIDATION_CLASSIFICATION_TEMPLATE; the xfail marker has
    been removed and this is now a regular passing parity test."""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    X_train, y_train, X_test = _make_multiclass_classification_data(n_classes=3)
    params = dict(_CATBOOST_BASE_PARAMS)
    _run_parity(
        model=CatBoostClassifier(**params),
        model_name="CatBoost",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# PLS-DA — Pipeline of PLS scores -> StandardScaler -> LogisticRegression (T-20c)
# ---------------------------------------------------------------------------


def test_pls_da_binary_classification_parity_no_imbalance(tmp_path):
    """T-20c: PLS-DA pipeline parity (PLS scores -> StandardScaler -> LR).

    PLS-DA is the canonical chemometrics classifier and the export path
    (``code_generator._render_pls_da_pipeline``) emits a three-step Pipeline:

        Pipeline([
            ('pls',    PLSTransformer(n_components=N, scale=False)),
            ('scaler', StandardScaler()),
            ('lr',     LogisticRegression(C=, solver=, max_iter=, random_state=42))
        ])

    The runtime builds the same Pipeline at ``search.py:373-387`` using
    ``spectral_predict.models.PLSTransformer``. The exported script defines
    its own minimal ``PLSTransformer`` inline (codegen lines 1314-1355) but
    the fit/transform behavior is equivalent for 1-D y (no ndim>2 branch),
    so predict_proba parity holds.

    Param-routing contract: the codegen splits prefixed keys via
    ``_split_pls_da_params`` — ``pls__*`` keys go to PLSTransformer kwargs,
    ``lr__*`` keys go to LogisticRegression. The in-process test must use
    the same prefixes in the params dict and instantiate the components
    with the same effective values.

    ``scale=False`` is mandatory for the PLS step (chemometrics convention,
    avoids double-scaling SNV-preprocessed spectra; matches the runtime's
    hardcoded ``PLSTransformer(scale=False)`` at ``search.py:373-387`` and
    ``models.py:244``). ``n_components=3`` keeps the PLS stage well below
    the rank ceiling for the synthetic 60-sample/80-feature data.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from spectral_predict.models import PLSTransformer

    X_train, y_train, X_test = _make_binary_classification_data()

    # Params mirror what the runtime emits: prefixed keys routed by
    # _split_pls_da_params. n_components=3 picks a rank well below the
    # 60-sample / 80-feature regime. lr__random_state is explicit (rather
    # than relying on _render_pls_da_pipeline's internal lr_defaults at
    # code_generator.py:1300) so the contract is documented at the test
    # site — if a future codegen change drops the internal random_state
    # default, this test still pins the deterministic-LR contract.
    params = {
        "pls__n_components": 3,
        "lr__C": 1.0,
        "lr__solver": "lbfgs",
        "lr__max_iter": 1000,
        "lr__random_state": 42,
    }

    # In-process Pipeline: same shape as what _render_pls_da_pipeline emits.
    # PLSTransformer defaults to max_iter=500, tol=1e-6, scale=False — the
    # codegen pops the same defaults from pls_params (code_generator.py:1293-1296).
    pls = PLSTransformer(n_components=3, max_iter=500, tol=1e-6, scale=False)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model = Pipeline(
        [("pls", pls), ("scaler", StandardScaler()), ("lr", lr)]
    )

    _run_parity(
        model=model,
        model_name="PLS-DA",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method=None,
        tmp_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# MLP — no-op imbalance marker (T-20c follow-up)
# ---------------------------------------------------------------------------


def test_mlp_binary_classification_imbalance_is_no_op_marker(tmp_path):
    """Marker: MLP + ``imbalance_method='class_weight'`` MUST be a no-op.

    MLPClassifier accepts neither ``class_weight`` (TypeError on
    ``__init__``) nor ``sample_weight`` at fit() under the pyproject's
    sklearn floor. The codegen at ``code_generator.py:918-923`` explicitly
    excludes MLP from class_weight injection in the StandardScaler-wrapped
    path, mirroring the runtime fallback at ``search.py:4439-4444``
    (unweighted training, with a warning emitted to the user).

    This test pins the no-op behavior end-to-end via the T-20 parity
    contract. The in-process MLP has no imbalance kwargs (mirrors the
    runtime fallback). The codegen is asked for ``imbalance_method=
    'class_weight'``; if it ever starts injecting class_weight or
    sample_weight into the MLP path, the exported script's predict_proba
    will diverge from the saved model's and this test fails.

    Pairs with the static-source-check tests in
    ``tests/test_t19_class_weight_per_library.py`` (which assert
    ``'class_weight': 'balanced'`` is NOT in the generated script) — this
    parity test catches the same bug class via a different angle (numerical
    equivalence rather than source inspection), guarding against regressions
    that look correct in source but fit differently at runtime.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from spectral_predict.templates.models import DEFAULT_PARAMS

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)

    # Mirror the codegen's DEFAULT_PARAMS merge: copy the MLPClassifier
    # defaults, then override with our test-specific params (small network
    # + low max_iter to keep subprocess startup the bottleneck). The codegen
    # does the same in code_generator._render_scaled_pipeline.
    params = {"hidden_layer_sizes": (10,), "max_iter": 50, "random_state": 42}
    mlp_params_full = DEFAULT_PARAMS["MLPClassifier"].copy()
    mlp_params_full.update(params)

    # In-process model: no imbalance kwargs (mirrors runtime fallback).
    # StandardScaler wraps because MLP is in SCALE_SENSITIVE_MODELS
    # (search.py:113); the codegen wraps via _render_scaled_pipeline.
    mlp = MLPClassifier(**mlp_params_full)
    model = Pipeline([("scaler", StandardScaler()), ("model", mlp)])

    _run_parity(
        model=model,
        model_name="MLP",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method="class_weight",  # Marker: must be no-op for MLP.
        tmp_path=tmp_path,
        # Pin the user-facing warning. The codegen at code_generator.py:1510-1516
        # emits "[Imbalance] note: MLP does not support class_weight; ..." under
        # IMBALANCE_METHOD == 'class_weight' so users know imbalance was silently
        # dropped. If a future refactor silences this warning while keeping
        # numerical no-op behavior, parity would still pass but users would be
        # misled (they'd think balancing was applied).
        expect_stdout_marker="MLP does not support class_weight",
    )


# ---------------------------------------------------------------------------
# Structural contracts (PR #33 fix-of-fixes): n_jobs must not be re-added to
# either PIPELINE_PARAMS strip set. n_jobs is a model constructor kwarg, not
# a sklearn Pipeline kwarg, and the downstream setdefault('n_jobs', -1) at
# code_generator.py:988 silently overrides user determinism choices when the
# strip happens. GLM + Codex convergent finding (PR #33 cross-family review).
# ---------------------------------------------------------------------------


def test_codegen_pipeline_params_excludes_n_jobs():
    """Regression pin: CodeGenerator._PIPELINE_PARAMS must not strip n_jobs."""
    from spectral_predict.code_generator import CodeGenerator
    assert "n_jobs" not in CodeGenerator._PIPELINE_PARAMS, (
        "n_jobs is a model constructor kwarg (XGBoost / LightGBM / "
        "RandomForest / sklearn) and a determinism choice. Stripping it "
        "via _PIPELINE_PARAMS combined with the setdefault('n_jobs', -1) at "
        "code_generator.py:988 silently overrides user-set n_jobs in "
        "exported scripts."
    )


def test_unified_bayesian_pipeline_params_excludes_n_jobs():
    """Regression pin: unified_bayesian.PIPELINE_PARAMS must not strip n_jobs.

    Same bug class on the Bayesian sister site: _capture_serializable_params
    strips listed keys from model.get_params() before writing the captured
    params to the Optuna trial. Stripping n_jobs there means a Bayesian-
    trained model exported via the codegen path loses the user's
    determinism choice."""
    from spectral_predict.unified_bayesian import PIPELINE_PARAMS
    assert "n_jobs" not in PIPELINE_PARAMS, (
        "Sister site of the codegen _PIPELINE_PARAMS bug — same architectural "
        "smell, same overwrite hazard via the codegen export path."
    )


# ---------------------------------------------------------------------------
# PR #33 deferred HIGHs: behavioral + architectural pins on top of the
# structural ones above. Structural pins assert "n_jobs not in
# {_,}PIPELINE_PARAMS"; behavioral pin asserts "user-set n_jobs survives
# end-to-end into the generated script content"; architectural pin asserts
# "no third PIPELINE_PARAMS-named set in the codebase contains n_jobs."
#
# Why all three layers: a refactor that moves the bug to a different
# mechanism (e.g. a new strip path or a setdefault tweak that overrides
# user-set values) would pass the existing structural pins but fail the
# behavioral pin. A new sister site being silently born (a third
# *PIPELINE_PARAMS* set elsewhere in the codebase) would pass both
# existing pins but fail the architectural pin.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_name",
    ["XGBoost", "RandomForest", "LightGBM"],
)
def test_user_set_n_jobs_survives_codegen(model_name, tmp_path):
    """Behavioral pin (PR #33 deferred HIGH #1): user-set n_jobs=1 must
    appear literally in the generated script content for all three of
    XGBoost, RandomForest, and LightGBM. The structural pins above prove
    n_jobs isn't stripped via the named PIPELINE_PARAMS sets — but a
    refactor moving the bug to a different mechanism (e.g. an unconditional
    setdefault, a new strip path, or a model-specific gate inside
    `_render_model` like ``if model_class.startswith('LightGBM'):
    params_full.pop('n_jobs', None)``) would still pass those pins while
    silently overwriting the user's determinism choice in the exported
    script.

    This test catches the runtime-vs-export drift end-to-end by inspecting
    the script content itself, not just the input params or the prediction
    output (the existing parity matrix passes even when n_jobs is dropped
    because tree determinism doesn't always depend on n_jobs at runtime
    with random_state fixed).

    Coverage notes (DeepSeek 2026-05-07 review of PR #47):

    - All three parametrized models hit the ``else`` branch of
      ``_render_model``. Coverage of that branch's model-specific gates is
      complete for the n_jobs-accepting model set: each row catches gates
      targeting that specific model name.
    - **CatBoost is NOT included** because CatBoost uses ``thread_count``
      rather than ``n_jobs``. A separate behavioural pin would be needed
      for ``thread_count`` survival (deferred — file as a follow-up if a
      thread_count strip ever surfaces).
    - **One-class branch (IsolationForest, LocalOutlierFactor) is NOT
      covered** because the parity infrastructure (``_run_parity``,
      ``_build_export_config``) does not currently support
      ``task_type='one_class'``. Building that out is out-of-scope for
      this pin; deferred as infrastructure work.
    - **StandardScaler-pipeline branch (SVR/SVC/MLP/Ridge/Lasso/
      ElasticNet) is NOT covered** because none of those models accept
      ``n_jobs`` — the bug class doesn't apply there.
    - **PLS-DA branch is NOT covered** for the same reason (PLS doesn't
      accept ``n_jobs``)."""
    n_vars = 20
    params = {
        "n_estimators": 30,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": 1,
    }

    config = _build_export_config(
        model_name, "regression", n_vars, params, imbalance_method=None
    )
    options = ExportOptions(
        format="script",
        include_data=False,
        include_visualization=False,
        include_prediction_template=False,
    )
    script = CodeGenerator(config, options).generate_script()

    # Accept any quoting style that Python may produce when rendering the
    # params dict (single quotes, double quotes, kwarg form). The bug shape
    # is "n_jobs gets silently rewritten to -1"; pinning that the literal
    # "1" appears in proximity to "n_jobs" is sufficient.
    n_jobs_present = (
        "'n_jobs': 1" in script
        or '"n_jobs": 1' in script
        or "n_jobs=1" in script
    )
    assert n_jobs_present, (
        f"PR #33 regression: user-set n_jobs=1 was lost during codegen for "
        f"{model_name}. Search the generated script for 'n_jobs' to see what "
        f"it ended up as. The structural pins above only catch direct strip-"
        f"set membership; this test catches refactors that move the bug to "
        f"a different mechanism. Script length: {len(script)} chars."
    )

    # Negative pin: if a future refactor switches the setdefault default
    # back to -1 AND drops the user's value, we'd see 'n_jobs': -1 in the
    # script. The positive pin above catches this implicitly (1 != -1) but
    # an explicit negative makes the failure mode obvious to readers.
    assert "'n_jobs': -1" not in script and "n_jobs=-1" not in script, (
        f"PR #33 regression: generated script contains n_jobs=-1 for "
        f"{model_name} despite the user passing n_jobs=1. The setdefault-"
        f"overrides-user-value bug shape is back."
    )


def test_no_codebase_pipeline_params_set_contains_n_jobs():
    """Architectural pin (PR #33 deferred HIGH #2): walk the entire src/
    tree and verify no module-level or class-level *PIPELINE_PARAMS* set
    contains 'n_jobs'. Catches a future third sister site being silently
    born — the existing two structural pins (CodeGenerator._PIPELINE_PARAMS
    and unified_bayesian.PIPELINE_PARAMS) only cover the two known sites.

    The bug class is "code that pre-strips kwargs assumed to be Pipeline-
    only includes a model-constructor kwarg in its strip set." This pattern
    has surfaced twice in the dasp codebase (both fixed by PR #33) and the
    name *PIPELINE_PARAMS* is the calling convention; future contributors
    introducing a third set under the same name should not be able to
    accidentally re-introduce the n_jobs strip."""
    import re

    src_dir = Path(__file__).parent.parent / "src" / "spectral_predict"
    assert src_dir.exists(), f"src dir not found at {src_dir}"

    # Match `[_]PIPELINE_PARAMS = {...}` or `PIPELINE_PARAMS = frozenset({...})`
    # spanning multi-line set literals. The {...} group captures contents
    # for membership inspection.
    #
    # Known limitations (worth documenting so a future contributor doesn't
    # silently bypass via a less-common syntax form):
    # - Does NOT match `frozenset(["n_jobs", ...])` (list-input form).
    # - Does NOT match `set(["n_jobs", ...])` or `{*, *}` set unions.
    # - Does NOT match `dict.fromkeys([...])` or `globals()[...] = ...`.
    # If a sister PIPELINE_PARAMS-equivalent set ever lands in one of those
    # forms, the regex misses it and the architectural pin silently passes.
    # The naming-convention assumption (``*PIPELINE_PARAMS*`` literal) is
    # the load-bearing piece — extend the pattern if conventions diverge.
    pattern = re.compile(
        r"[A-Za-z_]*PIPELINE_PARAMS\b\s*=\s*"
        r"(?:frozenset\s*\(\s*)?"
        r"\{(?P<contents>[^}]*)\}",
        re.MULTILINE | re.DOTALL,
    )

    offenders: list[tuple[str, str]] = []
    found_any = False
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            found_any = True
            contents = match.group("contents")
            if "n_jobs" in contents:
                offenders.append(
                    (
                        str(py_file.relative_to(src_dir.parent.parent)),
                        contents.strip().replace("\n", " "),
                    )
                )

    # Sanity: regex must have caught the two known sites — if the count
    # drops to zero, the regex is wrong (silent test bypass), not the
    # codebase suddenly free of PIPELINE_PARAMS sets.
    assert found_any, (
        "Architectural pin regex matched zero PIPELINE_PARAMS sets in src/. "
        "Either both known sites (code_generator._PIPELINE_PARAMS, "
        "unified_bayesian.PIPELINE_PARAMS) were renamed or the regex broke. "
        "Check the regex pattern and update if the naming convention changed."
    )

    assert not offenders, (
        "PR #33 regression: a *PIPELINE_PARAMS* set in the codebase contains "
        "'n_jobs'. n_jobs is a model constructor kwarg (XGBoost / LightGBM / "
        "RandomForest / sklearn) and a determinism choice — stripping it "
        "combined with the downstream setdefault('n_jobs', -1) silently "
        "overrides user-set values in the exported script. Offenders:\n"
        + "\n".join(f"  - {f}: {{ {c} }}" for f, c in offenders)
    )
