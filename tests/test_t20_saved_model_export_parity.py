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
    if imbalanced:
        # ~80/20 split so class_weight='balanced' has something to do.
        y_train = (rng.uniform(size=n_train) > 0.8).astype(int)
        if y_train.sum() == 0:
            y_train[0] = 1
        if y_train.sum() == n_train:
            y_train[0] = 0
    else:
        y_train = (rng.uniform(size=n_train) > 0.5).astype(int)
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
) -> None:
    """Run save→export→exec→compare parity for one (model, config) row.

    For classification, predictions are compared via ``predict_proba`` (full
    probability vectors), not ``predict`` (hard labels). Hard labels mask
    probability drift up to the nearest class boundary — DeepSeek's T-20
    review empirically demonstrated 0.296 probability drift with identical
    hard labels. ``predict_proba`` makes the parity assertion sensitive to
    the kind of regressions T-19/T-32 actually fixed.

    For regression, ``predict`` is the right comparison (continuous output,
    no quantisation).

    ``fit_kwargs`` is forwarded to the in-process ``model.fit(...)`` call.
    For XGBoost + ``class_weight`` (or auto-resolves-to-class_weight), pass
    ``{'sample_weight': compute_sample_weight('balanced', y_train)}`` to
    match what the codegen emits for the runtime fit (templates/validation
    + code_generator._render_final_model_with_imbalance:1880-1886).
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
    # spectral_predict/templates/validation.py:226: ``model.fit(X_final, y)``).
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
    # that would defeat the parity assertion.
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"exported script failed (returncode={result.returncode})\n"
        f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
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
    merge injects ``scale=False`` (templates/models.py:280) because the
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
    params = {"n_estimators": 30, "max_depth": 6, "random_state": 42}
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
    params = {"n_estimators": 30, "max_depth": 6, "random_state": 42}
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
    runtime did."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, y_train, X_test = _make_binary_classification_data(imbalanced=True)
    params = {
        "n_estimators": 30,
        "max_depth": 6,
        "random_state": 42,
        "class_weight": "balanced",
    }
    _run_parity(
        model=RandomForestClassifier(**params),
        model_name="RandomForest",
        task_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        params=params,
        imbalance_method="class_weight",
        tmp_path=tmp_path,
    )


def test_random_forest_multiclass_classification_parity(tmp_path):
    """Multi-class baseline — verifies the parity contract holds for
    multi-output prediction (classes > 2)."""
    from sklearn.ensemble import RandomForestClassifier

    X_train, y_train, X_test = _make_multiclass_classification_data(n_classes=3)
    params = {"n_estimators": 30, "max_depth": 6, "random_state": 42}
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
# (code_generator.py:962-964, 987-988), so test params explicitly setting
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

    The codegen path emits ``model.fit(X, y, sample_weight=
    compute_sample_weight('balanced', y))`` for XGBoost +
    imbalance_method='class_weight'
    (code_generator._render_final_model_with_imbalance:1880-1886). The
    in-process model must apply the same sample_weight at fit time;
    using ``scale_pos_weight`` instead would mean the two paths use
    different weighting mechanisms — the test would only pass on
    datasets where both produce trivially identical predictions
    (DeepSeek HIGH-2 from T-20 review)."""
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
    (code_generator._render_imbalance_handling:1518-1548) mutates
    IMBALANCE_METHOD to None. Downstream gates do not fire; the script
    fits without sample_weight. The in-process model must do the same.

    Originally this test used imbalanced data so auto resolved to
    class_weight, which made the test mechanically identical to
    ``test_xgboost_binary_classification_parity_class_weight`` (DeepSeek
    HIGH-3 from T-20 review). Reworked to balanced data so the auto path
    is now a genuinely distinct code path."""
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
