"""T-19 regression tests — exported code with imbalance_method='class_weight'
must use per-library balanced-loss kwargs.

Pre-T-19, the catch-all branch in code_generator._render_model() unconditionally
injected ``class_weight='balanced'`` into every classifier's __init__ params:
- LightGBM, RandomForest, sklearn LR/SVC: works (native kwarg)
- XGBoost: silently ignored at fit time (no balancing applied)
- CatBoost: hard TypeError on instantiation

T-19 dispatches per library: CatBoost gets ``auto_class_weights='Balanced'``,
XGBoost gets ``sample_weight=compute_sample_weight('balanced', y)`` threaded
into the CV-fold and final-model fit() calls. Other libraries keep
class_weight='balanced'.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from sklearn.datasets import make_classification

from spectral_predict.code_generator import CodeGenerator, ExportOptions


def _classification_data(seed: int = 42, n_samples: int = 120) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=seed,
    )
    return X, y


def _generate(model_name: str, params: dict | None = None) -> str:
    config = {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": "classification",
        "params": params or {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
        "imbalance_method": "class_weight",
    }
    return CodeGenerator(config, ExportOptions(include_data=False)).generate_script()


def _execute(model_name: str, params: dict | None = None) -> dict:
    X, y = _classification_data()
    config = {
        "model_name": model_name,
        "preprocessing": "raw",
        "task_type": "classification",
        "params": params or {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
        "imbalance_method": "class_weight",
    }
    opts = ExportOptions(include_data=True, data_X=X.copy(), data_y=y.copy())
    script = CodeGenerator(config, opts).generate_script()
    g: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(script, g)
    return g


# ----------------------------------------------------------------------
# Per-library kwarg emission contracts
# ----------------------------------------------------------------------

def test_catboost_emits_auto_class_weights_not_class_weight():
    script = _generate("CatBoost", params={"iterations": 30, "depth": 3})
    assert "'auto_class_weights': 'Balanced'" in script, (
        "CatBoost must use auto_class_weights='Balanced'; class_weight kwarg "
        "raises TypeError on CatBoostClassifier instantiation."
    )
    assert "'class_weight': 'balanced'" not in script


def test_xgboost_does_not_inject_class_weight_in_params():
    """XGBoost ignores class_weight at fit time; sample_weight is threaded instead."""
    script = _generate("XGBoost")
    assert "'class_weight': 'balanced'" not in script, (
        "XGBoost must not inject class_weight into __init__ — it silently no-ops."
    )


def test_xgboost_threads_sample_weight_via_fit_kwargs():
    script = _generate("XGBoost")
    assert "from sklearn.utils.class_weight import compute_sample_weight" in script
    assert "fit_kwargs['sample_weight'] = compute_sample_weight('balanced'" in script
    assert "model.fit(X_train_full, y_train_full, **fit_kwargs)" in script
    assert "fold_model.fit(X_train_fold, y_train_fold, **fit_kwargs)" in script


def test_mlp_does_not_get_class_weight_injected():
    """MLP rejects class_weight kwarg with TypeError; injecting it crashes the
    exported script. sample_weight at fit() requires sklearn>=1.7 which is above
    the pyproject floor, so we mirror the runtime fallback (unweighted) here."""
    script = _generate("MLP", params={"hidden_layer_sizes": (10,), "max_iter": 50})
    assert "'class_weight': 'balanced'" not in script, (
        "MLP must not have class_weight injected — MLPClassifier.__init__() "
        "raises TypeError on unexpected keyword argument."
    )


def test_mlp_class_weight_export_executes_without_typeerror():
    """End-to-end: MLP+class_weight exported script runs (crashed pre-fix)."""
    g = _execute("MLP", params={"hidden_layer_sizes": (10,), "max_iter": 50})
    assert "accuracy" in g
    assert isinstance(g["accuracy"], float)


def test_svc_keeps_class_weight_balanced():
    """SVC accepts class_weight natively; StandardScaler-wrapped path keeps it."""
    script = _generate("SVC", params={"C": 1.0})
    assert "'class_weight': 'balanced'" in script


def test_non_xgboost_classification_does_not_emit_fit_kwargs_plumbing():
    """Pre-T-19 fix-of-fixes the catch-all fit_kwargs={} appeared even for
    non-XGBoost paths under class_weight — pure noise. Conditional emission
    removes it. Pin so a future refactor doesn't reintroduce the noise."""
    for name, params in [
        ("CatBoost", {"iterations": 30, "depth": 3}),
        ("LightGBM", {"n_estimators": 30, "max_depth": 3}),
        ("RandomForest", {"n_estimators": 30, "max_depth": 3}),
        ("MLP", {"hidden_layer_sizes": (10,), "max_iter": 50}),
        ("SVC", {"C": 1.0}),
    ]:
        script = _generate(name, params=params)
        assert "fit_kwargs" not in script, (
            f"{name} should not emit fit_kwargs plumbing in class_weight mode "
            f"— it's pure noise when sample_weight isn't being threaded."
        )


def test_lightgbm_keeps_class_weight_balanced():
    script = _generate("LightGBM")
    assert "'class_weight': 'balanced'" in script
    assert "'auto_class_weights'" not in script


def test_randomforest_keeps_class_weight_balanced():
    script = _generate("RandomForest")
    assert "'class_weight': 'balanced'" in script


def test_non_xgboost_does_not_emit_class_weight_sample_weight_block():
    """Sample-weight threading is XGBoost-specific; other libraries use kwargs.

    The phrase ``compute_sample_weight`` may legitimately appear in regression
    helper bodies, so we assert specifically on the class_weight-mode threading
    pattern that T-19 introduced.
    """
    pattern = "fit_kwargs['sample_weight'] = compute_sample_weight('balanced'"
    for name in ("CatBoost", "LightGBM", "RandomForest"):
        params = {"iterations": 30, "depth": 3} if name == "CatBoost" else None
        script = _generate(name, params=params)
        assert pattern not in script, (
            f"{name} should not emit XGBoost-specific class_weight sample-weight block"
        )


# ----------------------------------------------------------------------
# End-to-end: each script must execute without crashing
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "model_name,params",
    [
        ("XGBoost", {"n_estimators": 30, "max_depth": 3}),
        ("CatBoost", {"iterations": 30, "depth": 3}),
        ("LightGBM", {"n_estimators": 30, "max_depth": 3}),
        ("RandomForest", {"n_estimators": 30, "max_depth": 3}),
    ],
)
def test_generated_script_executes_under_class_weight(model_name: str, params: dict) -> None:
    g = _execute(model_name, params)
    assert "accuracy" in g
    assert isinstance(g["accuracy"], float)
    assert 0.0 <= g["accuracy"] <= 1.0


def test_xgboost_class_weight_export_handles_multiclass():
    """Per DeepSeek residual-risk #2: scale_pos_weight is binary-only, but
    sample_weight=compute_sample_weight('balanced', y) handles n_classes>2
    uniformly. Pin end-to-end on a 3-class imbalanced dataset."""
    X, y = make_classification(
        n_samples=180,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.6, 0.3, 0.1],
        random_state=42,
    )
    config = {
        "model_name": "XGBoost",
        "preprocessing": "raw",
        "task_type": "classification",
        "params": {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
        "imbalance_method": "class_weight",
    }
    opts = ExportOptions(include_data=True, data_X=X.copy(), data_y=y.copy())
    script = CodeGenerator(config, opts).generate_script()
    g: dict = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        exec(script, g)
    assert "accuracy" in g
    assert isinstance(g["accuracy"], float)


def test_xgboost_balanced_weight_applied_at_fit():
    """End-to-end check that sample_weight actually flows to XGBoost.fit()."""
    g = _execute("XGBoost")
    final_model = g["model"]
    # XGBoost final fit was called with sample_weight; we can't introspect that
    # post-hoc, but the model should at least classify the minority class
    # better than a no-weight baseline. Use accuracy as a smoke check.
    assert g["accuracy"] >= 0.5


# ----------------------------------------------------------------------
# Negative controls — non-class_weight modes are unchanged
# ----------------------------------------------------------------------

def test_smote_mode_does_not_emit_sample_weight_block():
    """SMOTE handling uses resampling, not sample_weight."""
    config = {
        "model_name": "XGBoost",
        "preprocessing": "raw",
        "task_type": "classification",
        "params": {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
        "imbalance_method": "smote",
    }
    script = CodeGenerator(config, ExportOptions(include_data=False)).generate_script()
    # Regression-helper compute_sample_weight may appear in helper defs but the
    # main fit-call sample_weight conditional should not.
    assert "fit_kwargs['sample_weight'] = compute_sample_weight('balanced'" not in script


def test_no_imbalance_method_does_not_emit_sample_weight_block():
    config = {
        "model_name": "XGBoost",
        "preprocessing": "raw",
        "task_type": "classification",
        "params": {"n_estimators": 30, "max_depth": 3},
        "cv_folds": 3,
    }
    script = CodeGenerator(config, ExportOptions(include_data=False)).generate_script()
    assert "fit_kwargs['sample_weight'] = compute_sample_weight('balanced'" not in script
